"""Observation handling: batch splitting, image preparation, proprio-to-state helpers, hold action, fingerprints.

Input format (see b1k26.constants for key names): the evaluator sends one flattened dict per query.
- v3.9.3-post1/post2 (single port, possibly batched over N logical environments): every value has a leading
  batch dim N, e.g. proprio (N, 61) float32, RGB (N, H, W, 4) uint8, task_id (N, 1) int64.
- 2026/eval multi-port (--policy-endpoints): one connection per environment, values unsqueezed to N = 1.
- v3.9.2: unbatched values, proprio (61,), RGB (H, W, 4), task_id (1,).
Arrays decoded by b1k26.protocol are read-only views over the message bytes; nothing here writes into them.
By default every array placed in an EnvObs is a fresh, writeable copy. The front server passes
``copy_images=False``: uint8 camera images and depth maps then stay read-only views of the message (no per-step
copy of the ~6.8 MB of full-resolution images and depth); images are only copied, resized and padded when a plan
needs them (``prepare_images``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image

from b1k26 import constants as C

CAMERA_ROLES: tuple[str, ...] = ("head", "left_wrist", "right_wrist")

# Fingerprint image subsampling: about this many rows/cols per image, taken with a fixed stride.
_FP_GRID = 37


@dataclass
class EnvObs:
    task_id: int  # from obs["task_id"] (shape (N,1) int64 on v3.9.3+, may be (1,) or scalar)
    proprio: np.ndarray  # (61,) float32
    rgb: dict[str, np.ndarray]  # role -> (H, W, 3) uint8; only roles present in the message ("head" always);
    #                             read-only views of the message with split_batch(copy_images=False)
    depth: dict[str, np.ndarray] = field(default_factory=dict)  # role -> (H, W) float32, empty if not sent
    cam_rel_poses: np.ndarray | None = None  # (21,) float32
    fingerprint: bytes = b""  # cheap hash of proprio + task_id + strided image bytes


# ----------------------------------------------------------------------------------------------------------
# Batch splitting
# ----------------------------------------------------------------------------------------------------------
def _as_array(value: Any, key: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):  # torch tensor (local test clients)
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"obs[{key!r}] has unsupported dtype {arr.dtype}")
    return arr


def _compact_copy(img: np.ndarray) -> np.ndarray:
    """New C-contiguous copy of an (..., C) image. Copying channel by channel is ~4x faster than
    ``np.ascontiguousarray`` for the strided RGB view of an RGBA buffer (0.4 vs 1.8 ms at 720x720)."""
    if img.flags.c_contiguous:
        return img.copy()
    out = np.empty(img.shape, dtype=img.dtype)
    for c in range(img.shape[-1]):
        out[..., c] = img[..., c]
    return out


def _to_rgb_uint8(img: np.ndarray, key: str, copy: bool = True) -> np.ndarray:
    """(H, W, C) with C in {3, 4} (or (H, W) gray) -> (H, W, 3) uint8.

    Returns a new contiguous array, except for uint8 input with ``copy=False``: then the result is a (strided,
    possibly read-only) view that drops the alpha channel without copying.
    """
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3 or img.shape[-1] not in (1, 3, 4):
        raise ValueError(f"obs[{key!r}]: expected an (H, W, 3|4) image per env, got shape {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = img[..., :3]  # drop alpha (RGBA from the simulator)
    if img.dtype == np.uint8:
        return _compact_copy(img) if copy else img
    if img.dtype.kind == "f":
        x = np.nan_to_num(img.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        # [0, 1] floats are the documented case; a float image with values clearly above 1 is taken as 0-255.
        if x.size and float(x.max()) > 1.5:
            x = x / 255.0
        # Same conversion as openpi_client.image_tools.convert_to_uint8 (truncation), after clipping.
        return np.ascontiguousarray((255 * np.clip(x, 0.0, 1.0)).astype(np.uint8))
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def _parse_task_ids(value: Any, n: int, batched: bool) -> list[int]:
    arr = _as_array(value, C.TASK_ID_KEY)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
            raise ValueError(f"obs['task_id'] is not integral: {arr!r}")
    flat = arr.reshape(-1)
    if flat.size == n:
        ids = [int(v) for v in flat]
    elif flat.size == 1:
        ids = [int(flat[0])] * n
    else:
        raise ValueError(f"obs['task_id'] has shape {arr.shape}, incompatible with batch size {n} "
                         f"({'batched' if batched else 'unbatched'} message)")
    for t in ids:
        if not 0 <= t < C.NUM_TASKS:
            raise ValueError(f"obs['task_id'] = {t} is outside [0, {C.NUM_TASKS})")
    return ids


def split_batch(msg: Mapping[str, Any], default_task_id: int | None = None, copy_images: bool = True
                ) -> list[EnvObs]:
    """Split the evaluator's flattened obs dict into one EnvObs per environment.

    Batched (v3.9.3+, 2026/eval multi-port with N=1) and unbatched (v3.9.2, detected by proprio.ndim == 1)
    messages are both accepted. Unknown keys are ignored. Raises ValueError with a clear message when proprio
    or the head RGB image is missing or malformed, or when task_id is missing and no ``default_task_id`` is
    given. Never mutates the input arrays.

    ``copy_images=False`` keeps uint8 RGB images and float32 depth maps as views of the input arrays (read-only
    when the input came from b1k26.protocol) instead of copying them; values, shapes and fingerprints are
    identical either way. proprio and cam_rel_poses are always copied (they are tiny).
    """
    if C.PROPRIO_KEY not in msg:
        raise ValueError(f"observation is missing {C.PROPRIO_KEY!r} (keys: {sorted(map(str, msg))[:12]})")
    proprio = _as_array(msg[C.PROPRIO_KEY], C.PROPRIO_KEY)
    if proprio.ndim == 1:
        batched = False
        proprio = proprio[None]
    elif proprio.ndim == 2:
        batched = True
    else:
        raise ValueError(f"{C.PROPRIO_KEY} must be (61,) or (N, 61), got shape {proprio.shape}")
    if proprio.shape[-1] != C.PROPRIO_DIM:
        raise ValueError(f"{C.PROPRIO_KEY} last dim must be {C.PROPRIO_DIM}, got shape {proprio.shape}")
    n = proprio.shape[0]
    if n < 1:
        raise ValueError(f"{C.PROPRIO_KEY} has an empty batch")

    if C.TASK_ID_KEY in msg:
        task_ids = _parse_task_ids(msg[C.TASK_ID_KEY], n, batched)
    elif default_task_id is not None:
        task_ids = [int(default_task_id)] * n
    else:
        raise ValueError("observation is missing 'task_id' and no default_task_id was given")

    def per_env(key: str, value: Any, per_env_ndim: tuple[int, ...]) -> np.ndarray:
        """Return value with a leading batch dim of size n; per_env_ndim lists the accepted unbatched ndims."""
        arr = _as_array(value, key)
        if batched:
            if arr.ndim - 1 not in per_env_ndim or arr.shape[0] != n:
                raise ValueError(f"obs[{key!r}] has shape {arr.shape}; expected batch dim {n} followed by "
                                 f"{'/'.join(map(str, per_env_ndim))} dims")
            return arr
        if arr.ndim in per_env_ndim:
            return arr[None]
        if arr.ndim - 1 in per_env_ndim and arr.shape[0] == 1:  # unbatched proprio but batched value: tolerate
            return arr
        raise ValueError(f"obs[{key!r}] has shape {arr.shape}; expected {'/'.join(map(str, per_env_ndim))} dims")

    images: dict[str, np.ndarray] = {}
    for role in CAMERA_ROLES:
        key = C.rgb_key(role)
        if key in msg and msg[key] is not None:
            images[role] = per_env(key, msg[key], (3, 2))
        elif role == "head":
            raise ValueError(f"observation is missing the head camera image {key!r}")
    depths: dict[str, np.ndarray] = {}
    for role in CAMERA_ROLES:
        key = C.depth_key(role)
        if key in msg and msg[key] is not None:
            depths[role] = per_env(key, msg[key], (2, 3))
    cam = None
    if C.CAM_REL_POSES_KEY in msg and msg[C.CAM_REL_POSES_KEY] is not None:
        cam = per_env(C.CAM_REL_POSES_KEY, msg[C.CAM_REL_POSES_KEY], (1,))

    out: list[EnvObs] = []
    for b in range(n):
        p = np.array(proprio[b], dtype=np.float32, copy=True)
        rgb = {role: _to_rgb_uint8(arr[b], C.rgb_key(role), copy=copy_images) for role, arr in images.items()}
        depth = {}
        for role, arr in depths.items():
            d = arr[b]
            if d.ndim == 3:
                if d.shape[-1] != 1:
                    raise ValueError(f"depth for {role} must be (H, W) or (H, W, 1), got {d.shape}")
                d = d[..., 0]
            if copy_images:
                depth[role] = np.array(d, dtype=np.float32, copy=True)
            else:
                depth[role] = d if d.dtype == np.float32 else d.astype(np.float32)
        cam_b = None if cam is None else np.array(cam[b], dtype=np.float32, copy=True).reshape(-1)
        env = EnvObs(task_id=task_ids[b], proprio=p, rgb=rgb, depth=depth, cam_rel_poses=cam_b)
        env.fingerprint = compute_fingerprint(env.task_id, env.proprio, env.rgb)
        out.append(env)
    return out


def compute_fingerprint(task_id: int, proprio: np.ndarray, rgb: Mapping[str, np.ndarray]) -> bytes:
    """16-byte blake2b over proprio bytes, task_id and a fixed strided subsample of each RGB image.

    Costs well under 1 ms for full-resolution (720x720 + 2x480x480) inputs. Two observations with equal
    fingerprints are treated as the same observation (reconnect replay).
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(proprio, dtype=np.float32).tobytes())
    h.update(int(task_id).to_bytes(8, "little", signed=True))
    for role in sorted(rgb):
        img = rgb[role]
        sh = max(1, img.shape[0] // _FP_GRID)
        sw = max(1, img.shape[1] // _FP_GRID)
        h.update(role.encode())
        h.update(np.asarray(img.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(img[::sh, ::sw]).tobytes())
    return h.digest()


# ----------------------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------------------
_RESAMPLE = {
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
}


def _resample(method: str | int) -> int:
    if isinstance(method, (int, np.integer)):
        return int(method)
    m = method.lower()
    if m.endswith("_pad"):
        m = m[: -len("_pad")]
    if m not in _RESAMPLE:
        raise ValueError(f"unknown resize method {method!r}; known: {sorted(_RESAMPLE)}")
    return _RESAMPLE[m]


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    # Line-by-line port of openpi_client.image_tools._resize_with_pad_pil.
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_with_pad(img: np.ndarray, height: int, width: int, method: str | int = "bilinear") -> np.ndarray:
    """Bit-exact port of openpi_client.image_tools.resize_with_pad (PIL; centered zero padding).

    Accepts (..., H, W, C) uint8. Like the original, returns the input unchanged (same object) when it is
    already (height, width). ``method``: "bilinear" (default, = Image.BILINEAR), "nearest", "lanczos",
    "bicubic" (a "_pad" suffix such as "bilinear_pad" is accepted), or a PIL resampling constant.
    """
    resample = _resample(method)
    if img.shape[-3:-1] == (height, width):
        return img
    original_shape = img.shape
    images = img.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [np.asarray(_resize_with_pad_pil(Image.fromarray(im), height, width, method=resample)) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def prepare_images(
    env: EnvObs, size: int, method: str = "bilinear", roles: tuple[str, ...] = CAMERA_ROLES
) -> dict[str, np.ndarray]:
    """Resize-with-pad every camera of ``env`` to (size, size, 3) uint8.

    Roles missing from the observation (wrist cameras with a custom env wrapper) are filled with zeros so a
    backend always receives every role in ``roles``. Returned arrays are C-contiguous and never alias ``env.rgb``
    (which may hold strided read-only views, see ``split_batch(copy_images=False)``).
    """
    out: dict[str, np.ndarray] = {}
    for role in roles:
        img = env.rgb.get(role)
        if img is None:
            out[role] = np.zeros((size, size, 3), dtype=np.uint8)
            continue
        src = img
        if not img.flags.c_contiguous:
            img = _compact_copy(img)  # one compact copy (drops the alpha stride) before PIL sees it
        r = resize_with_pad(img, size, size, method)
        # r is src only when src was already contiguous and already (size, size): copy so we never alias env.rgb.
        out[role] = np.array(r, dtype=np.uint8, copy=True) if r is src else np.ascontiguousarray(r, dtype=np.uint8)
    return out


# ----------------------------------------------------------------------------------------------------------
# Proprio -> state / hold
# ----------------------------------------------------------------------------------------------------------
_P = C.PROPRIO_INDICES_2026


def gripper_width_normalized(proprio: np.ndarray, side: str) -> np.ndarray:
    """2 * (finger_sum / 0.1) - 1 for side "left" or "right" (unclipped), shape (..., 1)."""
    fingers = np.asarray(proprio)[..., _P[f"gripper_{side}_qpos"]]
    return 2.0 * (fingers.sum(axis=-1, keepdims=True) / C.GRIPPER_MAX_WIDTH) - 1.0


def state23_action_order(proprio: np.ndarray) -> np.ndarray:
    """RLC extract_state_from_proprio on the 61-D proprio: (..., 23) float32.

    Order [base_qvel 3, trunk_qpos 4, L arm qpos 7, L grip, R arm qpos 7, R grip] where grip is the
    unclipped normalized width 2 * (finger_sum / 0.1) - 1. base_qvel is the measured robot-frame velocity.
    """
    p = np.asarray(proprio, dtype=np.float32)
    if p.shape[-1] != C.PROPRIO_DIM:
        raise ValueError(f"proprio must be (..., {C.PROPRIO_DIM}), got {p.shape}")
    return np.concatenate(
        [
            p[..., _P["base_qvel"]],
            p[..., _P["trunk_qpos"]],
            p[..., _P["arm_left_qpos"]],
            gripper_width_normalized(p, "left"),
            p[..., _P["arm_right_qpos"]],
            gripper_width_normalized(p, "right"),
        ],
        axis=-1,
    ).astype(np.float32)


def hold_from_state23(state23: np.ndarray) -> np.ndarray:
    """(23,) action that holds the pose described by a state23 vector: base 0, joints = state, grippers clipped.

    Non-finite entries become 0 so the result is always finite.
    """
    s = np.asarray(state23, dtype=np.float32).reshape(-1)
    if s.shape[0] != C.ACTION_DIM:
        raise ValueError(f"state23 must have {C.ACTION_DIM} entries, got {s.shape}")
    a = s.copy()
    a[C.ACTION_SLICES["base"]] = 0.0
    for idx in (C.LEFT_GRIPPER_ACTION_IDX, C.RIGHT_GRIPPER_ACTION_IDX):
        a[idx] = np.clip(a[idx], -1.0, 1.0)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def hold_action(proprio: np.ndarray) -> np.ndarray:
    """(23,) float32 that holds the current pose.

    Base velocity 0, torso/arms = current joint positions, grippers = normalized current width clipped to
    [-1, 1]. Never zeros: a zero torso command stands the robot upright. Non-finite proprio entries map to 0.
    """
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    return hold_from_state23(state23_action_order(p))
