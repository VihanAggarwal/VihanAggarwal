"""openpi-family worker backends: shared base class plus the organizers' ``wensi-ai/openpi@behavior`` fork.

This module hosts two things:

1. ``OpenPIBackendBase``: everything the openpi forks have in common (lazy import, policy creation through the
   fork's own ``create_trained_policy``, norm-stats resolution, flow-step override, optional float32 restore,
   optional true batching, output post-processing, warmup). ``b1k26.backends.openpi_comet.CometBackend`` builds
   on it too.
2. ``OpenPIB1KBackend``: the organizers' ``pi05_b1k`` config (wensi-ai/openpi branch ``behavior``, commit
   ``0cc8e35``) and checkpoints compatible with it: the provided ``turning_on_radio`` baseline and the
   ``Hoshipu/pi05-b1k100t-2026-*`` multi-task checkpoints.

The per-item input dict is exactly what the fork's ``B1KPolicyWrapper.process_input``
(``src/openpi/shared/eval_b1k_wrapper.py``) hands to ``policy.infer``::

    {"observation/image_0": head (224,224,3) uint8,
     "observation/image_1": left wrist, "observation/image_2": right wrist,
     "observation/state": raw 61-D proprio, "prompt": str}

The fork's own input transforms then extract the 23-D state (``b1k_policy.extract_state_from_proprio`` with
the ``b1k/R1Pro`` robot config: base_qvel, trunk, left arm, left finger-width sum, right arm, right finger-width
sum), normalize (z-score: ``LeRobotB1KDataConfig`` sets ``use_quantile_norm=False``), tokenize the prompt with
the discretized state (pi0.5), and the output transforms un-normalize, add the current state back to the
torso/arm dims (``MappedAbsoluteActions``, from ``extra_delta_transform=True``) and keep the first 23 dims.
The images arrive already resized (``resize_with_pad`` to 224) by the front server; the robot config's
``robot`` vs ``robot_r1`` name bug in the fork only affects its own wrapper, which we do not use.

Nothing here imports JAX or openpi at module import time. The fork is imported inside ``__init__`` (or never,
when a ``policy`` object is injected, which is how the CPU tests exercise this code).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import types
from typing import Any, Callable

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut, InferItem

logger = logging.getLogger(__name__)

# Pinned upstream sources (scripts/envs/openpi_b1k.sh installs exactly this commit).
OPENPI_B1K_REPO = "https://github.com/wensi-ai/openpi"
OPENPI_B1K_COMMIT = "0cc8e355f7bac0976db1cc3139b1ff0379feea60"  # branch "behavior" head, 2026-06-28

IMAGE_SIZE = 224
ROLES = ("head", "left_wrist", "right_wrist")
PROMPT_MODES = ("comet2025", "instruction", "snake_case")
DTYPES = ("bfloat16", "float32")
GRIPPER_STATE_MODES = ("auto", "width", "pm1")
MAX_BATCHED_FAILURES = 3  # consecutive batched-path errors before falling back to the loop for good

# Indices into the 61-D proprio (constants.PROPRIO_INDICES_2026), resolved once.
_P = C.PROPRIO_INDICES_2026
_LEFT_FINGERS = _P["gripper_left_qpos"]
_RIGHT_FINGERS = _P["gripper_right_qpos"]

# Robot config "b1k/R1Pro" camera keys -> our image roles (src/openpi/configs/robots/b1k.py: image_0 = head
# "zed_link", image_1 = left wrist, image_2 = right wrist; ObservationConfig.name equals our role names).
DEFAULT_B1K_CAMERA_KEYS = {"image_0": "head", "image_1": "left_wrist", "image_2": "right_wrist"}

# Gripper columns of the 23-D state in action order (b1k/R1Pro proprio order puts them at 14 and 22).
B1K_STATE_GRIPPER_COLUMNS = (14, 22)


# ------------------------------------------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested on CPU)
# ------------------------------------------------------------------------------------------------------------
def resolve_prompt(prompt: str | None, task_id: int, fallback_mode: str) -> str:
    """Return the prompt to send to the model.

    The front server resolves prompts per profile (``comet2025 | instruction | snake_case``) and the backend
    passes a non-empty ``prompt`` through verbatim. Only an empty/missing prompt falls back to the task table:
    ``comet2025`` = Comet's 2025 ``scripts/task_mapping.json`` text (tasks 0-49; the 2026 instruction for new
    tasks), ``instruction`` = 2026 natural-language instruction, ``snake_case`` = the task name (what LeRobot's
    ``meta/tasks.parquet`` and therefore ``prompt_from_task=True`` training used for the 2026 demos).
    """
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    if fallback_mode not in PROMPT_MODES:
        raise ValueError(f"unknown prompt mode {fallback_mode!r}; expected one of {PROMPT_MODES}")
    info = C.task(int(task_id))
    if fallback_mode == "snake_case":
        return info.name
    if fallback_mode == "comet2025" and info.instruction_comet2025:
        return info.instruction_comet2025
    return info.instruction


def b1k_state_from_proprio(proprio: np.ndarray, gripper_state: str = "width") -> np.ndarray:
    """23-D state the ``b1k/R1Pro`` robot config extracts, in action order.

    ``[base_qvel 3, trunk 4, left arm 7, left grip, right arm 7, right grip]``. ``gripper_state="width"`` is the
    fork's own extraction (finger sum in metres, ~[0, 0.1]); ``"pm1"`` is ``2 * width / 0.1 - 1`` (~[-1, 1]),
    which is what the Hoshipu 100-task norm stats imply. Reference implementation used for self-checks/tests.
    """
    p = np.asarray(proprio, dtype=np.float32)
    left = p[..., _LEFT_FINGERS].sum(axis=-1, keepdims=True)
    right = p[..., _RIGHT_FINGERS].sum(axis=-1, keepdims=True)
    if gripper_state == "pm1":
        left = 2.0 * (left / C.GRIPPER_MAX_WIDTH) - 1.0
        right = 2.0 * (right / C.GRIPPER_MAX_WIDTH) - 1.0
    elif gripper_state != "width":
        raise ValueError(f"gripper_state must be 'width' or 'pm1', got {gripper_state!r}")
    return np.concatenate(
        [p[..., _P["base_qvel"]], p[..., _P["trunk_qpos"]], p[..., _P["arm_left_qpos"]], left,
         p[..., _P["arm_right_qpos"]], right],
        axis=-1,
    ).astype(np.float32)


def apply_gripper_pm1(proprio: np.ndarray) -> np.ndarray:
    """Rewrite a copy of the 61-D proprio so that the fork's finger-sum extraction yields ``2*w/0.1 - 1``.

    The fork's ``B1KInputs`` sums ``proprio[24:26]`` / ``proprio[49:51]``; putting the rescaled width in the
    first finger slot and 0 in the second makes that sum equal the [-1, 1] gripper state without touching the
    fork. Only the finger qpos entries change; they feed nothing else in the input transforms.
    """
    p = np.array(proprio, dtype=np.float32, copy=True)
    for sl in (_LEFT_FINGERS, _RIGHT_FINGERS):
        width = p[sl].sum()
        p[sl.start] = 2.0 * (width / C.GRIPPER_MAX_WIDTH) - 1.0
        p[sl.start + 1 : sl.stop] = 0.0
    return p


def _stat(stats: Any, name: str) -> np.ndarray | None:
    value = stats.get(name) if isinstance(stats, dict) else getattr(stats, name, None)
    return None if value is None else np.asarray(value, dtype=np.float64)


def detect_gripper_state_mode(state_stats: Any, columns: tuple[int, ...] = B1K_STATE_GRIPPER_COLUMNS) -> str | None:
    """Infer ``"width"`` or ``"pm1"`` from the state norm stats of the gripper columns, or None if ambiguous.

    Width-based stats (radio baseline: q01 0, q99 0.1) and [-1, 1] stats (Hoshipu 100t: q01 ~-0.8, q99 1.0) are
    an order of magnitude apart, so the rule is simply the upper quantile (or mean + 3 std) of each column.
    """
    q99 = _stat(state_stats, "q99")
    if q99 is None:
        mean, std = _stat(state_stats, "mean"), _stat(state_stats, "std")
        if mean is None or std is None:
            return None
        q99 = mean + 3.0 * std
    if q99.shape[-1] <= max(columns):
        return None
    highs = [float(q99[c]) for c in columns]
    if all(h <= 0.2 for h in highs):
        return "width"
    if all(h >= 0.5 for h in highs):
        return "pm1"
    return None


def postprocess_actions(actions: Any, action_dim: int = C.ACTION_DIM) -> np.ndarray:
    """Model output -> ``(T, 23)`` float32 absolute evaluator actions (a fresh, contiguous array).

    ``policy.infer`` returns ``{"actions": (T, 23)}`` after the forks' output transforms (``B1K*Outputs`` already
    keeps 23 dims); a 32-wide or ``(1, T, A)`` array is accepted defensively. Non-finite values are reported but
    left in place: the front server's ``control.sanitize`` replaces them with hold actions.
    """
    if hasattr(actions, "detach"):  # torch tensor from a PyTorch checkpoint path
        actions = actions.detach().cpu().numpy()
    a = np.asarray(actions)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected (T, A) actions, got shape {a.shape}")
    if a.shape[0] < 1 or a.shape[1] < action_dim:
        raise ValueError(f"expected (T>=1, A>={action_dim}) actions, got shape {a.shape}")
    out = np.array(a[:, :action_dim], dtype=np.float32, copy=True)
    bad = int(np.count_nonzero(~np.isfinite(out)))
    if bad:
        logger.warning("policy returned %d non-finite action values (left for the front server to sanitize)", bad)
    return out


def validate_item(item: InferItem, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Check one worker item and return ``(proprio copy (61,) float32, {role: (S,S,3) uint8})``.

    Raises ValueError on a malformed item (missing camera, wrong channel count, wrong proprio size). Non-finite
    proprio values are zeroed with a warning: NaN would otherwise poison the discretized state tokens.
    """
    proprio = np.array(item.proprio, dtype=np.float32, copy=True).reshape(-1)
    if proprio.shape != (C.PROPRIO_DIM,):
        raise ValueError(f"proprio must have {C.PROPRIO_DIM} values, got shape {np.shape(item.proprio)}")
    if not np.all(np.isfinite(proprio)):
        logger.warning("non-finite proprio values for task %s; replacing them with 0", item.task_id)
        proprio = np.nan_to_num(proprio, nan=0.0, posinf=0.0, neginf=0.0)
    images: dict[str, np.ndarray] = {}
    for role in ROLES:
        if role not in item.images:
            raise ValueError(f"missing image role {role!r}; got {sorted(item.images)}")
        img = np.asarray(item.images[role])
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"image {role!r} must be (H, W, 3), got {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"image {role!r} must be uint8, got {img.dtype}")
        if img.shape[:2] != (image_size, image_size):
            # The model transforms resize_with_pad to 224 anyway, so this is not fatal, but it means the front
            # server's profile.image_size does not match this backend's info().
            logger.warning("image %r is %s, expected %dx%d", role, img.shape[:2], image_size, image_size)
        images[role] = np.ascontiguousarray(img)
    return proprio, images


def reset_pose_proprio() -> np.ndarray:
    """A plausible 61-D proprio at the R1Pro reset pose (eval/r1pro.yaml reset_joint_pos), for warmup."""
    p = np.zeros(C.PROPRIO_DIM, dtype=np.float32)
    p[_P["trunk_qpos"]] = (1.025, -1.45, -0.47, 0.0)
    p[_LEFT_FINGERS] = 0.05
    p[_RIGHT_FINGERS] = 0.05
    p[_P["eef_left_quat"].stop - 1] = 1.0  # xyzw identity; unused by the state extraction
    p[_P["eef_right_quat"].stop - 1] = 1.0
    return p


def bucket_sizes(max_batch: int) -> list[int]:
    """Batch sizes compiled by the batched path: powers of two below ``max_batch``, then ``max_batch``."""
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    sizes, b = [], 1
    while b < max_batch:
        sizes.append(b)
        b *= 2
    sizes.append(max_batch)
    return sizes


def bucket_for(n: int, max_batch: int) -> int:
    """Smallest compiled batch size that holds ``n`` items (``n <= max_batch``)."""
    for size in bucket_sizes(max_batch):
        if size >= n:
            return size
    raise ValueError(f"group of {n} exceeds max_batch {max_batch}")


def copy_tree(tree: Any) -> Any:
    """Copy the dict containers of a nested example (leaves shared), like ``jax.tree.map(lambda x: x, obs)``."""
    if isinstance(tree, dict):
        return {k: copy_tree(v) for k, v in tree.items()}
    return tree


def stack_trees(trees: list[Any]) -> Any:
    """Stack identically structured nested dicts of arrays along a new leading batch axis."""
    first = trees[0]
    if isinstance(first, dict):
        keys = list(first)
        for t in trees[1:]:
            if not isinstance(t, dict) or list(t) != keys:
                raise ValueError("cannot batch examples with different structures")
        return {k: stack_trees([t[k] for t in trees]) for k in keys}
    return np.stack([np.asarray(t) for t in trees])


def gpu_compute_capability(jax_module: Any | None = None) -> tuple[int, int] | None:
    """Compute capability of the first GPU, from JAX if possible, else ``nvidia-smi``; None if unknown."""
    if jax_module is not None:
        try:
            for dev in jax_module.devices():
                # VERIFY: jaxlib CUDA devices expose `compute_capability` as a string such as "8.6".
                cc = getattr(dev, "compute_capability", None)
                if cc:
                    major, minor = str(cc).split(".")[:2]
                    return int(major), int(minor)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        major, minor = out[0].strip().split(".")[:2]
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - older drivers do not know compute_cap
        return None


class _DataFactoryOverride:
    """Wraps a fork ``DataConfigFactory`` so ``create()`` returns a DataConfig with ``use_quantile_norm`` forced.

    ``create_trained_policy`` reads ``data_config.use_quantile_norm`` for Normalize/Unnormalize, and the B1K
    factories hard-code it in ``create()``, so the override has to sit on the factory. Every other attribute is
    delegated to the wrapped factory.
    """

    def __init__(self, inner: Any, use_quantile_norm: bool):
        self._inner = inner
        self._use_quantile_norm = bool(use_quantile_norm)

    def create(self, assets_dirs: Any, model_config: Any) -> Any:
        data_config = self._inner.create(assets_dirs, model_config)
        return dataclasses.replace(data_config, use_quantile_norm=self._use_quantile_norm)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ------------------------------------------------------------------------------------------------------------
# Shared base class
# ------------------------------------------------------------------------------------------------------------
class OpenPIBackendBase(Backend):
    """One openpi pi0/pi0.5 checkpoint served through its fork's own ``Policy`` and transforms.

    Constructor arguments (all keyword, JSON-serializable so the worker can pass them through):
      checkpoint         checkpoint dir (``params/`` + ``assets/``); gs:// works via openpi ``maybe_download``.
      config_name        the fork's TrainConfig name. Validated against ``_CONFIGS_DICT`` (the Comet fork's
                         ``get_config`` silently falls back to ``pi05_b1k-base`` for unknown names).
      asset_id           norm stats sub-dir under ``<checkpoint>/assets`` (default: the config's asset id).
      norm_stats_dir     explicit dir holding ``norm_stats.json``; relative paths are relative to the checkpoint.
                         Wins over ``asset_id`` (use ``"assets"`` for checkpoints that store it there directly).
      num_steps          flow-matching denoise steps (``sample_kwargs={"num_steps": n}``; model default 10).
      dtype              ``bfloat16`` (upstream ``create_trained_policy``, params restored as bf16) or
                         ``float32`` (same pipeline, params restored as f32 and ``Pi0Config.dtype="float32"``).
      action_horizon     override ``model.action_horizon`` (pi0.5 has no horizon-shaped params; only for
                         checkpoints whose training horizon differs from the config's).
      use_quantile_norm  force quantile (True) or z-score (False) normalization; None keeps the config's.
      batched            run a micro-batch as one padded ``sample_actions`` call instead of a loop.
      max_batch          largest compiled batch (buckets 1, 2, 4, ..., max_batch are compiled at warmup).
      default_prompt_mode prompt fallback when an item arrives without a prompt (see ``resolve_prompt``).
      mem_fraction       sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` before JAX is imported (None: leave as is).
      warmup_task_id     task used for the warmup dummy inference.
      policy             inject a ready policy object (``infer(dict) -> {"actions": ...}``); skips openpi.
    """

    flavor = "openpi"
    default_config_name = ""
    default_prompt_mode = "instruction"
    # Whether checkpoints of this family were trained on 2025-style base velocities (~0) and should get
    # proprio[0:3] zeroed by the front server (profile.mask_base_qvel). Informational; the front server owns it.
    recommended_mask_base_qvel = False

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        config_name: str | None = None,
        asset_id: str | None = None,
        norm_stats_dir: str | None = None,
        num_steps: int | None = None,
        dtype: str = "bfloat16",
        action_horizon: int | None = None,
        use_quantile_norm: bool | None = None,
        batched: bool = False,
        max_batch: int = 8,
        default_prompt_mode: str | None = None,
        mem_fraction: float | None = None,
        warmup_task_id: int = 0,
        policy: Any = None,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        if num_steps is not None and int(num_steps) < 1:
            raise ValueError("num_steps must be >= 1")
        if action_horizon is not None and int(action_horizon) < 1:
            raise ValueError("action_horizon must be >= 1")
        self.default_prompt_mode = default_prompt_mode or type(self).default_prompt_mode
        if self.default_prompt_mode not in PROMPT_MODES:
            raise ValueError(f"default_prompt_mode must be one of {PROMPT_MODES}")
        self.config_name = config_name or self.default_config_name
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.asset_id = asset_id
        self.norm_stats_dir = norm_stats_dir
        self.num_steps = None if num_steps is None else int(num_steps)
        self.dtype = dtype
        self.use_quantile_norm = use_quantile_norm
        self.max_batch = int(max_batch)
        bucket_sizes(self.max_batch)  # validates
        self.warmup_task_id = int(warmup_task_id)
        self.train_config: Any = None
        self.norm_stats: Any = None
        self._fork: types.SimpleNamespace | None = None
        self._jax: Any = None
        self._jnp: Any = None
        self._warned_extras = False
        self._batched_failures = 0

        if policy is not None:
            # Dependency injection (tests, or a caller that built the policy itself).
            if action_horizon is None:
                raise ValueError("action_horizon is required when a policy object is injected")
            self.policy = policy
            self.action_horizon = int(action_horizon)
            self.batched = False
            if batched:
                logger.warning("batched=True ignored for an injected policy; using the per-item loop")
            return

        if self.checkpoint is None:
            raise ValueError("checkpoint is required")
        if mem_fraction is not None:
            self._set_mem_fraction(float(mem_fraction))
        t0 = time.monotonic()
        self._fork = self._import_fork()
        self._jax = self._fork.jax
        self._jnp = self._fork.jnp
        self._log_gpu()
        self.policy = self._load_policy(action_horizon)
        self.action_horizon = int(self.train_config.model.action_horizon)
        self.batched = bool(batched) and self._batched_path_available()
        logger.info(
            "%s ready in %.1fs: config=%s checkpoint=%s horizon=%d num_steps=%s dtype=%s batched=%s",
            self.flavor, time.monotonic() - t0, self.config_name, self.checkpoint, self.action_horizon,
            self.num_steps or "default(10)", self.dtype, self.batched,
        )

    # ---- hooks for subclasses -------------------------------------------------------------------------------
    def _import_fork(self) -> types.SimpleNamespace:
        """Import the fork lazily and return a namespace with the modules the base class uses."""
        raise NotImplementedError

    def _check_train_config(self, train_config: Any) -> None:
        """Reject configs this backend cannot feed (e.g. depth/point-cloud variants)."""

    def _override_train_config(self, train_config: Any) -> Any:
        """Fork-specific config overrides (e.g. repo_id). Called before the generic ones."""
        return train_config

    def _after_load(self) -> None:
        """Fork-specific checks once ``self.train_config``/``self.norm_stats`` are known."""

    def build_example(self, item: InferItem) -> dict[str, Any]:
        """The exact dict the fork's wrapper passes to ``policy.infer`` for one item."""
        raise NotImplementedError

    # ---- loading ----------------------------------------------------------------------------------------------
    @staticmethod
    def _set_mem_fraction(fraction: float) -> None:
        if not 0.05 <= fraction <= 1.0:
            raise ValueError("mem_fraction must be in [0.05, 1.0]")
        if "jax" in sys.modules:
            logger.warning("jax already imported; mem_fraction=%s may have no effect", fraction)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.3f}"

    def _log_gpu(self) -> None:
        jax = self._jax
        try:
            devices = jax.devices()
            logger.info("JAX %s devices: %s", jax.__version__, [str(d) for d in devices])
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not list JAX devices: %s", exc)
            return
        cc = gpu_compute_capability(jax)
        if cc is None:
            if all(getattr(d, "platform", "") != "gpu" for d in devices):
                logger.warning("no GPU visible to JAX: inference will run on CPU (very slow)")
            return
        logger.info("GPU compute capability %d.%d", *cc)
        if cc < (8, 0):
            # Turing (e.g. TITAN RTX, sm_75) has no bf16 tensor cores. XLA's FloatNormalization rewrites bf16
            # matmuls/convs to f32 there, so bf16 still runs correctly, just at f32 SIMT speed.
            logger.warning(
                "GPU sm_%d%d < sm_80: bf16 matmuls are upcast to f32 by XLA (correct but slower). dtype=%s; "
                "dtype=float32 avoids the bf16 rounding at a higher memory cost (~14 GB).", cc[0], cc[1], self.dtype,
            )

    def _resolve_checkpoint(self) -> pathlib.Path:
        path = self._fork.download.maybe_download(str(self.checkpoint))
        path = pathlib.Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {path}")
        is_pytorch = (path / "model.safetensors").exists()
        if not is_pytorch and not (path / "params").is_dir():
            raise FileNotFoundError(f"{path} has neither params/ (JAX orbax) nor model.safetensors (PyTorch)")
        return path

    def _get_train_config(self) -> Any:
        cfg_mod = self._fork.config
        known = getattr(cfg_mod, "_CONFIGS_DICT", None)
        if known is not None and self.config_name not in known:
            import difflib

            close = difflib.get_close_matches(self.config_name, list(known), n=3, cutoff=0.0)
            raise ValueError(f"config {self.config_name!r} not found in this fork; closest: {close}")
        return cfg_mod.get_config(self.config_name)

    def _apply_overrides(self, train_config: Any, action_horizon: int | None) -> Any:
        train_config = self._override_train_config(train_config)
        if self.asset_id is not None:
            data = train_config.data
            assets = dataclasses.replace(data.assets, asset_id=self.asset_id)
            train_config = dataclasses.replace(train_config, data=dataclasses.replace(data, assets=assets))
        model_updates: dict[str, Any] = {}
        if action_horizon is not None:
            model_updates["action_horizon"] = int(action_horizon)
        if self.dtype == "float32":
            model_updates["dtype"] = "float32"
        if model_updates:
            model = dataclasses.replace(train_config.model, **model_updates)
            train_config = dataclasses.replace(train_config, model=model)
        if self.use_quantile_norm is not None:
            train_config = dataclasses.replace(
                train_config, data=_DataFactoryOverride(train_config.data, self.use_quantile_norm)
            )
        return train_config

    def _load_norm_stats(self, train_config: Any, ckpt: pathlib.Path) -> dict[str, Any]:
        if self.norm_stats_dir is not None:
            directory = pathlib.Path(self.norm_stats_dir)
            if not directory.is_absolute():
                directory = ckpt / directory
            source = str(directory)
            loader: Callable[[], Any] = lambda: self._fork.normalize.load(directory)  # noqa: E731
        else:
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
            asset_id = data_config.asset_id
            if asset_id is None:
                raise ValueError("the config has no asset id; pass asset_id or norm_stats_dir")
            if isinstance(asset_id, (list, tuple)):  # wensi-ai fork allows a list; it uses the first entry
                asset_id = asset_id[0]
            source = str(ckpt / "assets" / asset_id)
            loader = lambda: self._fork.checkpoints.load_norm_stats(ckpt / "assets", asset_id)  # noqa: E731
        try:
            norm_stats = loader()
        except FileNotFoundError as exc:
            found = sorted(str(p.relative_to(ckpt)) for p in ckpt.glob("assets/**/norm_stats.json"))
            raise FileNotFoundError(
                f"norm_stats.json not found under {source}. Found in checkpoint: {found or 'none'}. "
                "Set asset_id (sub-dir of assets/) or norm_stats_dir (e.g. 'assets')."
            ) from exc
        for key in ("state", "actions"):
            if key not in norm_stats:
                raise ValueError(f"norm stats at {source} lack {key!r} (keys: {sorted(norm_stats)})")
        state_dim = int(np.asarray(norm_stats["state"].mean).shape[-1])
        action_dim = int(np.asarray(norm_stats["actions"].mean).shape[-1])
        if state_dim < 23 or action_dim < C.ACTION_DIM:
            raise ValueError(f"norm stats at {source} have state dim {state_dim} / action dim {action_dim} (< 23)")
        logger.info("norm stats from %s (state %d, actions %d)", source, state_dim, action_dim)
        return norm_stats

    def _sample_kwargs(self) -> dict[str, Any] | None:
        return None if self.num_steps is None else {"num_steps": self.num_steps}

    def _load_policy(self, action_horizon: int | None) -> Any:
        fork = self._fork
        train_config = self._get_train_config()
        self._check_train_config(train_config)
        train_config = self._apply_overrides(train_config, action_horizon)
        self.train_config = train_config
        ckpt = self._resolve_checkpoint()
        self.norm_stats = self._load_norm_stats(train_config, ckpt)
        self._after_load()
        is_pytorch = (ckpt / "model.safetensors").exists()
        if self.dtype == "bfloat16" or is_pytorch:
            if is_pytorch and self.dtype != "bfloat16":
                logger.warning("dtype=%s ignored for a PyTorch checkpoint (openpi casts it to bf16)", self.dtype)
            # Upstream path, unchanged: restore params as bf16, build the transforms from the config.
            return fork.policy_config.create_trained_policy(
                train_config, ckpt, sample_kwargs=self._sample_kwargs(), default_prompt=None, norm_stats=self.norm_stats
            )
        return self._create_policy_float32(train_config, ckpt)

    def _create_policy_float32(self, train_config: Any, ckpt: pathlib.Path) -> Any:
        """``create_trained_policy`` (identical in both forks) with params restored as float32.

        Mirrors ``src/openpi/policies/policy_config.py`` line for line except the restore dtype; the model config
        already carries ``dtype="float32"`` (``_apply_overrides``), so activations are f32 too.
        """
        fork = self._fork
        transforms = fork.transforms
        params = fork.model.restore_params(ckpt / "params", dtype=self._jnp.float32)
        model = train_config.model.load(params)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = self.norm_stats
        return fork.policy.Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            sample_kwargs=self._sample_kwargs(),
            metadata=train_config.policy_metadata,
        )

    def _batched_path_available(self) -> bool:
        needed = ("_input_transform", "_output_transform", "_sample_actions", "_rng", "_sample_kwargs")
        missing = [a for a in needed if not hasattr(self.policy, a)]
        if missing or getattr(self.policy, "_is_pytorch_model", False):
            logger.warning("batched path unavailable (missing %s or PyTorch model); using the loop", missing)
            return False
        return True

    # ---- Backend API ------------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "action_horizon": int(self.action_horizon),
            "image_size": IMAGE_SIZE,
            "num_stages": None,
            "supports_inpaint": False,
            "supports_stage": False,
        }

    def _dummy_item(self) -> InferItem:
        blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return InferItem(
            task_id=self.warmup_task_id,
            prompt=resolve_prompt(None, self.warmup_task_id, self.default_prompt_mode),
            proprio=reset_pose_proprio(),
            images={role: blank for role in ROLES},
        )

    def warmup(self) -> float:
        """Compile every batch size the server may use. Returns total milliseconds."""
        t0 = time.monotonic()
        sizes = bucket_sizes(self.max_batch) if self.batched else [1]
        for size in sizes:
            t1 = time.monotonic()
            out = self.infer([self._dummy_item() for _ in range(size)])
            if len(out) != size:
                raise RuntimeError(f"warmup returned {len(out)} chunks for {size} items")
            logger.info("warmup batch %d: %.0f ms, chunk %s", size, (time.monotonic() - t1) * 1e3, out[0].actions.shape)
        return (time.monotonic() - t0) * 1e3

    def infer(self, items: list[InferItem]) -> list[ChunkOut]:
        if not items:
            return []
        if not self._warned_extras and any(i.initial_actions is not None or i.stage is not None for i in items):
            logger.info("%s ignores stage / initial_actions (supports_stage=supports_inpaint=False)", self.flavor)
            self._warned_extras = True
        examples = [self.build_example(item) for item in items]
        results = None
        if self.batched and len(examples) > 1:
            try:
                results = self._infer_batched(examples)
                self._batched_failures = 0
            except Exception:  # noqa: BLE001 - e.g. XLA OOM at the largest bucket: serve this request item by item
                self._batched_failures += 1
                logger.exception("batched inference failed (%d in a row); falling back to the loop",
                                 self._batched_failures)
                if self._batched_failures >= MAX_BATCHED_FAILURES:
                    logger.error("disabling the batched path after %d consecutive failures", self._batched_failures)
                    self.batched = False
        if results is None:
            results = [self.policy.infer(example) for example in examples]
        if len(results) != len(items):
            raise RuntimeError(f"policy returned {len(results)} results for {len(items)} items")
        return [ChunkOut(actions=postprocess_actions(r["actions"])) for r in results]

    def _infer_batched(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One padded ``sample_actions`` call per group of ``max_batch`` examples.

        Re-implements ``Policy.infer`` (identical in both forks) for B > 1 using the policy's own transforms and
        jitted sampler: per-item input transforms, stack (instead of ``[np.newaxis]``), split the policy RNG,
        sample, then per-item output transforms on ``{"state", "actions"}``. Groups are padded to a power-of-two
        bucket by repeating the last item so XLA compiles at most ``len(bucket_sizes(max_batch))`` shapes.
        Noise is drawn for the whole batch at once, so samples differ from the loop path for the same seed but
        follow the same distribution.
        """
        jax, jnp = self._jax, self._jnp
        policy = self.policy
        observation_cls = self._fork.model.Observation
        results: list[dict[str, Any]] = []
        for start in range(0, len(examples), self.max_batch):
            group = examples[start : start + self.max_batch]
            inputs = [policy._input_transform(copy_tree(ex)) for ex in group]
            size = bucket_for(len(inputs), self.max_batch)
            batch = stack_trees(inputs + [inputs[-1]] * (size - len(inputs)))
            batch = jax.tree.map(jnp.asarray, batch)
            policy._rng, sample_rng = jax.random.split(policy._rng)
            observation = observation_cls.from_dict(batch)
            actions = np.asarray(policy._sample_actions(sample_rng, observation, **policy._sample_kwargs))
            state = np.asarray(batch["state"])
            for i in range(len(group)):
                results.append(policy._output_transform({"state": state[i], "actions": actions[i]}))
        return results


# ------------------------------------------------------------------------------------------------------------
# wensi-ai/openpi@behavior: pi05_b1k
# ------------------------------------------------------------------------------------------------------------
class OpenPIB1KBackend(OpenPIBackendBase):
    """The organizers' ``pi05_b1k`` (wensi-ai/openpi ``behavior``) and compatible checkpoints.

    Extra constructor arguments on top of ``OpenPIBackendBase``:
      repo_id        replaces ``config.data.repo_id`` like ``serve_b1k.py --repo-id`` (the asset id defaults to it).
                     The radio baseline needs nothing: ``pi05_b1k`` already has ``repo_id="turning_on_radio"``.
      gripper_state  ``auto`` (from the norm stats), ``width`` (the fork's own finger-sum state, radio baseline)
                     or ``pm1`` (``2*w/0.1-1``; Hoshipu 100t stats). See ``apply_gripper_pm1``.
    """

    flavor = "openpi_b1k"
    default_config_name = "pi05_b1k"
    default_prompt_mode = "snake_case"  # what LeRobot prompt_from_task=True yields for the 2026 demos
    recommended_mask_base_qvel = False  # trained on 2026 robot-frame base velocities

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        repo_id: str | None = None,
        gripper_state: str = "auto",
        **kwargs: Any,
    ) -> None:
        if gripper_state not in GRIPPER_STATE_MODES:
            raise ValueError(f"gripper_state must be one of {GRIPPER_STATE_MODES}, got {gripper_state!r}")
        self.repo_id = repo_id
        self.gripper_state_setting = gripper_state
        self.gripper_state = "width" if gripper_state == "auto" else gripper_state
        self.camera_keys = dict(DEFAULT_B1K_CAMERA_KEYS)
        super().__init__(checkpoint, **kwargs)

    def _import_fork(self) -> types.SimpleNamespace:
        import jax
        import jax.numpy as jnp

        # Module paths checked at wensi-ai/openpi 0cc8e35 (CPU integration run through the real fork).
        from openpi import transforms
        from openpi.configs import ROBOT_REGISTRY
        from openpi.models import model
        from openpi.policies import b1k_policy, policy, policy_config
        from openpi.shared import download, normalize
        from openpi.training import checkpoints
        from openpi.training import config

        return types.SimpleNamespace(
            jax=jax, jnp=jnp, transforms=transforms, robot_registry=ROBOT_REGISTRY, model=model,
            b1k_policy=b1k_policy, policy=policy, policy_config=policy_config, download=download,
            normalize=normalize, checkpoints=checkpoints, config=config,
        )

    def _check_train_config(self, train_config: Any) -> None:
        factory_cls = getattr(self._fork.config, "LeRobotB1KDataConfig", None)
        if factory_cls is None or not isinstance(train_config.data, factory_cls):
            raise ValueError(f"config {self.config_name!r} does not use LeRobotB1KDataConfig")

    def _override_train_config(self, train_config: Any) -> Any:
        if self.repo_id is not None:
            data = dataclasses.replace(train_config.data, repo_id=self.repo_id)
            train_config = dataclasses.replace(train_config, data=data)
        return train_config

    def _after_load(self) -> None:
        robot_name = self.train_config.data.robot_config_name
        robot = self._fork.robot_registry[robot_name]
        # Camera keys exactly as B1KInputs iterates them (robot_config.observations order).
        cams = {key: obs.name for key, obs in robot.observations.items()}
        if sorted(cams.values()) != sorted(ROLES) or len(cams) != 3:
            raise ValueError(f"robot config {robot_name!r} cameras {cams} do not map onto {ROLES}")
        self.camera_keys = cams
        # Gripper state representation: from the norm stats unless set explicitly.
        detected = detect_gripper_state_mode(self.norm_stats["state"])
        if self.gripper_state_setting == "auto":
            if detected is None:
                raise ValueError("cannot infer the gripper state representation from the norm stats; "
                                 "set gripper_state to 'width' or 'pm1'")
            self.gripper_state = detected
        elif detected is not None and detected != self.gripper_state:
            logger.warning("gripper_state=%s but the norm stats look like %r", self.gripper_state, detected)
        logger.info("gripper state representation: %s (norm stats suggest %s)", self.gripper_state, detected)
        self._check_state_extraction(robot)

    def _check_state_extraction(self, robot: Any) -> None:
        """Run the fork's own state extraction on a probe and compare with the expected 23-D layout."""
        probe = (np.arange(C.PROPRIO_DIM, dtype=np.float32) + 1.0) * 0.013
        fed = apply_gripper_pm1(probe) if self.gripper_state == "pm1" else probe
        got = np.asarray(self._fork.b1k_policy.extract_state_from_proprio(fed, robot), dtype=np.float32)
        want = b1k_state_from_proprio(probe, self.gripper_state)
        if got.shape != want.shape or not np.allclose(got, want, atol=1e-5):
            raise RuntimeError(f"fork state extraction mismatch:\n got  {got}\n want {want}")

    def build_example(self, item: InferItem) -> dict[str, Any]:
        proprio, images = validate_item(item)
        if self.gripper_state == "pm1":
            proprio = apply_gripper_pm1(proprio)
        example: dict[str, Any] = {f"observation/{key}": images[role] for key, role in self.camera_keys.items()}
        example["observation/state"] = proprio
        example["prompt"] = resolve_prompt(item.prompt, item.task_id, self.default_prompt_mode)
        return example
