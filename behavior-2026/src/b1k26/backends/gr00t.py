"""GR00T N1.7 worker backend (``wensi-ai/Isaac-GR00T`` branch ``behavior``): ``Gr00tPolicy`` used directly.

Checkpoints: ``kmy17518/gr00t-n1.7-b1k-multitask`` (all 100 tasks; e.g. ``checkpoint-238000``) and the organizers'
``turning_on_radio`` baseline (Google Drive ``turning_on_radio_GR00T-checkpoint-150000.zip``). Both are
``Gr00tN1d7`` HF checkpoints (``config.json``, ``model-*.safetensors``, ``processor_config.json``,
``statistics.json``, ``embodiment_id.json``) for embodiment tag ``NEW_EMBODIMENT`` with the R1Pro modality config
``examples/b1k/r1pro.py`` / ``r1pro.json`` (serialized into ``processor_config.json``, so it is not imported here).

Observation, built exactly as ``gr00t/eval/eval_b1k_wrapper.py::B1KPolicyWrapper.process_input`` does for one env
(and batched the way the fork's ``andi/vector`` fix does it for N envs)::

    {"video": {"head" | "left_wrist" | "right_wrist": (B, 1, S, S, 3) uint8},     # S = 224 (wrapper obs_size)
     "state": {"base_qvel": proprio[0:3], "torso": proprio[53:57], "left_arm": proprio[3:10],
               "left_gripper": proprio[24:26], "right_arm": proprio[28:35], "right_gripper": proprio[49:51]},
               # each (B, 1, D) float32: raw slices, the two finger positions per gripper are NOT summed
     "language": {"annotation.human.task_description": [[prompt] for each item]}}

``Gr00tPolicy.get_action`` validates that dict (``strict``), normalizes the state with the checkpoint's percentile
stats, resizes the images inside its processor (letterbox, shortest edge 256, center crop 0.95, 256), denoises the
16-step chunk and ``decode_action`` un-normalizes it and converts the RELATIVE groups (torso, left_arm, right_arm,
``state_key`` = the same group) back to absolute by adding the current state (``StateActionProcessor.unapply_action``
with the raw state; checked against the fork's processor with the kmy statistics). The returned dict has
``base (B,16,3), torso (B,16,4), left_arm (B,16,7), left_gripper (B,16,1), right_arm (B,16,7), right_gripper
(B,16,1)``; the action group names equal ``b1k26.constants.ACTION_SLICES``, so the 23-D layout is filled by name.

Prompt: the policy is conditioned on the dataset task string (``meta/tasks.parquet`` = snake_case task name, e.g.
``turning_on_radio``; the multitask card says "serve with the matching string"). The upstream wrapper sends its
default ``"pick up the object and place it on the table"`` (a known upstream bug). ``prompt_style`` defaults to
``snake_case`` derived from ``task_id``; ``item`` passes the front server's profile prompt through.

Turing (sm_75, e.g. TITAN RTX): flash-attn 2.7.4 refuses GPUs below sm_80 at runtime, and the backbone only falls back
to SDPA when ``import flash_attn`` fails (``gr00t/model/modules/qwen3_backbone.py``). Below sm_80 (or on CPU) the
backend therefore blocks the ``flash_attn`` import before the model is built, so the backbone requests
``attn_implementation="sdpa"`` (same effect as ``use_flash_attention: false`` in the checkpoint's ``config.json``).
``dtype="float32"`` (or ``"auto"`` below sm_80) upcasts the model after loading and makes the policy cast its inputs
to float32 instead of the hard-coded bfloat16.

Nothing here imports torch or gr00t at module import time.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut, InferItem
from b1k26.backends.openpi_b1k import postprocess_actions, reset_pose_proprio, validate_item

logger = logging.getLogger(__name__)

# Pinned upstream sources (scripts/envs/gr00t.sh installs exactly this commit).
GR00T_REPO = "https://github.com/wensi-ai/Isaac-GR00T"
GR00T_COMMIT = "ace36d935b376fbf25cd56371e23877b95407c40"  # branch "behavior" head (2026-07-02)

DEFAULT_EMBODIMENT_TAG = "NEW_EMBODIMENT"
LANGUAGE_KEY = "annotation.human.task_description"
IMAGE_SIZE = 224  # B1KPolicyWrapper obs_size default; the processor then resizes to 256 itself
VIDEO_KEYS = ("head", "left_wrist", "right_wrist")  # r1pro.py video modality_keys == b1k26 camera roles
# examples/b1k/r1pro.json "state" start/end: slices of the 61-D proprio (constants.PROPRIO_INDICES_2026 names).
_P = C.PROPRIO_INDICES_2026
STATE_SLICES = {
    "base_qvel": _P["base_qvel"],
    "torso": _P["trunk_qpos"],
    "left_arm": _P["arm_left_qpos"],
    "left_gripper": _P["gripper_left_qpos"],
    "right_arm": _P["arm_right_qpos"],
    "right_gripper": _P["gripper_right_qpos"],
}
# r1pro.py action modality order; names equal b1k26.constants.ACTION_SLICES keys.
ACTION_KEYS = ("base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper")
ACTION_DIMS = {k: C.ACTION_SLICES[k].stop - C.ACTION_SLICES[k].start for k in ACTION_KEYS}
PROMPT_STYLES = ("snake_case", "instruction", "item")
ATTN_CHOICES = ("auto", "flash_attention_2", "sdpa")
DTYPES = ("bfloat16", "float32", "auto")
MAX_BATCHED_FAILURES = 3


# ------------------------------------------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested on CPU)
# ------------------------------------------------------------------------------------------------------------
def gr00t_prompt(style: str, task_id: int, item_prompt: str | bytes | None = None) -> str:
    """Language instruction for the policy.

    ``snake_case`` (default): the dataset task string the checkpoints were trained on. ``instruction``: the 2026
    natural-language instruction. ``item``: the front server's prompt verbatim (falls back to snake_case if empty).
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, got {style!r}")
    info = C.task(int(task_id))
    if style == "item":
        if isinstance(item_prompt, bytes):
            item_prompt = item_prompt.decode("utf-8")
        if isinstance(item_prompt, str) and item_prompt.strip():
            return item_prompt
        return info.name
    return info.name if style == "snake_case" else info.instruction


def state_from_proprio(proprio: np.ndarray) -> dict[str, np.ndarray]:
    """GR00T state groups from the 61-D proprio: ``{key: (D,) float32}`` (copies)."""
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if p.shape[0] != C.PROPRIO_DIM:
        raise ValueError(f"proprio must have {C.PROPRIO_DIM} values, got {p.shape}")
    return {k: np.array(p[sl], dtype=np.float32, copy=True) for k, sl in STATE_SLICES.items()}


def build_observation(prepared: list[tuple[np.ndarray, dict[str, np.ndarray], str]],
                      language_key: str = LANGUAGE_KEY) -> dict[str, Any]:
    """Batched ``Gr00tPolicy`` observation from ``[(proprio (61,), {role: (S,S,3) uint8}, prompt), ...]``.

    Video ``(B, T=1, S, S, 3)`` uint8, state ``(B, T=1, D)`` float32, language ``[[prompt], ...]`` (B lists of one
    string: the ``(B, T)`` layout ``Gr00tPolicy.check_observation`` requires; the upstream wrapper's ``[[p] * B]``
    is only right for B = 1).
    """
    if not prepared:
        raise ValueError("empty batch")
    video = {k: np.stack([images[k] for _, images, _ in prepared])[:, None] for k in VIDEO_KEYS}
    for k, v in video.items():
        if v.dtype != np.uint8 or v.ndim != 5 or v.shape[-1] != 3:
            raise ValueError(f"video {k!r} must be (B, 1, H, W, 3) uint8, got {v.shape} {v.dtype}")
    states = [state_from_proprio(p) for p, _, _ in prepared]
    state = {k: np.stack([s[k] for s in states]).astype(np.float32)[:, None] for k in STATE_SLICES}
    language = {language_key: [[str(prompt)] for _, _, prompt in prepared]}
    return {"video": video, "state": state, "language": language}


def actions_to_23(action: dict[str, Any], batch: int) -> np.ndarray:
    """``Gr00tPolicy.get_action`` output dict -> ``(B, T, 23)`` float32 in the evaluator layout (filled by name)."""
    missing = [k for k in ACTION_KEYS if k not in action]
    if missing:
        raise ValueError(f"policy action dict lacks {missing} (got {sorted(action)})")
    arrays = {}
    horizon = None
    for k in ACTION_KEYS:
        a = action[k]
        if hasattr(a, "detach"):
            a = a.detach().float().cpu().numpy()
        a = np.asarray(a)
        if a.ndim == 2:  # (T, D) for a single env
            a = a[None]
        if a.ndim != 3 or a.shape[0] != batch or a.shape[2] != ACTION_DIMS[k]:
            raise ValueError(f"action {k!r} has shape {a.shape}; expected ({batch}, T, {ACTION_DIMS[k]})")
        if horizon is None:
            horizon = a.shape[1]
        elif a.shape[1] != horizon:
            raise ValueError(f"action groups disagree on the horizon: {k!r} has {a.shape[1]}, expected {horizon}")
        arrays[k] = a
    out = np.empty((batch, int(horizon), C.ACTION_DIM), dtype=np.float32)
    for k in ACTION_KEYS:
        out[:, :, C.ACTION_SLICES[k]] = arrays[k]
    return out


def decide_attention(requested: str, capability: tuple[int, int] | None, flash_available: bool) -> str:
    """``"flash_attention_2"`` (leave the checkpoint config alone) or ``"sdpa"`` (block flash-attn).

    ``auto``: flash only on a CUDA device of sm_80+ with flash-attn installed. An explicit ``flash_attention_2`` on
    a device that cannot run it is an error rather than a crash at the first inference.
    """
    if requested not in ATTN_CHOICES:
        raise ValueError(f"attn_implementation must be one of {ATTN_CHOICES}, got {requested!r}")
    capable = capability is not None and tuple(capability) >= (8, 0)
    if requested == "sdpa":
        return "sdpa"
    if requested == "flash_attention_2":
        if not capable:
            raise ValueError(f"flash_attention_2 needs a CUDA GPU of sm_80+ (device capability {capability}); "
                             "use attn_implementation 'sdpa' or 'auto'")
        if not flash_available:
            raise ValueError("flash_attention_2 requested but flash_attn is not installed")
        return "flash_attention_2"
    return "flash_attention_2" if capable and flash_available else "sdpa"


def decide_dtype(requested: str, capability: tuple[int, int] | None) -> str:
    """``auto``: float32 below sm_80 (no bf16 tensor cores; PyTorch bf16 SDPA falls back to the math kernel) and on
    CPU, else bfloat16 (the upstream policy's fixed choice)."""
    if requested not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {requested!r}")
    if requested != "auto":
        return requested
    return "bfloat16" if capability is not None and tuple(capability) >= (8, 0) else "float32"


def block_flash_attn() -> None:
    """Make ``import flash_attn`` raise ImportError in this process (and ``find_spec`` return None).

    ``Qwen3Backbone`` then logs "flash_attn is not installed. Falling back to sdpa attention" and passes
    ``attn_implementation="sdpa"``. Must run before the model is constructed. Idempotent.
    """
    for name in [m for m in sys.modules if m == "flash_attn" or m.startswith("flash_attn.")]:
        if sys.modules[name] is not None:
            logger.warning("flash_attn was already imported; blocking it for later imports only")
    sys.modules["flash_attn"] = None  # type: ignore[assignment]


def flash_attn_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def check_modality_json(path: str | os.PathLike[str]) -> None:
    """Cross-check a fork ``r1pro.json`` (state/action start/end) against the slices hard-coded here."""
    import json

    doc = json.loads(pathlib.Path(path).read_text())
    for key, sl in STATE_SLICES.items():
        entry = doc["state"][key]
        if (entry["start"], entry["end"]) != (sl.start, sl.stop):
            raise ValueError(f"{path}: state {key} is {entry['start']}:{entry['end']}, expected {sl.start}:{sl.stop}")
    for key in ACTION_KEYS:
        entry, sl = doc["action"][key], C.ACTION_SLICES[key]
        if (entry["start"], entry["end"]) != (sl.start, sl.stop):
            raise ValueError(f"{path}: action {key} is {entry['start']}:{entry['end']}, expected {sl.start}:{sl.stop}")
    if sorted(doc["video"]) != sorted(VIDEO_KEYS):
        raise ValueError(f"{path}: video keys {sorted(doc['video'])} != {sorted(VIDEO_KEYS)}")


def _modality_keys(cfg: Any) -> list[str]:
    keys = cfg.get("modality_keys") if isinstance(cfg, dict) else getattr(cfg, "modality_keys", None)
    return list(keys or [])


def _delta_indices(cfg: Any) -> list[int]:
    idx = cfg.get("delta_indices") if isinstance(cfg, dict) else getattr(cfg, "delta_indices", None)
    return list(idx or [])


def check_policy_modalities(modality_configs: dict[str, Any]) -> int:
    """Validate the checkpoint's modality config against the R1Pro layout used here; return the action horizon."""
    for modality in ("video", "state", "action", "language"):
        if modality not in modality_configs:
            raise ValueError(f"checkpoint modality config lacks {modality!r}")
    video, state, action = (_modality_keys(modality_configs[m]) for m in ("video", "state", "action"))
    if sorted(video) != sorted(VIDEO_KEYS):
        raise ValueError(f"checkpoint video keys {video} != {list(VIDEO_KEYS)}")
    if sorted(state) != sorted(STATE_SLICES):
        raise ValueError(f"checkpoint state keys {state} != {list(STATE_SLICES)}")
    if action != list(ACTION_KEYS):
        raise ValueError(f"checkpoint action keys {action} != {list(ACTION_KEYS)}")
    for m in ("video", "state", "language"):
        if len(_delta_indices(modality_configs[m])) != 1:
            raise ValueError(f"checkpoint {m} uses {len(_delta_indices(modality_configs[m]))} timesteps; "
                             "this backend sends one")
    horizon = len(_delta_indices(modality_configs["action"]))
    if horizon < 1:
        raise ValueError("checkpoint action horizon is 0")
    return horizon


# ------------------------------------------------------------------------------------------------------------
# Backend
# ------------------------------------------------------------------------------------------------------------
class Gr00tBackend(Backend):
    """GR00T N1.7 checkpoint served through the fork's ``Gr00tPolicy``.

    Constructor arguments (keyword, JSON-serializable):
      checkpoint           checkpoint dir (``config.json``, safetensors, ``processor_config.json`` or ``processor/``).
      embodiment_tag       ``NEW_EMBODIMENT`` (what the B1K fine-tunes use).
      device               torch device (default ``cuda``; ``cuda:1`` etc. work).
      prompt_style         ``snake_case`` (default) | ``instruction`` | ``item`` (front server prompt verbatim).
      attn_implementation  ``auto`` (flash-attn only on sm_80+ with flash-attn installed, else SDPA) | ``sdpa`` |
                           ``flash_attention_2``.
      dtype                ``bfloat16`` (upstream) | ``float32`` | ``auto`` (float32 below sm_80 and on CPU).
      num_steps            override ``num_inference_timesteps`` of the flow-matching head (checkpoint: 4).
      image_size           expected input size (the front server's ``profile.image_size``; default 224).
      batched              one ``get_action`` call per micro-batch (``max_batch`` items) instead of a loop.
      strict               ``Gr00tPolicy`` input/output validation (default on).
      seed                 ``torch.manual_seed`` at load (the denoiser samples torch noise).
      modality_json        optional ``examples/b1k/r1pro.json`` to cross-check the hard-coded slices.
      warmup_task_id       task for the warmup inference.
      policy, action_horizon  test injection (``get_action(obs) -> (dict, info)`` object); skips torch / gr00t.
    """

    flavor = "gr00t"

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        device: str = "cuda",
        prompt_style: str = "snake_case",
        attn_implementation: str = "auto",
        dtype: str = "bfloat16",
        num_steps: int | None = None,
        image_size: int = IMAGE_SIZE,
        batched: bool = True,
        max_batch: int = 8,
        strict: bool = True,
        seed: int | None = None,
        modality_json: str | None = None,
        warmup_task_id: int = 0,
        policy: Any = None,
        action_horizon: int | None = None,
    ) -> None:
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, got {prompt_style!r}")
        if attn_implementation not in ATTN_CHOICES:
            raise ValueError(f"attn_implementation must be one of {ATTN_CHOICES}, got {attn_implementation!r}")
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        if num_steps is not None and int(num_steps) < 1:
            raise ValueError("num_steps must be >= 1")
        if int(image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(max_batch) < 1:
            raise ValueError("max_batch must be >= 1")
        if not 0 <= int(warmup_task_id) < C.NUM_TASKS:
            raise ValueError(f"warmup_task_id must be in [0, {C.NUM_TASKS})")
        if modality_json is not None:
            check_modality_json(modality_json)
        self.prompt_style = prompt_style
        self.image_size = int(image_size)
        self.batched = bool(batched)
        self.max_batch = int(max_batch)
        self.warmup_task_id = int(warmup_task_id)
        self.num_steps = None if num_steps is None else int(num_steps)
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.attention = "injected"
        self.dtype = "injected"
        self.capability: tuple[int, int] | None = None
        self.language_key = LANGUAGE_KEY
        self._batched_failures = 0
        self._torch: Any = None

        if policy is not None:
            self.policy = policy
            configs = getattr(policy, "modality_configs", None)
            if configs is not None:
                horizon = check_policy_modalities(configs)
                if action_horizon is not None and int(action_horizon) != horizon:
                    raise ValueError(f"action_horizon {action_horizon} != policy horizon {horizon}")
                self.action_horizon = horizon
            elif action_horizon is None:
                raise ValueError("action_horizon is required for an injected policy without modality_configs")
            else:
                self.action_horizon = int(action_horizon)
            self.language_key = getattr(policy, "language_key", LANGUAGE_KEY)
            return

        if self.checkpoint is None:
            raise ValueError("checkpoint is required")
        t0 = time.monotonic()
        self._check_checkpoint_dir(pathlib.Path(self.checkpoint))
        self.policy = self._load_policy(embodiment_tag, device, attn_implementation, dtype, strict, seed)
        self.action_horizon = check_policy_modalities(self.policy.modality_configs)
        self.language_key = getattr(self.policy, "language_key", LANGUAGE_KEY)
        if self.language_key != LANGUAGE_KEY:
            logger.warning("checkpoint language key is %r (expected %r)", self.language_key, LANGUAGE_KEY)
        logger.info("gr00t ready in %.1fs: %s horizon=%d attention=%s dtype=%s num_steps=%s prompt=%s batched=%s",
                    time.monotonic() - t0, self.checkpoint, self.action_horizon, self.attention, self.dtype,
                    self.num_steps or "checkpoint", self.prompt_style, self.batched)

    # ---- loading -------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_checkpoint_dir(path: pathlib.Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {path}")
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"{path} has no config.json (expected a Gr00tN1d7 HF checkpoint dir)")
        if not (path / "processor_config.json").exists() and not (path / "processor").is_dir():
            raise FileNotFoundError(f"{path} has neither processor_config.json nor processor/")
        if not any(path.glob("*.safetensors")):
            raise FileNotFoundError(f"{path} has no *.safetensors weights")

    @staticmethod
    def _cuda_capability(torch: Any, device: str) -> tuple[int, int] | None:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        return int(major), int(minor)

    def _load_policy(self, embodiment_tag: str, device: str, attn: str, dtype: str, strict: bool,
                     seed: int | None) -> Any:
        import torch

        self._torch = torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available to torch")
        self.capability = self._cuda_capability(torch, device)
        flash_ok = flash_attn_available()
        self.attention = decide_attention(attn, self.capability, flash_ok)
        self.dtype = decide_dtype(dtype, self.capability)
        logger.info("device %s capability %s, flash_attn installed: %s -> attention %s, dtype %s", device,
                    self.capability, flash_ok, self.attention, self.dtype)
        if self.attention == "sdpa":
            if self.capability is not None and self.capability < (8, 0):
                logger.warning("GPU sm_%d%d < sm_80: flash-attn is disabled (SDPA attention); see docs/BACKENDS.md",
                               *self.capability)
            block_flash_attn()
        if seed is not None:
            torch.manual_seed(int(seed))

        gp = importlib.import_module("gr00t.policy.gr00t_policy")
        if self.dtype == "float32":
            self._patch_input_dtype(gp, torch.float32)
        try:
            policy = gp.Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=self.checkpoint, device=device,
                                    strict=strict)
        except OSError as exc:
            raise OSError(f"{exc}\nGR00T N1.7 builds its backbone from nvidia/Cosmos-Reason2-2B (gated): accept the "
                          "license on Hugging Face and pre-download it (scripts/envs/gr00t.sh --download-backbone) "
                          "with HF_TOKEN set, then serve with HF_HUB_OFFLINE=1") from exc
        if self.dtype == "float32":
            policy.model.to(dtype=torch.float32)  # loads as bf16 (Gr00tPolicy.__init__), upcast losslessly
        self._check_attention(policy)
        if self.num_steps is not None:
            head = policy.model.action_head
            # Gr00tN1d7ActionHead reads self.num_inference_timesteps at every call (gr00t_n1d7.py:355/397, ace36d9).
            logger.info("num_inference_timesteps %s -> %d", getattr(head, "num_inference_timesteps", "?"),
                        self.num_steps)
            head.num_inference_timesteps = self.num_steps
        return policy

    @staticmethod
    def _patch_input_dtype(gp_module: Any, dtype: Any) -> None:
        """``Gr00tPolicy._get_action`` casts its collated inputs with ``_rec_to_dtype(x, dtype=torch.bfloat16)``;
        redirect that module-level helper to ``dtype`` so a float32 model gets float32 inputs (no bf16 rounding of
        pixels and normalized states). Helper and call site checked at ace36d9 (gr00t/policy/gr00t_policy.py:37, 404)
        and exercised on CPU against that module."""
        original = getattr(gp_module, "_rec_to_dtype", None)
        if original is None:
            raise RuntimeError("gr00t.policy.gr00t_policy._rec_to_dtype not found; cannot run in float32")
        if getattr(original, "_b1k26_dtype", None) is not None:
            original = original._b1k26_original

        target = dtype

        def cast_to(x: Any, dtype: Any = None) -> Any:  # noqa: ARG001 - the requested dtype is overridden
            return original(x, dtype=target)

        cast_to._b1k26_dtype = target  # type: ignore[attr-defined]
        cast_to._b1k26_original = original  # type: ignore[attr-defined]
        gp_module._rec_to_dtype = cast_to

    def _check_attention(self, policy: Any) -> None:
        # VERIFY: Qwen3Backbone keeps the HF model at .backbone.model and transformers 4.57 records the choice in
        # config._attn_implementation.
        impl = None
        try:
            impl = policy.model.backbone.model.config._attn_implementation
        except AttributeError:
            pass
        logger.info("backbone attention implementation: %s", impl)
        if impl == "flash_attention_2" and self.attention == "sdpa":
            raise RuntimeError("the backbone still uses flash_attention_2 although SDPA was required (sm < 80)")

    # ---- Backend API ------------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "action_horizon": int(self.action_horizon),
            "image_size": self.image_size,
            "num_stages": None,
            "supports_inpaint": False,
            "supports_stage": False,
        }

    def _prepare(self, item: InferItem) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
        if not 0 <= int(item.task_id) < C.NUM_TASKS:
            raise ValueError(f"task_id {item.task_id} outside [0, {C.NUM_TASKS})")
        proprio, images = validate_item(item, self.image_size)
        return proprio, images, gr00t_prompt(self.prompt_style, int(item.task_id), item.prompt)

    def _run(self, group: list[tuple[np.ndarray, dict[str, np.ndarray], str]]) -> np.ndarray:
        obs = build_observation(group, self.language_key)
        result = self.policy.get_action(obs)
        action = result[0] if isinstance(result, tuple) else result
        return actions_to_23(action, len(group))

    def infer(self, items: list[InferItem]) -> list[ChunkOut]:
        if not items:
            return []
        prepared = [self._prepare(item) for item in items]
        # Items can only share a forward pass when their images have the same size.
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for i, (_, images, _) in enumerate(prepared):
            by_shape.setdefault(tuple(images["head"].shape), []).append(i)
        chunks: list[np.ndarray | None] = [None] * len(items)
        for indices in by_shape.values():
            step = self.max_batch if self.batched else 1
            for start in range(0, len(indices), step):
                idx = indices[start : start + step]
                group = [prepared[i] for i in idx]
                out = None
                if len(group) > 1:
                    try:
                        out = self._run(group)
                        self._batched_failures = 0
                    except Exception:  # noqa: BLE001 - e.g. CUDA OOM at the larger batch: serve item by item
                        self._batched_failures += 1
                        logger.exception("batched get_action failed (%d in a row); falling back to single items",
                                         self._batched_failures)
                        if self._batched_failures >= MAX_BATCHED_FAILURES:
                            logger.error("disabling batched inference after %d failures", self._batched_failures)
                            self.batched = False
                if out is None:
                    out = np.concatenate([self._run([g]) for g in group], axis=0)
                for j, i in enumerate(idx):
                    chunks[i] = out[j]
        return [ChunkOut(actions=postprocess_actions(c)) for c in chunks]  # type: ignore[arg-type]

    def _dummy_item(self) -> InferItem:
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return InferItem(task_id=self.warmup_task_id, prompt="", proprio=reset_pose_proprio(),
                         images={r: blank for r in VIDEO_KEYS})

    def warmup(self) -> float:
        """Two single-item inferences (the first pays CUDA/cuDNN initialization). Returns total milliseconds."""
        t0 = time.monotonic()
        for n in range(2):
            t1 = time.monotonic()
            out = self.infer([self._dummy_item()])
            if len(out) != 1 or out[0].actions.shape != (self.action_horizon, C.ACTION_DIM):
                raise RuntimeError(f"warmup produced {[o.actions.shape for o in out]}")
            logger.info("warmup %d: %.0f ms", n, (time.monotonic() - t1) * 1e3)
        return (time.monotonic() - t0) * 1e3


__all__ = [
    "ACTION_KEYS",
    "GR00T_COMMIT",
    "GR00T_REPO",
    "Gr00tBackend",
    "LANGUAGE_KEY",
    "STATE_SLICES",
    "VIDEO_KEYS",
    "actions_to_23",
    "block_flash_attn",
    "build_observation",
    "check_modality_json",
    "check_policy_modalities",
    "decide_attention",
    "decide_dtype",
    "gr00t_prompt",
    "state_from_proprio",
]
