"""Chunk post-processing: temporal compression, inpainting tail, gripper-variation check, sanitizing.

Semantics follow the RLC 2025 wrapper (shared/eval_b1k_wrapper.py, B1KPolicyWrapper.act):
- Compression applies when ``execute_steps < predicted_steps_to_use``: the first ``predicted_steps_to_use``
  predicted actions are resampled with a cubic spline (scipy ``interp1d(kind="cubic")`` over
  ``np.linspace`` indices) onto ``execute_steps`` samples, and the base velocity dims [:3] are multiplied by
  ``predicted_steps_to_use / execute_steps`` so the base covers the same distance in fewer steps.
- Compression is disabled when either gripper's range (max - min) over ``raw[:predicted_steps_to_use]``
  exceeds ``disable_compression_gripper_range`` (RLC GRIPPER_VARIATION_THRESHOLD = 0.2). Then
  ``raw[:execute_steps]`` is executed unchanged.
- The inpainting tail is ``raw[used:used + keep_for_inpaint]`` of the uncompressed chunk, where ``used`` is the
  number of predicted actions consumed (``predicted_steps_to_use`` when compressed, ``execute_steps`` otherwise).

Additions over RLC (documented, behavior-preserving for well-formed chunks):
- Short chunks: only the available actions are used; the tail is None when the chunk does not contain it.
- A chunk segment with non-finite values is never compressed (a spline would smear one NaN over every step);
  ``sanitize`` then replaces only the bad rows.
- Base dims are clipped to [-clip_base, clip_base] after scaling (the controller clips its input to the same
  limits, so this does not change what the robot does).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b1k26 import constants as C
from b1k26.obs import hold_action

_BASE = C.ACTION_SLICES["base"]
_GRIPPERS = (C.LEFT_GRIPPER_ACTION_IDX, C.RIGHT_GRIPPER_ACTION_IDX)


@dataclass
class ExecutionConfig:
    execute_steps: int = 20  # actions sent to the robot per plan (the replan period)
    predicted_steps_to_use: int = 26  # predicted actions consumed per plan; > execute_steps means compression
    keep_for_inpaint: int = 4  # predicted actions after the consumed ones, kept for the next plan
    base_velocity_scale_with_compression: bool = True
    disable_compression_gripper_range: float | None = 0.2  # RLC GRIPPER_VARIATION_THRESHOLD; None = never
    clip_base: float = 1.0

    def __post_init__(self) -> None:
        if int(self.execute_steps) < 1:
            raise ValueError(f"execute_steps must be >= 1, got {self.execute_steps}")
        if int(self.predicted_steps_to_use) < 1:
            raise ValueError(f"predicted_steps_to_use must be >= 1, got {self.predicted_steps_to_use}")
        if int(self.keep_for_inpaint) < 0:
            raise ValueError(f"keep_for_inpaint must be >= 0, got {self.keep_for_inpaint}")
        if not self.clip_base > 0:
            raise ValueError(f"clip_base must be > 0, got {self.clip_base}")
        self.execute_steps = int(self.execute_steps)
        self.predicted_steps_to_use = int(self.predicted_steps_to_use)
        self.keep_for_inpaint = int(self.keep_for_inpaint)

    @property
    def compresses(self) -> bool:
        return self.execute_steps < self.predicted_steps_to_use


@dataclass
class PlannedChunk:
    actions: np.ndarray  # (execute_steps or fewer, 23) float32, ready to send
    inpaint_tail: np.ndarray | None  # (keep_for_inpaint, 23) absolute actions for the next plan, or None
    compressed: bool


def as_chunk(raw: np.ndarray) -> np.ndarray:
    """Validate a model chunk and return it as (T, 23) float64 (a (1, T, D) batch and D > 23 are accepted)."""
    a = np.asarray(raw)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] < C.ACTION_DIM:
        raise ValueError(f"action chunk must be (T, >= {C.ACTION_DIM}), got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError("action chunk is empty")
    if a.dtype.kind not in "biuf":
        raise ValueError(f"action chunk has unsupported dtype {a.dtype}")
    return a[:, : C.ACTION_DIM].astype(np.float64)


def gripper_variation(actions: np.ndarray, num_actions_to_check: int) -> tuple[float, float]:
    """(left, right) gripper range max - min over actions[:num_actions_to_check] (RLC check_gripper_variation)."""
    a = np.asarray(actions)[:num_actions_to_check]
    left = a[:, C.LEFT_GRIPPER_ACTION_IDX]
    right = a[:, C.RIGHT_GRIPPER_ACTION_IDX]
    return float(np.max(left) - np.min(left)), float(np.max(right) - np.min(right))


def interpolate_actions(actions: np.ndarray, target_steps: int) -> np.ndarray:
    """Resample (T, D) actions onto ``target_steps`` samples spanning the same index range (float64).

    RLC B1KPolicyWrapper._interpolate_actions: scipy interp1d(kind="cubic") per dim over np.linspace indices.
    Falls back to linear interpolation when scipy is missing or T < 4 (cubic needs four points).
    """
    a = np.asarray(actions, dtype=np.float64)
    n = a.shape[0]
    if n == 1:
        return np.repeat(a, target_steps, axis=0)
    original_indices = np.linspace(0, n - 1, n)
    target_indices = np.linspace(0, n - 1, target_steps)
    interp1d = None
    if n >= 4:
        try:
            from scipy.interpolate import interp1d
        except ImportError:  # pragma: no cover - scipy is a declared dependency of the front server
            interp1d = None
    out = np.zeros((target_steps, a.shape[1]))
    for dim in range(a.shape[1]):
        if interp1d is not None:
            out[:, dim] = interp1d(original_indices, a[:, dim], kind="cubic")(target_indices)
        else:
            out[:, dim] = np.interp(target_indices, original_indices, a[:, dim])
    return out


def plan_execution(raw: np.ndarray, cfg: ExecutionConfig) -> PlannedChunk:
    """Turn a corrected model chunk (T, 23) of absolute actions into the actions to execute and the next tail.

    See the module docstring for the exact semantics. The returned arrays never alias ``raw``.
    """
    a = as_chunk(raw)
    t = a.shape[0]
    e = cfg.execute_steps
    used_pred = min(t, cfg.predicted_steps_to_use)  # predicted actions available for compression
    compress = e < used_pred
    if compress and not np.all(np.isfinite(a[:used_pred])):
        compress = False
    if compress and cfg.disable_compression_gripper_range is not None:
        left, right = gripper_variation(a, cfg.predicted_steps_to_use)
        if left > cfg.disable_compression_gripper_range or right > cfg.disable_compression_gripper_range:
            compress = False

    if compress:
        used = used_pred
        actions = interpolate_actions(a[:used], e)
        if cfg.base_velocity_scale_with_compression:
            actions[:, _BASE] *= used / e
        tail_start = used
    else:
        used = min(t, e)
        actions = a[:used].copy()
        tail_start = e
    actions[:, _BASE] = np.clip(actions[:, _BASE], -cfg.clip_base, cfg.clip_base)

    k = cfg.keep_for_inpaint
    tail = None
    if k > 0 and t >= tail_start + k:
        seg = a[tail_start: tail_start + k]
        if np.all(np.isfinite(seg)):
            tail = seg.astype(np.float32)
    return PlannedChunk(actions=actions.astype(np.float32), inpaint_tail=tail, compressed=compress)


def sanitize(actions: np.ndarray, proprio: np.ndarray, clip_base: float = 1.0) -> np.ndarray:
    """Return a finite float32 copy of ``actions`` ((n, 23) or (23,)) that is safe to send.

    Rows with any non-finite value are replaced by ``hold_action(proprio)``; base dims are clipped to
    [-clip_base, clip_base] and grippers to [-1, 1]. Extra trailing dims beyond 23 are dropped.
    """
    a = np.asarray(actions)
    single = a.ndim == 1
    a2 = np.array(a.reshape(1, -1) if single else a, dtype=np.float32, copy=True)
    if a2.ndim != 2 or a2.shape[1] < C.ACTION_DIM:
        raise ValueError(f"actions must be (n, >= {C.ACTION_DIM}) or ({C.ACTION_DIM},), got shape {a.shape}")
    a2 = a2[:, : C.ACTION_DIM]
    bad = ~np.all(np.isfinite(a2), axis=1)
    if bad.any():
        a2[bad] = hold_action(proprio)
    a2[:, _BASE] = np.clip(a2[:, _BASE], -clip_base, clip_base)
    for idx in _GRIPPERS:
        a2[:, idx] = np.clip(a2[:, idx], -1.0, 1.0)
    a2 = np.ascontiguousarray(a2)
    return a2[0] if single else a2
