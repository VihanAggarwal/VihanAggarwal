"""Deterministic fake backends for tests, smoke runs and the Docker smoke test.

- ``HoldBackend`` (``fake_hold``): every chunk holds the current pose (``obs.hold_action``).
- ``SineBackend`` (``fake_sine``): small smooth motion around the current pose. Optional stage logits,
  inpainting prefix, simulated latency and injected failures.
- ``ReplayBackend`` (``fake_replay``): replays an ``.npy`` action file window by window.

Outputs depend only on the inputs and constructor arguments (``ReplayBackend`` also on its per-task cursor), never
on time or randomness, so a test can compare two runs action by action. Constructor arguments are plain JSON
values so ``b1k26-worker --backend fake_sine --backend-arg period=40`` works.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut, InferItem
from b1k26.obs import hold_action


def _num_stages_list(num_stages: int | list[int] | None) -> list[int] | None:
    if num_stages is None:
        return None
    if isinstance(num_stages, (list, tuple)):
        if len(num_stages) != C.NUM_TASKS:
            raise ValueError(f"num_stages list must have {C.NUM_TASKS} entries, got {len(num_stages)}")
        out = [int(s) for s in num_stages]
    else:
        out = [int(num_stages)] * C.NUM_TASKS
    if any(s < 1 for s in out):
        raise ValueError("num_stages entries must be >= 1")
    return out


class _FakeBase(Backend):
    flavor = "fake"

    def __init__(
        self,
        horizon: int = 32,
        image_size: int = 224,
        delay_ms: float = 0.0,
        warmup_ms: float = 0.0,
        fail_task_ids: list[int] | None = None,
        slow_task_ids: list[int] | None = None,
        slow_ms: float = 0.0,
        checkpoint: str | None = None,
    ):
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        self.delay_ms = float(delay_ms)
        self.warmup_ms = float(warmup_ms)
        self.fail_task_ids = {int(t) for t in (fail_task_ids or [])}
        self.slow_task_ids = {int(t) for t in (slow_task_ids or [])}
        self.slow_ms = float(slow_ms)
        self.checkpoint = checkpoint
        self.calls = 0  # infer() calls (micro-batches), for tests
        self.items_seen = 0

    # ---- Backend API ---------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "action_horizon": self.horizon,
            "image_size": self.image_size,
            "num_stages": None,
            "supports_inpaint": False,
            "supports_stage": False,
        }

    def warmup(self) -> float:
        t0 = time.monotonic()
        if self.warmup_ms > 0:
            time.sleep(self.warmup_ms / 1e3)
        dummy = InferItem(
            task_id=0, prompt="warmup", proprio=np.zeros(C.PROPRIO_DIM, np.float32),
            images={"head": np.zeros((self.image_size, self.image_size, 3), np.uint8)},
        )
        self._chunk(dummy)
        return (time.monotonic() - t0) * 1e3

    def infer(self, items: list[InferItem]) -> list[ChunkOut]:
        self.calls += 1
        self.items_seen += len(items)
        delay = self.delay_ms
        for item in items:
            if int(item.task_id) in self.fail_task_ids:
                raise RuntimeError(f"injected failure for task {item.task_id}")
            if int(item.task_id) in self.slow_task_ids:
                delay = max(delay, self.slow_ms)
        if delay > 0:
            time.sleep(delay / 1e3)
        return [self._chunk(item) for item in items]

    # ---- helpers -------------------------------------------------------------------------------------------
    def _chunk(self, item: InferItem) -> ChunkOut:
        raise NotImplementedError

    @staticmethod
    def _hold(item: InferItem) -> np.ndarray:
        return hold_action(np.asarray(item.proprio, dtype=np.float32))


class HoldBackend(_FakeBase):
    """Every predicted action holds the current pose (base 0, joints = current qpos, grippers = current width)."""

    flavor = "fake_hold"

    def _chunk(self, item: InferItem) -> ChunkOut:
        return ChunkOut(actions=np.tile(self._hold(item), (self.horizon, 1)).astype(np.float32))


class SineBackend(_FakeBase):
    """Smooth, deterministic motion around the current pose.

    ``actions[t] = hold + amplitude * sin(2 pi (t + 1) / period + phase_j)`` on the torso and arm joints (each
    joint j has its own phase), base velocity ``base_amplitude * sin(...)``; grippers stay at the current width
    unless ``gripper_open`` is set (then +1). The phase also depends on the task id, so two tasks differ.

    ``initial_actions`` (k, 23), when given and ``supports_inpaint`` is on, become the first k actions verbatim
    and the rest continues from the last of them. With ``num_stages`` (int or a 100-entry list) the backend
    reports ``supports_stage`` and emits one-hot ``subtask_logits`` (length ``max(num_stages)``) peaking at
    ``min(stage + stage_step, num_stages[task] - 1)``.
    """

    flavor = "fake_sine"

    def __init__(
        self,
        amplitude: float = 0.05,
        base_amplitude: float = 0.2,
        period: float = 64.0,
        gripper_open: bool = False,
        num_stages: int | list[int] | None = None,
        stage_step: int = 1,
        supports_inpaint: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.amplitude = float(amplitude)
        self.base_amplitude = float(base_amplitude)
        self.period = float(period)
        if self.period <= 0:
            raise ValueError("period must be > 0")
        self.gripper_open = bool(gripper_open)
        self.num_stages = _num_stages_list(num_stages)
        self.stage_step = int(stage_step)
        self.supports_inpaint = bool(supports_inpaint)

    def info(self) -> dict[str, Any]:
        info = super().info()
        info["num_stages"] = self.num_stages
        info["supports_stage"] = self.num_stages is not None
        info["supports_inpaint"] = self.supports_inpaint
        return info

    def _chunk(self, item: InferItem) -> ChunkOut:
        hold = self._hold(item).astype(np.float64)
        t = np.arange(1, self.horizon + 1, dtype=np.float64)[:, None]  # (H, 1)
        joint_phase = np.linspace(0.0, np.pi, C.ACTION_DIM)[None, :] + 0.1 * (int(item.task_id) % 17)
        wave = np.sin(2.0 * np.pi * t / self.period + joint_phase)  # (H, 23)
        actions = np.tile(hold, (self.horizon, 1))
        joints = np.r_[3:14, 15:22]
        actions[:, joints] += self.amplitude * wave[:, joints]
        actions[:, 0:3] = self.base_amplitude * wave[:, 0:3]
        if self.gripper_open:
            actions[:, [C.LEFT_GRIPPER_ACTION_IDX, C.RIGHT_GRIPPER_ACTION_IDX]] = 1.0

        init = item.initial_actions
        if self.supports_inpaint and init is not None:
            init = np.asarray(init, dtype=np.float64)
            if init.ndim != 2 or init.shape[1] < C.ACTION_DIM:
                raise ValueError(f"initial_actions must be (k, >= 23), got {init.shape}")
            k = min(init.shape[0], self.horizon)
            if k > 0:
                # Shift the motion so it continues from the last prefix action, then splice the prefix in.
                offset = init[k - 1, : C.ACTION_DIM] - actions[k - 1]
                actions[k:] += offset
                actions[:k] = init[:k, : C.ACTION_DIM]

        logits = None
        if self.num_stages is not None:
            n = self.num_stages[int(item.task_id)]
            cur = 0 if item.stage is None else int(item.stage)
            target = min(max(cur + self.stage_step, 0), n - 1)
            logits = np.full(max(self.num_stages), -10.0, dtype=np.float32)
            logits[target] = 10.0
        return ChunkOut(actions=actions.astype(np.float32), subtask_logits=logits)


class ReplayBackend(_FakeBase):
    """Replays an ``.npy`` file of absolute actions (N, >= 23).

    Each task id has a cursor that starts at 0 and advances by ``stride`` (default: the horizon) per returned
    chunk, so with ``stride == execute_steps`` consecutive plans of one rollout replay the file contiguously.
    Past the end, the last row is repeated with zero base velocity. ``loop=True`` wraps around instead. The cursor
    is per task, not per rollout (the worker protocol has no rollout id): use one rollout per task at a time.
    """

    flavor = "fake_replay"

    def __init__(self, path: str | None = None, stride: int | None = None, loop: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        path = path or kwargs.get("checkpoint") or self.checkpoint
        if not path:
            raise ValueError("ReplayBackend needs path= (or --checkpoint) pointing to an .npy action file")
        data = np.load(path, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] < C.ACTION_DIM or data.shape[0] < 1:
            raise ValueError(f"{path}: expected (N, >= {C.ACTION_DIM}) actions, got {data.shape}")
        if not np.all(np.isfinite(data[:, : C.ACTION_DIM])):
            raise ValueError(f"{path}: contains non-finite actions")
        self.path = path
        self.actions = np.ascontiguousarray(data[:, : C.ACTION_DIM], dtype=np.float32)
        self.stride = self.horizon if stride is None else int(stride)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.loop = bool(loop)
        self.cursors: dict[int, int] = {}

    def reset_cursors(self) -> None:
        self.cursors.clear()

    def warmup(self) -> float:
        return 0.0  # must not move any cursor

    def _chunk(self, item: InferItem) -> ChunkOut:
        tid = int(item.task_id)
        start = self.cursors.get(tid, 0)
        n = self.actions.shape[0]
        if self.loop:
            idx = (start + np.arange(self.horizon)) % n
            out = self.actions[idx].copy()
        else:
            idx = start + np.arange(self.horizon)
            out = self.actions[np.minimum(idx, n - 1)].copy()
            out[idx >= n, 0:3] = 0.0
        self.cursors[tid] = start + self.stride
        return ChunkOut(actions=out.astype(np.float32))
