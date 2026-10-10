"""Per-rollout state kept by the front server.

A ``RolloutSession`` belongs to one evaluator environment slot: ``key = (port, group, batch index)``. It lives in
a slot group owned by the port, not by the websocket connection, so it survives a reconnect (see b1k26.server).
"""

from __future__ import annotations

import array
import collections
import time
from dataclasses import dataclass, field

import numpy as np

from b1k26.stage import StageTracker


@dataclass
class RolloutStats:
    """Counters for the per-rollout log line (reset with the session)."""

    started: float = field(default_factory=time.monotonic)
    queries: int = 0  # engine.step calls that included this session
    plans: int = 0  # successful plans
    plan_failures: int = 0  # worker errors / timeouts (a hold action was sent instead)
    hold_steps: int = 0  # actions that were hold fallbacks
    corrections: int = 0  # chunks replaced by the gripper rule
    compressed: int = 0  # plans executed with temporal compression
    replays: int = 0  # cached responses re-sent after a reconnect
    padded: int = 0  # actions padded because a plan was shorter than the requested chunk
    restart_wait_ms: float = 0.0  # time queries spent waiting for a (re)starting worker instead of holding
    # Compact float arrays (a rollout can have tens of thousands of queries).
    plan_ms: array.array = field(default_factory=lambda: array.array("f"))
    query_ms: array.array = field(default_factory=lambda: array.array("f"))  # server-side time per query

    def summary(self) -> dict[str, float | int]:
        def stats(xs: array.array) -> tuple[float, float, float]:
            if not len(xs):
                return 0.0, 0.0, 0.0
            a = np.frombuffer(xs, dtype=np.float32).astype(np.float64)
            return float(a.mean()), float(np.percentile(a, 50)), float(a.max())

        qm, q50, qx = stats(self.query_ms)
        pm, _, px = stats(self.plan_ms)
        return {
            "queries": self.queries, "plans": self.plans, "plan_failures": self.plan_failures,
            "hold_steps": self.hold_steps, "corrections": self.corrections, "compressed": self.compressed,
            "replays": self.replays, "padded": self.padded, "restart_wait_s": round(self.restart_wait_ms / 1e3, 1),
            "query_ms_mean": round(qm, 2), "query_ms_p50": round(q50, 2), "query_ms_max": round(qx, 2),
            "plan_ms_mean": round(pm, 2), "plan_ms_max": round(px, 2),
            "wall_s": round(time.monotonic() - self.started, 1),
        }


class RolloutSession:
    """State of one rollout slot: action queue, stage, last action and the reconnect replay cache."""

    def __init__(self, key: tuple):
        self.key: tuple = key
        self.task_id: int | None = None
        self.profile_name: str | None = None
        self.queue: collections.deque[np.ndarray] = collections.deque()  # pending (23,) float32 actions
        self.inpaint_tail: np.ndarray | None = None
        self.stage: StageTracker | None = None
        self.step: int = 0  # actions handed to the evaluator in this rollout (= executed steps)
        self.last_action: np.ndarray | None = None
        self.last_fingerprint: bytes | None = None
        self.last_response: tuple | None = None  # (action (23,), chunk (K, 23) | None) for last_fingerprint
        self.fallback_note: str | None = None  # why the routed profile was replaced (logged once)
        self.stats = RolloutStats()

    def reset_plan_state(self) -> None:
        """Forget the current plan (queue, inpainting tail) and the stage; keep identity and counters."""
        self.queue.clear()
        self.inpaint_tail = None
        if self.stage is not None:
            self.stage.reset()

    def reset(self) -> None:
        """Start a new rollout in this slot."""
        self.task_id = None
        self.profile_name = None
        self.queue.clear()
        self.inpaint_tail = None
        self.stage = None
        self.step = 0
        self.last_action = None
        self.last_fingerprint = None
        self.last_response = None
        self.fallback_note = None
        self.stats = RolloutStats()

    @property
    def active(self) -> bool:
        """Whether this slot has served any step since its last reset."""
        return self.stats.queries > 0

    def describe(self) -> str:
        stage = None if self.stage is None else self.stage.stage
        return (f"slot={self.key} task={self.task_id} profile={self.profile_name} steps={self.step} "
                f"stage={stage} queue={len(self.queue)}")
