"""Gripper-reopen correction rule, generalized from RLC's 2025 correction_rules.py to 100 tasks.

When a gripper is fully closed (closed on nothing: normalized width < closed_threshold = -0.98) in a task, stage
or task phase where the human demonstrations never close it fully, the policy has almost certainly missed a
grasp. The predicted chunk is then replaced by a "hold pose" chunk with that gripper commanded fully open (+1),
which lets the policy retry. Rules are data in ``b1k26/data/gripper_rules.json``:

    {"closed_threshold": -0.98,
     "tasks": {"<id>": {"left":  {"always_open": bool, "min_stage": int|null, "min_progress": float|null},
                        "right": {...},
                        "exempt_right": bool,           # right gripper never corrected (RLC spray tasks 38, 39)
                        "rlc_task0_rule": bool,         # optional, task 0 only: RLC's stage-4 -> 2 reset rule
                        "name"/"source"/"demo_stats": informational, ignored here}}}

A side rule triggers when that gripper is closed and
- ``always_open`` is true; or
- ``min_stage`` is set, a stage is known (stage-tracking models) and stage < min_stage (the stage then decides
  alone); or
- the stage rule does not apply (no ``min_stage`` or no known stage), ``min_progress`` is set, a progress is
  given and progress < min_progress, where progress is ``step / human_mean_len`` of the task (``task_progress``).
A task (or side) without a rule is never corrected.

Differences from RLC (2025), on purpose:
- The hold chunk commands base velocity 0. RLC tiled the 23-D state into every action, which put the measured
  base_qvel into the base command; that was ~0 in the 2025 data but is a real velocity in 2026.
- Gripper values in the hold chunk are clipped to [-1, 1] (the controller clips anyway).
- Output chunks that were replaced are (T, 23) float32 even if the model emitted more than 23 dims.
The other gripper keeps its current normalized width, as in RLC. With the R1Pro's assisted grasping
(position control, grasping direction "lower"), any command below fully open keeps an assisted grasp alive,
so holding the other gripper at its current width does not drop a held object.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping

import numpy as np

from b1k26 import constants as C
from b1k26.obs import hold_from_state23

logger = logging.getLogger(__name__)

DEFAULT_CLOSED_THRESHOLD = -0.98  # RLC CLOSED_THRESHOLD
OPEN_THRESHOLD = 0.90  # RLC OPEN_THRESHOLD (used by the task-0 rule only)
_SIDE_IDX = {"left": C.LEFT_GRIPPER_ACTION_IDX, "right": C.RIGHT_GRIPPER_ACTION_IDX}


@dataclass(frozen=True)
class SideRule:
    always_open: bool = False
    min_stage: int | None = None
    min_progress: float | None = None


@dataclass(frozen=True)
class TaskRule:
    left: SideRule | None = None
    right: SideRule | None = None
    exempt_right: bool = False
    rlc_task0_rule: bool = False


def task_progress(task_id: int, step: int) -> float:
    """Progress measure used with ``min_progress``: rollout step / mean human demo length of the task."""
    return float(step) / float(C.task(int(task_id)).human_mean_len)


def _parse_side(raw: Any, where: str) -> SideRule | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected an object, got {type(raw).__name__}")
    min_stage = raw.get("min_stage")
    min_progress = raw.get("min_progress")
    if min_stage is not None and (isinstance(min_stage, bool) or int(min_stage) != min_stage or min_stage < 0):
        raise ValueError(f"{where}.min_stage must be a non-negative int or null, got {min_stage!r}")
    if min_progress is not None:
        min_progress = float(min_progress)
        if not np.isfinite(min_progress) or min_progress < 0:
            raise ValueError(f"{where}.min_progress must be a non-negative number or null, got {min_progress!r}")
    always_open = raw.get("always_open", False)
    if not isinstance(always_open, bool):
        raise ValueError(f"{where}.always_open must be a bool, got {always_open!r}")
    return SideRule(
        always_open=always_open,
        min_stage=None if min_stage is None else int(min_stage),
        min_progress=min_progress,
    )


class GripperRules:
    """Per-task gripper correction rules (see the module docstring)."""

    def __init__(self, tasks: Mapping[int, TaskRule] | None = None, closed_threshold: float = DEFAULT_CLOSED_THRESHOLD):
        self.tasks: dict[int, TaskRule] = dict(tasks or {})
        self.closed_threshold = float(closed_threshold)

    # ------------------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "GripperRules":
        if not isinstance(doc, Mapping) or not isinstance(doc.get("tasks", {}), Mapping):
            raise ValueError("gripper rules must be an object with a 'tasks' object")
        thr = float(doc.get("closed_threshold", DEFAULT_CLOSED_THRESHOLD))
        if not -1.0 <= thr < 1.0:
            raise ValueError(f"closed_threshold must be in [-1, 1), got {thr}")
        tasks: dict[int, TaskRule] = {}
        for key, entry in doc.get("tasks", {}).items():
            try:
                tid = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"gripper rules: task key {key!r} is not an int") from None
            if not 0 <= tid < C.NUM_TASKS:
                raise ValueError(f"gripper rules: task id {tid} outside [0, {C.NUM_TASKS})")
            if not isinstance(entry, Mapping):
                raise ValueError(f"gripper rules: task {tid} entry must be an object")
            tasks[tid] = TaskRule(
                left=_parse_side(entry.get("left"), f"tasks.{tid}.left"),
                right=_parse_side(entry.get("right"), f"tasks.{tid}.right"),
                exempt_right=bool(entry.get("exempt_right", False)),
                rlc_task0_rule=bool(entry.get("rlc_task0_rule", False)) and tid == 0,
            )
        return cls(tasks, thr)

    @classmethod
    def load(cls, path: str | None = None) -> "GripperRules":
        """Load rules from ``path`` (default: the package's data/gripper_rules.json)."""
        if path is None:
            text = resources.files("b1k26.data").joinpath("gripper_rules.json").read_text()
        else:
            with open(path) as f:
                text = f.read()
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------------------------------------------
    def _side_needs_opening(self, rule: SideRule | None, stage: int | None, progress: float | None) -> bool:
        if rule is None:
            return False
        if rule.always_open:
            return True
        if rule.min_stage is not None and stage is not None:
            return stage < rule.min_stage  # a known stage decides alone (RLC MIN_STAGE_FOR_CLOSURE)
        if rule.min_progress is not None and progress is not None:
            return progress < rule.min_progress
        return False

    def needs_opening(
        self, task_id: int, stage: int | None, state23: np.ndarray, progress: float | None = None
    ) -> tuple[bool, bool]:
        """(left, right): whether the general rule wants to reopen each gripper (RLC general_gripper_correction)."""
        rule = self.tasks.get(int(task_id))
        if rule is None:
            return False, False
        s = np.asarray(state23, dtype=np.float64).reshape(-1)
        left_closed = bool(s[C.LEFT_GRIPPER_ACTION_IDX] < self.closed_threshold)
        right_closed = bool(s[C.RIGHT_GRIPPER_ACTION_IDX] < self.closed_threshold)
        left = left_closed and self._side_needs_opening(rule.left, stage, progress)
        right = right_closed and not rule.exempt_right and self._side_needs_opening(rule.right, stage, progress)
        return left, right

    @staticmethod
    def _chunk_len(actions: np.ndarray) -> int:
        shape = np.shape(actions)
        if len(shape) == 3 and shape[0] == 1:  # (1, T, D) batch of one
            return int(shape[1])
        return int(shape[0]) if len(shape) >= 2 else 1

    @staticmethod
    def _hold_chunk(state23: np.ndarray, n: int, open_sides: tuple[str, ...]) -> np.ndarray:
        hold = hold_from_state23(state23)
        for side in open_sides:
            hold[_SIDE_IDX[side]] = 1.0
        return np.tile(hold, (max(int(n), 1), 1)).astype(np.float32)

    def _task0_rule(self, stage: int, state23: np.ndarray, actions: np.ndarray
                    ) -> tuple[np.ndarray, bool, int] | None:
        """RLC task0_stage4_reset_to_stage2 (stage known). Returns (actions, changed, stage) or None."""
        if stage < 2:
            return None
        corrected_stage = 2 if stage == 4 else stage
        s = np.asarray(state23, dtype=np.float64).reshape(-1)
        lg, rg = s[C.LEFT_GRIPPER_ACTION_IDX], s[C.RIGHT_GRIPPER_ACTION_IDX]
        left_closed, right_closed = lg < self.closed_threshold, rg < self.closed_threshold
        left_middle = not (lg > OPEN_THRESHOLD or left_closed)
        right_middle = not (rg > OPEN_THRESHOLD or right_closed)
        open_sides = []
        if left_closed and not right_middle:
            open_sides.append("left")
        if right_closed and not left_middle:
            open_sides.append("right")
        if not open_sides:
            if stage == corrected_stage:
                return None
            return actions, False, corrected_stage
        logger.info("gripper correction (task 0 rule): stage %d -> %d, opening %s", stage, corrected_stage,
                    "+".join(open_sides))
        return self._hold_chunk(s, self._chunk_len(actions), tuple(open_sides)), True, corrected_stage

    def apply_with_stage(
        self,
        task_id: int,
        stage: int | None,
        state23: np.ndarray,
        actions: np.ndarray,
        progress: float | None = None,
    ) -> tuple[np.ndarray, bool, int | None]:
        """Like ``apply`` but also returns the corrected stage (only RLC's task-0 rule changes it), else None.

        A caller that tracks stages must apply a returned stage with ``StageTracker.set_stage`` (which clears the
        vote history, as RLC does) before voting with this prediction's logits.
        """
        tid = int(task_id)
        rule = self.tasks.get(tid)
        if rule is None:
            return actions, False, None
        new_stage: int | None = None
        if rule.rlc_task0_rule and stage is not None:
            res = self._task0_rule(int(stage), state23, actions)
            if res is not None:
                out, changed, st = res
                return out, changed, (st if st != stage else None)
        left, right = self.needs_opening(tid, stage, state23, progress)
        if not (left or right):
            return actions, False, new_stage
        sides = tuple(s for s, flag in (("left", left), ("right", right)) if flag)
        logger.info("gripper correction: task %d stage %s progress %s: opening %s", tid, stage,
                    None if progress is None else round(progress, 3), "+".join(sides))
        return self._hold_chunk(state23, self._chunk_len(actions), sides), True, new_stage

    def apply(
        self,
        task_id: int,
        stage: int | None,
        state23: np.ndarray,
        actions: np.ndarray,
        progress: float | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Return (actions, changed). If a gripper is fully closed where the demos never close it, the chunk is
        replaced by a (T, 23) float32 hold chunk (base 0, joints = state23, grippers = state23 clipped) with that
        gripper set to +1 (open). Otherwise ``actions`` is returned unchanged (same object).

        ``state23`` is ``obs.state23_action_order(proprio)``. Stage rules apply only when ``stage`` is not None;
        progress rules (``min_progress``) only when ``progress`` is given and no stage rule applies.
        """
        out, changed, _ = self.apply_with_stage(task_id, stage, state23, actions, progress)
        return out, changed
