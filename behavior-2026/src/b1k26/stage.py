"""Stage tracking by majority voting over the model's predicted subtask (RLC 2025).

Exact port of B1KPolicyWrapper.update_current_stage from the RLC 2025 wrapper (shared/eval_b1k_wrapper.py):
- the predicted stage is argmax(logits), clamped to num_stages - 1;
- predictions go into a history of length ``history`` (3); decisions are taken only when it is full;
- with next = stage + 1 <= max_stage: promote to next on >= ``votes_to_promote`` (2) votes for next;
  otherwise, on ``history`` unanimous votes for next + 1 ("skip" in RLC), also move to next (one stage only);
  otherwise go back to stage - 1 on ``history`` unanimous votes for stage - 1 (only when stage > 0);
- at the last stage (next > max_stage) nothing changes, including going back;
- the history is cleared on every transition.
"""

from __future__ import annotations

import collections

import numpy as np


class StageTracker:
    def __init__(self, num_stages: int, history: int = 3, votes_to_promote: int = 2):
        if int(num_stages) < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        if int(history) < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.num_stages = int(num_stages)
        self.history = int(history)
        self.votes_to_promote = int(votes_to_promote)
        self.stage = 0
        self.predictions: collections.deque[int] = collections.deque(maxlen=self.history)

    @property
    def max_stage(self) -> int:
        return self.num_stages - 1

    def reset(self) -> None:
        self.stage = 0
        self.predictions.clear()

    def set_stage(self, stage: int) -> None:
        """Force the stage (e.g. a correction rule's stage reset); clears the history like RLC does."""
        self.stage = int(min(max(int(stage), 0), self.max_stage))
        self.predictions.clear()

    def update(self, logits: np.ndarray | None) -> int:
        """Vote with one prediction's subtask logits and return the (possibly updated) stage.

        ``None``, empty or non-finite logits are ignored (the stage and history do not change).
        """
        if logits is None:
            return self.stage
        x = np.asarray(logits, dtype=np.float64).reshape(-1)
        if x.size == 0 or not np.all(np.isfinite(x)):
            return self.stage
        predicted = int(np.argmax(x))
        if predicted > self.max_stage:
            predicted = self.max_stage
        self.predictions.append(predicted)
        if len(self.predictions) == self.history:
            next_stage = self.stage + 1
            if next_stage <= self.max_stage:
                votes_for_next = sum(1 for p in self.predictions if p == next_stage)
                votes_to_skip = sum(1 for p in self.predictions if p == next_stage + 1)
                votes_to_go_back = sum(1 for p in self.predictions if p == self.stage - 1)
                if votes_for_next >= self.votes_to_promote:
                    self.stage = next_stage
                    self.predictions.clear()
                elif votes_to_skip == self.history:
                    self.stage = next_stage
                    self.predictions.clear()
                elif votes_to_go_back == self.history and self.stage > 0:
                    self.stage -= 1
                    self.predictions.clear()
        return self.stage
