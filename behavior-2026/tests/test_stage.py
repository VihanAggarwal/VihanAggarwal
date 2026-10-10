"""Tests for b1k26.stage (RLC stage voting)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from b1k26.stage import StageTracker


class RLCVoter:
    """Verbatim logic of RLC B1KPolicyWrapper.update_current_stage (shared/eval_b1k_wrapper.py)."""

    def __init__(self, num_stages, history_len=3, votes_to_promote=2):
        self.num_stages = num_stages
        self.history_len = history_len
        self.votes_to_promote = votes_to_promote
        self.current_stage = 0
        self.prediction_history = deque([], maxlen=history_len)

    def update_current_stage(self, predicted_subtask_logits):
        max_stage = self.num_stages - 1
        predicted_stage = int(np.argmax(predicted_subtask_logits))
        if predicted_stage > max_stage:
            predicted_stage = max_stage
        self.prediction_history.append(predicted_stage)
        if len(self.prediction_history) == self.history_len:
            next_stage = self.current_stage + 1
            if next_stage <= max_stage:
                votes_for_next = sum(1 for pred in self.prediction_history if pred == next_stage)
                votes_to_skip = sum(1 for pred in self.prediction_history if pred == next_stage + 1)
                votes_to_go_back = sum(1 for pred in self.prediction_history if pred == self.current_stage - 1)
                if votes_for_next >= self.votes_to_promote:
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                elif votes_to_skip == self.history_len:
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                elif votes_to_go_back == self.history_len and self.current_stage > 0:
                    self.current_stage -= 1
                    self.prediction_history.clear()


def onehot(i, n=15):
    x = np.zeros(n, np.float32)
    x[i] = 1.0
    return x


@pytest.mark.parametrize("num_stages", [1, 2, 5, 9, 15])
@pytest.mark.parametrize("seed", range(5))
def test_matches_rlc_on_random_sequences(num_stages, seed):
    rng = np.random.default_rng(seed * 100 + num_stages)
    ours = StageTracker(num_stages)
    ref = RLCVoter(num_stages)
    for _ in range(400):
        # Bias predictions around the current stage so that every branch gets exercised.
        center = ref.current_stage + rng.integers(-1, 3)
        pred = int(np.clip(center, 0, 14)) if rng.random() < 0.8 else int(rng.integers(0, 15))
        logits = rng.normal(scale=0.1, size=15).astype(np.float32)
        logits[pred] += 5.0
        ref.update_current_stage(logits)
        assert ours.update(logits) == ref.current_stage
        assert list(ours.predictions) == list(ref.prediction_history)


def test_promote_on_two_of_three():
    t = StageTracker(5)
    assert t.update(onehot(1)) == 0
    assert t.update(onehot(0)) == 0
    assert t.update(onehot(1)) == 1  # 2 votes for next
    assert len(t.predictions) == 0  # cleared on transition


def test_needs_full_history():
    t = StageTracker(5)
    t.update(onehot(1))
    assert t.update(onehot(1)) == 0  # only 2 predictions so far, history is 3
    assert t.update(onehot(3)) == 1


def test_skip_moves_one_stage_only():
    t = StageTracker(5)
    for _ in range(3):
        t.update(onehot(2))
    assert t.stage == 1  # unanimous votes for s+2 advance by one, like RLC


def test_go_back_and_not_from_last_stage():
    t = StageTracker(5)
    t.set_stage(2)
    for _ in range(3):
        t.update(onehot(1))
    assert t.stage == 1
    t.set_stage(4)  # last stage: no decision at all, including going back
    for _ in range(6):
        t.update(onehot(3))
    assert t.stage == 4


def test_clamp_predictions_beyond_num_stages():
    t = StageTracker(2)  # stages 0, 1
    t.update(onehot(14))
    t.update(onehot(9))
    assert list(t.predictions) == [1, 1]
    t.update(onehot(0))
    assert t.stage == 1


def test_reset_set_stage_and_invalid_logits():
    t = StageTracker(5)
    t.update(onehot(1))
    t.update(None)
    t.update(np.array([]))
    t.update(np.full(15, np.nan))
    assert list(t.predictions) == [1]
    t.set_stage(10)
    assert t.stage == 4 and len(t.predictions) == 0
    t.set_stage(-3)
    assert t.stage == 0
    t.update(onehot(1))
    t.reset()
    assert t.stage == 0 and len(t.predictions) == 0
    with pytest.raises(ValueError):
        StageTracker(0)


def test_custom_history_and_votes():
    t = StageTracker(6, history=5, votes_to_promote=3)
    for p in (1, 0, 1, 0):
        t.update(onehot(p))
    assert t.stage == 0
    assert t.update(onehot(1)) == 1
