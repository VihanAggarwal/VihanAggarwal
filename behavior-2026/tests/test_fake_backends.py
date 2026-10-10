"""Deterministic fake backends (b1k26.backends.fake)."""

from __future__ import annotations

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.backends.base import InferItem, create_backend
from b1k26.backends.fake import HoldBackend, ReplayBackend, SineBackend
from b1k26.obs import hold_action

from runtime_helpers import proprio_at


def mk(task_id: int = 3, step: int = 0, stage=None, init=None) -> InferItem:
    img = np.zeros((32, 32, 3), np.uint8)
    return InferItem(task_id=task_id, prompt="p", proprio=proprio_at(step), images={"head": img}, stage=stage,
                     initial_actions=init)


def test_registry_names() -> None:
    assert isinstance(create_backend("fake_hold"), HoldBackend)
    assert isinstance(create_backend("fake_sine", period=10), SineBackend)
    with pytest.raises(KeyError):
        create_backend("nope")


def test_hold_backend() -> None:
    b = HoldBackend(horizon=10)
    out = b.infer([mk(step=4), mk(step=9)])
    assert len(out) == 2
    np.testing.assert_array_equal(out[0].actions, np.tile(hold_action(proprio_at(4)), (10, 1)))
    assert out[0].actions.dtype == np.float32 and out[0].subtask_logits is None
    info = b.info()
    assert info["action_horizon"] == 10 and info["supports_stage"] is False and b.warmup() >= 0


def test_sine_is_deterministic_smooth_and_near_pose() -> None:
    b1, b2 = SineBackend(horizon=32), SineBackend(horizon=32)
    a1 = b1.infer([mk(3, 5)])[0].actions
    a2 = b2.infer([mk(3, 5)])[0].actions
    np.testing.assert_array_equal(a1, a2)
    assert a1.shape == (32, 23) and a1.dtype == np.float32 and np.all(np.isfinite(a1))
    hold = hold_action(proprio_at(5))
    assert np.max(np.abs(a1[:, 3:14] - hold[3:14])) <= 0.05 + 1e-6
    assert np.max(np.abs(np.diff(a1[:, 3:14], axis=0))) < 0.01  # smooth
    assert np.all(np.abs(a1[:, 0:3]) <= 0.2 + 1e-6)
    np.testing.assert_array_equal(a1[:, 14], hold[14])  # grippers stay put
    assert not np.array_equal(a1, b1.infer([mk(4, 5)])[0].actions)  # task-dependent phase


def test_sine_inpaint_prefix_and_stages() -> None:
    b = SineBackend(horizon=20, num_stages=[2] * 50 + [5] * 50, stage_step=1)
    init = np.full((4, 23), 0.3, np.float32)
    out = b.infer([mk(60, 0, stage=3, init=init)])[0]
    np.testing.assert_array_equal(out.actions[:4], init)
    assert np.all(np.isfinite(out.actions))
    assert out.subtask_logits.shape == (5,) and int(np.argmax(out.subtask_logits)) == 4
    out = b.infer([mk(10, 0, stage=1)])[0]
    assert int(np.argmax(out.subtask_logits)) == 1  # clamped to num_stages[10] - 1
    info = b.info()
    assert info["supports_stage"] and info["num_stages"][60] == 5 and info["supports_inpaint"]
    no_inpaint = SineBackend(horizon=20, supports_inpaint=False)
    a = no_inpaint.infer([mk(60, 0, init=init)])[0].actions
    assert not np.array_equal(a[:4], init)
    with pytest.raises(ValueError):
        SineBackend(num_stages=[3, 3])


def test_injected_failures() -> None:
    b = SineBackend(horizon=8, fail_task_ids=[7])
    with pytest.raises(RuntimeError, match="injected"):
        b.infer([mk(3), mk(7)])
    assert len(b.infer([mk(3)])) == 1


def test_replay_backend(tmp_path) -> None:
    data = np.arange(10 * 23, dtype=np.float32).reshape(10, 23) / 100.0
    path = tmp_path / "a.npy"
    np.save(path, data)
    b = ReplayBackend(path=str(path), horizon=4)
    np.testing.assert_array_equal(b.infer([mk(1)])[0].actions, data[0:4])
    np.testing.assert_array_equal(b.infer([mk(1)])[0].actions, data[4:8])
    np.testing.assert_array_equal(b.infer([mk(2)])[0].actions, data[0:4])  # separate cursor per task
    tail = b.infer([mk(1)])[0].actions
    np.testing.assert_array_equal(tail[:2], data[8:10])
    np.testing.assert_array_equal(tail[2:, 3:], np.tile(data[9, 3:], (2, 1)))
    np.testing.assert_array_equal(tail[2:, :3], 0.0)  # base stopped past the end
    assert b.warmup() == 0.0
    looped = ReplayBackend(path=str(path), horizon=4, stride=8, loop=True)
    looped.infer([mk(1)])
    np.testing.assert_array_equal(looped.infer([mk(1)])[0].actions, data[[8, 9, 0, 1]])
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((3, 5)))
    with pytest.raises(ValueError):
        ReplayBackend(path=str(bad))
    with pytest.raises(ValueError):
        ReplayBackend()
    assert C.ACTION_DIM == 23
