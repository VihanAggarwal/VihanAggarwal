"""Tests for b1k26.control."""

from __future__ import annotations

import numpy as np
import pytest

from b1k26 import control as K
from b1k26.obs import hold_action

GRIPPER_VARIATION_THRESHOLD = 0.2


# ------------------------------------------------------------------------------------------------------------
# Reference: the chunk-processing part of RLC's B1KPolicyWrapper.act (shared/eval_b1k_wrapper.py), with
# apply_eval_tricks=True and no correction rule firing.
# ------------------------------------------------------------------------------------------------------------
def rlc_interpolate_actions(actions, target_steps):
    from scipy.interpolate import interp1d

    original_indices = np.linspace(0, len(actions) - 1, len(actions))
    target_indices = np.linspace(0, len(actions) - 1, target_steps)
    interpolated = np.zeros((target_steps, actions.shape[1]))
    for dim in range(actions.shape[1]):
        f = interp1d(original_indices, actions[:, dim], kind="cubic")
        interpolated[:, dim] = f(target_indices)
    return interpolated


def rlc_check_gripper_variation(actions, num_actions_to_check):
    actions_to_check = actions[:num_actions_to_check]
    left = actions_to_check[:, 14]
    right = actions_to_check[:, 22]
    lv = float(np.max(left) - np.min(left))
    rv = float(np.max(right) - np.min(right))
    return lv > GRIPPER_VARIATION_THRESHOLD or rv > GRIPPER_VARIATION_THRESHOLD, lv, rv


def rlc_process(actions, actions_to_execute=26, actions_to_keep=4, execute_in_n_steps=20):
    if len(actions.shape) == 3:
        actions = actions[0]
    if actions.shape[1] > 23:
        actions = actions[:, :23]
    should_compress = execute_in_n_steps < actions_to_execute
    if should_compress:
        has_high_variation, _, _ = rlc_check_gripper_variation(actions, actions_to_execute)
        if has_high_variation:
            should_compress = False
    n_exec = actions_to_execute if should_compress else execute_in_n_steps
    inpainting_start = n_exec
    inpainting_end = inpainting_start + actions_to_keep
    next_initial_actions = actions[inpainting_start:inpainting_end].copy() if len(actions) >= inpainting_end else None
    last_actions = actions[:n_exec].copy()
    if should_compress:
        compressed = rlc_interpolate_actions(last_actions, execute_in_n_steps)
        compressed[:, :3] *= n_exec / execute_in_n_steps
        last_actions = compressed
    # The wrapper returns torch.from_numpy(action).float() per step.
    return last_actions.astype(np.float32), next_initial_actions, should_compress


def smooth_chunk(rng, t=30, d=23, grip_const=True):
    """A smooth random chunk with base velocities well inside [-1, 1] after 1.3x scaling."""
    x = np.cumsum(rng.normal(scale=0.05, size=(t, d)), axis=0).astype(np.float32)
    x[:, :3] = np.clip(x[:, :3], -0.7, 0.7)
    if grip_const:
        x[:, 14] = 0.8 + 0.01 * rng.normal(size=t)
        x[:, 22] = -0.5 + 0.01 * rng.normal(size=t)
    return x


# ------------------------------------------------------------------------------------------------------------
def test_matches_rlc_wrapper_compressed():
    rng = np.random.default_rng(0)
    cfg = K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4)
    for _ in range(20):
        raw = smooth_chunk(rng)
        out = K.plan_execution(raw, cfg)
        ref_actions, ref_tail, ref_comp = rlc_process(raw)
        assert out.compressed and ref_comp
        assert out.actions.shape == (20, 23) and out.actions.dtype == np.float32
        np.testing.assert_array_equal(out.actions, ref_actions)
        np.testing.assert_array_equal(out.inpaint_tail, ref_tail)
        np.testing.assert_array_equal(out.inpaint_tail, raw[26:30])


def test_matches_rlc_wrapper_gripper_variation():
    rng = np.random.default_rng(1)
    cfg = K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4)
    raw = smooth_chunk(rng)
    raw[22:, 14] = -1.0  # left gripper closes inside the consumed window
    out = K.plan_execution(raw, cfg)
    ref_actions, ref_tail, ref_comp = rlc_process(raw)
    assert not out.compressed and not ref_comp
    np.testing.assert_array_equal(out.actions, ref_actions)
    np.testing.assert_array_equal(out.actions, raw[:20])
    # Tail starts after the execute_steps raw actions that were consumed.
    np.testing.assert_array_equal(out.inpaint_tail, raw[20:24])
    np.testing.assert_array_equal(out.inpaint_tail, ref_tail)


def test_gripper_variation_outside_window_or_at_threshold_keeps_compression():
    rng = np.random.default_rng(2)
    cfg = K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4)
    raw = smooth_chunk(rng)
    raw[26:, 22] = 1.0  # change after the consumed window: ignored
    assert K.plan_execution(raw, cfg).compressed
    raw = smooth_chunk(rng)
    raw[:, 14] = 0.5
    raw[10, 14] = 0.7  # range exactly 0.2 is not "> 0.2"
    raw[:, 14] = raw[:, 14].astype(np.float64)
    lv, _ = K.gripper_variation(raw, 26)
    assert lv <= 0.2 + 1e-6
    out = K.plan_execution(raw, K.ExecutionConfig(disable_compression_gripper_range=lv))
    assert out.compressed
    out = K.plan_execution(raw, K.ExecutionConfig(disable_compression_gripper_range=lv - 1e-4))
    assert not out.compressed
    out = K.plan_execution(raw, K.ExecutionConfig(disable_compression_gripper_range=None))
    assert out.compressed


def test_identity_when_predicted_equals_execute():
    rng = np.random.default_rng(3)
    raw = smooth_chunk(rng, t=50)
    raw[:, 14] = np.linspace(1, -1, 50)  # gripper variation is irrelevant without compression
    for e in (1, 20, 32):
        cfg = K.ExecutionConfig(execute_steps=e, predicted_steps_to_use=e, keep_for_inpaint=4)
        out = K.plan_execution(raw, cfg)
        assert not out.compressed
        np.testing.assert_array_equal(out.actions, raw[:e])
        np.testing.assert_array_equal(out.inpaint_tail, raw[e:e + 4])
        assert not np.shares_memory(out.actions, raw)


def test_no_tail_when_keep_zero():
    rng = np.random.default_rng(4)
    raw = smooth_chunk(rng, t=50)
    out = K.plan_execution(raw, K.ExecutionConfig(execute_steps=32, predicted_steps_to_use=32, keep_for_inpaint=0))
    assert out.inpaint_tail is None and out.actions.shape == (32, 23)


def test_compression_endpoints_and_base_scaling():
    rng = np.random.default_rng(5)
    raw = smooth_chunk(rng)
    raw[:, 0] = 0.5
    raw[:, 1] = -0.25
    raw[:, 2] = 0.9  # 0.9 * 1.3 = 1.17 -> clipped to 1
    cfg = K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4)
    out = K.plan_execution(raw, cfg)
    assert out.compressed
    np.testing.assert_allclose(out.actions[0, 3:], raw[0, 3:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.actions[-1, 3:], raw[25, 3:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.actions[:, 0], 0.5 * 26 / 20, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.actions[:, 1], -0.25 * 26 / 20, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(out.actions[:, 2], 1.0)
    cfg2 = K.ExecutionConfig(base_velocity_scale_with_compression=False)
    np.testing.assert_allclose(K.plan_execution(raw, cfg2).actions[:, 0], 0.5, rtol=0, atol=1e-6)


def test_linear_ramp_is_resampled_exactly():
    raw = np.zeros((30, 23), np.float32)
    raw[:, 7] = np.arange(30)
    out = K.plan_execution(raw, K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26))
    np.testing.assert_allclose(out.actions[:, 7], np.linspace(0, 25, 20), atol=1e-5)


def test_short_chunks():
    rng = np.random.default_rng(6)
    cfg = K.ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4)
    raw = smooth_chunk(rng, t=10)
    out = K.plan_execution(raw, cfg)  # fewer than execute_steps: everything, uncompressed, no tail
    assert not out.compressed and out.inpaint_tail is None
    np.testing.assert_array_equal(out.actions, raw)

    raw = smooth_chunk(rng, t=24)  # 24 > 20 available: compress 24 -> 20 with factor 1.2, no tail
    raw[:, 0] = 0.5
    out = K.plan_execution(raw, cfg)
    assert out.compressed and out.actions.shape == (20, 23) and out.inpaint_tail is None
    np.testing.assert_allclose(out.actions[:, 0], 0.5 * 24 / 20, atol=1e-6)
    np.testing.assert_allclose(out.actions[-1, 3:], raw[23, 3:], atol=1e-6)

    raw = smooth_chunk(rng, t=28)  # enough to compress, not enough for the tail
    out = K.plan_execution(raw, cfg)
    assert out.compressed and out.inpaint_tail is None
    ref_actions, ref_tail, _ = rlc_process(raw)
    np.testing.assert_array_equal(out.actions, ref_actions)
    assert ref_tail is None

    raw = smooth_chunk(rng, t=22)  # gripper variation, uncompressed: 20 actions, tail would need 24
    raw[5:, 22] = 1.0
    out = K.plan_execution(raw, cfg)
    assert not out.compressed and out.actions.shape == (20, 23) and out.inpaint_tail is None


def test_tiny_compression_uses_linear_fallback():
    raw = np.zeros((3, 23), np.float32)
    raw[:, 7] = [0.0, 1.0, 2.0]
    out = K.plan_execution(raw, K.ExecutionConfig(execute_steps=2, predicted_steps_to_use=3, keep_for_inpaint=0))
    assert out.compressed
    np.testing.assert_allclose(out.actions[:, 7], [0.0, 2.0])


def test_predicted_less_than_execute_is_uncompressed():
    rng = np.random.default_rng(7)
    raw = smooth_chunk(rng, t=40)
    out = K.plan_execution(raw, K.ExecutionConfig(execute_steps=26, predicted_steps_to_use=20, keep_for_inpaint=4))
    ref_actions, ref_tail, ref_comp = rlc_process(raw, actions_to_execute=20, execute_in_n_steps=26)
    assert not out.compressed and not ref_comp
    np.testing.assert_array_equal(out.actions, ref_actions)
    np.testing.assert_array_equal(out.inpaint_tail, ref_tail)


def test_chunk_shapes_accepted():
    rng = np.random.default_rng(8)
    raw32 = np.concatenate([smooth_chunk(rng), rng.normal(size=(30, 9)).astype(np.float32)], axis=1)
    out = K.plan_execution(raw32[None], K.ExecutionConfig())
    ref_actions, ref_tail, _ = rlc_process(raw32[None])
    np.testing.assert_array_equal(out.actions, ref_actions)
    assert out.inpaint_tail.shape == (4, 23)
    with pytest.raises(ValueError):
        K.plan_execution(np.zeros((0, 23)), K.ExecutionConfig())
    with pytest.raises(ValueError):
        K.plan_execution(np.zeros((10, 22)), K.ExecutionConfig())
    with pytest.raises(ValueError):
        K.plan_execution(np.zeros(23), K.ExecutionConfig())


def test_nonfinite_chunk_is_not_compressed():
    rng = np.random.default_rng(9)
    raw = smooth_chunk(rng)
    raw[3, 10] = np.nan
    raw[27, 0] = np.inf  # would be in the compressed tail, not in the uncompressed one
    out = K.plan_execution(raw, K.ExecutionConfig())
    assert not out.compressed
    assert np.isnan(out.actions[3, 10]) and np.all(np.isfinite(np.delete(out.actions, 3, axis=0)))
    np.testing.assert_array_equal(out.inpaint_tail, raw[20:24])
    # A non-finite tail is dropped instead of being fed to inpainting.
    raw = smooth_chunk(rng)
    raw[27, 5] = np.nan
    out = K.plan_execution(raw, K.ExecutionConfig())
    assert out.compressed and out.inpaint_tail is None and np.all(np.isfinite(out.actions))


def test_config_validation():
    with pytest.raises(ValueError):
        K.ExecutionConfig(execute_steps=0)
    with pytest.raises(ValueError):
        K.ExecutionConfig(predicted_steps_to_use=0)
    with pytest.raises(ValueError):
        K.ExecutionConfig(keep_for_inpaint=-1)
    assert K.ExecutionConfig().compresses
    assert not K.ExecutionConfig(execute_steps=32, predicted_steps_to_use=32).compresses


# ------------------------------------------------------------------------------------------------------------
# sanitize
# ------------------------------------------------------------------------------------------------------------
def test_sanitize_replaces_nonfinite_rows_with_hold():
    rng = np.random.default_rng(10)
    proprio = rng.normal(size=61).astype(np.float32)
    proprio[24:26] = 0.02
    proprio[49:51] = 0.0
    acts = rng.uniform(-0.5, 0.5, size=(5, 23)).astype(np.float32)
    acts[1, 4] = np.nan
    acts[3, 0] = -np.inf
    readonly = acts.copy()
    readonly.flags.writeable = False
    out = K.sanitize(readonly, proprio)
    assert out.dtype == np.float32 and out.shape == (5, 23) and np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[1], hold_action(proprio))
    np.testing.assert_array_equal(out[3], hold_action(proprio))
    np.testing.assert_array_equal(out[[0, 2, 4]], acts[[0, 2, 4]])
    assert np.isnan(readonly[1, 4])  # input untouched


def test_sanitize_clips_base_and_grippers():
    proprio = np.zeros(61, np.float32)
    acts = np.zeros((2, 23), np.float32)
    acts[0, :3] = [1.5, -2.0, 0.3]
    acts[0, 14] = 1.7
    acts[1, 22] = -3.0
    acts[1, 7] = 5.0  # joint positions are not clipped
    out = K.sanitize(acts, proprio)
    np.testing.assert_array_equal(out[0, :3], np.float32([1.0, -1.0, 0.3]))
    assert out[0, 14] == 1.0 and out[1, 22] == -1.0 and out[1, 7] == 5.0
    out = K.sanitize(acts, proprio, clip_base=0.5)
    np.testing.assert_array_equal(out[0, :3], np.float32([0.5, -0.5, 0.3]))


def test_sanitize_single_action_and_wide():
    proprio = np.zeros(61, np.float32)
    proprio[53:57] = [1.0, -1.5, -0.5, 0.0]
    a = np.full(23, np.nan, np.float32)
    out = K.sanitize(a, proprio)
    assert out.shape == (23,)
    np.testing.assert_array_equal(out, hold_action(proprio))
    wide = np.zeros((3, 32), np.float32)
    assert K.sanitize(wide, proprio).shape == (3, 23)
    with pytest.raises(ValueError):
        K.sanitize(np.zeros((3, 20)), proprio)
