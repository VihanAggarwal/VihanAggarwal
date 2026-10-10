"""Tests for b1k26.obs."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

from b1k26 import constants as C
from b1k26 import obs as O
from b1k26.protocol import packb, unpackb


# ------------------------------------------------------------------------------------------------------------
# Verbatim copy of openpi_client.image_tools (openpi-comet packages/openpi-client/src/openpi_client/image_tools.py)
# ------------------------------------------------------------------------------------------------------------
def ref_resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_ref_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _ref_resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def ref_convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


# ------------------------------------------------------------------------------------------------------------
# RLC extract_state_from_proprio (JackLiu port, src/b1k/policies/b1k_policy.py) with the 61-D index table of
# eval/utils/eval_utils.py PROPRIOCEPTION_INDICES["R1Pro"], written out independently of b1k26.constants.
# ------------------------------------------------------------------------------------------------------------
R1PRO_61 = {
    "base_qvel": np.s_[0:3],
    "arm_left_qpos": np.s_[3:10],
    "arm_left_qvel": np.s_[10:17],
    "eef_left_pos": np.s_[17:20],
    "eef_left_quat": np.s_[20:24],
    "gripper_left_qpos": np.s_[24:26],
    "gripper_left_qvel": np.s_[26:28],
    "arm_right_qpos": np.s_[28:35],
    "arm_right_qvel": np.s_[35:42],
    "eef_right_pos": np.s_[42:45],
    "eef_right_quat": np.s_[45:49],
    "gripper_right_qpos": np.s_[49:51],
    "gripper_right_qvel": np.s_[51:53],
    "trunk_qpos": np.s_[53:57],
    "trunk_qvel": np.s_[57:61],
}


def rlc_extract_state_from_proprio(proprio_data):
    base_qvel = proprio_data[..., R1PRO_61["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., R1PRO_61["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., R1PRO_61["arm_left_qpos"]]  # 7
    arm_right_qpos = proprio_data[..., R1PRO_61["arm_right_qpos"]]  # 7
    left_gripper_raw = proprio_data[..., R1PRO_61["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., R1PRO_61["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    MAX_GRIPPER_WIDTH = 0.1
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    return np.concatenate(
        [base_qvel, trunk_qpos, arm_left_qpos, left_gripper_width, arm_right_qpos, right_gripper_width], axis=-1
    )


# ------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------
def make_proprio(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    shape = (C.PROPRIO_DIM,) if n is None else (n, C.PROPRIO_DIM)
    p = rng.normal(size=shape).astype(np.float32)
    p[..., 24:26] = rng.uniform(0, 0.05, size=p[..., 24:26].shape)
    p[..., 49:51] = rng.uniform(0, 0.05, size=p[..., 49:51].shape)
    return p


def make_msg(rng, n=2, head=(32, 32), wrist=(24, 24), channels=4, batched=True, task_ids=None, depth=False,
             cam=False):
    def lead(shape):
        return ((n,) if batched else ()) + shape

    msg = {
        C.PROPRIO_KEY: make_proprio(rng, n if batched else None),
        C.HEAD_RGB_KEY: rng.integers(0, 256, size=lead(head + (channels,)), dtype=np.uint8),
        C.LEFT_RGB_KEY: rng.integers(0, 256, size=lead(wrist + (channels,)), dtype=np.uint8),
        C.RIGHT_RGB_KEY: rng.integers(0, 256, size=lead(wrist + (channels,)), dtype=np.uint8),
        C.TASK_ID_KEY: (np.array([[t] for t in (task_ids or [3] * n)], dtype=np.int64) if batched
                        else np.array([(task_ids or [3])[0]], dtype=np.int64)),
    }
    if depth:
        msg[C.depth_key("head")] = rng.uniform(0, 5, size=lead(head)).astype(np.float32)
    if cam:
        msg[C.CAM_REL_POSES_KEY] = rng.normal(size=lead((21,))).astype(np.float32)
    return msg


def roundtrip(msg):
    """Encode/decode like the evaluator -> server path: arrays come back read-only."""
    return unpackb(packb(msg))


# ------------------------------------------------------------------------------------------------------------
# split_batch
# ------------------------------------------------------------------------------------------------------------
def test_split_batch_batched_rgba_readonly():
    rng = np.random.default_rng(0)
    msg = make_msg(rng, n=3, task_ids=[0, 51, 99], depth=True, cam=True)
    dec = roundtrip(msg)
    assert not dec[C.PROPRIO_KEY].flags.writeable
    before = {k: np.array(v, copy=True) for k, v in dec.items()}
    envs = O.split_batch(dec)
    assert len(envs) == 3
    for b, env in enumerate(envs):
        assert env.task_id == [0, 51, 99][b]
        assert env.proprio.shape == (61,) and env.proprio.dtype == np.float32
        np.testing.assert_array_equal(env.proprio, msg[C.PROPRIO_KEY][b])
        assert set(env.rgb) == {"head", "left_wrist", "right_wrist"}
        np.testing.assert_array_equal(env.rgb["head"], msg[C.HEAD_RGB_KEY][b, ..., :3])
        assert env.rgb["head"].shape == (32, 32, 3) and env.rgb["head"].dtype == np.uint8
        assert env.rgb["left_wrist"].shape == (24, 24, 3)
        assert env.depth["head"].shape == (32, 32) and env.depth["head"].dtype == np.float32
        np.testing.assert_array_equal(env.depth["head"], msg[C.depth_key("head")][b])
        assert env.cam_rel_poses.shape == (21,)
        assert len(env.fingerprint) == 16
        # Outputs are writeable copies that do not alias the message.
        for arr in (env.proprio, env.rgb["head"], env.depth["head"], env.cam_rel_poses):
            assert arr.flags.writeable and arr.flags.c_contiguous
            assert not np.shares_memory(arr, dec[C.PROPRIO_KEY])
            assert not np.shares_memory(arr, dec[C.HEAD_RGB_KEY])
        env.rgb["head"][:] = 0
        env.proprio[:] = 0
    for k, v in dec.items():
        np.testing.assert_array_equal(v, before[k])


def test_split_batch_multiport_n1():
    rng = np.random.default_rng(1)
    msg = make_msg(rng, n=1, task_ids=[42])
    envs = O.split_batch(roundtrip(msg))
    assert len(envs) == 1 and envs[0].task_id == 42
    np.testing.assert_array_equal(envs[0].proprio, msg[C.PROPRIO_KEY][0])


def test_split_batch_unbatched_v392():
    rng = np.random.default_rng(2)
    msg = make_msg(rng, batched=False, task_ids=[7], depth=True, cam=True)
    envs = O.split_batch(roundtrip(msg))
    assert len(envs) == 1
    env = envs[0]
    assert env.task_id == 7
    np.testing.assert_array_equal(env.proprio, msg[C.PROPRIO_KEY])
    np.testing.assert_array_equal(env.rgb["head"], msg[C.HEAD_RGB_KEY][..., :3])
    assert env.depth["head"].shape == (32, 32)
    assert env.cam_rel_poses.shape == (21,)


@pytest.mark.parametrize(
    "task_id, n, expected",
    [
        (np.array([[5], [6]], dtype=np.int64), 2, [5, 6]),  # (N, 1)
        (np.array([5, 6], dtype=np.int64), 2, [5, 6]),  # (N,)
        (np.array([9], dtype=np.int64), 2, [9, 9]),  # (1,) broadcast
        (np.array([[9]], dtype=np.int64), 1, [9]),  # (1, 1)
        (np.int64(11), 2, [11, 11]),  # np.generic scalar
        (np.array(12), 1, [12]),  # 0-d array
        (13, 1, [13]),  # python int
        (np.array([[14.0]], dtype=np.float32), 1, [14]),  # integral float
        (np.array([[15]], dtype=np.int32), 1, [15]),
    ],
)
def test_task_id_shapes(task_id, n, expected):
    rng = np.random.default_rng(3)
    msg = make_msg(rng, n=n)
    msg[C.TASK_ID_KEY] = task_id
    assert [e.task_id for e in O.split_batch(roundtrip(msg))] == expected


def test_task_id_unbatched_scalar_generic_roundtrip():
    rng = np.random.default_rng(4)
    msg = make_msg(rng, batched=False)
    msg[C.TASK_ID_KEY] = np.int64(77)
    dec = roundtrip(msg)
    assert isinstance(dec[C.TASK_ID_KEY], np.generic)
    assert O.split_batch(dec)[0].task_id == 77


@pytest.mark.parametrize("bad", [np.array([1, 2, 3]), np.array([[150]]), np.array([[-1]]), np.array([[1.5]])])
def test_task_id_invalid(bad):
    rng = np.random.default_rng(5)
    msg = make_msg(rng, n=2 if bad.size == 3 else 1)
    msg[C.TASK_ID_KEY] = bad
    with pytest.raises(ValueError):
        O.split_batch(msg)


def test_task_id_missing():
    rng = np.random.default_rng(6)
    msg = make_msg(rng, n=2)
    del msg[C.TASK_ID_KEY]
    with pytest.raises(ValueError, match="task_id"):
        O.split_batch(msg)
    assert [e.task_id for e in O.split_batch(msg, default_task_id=4)] == [4, 4]


def test_images_rgb_and_float():
    rng = np.random.default_rng(7)
    msg = make_msg(rng, n=2, channels=3)
    envs = O.split_batch(roundtrip(msg))
    np.testing.assert_array_equal(envs[1].rgb["left_wrist"], msg[C.LEFT_RGB_KEY][1])
    fimg = rng.uniform(0, 1, size=(2, 32, 32, 4)).astype(np.float32)
    msg[C.HEAD_RGB_KEY] = fimg
    envs = O.split_batch(roundtrip(msg))
    assert envs[0].rgb["head"].dtype == np.uint8
    np.testing.assert_array_equal(envs[0].rgb["head"], ref_convert_to_uint8(fimg[0, ..., :3]))
    # Out-of-range / NaN floats never wrap around.
    bad = np.full((2, 32, 32, 3), np.nan, dtype=np.float32)
    bad[:, 0, 0, :] = -0.5
    bad[:, 1, 1, :] = 1.2
    msg[C.HEAD_RGB_KEY] = bad
    head = O.split_batch(msg)[0].rgb["head"]
    assert head[0, 0, 0] == 0 and head[1, 1, 0] == 255 and head[5, 5, 0] == 0


def test_optional_keys_missing():
    rng = np.random.default_rng(8)
    msg = make_msg(rng, n=2)
    del msg[C.LEFT_RGB_KEY]
    envs = O.split_batch(msg)
    assert envs[0].depth == {} and envs[0].cam_rel_poses is None
    assert set(envs[0].rgb) == {"head", "right_wrist"}
    prepared = O.prepare_images(envs[0], 16)
    assert set(prepared) == {"head", "left_wrist", "right_wrist"}
    assert prepared["left_wrist"].shape == (16, 16, 3) and not prepared["left_wrist"].any()


def test_required_keys_missing():
    rng = np.random.default_rng(9)
    msg = make_msg(rng, n=1)
    m1 = dict(msg)
    del m1[C.PROPRIO_KEY]
    with pytest.raises(ValueError, match="proprio"):
        O.split_batch(m1)
    m2 = dict(msg)
    del m2[C.HEAD_RGB_KEY]
    with pytest.raises(ValueError, match="head"):
        O.split_batch(m2)


def test_malformed_shapes():
    rng = np.random.default_rng(10)
    msg = make_msg(rng, n=2)
    m = dict(msg, **{C.PROPRIO_KEY: np.zeros((2, 60), np.float32)})
    with pytest.raises(ValueError):
        O.split_batch(m)
    m = dict(msg, **{C.HEAD_RGB_KEY: msg[C.HEAD_RGB_KEY][:1]})  # batch mismatch
    with pytest.raises(ValueError):
        O.split_batch(m)
    m = dict(msg, **{C.HEAD_RGB_KEY: np.zeros((2, 8, 8, 5), np.uint8)})
    with pytest.raises(ValueError):
        O.split_batch(m)


def test_depth_with_channel_dim():
    rng = np.random.default_rng(11)
    msg = make_msg(rng, n=2)
    d = rng.uniform(size=(2, 32, 32, 1)).astype(np.float32)
    msg[C.depth_key("left_wrist")] = d
    envs = O.split_batch(roundtrip(msg))
    np.testing.assert_array_equal(envs[1].depth["left_wrist"], d[1, ..., 0])


def test_unknown_keys_ignored():
    rng = np.random.default_rng(12)
    msg = make_msg(rng, n=1)
    msg["robot_r1::something_else"] = np.zeros((1, 3))
    msg["need_new_action"] = True
    assert len(O.split_batch(msg)) == 1


# ------------------------------------------------------------------------------------------------------------
# fingerprint
# ------------------------------------------------------------------------------------------------------------
def test_fingerprint_semantics():
    rng = np.random.default_rng(13)
    msg = make_msg(rng, n=2, task_ids=[1, 1])
    msg[C.PROPRIO_KEY][1] = msg[C.PROPRIO_KEY][0]
    msg[C.HEAD_RGB_KEY][1] = msg[C.HEAD_RGB_KEY][0]
    msg[C.LEFT_RGB_KEY][1] = msg[C.LEFT_RGB_KEY][0]
    msg[C.RIGHT_RGB_KEY][1] = msg[C.RIGHT_RGB_KEY][0]
    msg[C.HEAD_RGB_KEY][1, ..., 3] = 0  # alpha differs: ignored
    e = O.split_batch(roundtrip(msg))
    assert e[0].fingerprint == e[1].fingerprint
    assert O.split_batch(roundtrip(msg))[0].fingerprint == e[0].fingerprint  # deterministic

    m2 = {k: np.array(v, copy=True) for k, v in msg.items()}
    m2[C.PROPRIO_KEY][1, 5] += 1e-6
    assert O.split_batch(m2)[1].fingerprint != e[0].fingerprint
    m3 = {k: np.array(v, copy=True) for k, v in msg.items()}
    m3[C.HEAD_RGB_KEY][1, 0, 0, 0] ^= 1  # on the sampling grid
    assert O.split_batch(m3)[1].fingerprint != e[0].fingerprint
    m4 = {k: np.array(v, copy=True) for k, v in msg.items()}
    m4[C.TASK_ID_KEY][1] = 2
    assert O.split_batch(m4)[1].fingerprint != e[0].fingerprint


def test_fingerprint_is_cheap_full_res():
    rng = np.random.default_rng(14)
    rgb = {
        "head": rng.integers(0, 256, size=(720, 720, 3), dtype=np.uint8),
        "left_wrist": rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8),
        "right_wrist": rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8),
    }
    proprio = make_proprio(rng)
    O.compute_fingerprint(3, proprio, rgb)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        O.compute_fingerprint(3, proprio, rgb)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 1e-3


# ------------------------------------------------------------------------------------------------------------
# resize_with_pad
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, out",
    [
        ((720, 720, 3), (224, 224)),
        ((480, 480, 3), (224, 224)),
        ((300, 200, 3), (224, 224)),
        ((200, 300, 3), (224, 224)),
        ((100, 50, 3), (224, 224)),  # upscale + pad
        ((256, 256, 3), (180, 320)),  # non-square target
        ((2, 3, 120, 90, 3), (64, 64)),  # batch dims
    ],
)
@pytest.mark.parametrize("method, pil", [("bilinear", Image.BILINEAR), ("nearest", Image.NEAREST),
                                         ("lanczos", Image.LANCZOS)])
def test_resize_with_pad_bit_exact(shape, out, method, pil):
    rng = np.random.default_rng(hash((shape, out)) % 2**32)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    got = O.resize_with_pad(img, *out, method=method)
    ref = ref_resize_with_pad(img, *out, method=pil)
    assert got.dtype == ref.dtype == np.uint8
    assert got.shape == ref.shape == shape[:-3] + out + (3,)
    np.testing.assert_array_equal(got, ref)


def test_resize_with_pad_default_and_aliases():
    rng = np.random.default_rng(15)
    img = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    ref = ref_resize_with_pad(img, 224, 224)
    np.testing.assert_array_equal(O.resize_with_pad(img, 224, 224), ref)
    np.testing.assert_array_equal(O.resize_with_pad(img, 224, 224, "bilinear_pad"), ref)
    np.testing.assert_array_equal(O.resize_with_pad(img, 224, 224, Image.BILINEAR), ref)
    with pytest.raises(ValueError):
        O.resize_with_pad(img, 224, 224, "cubic-ish")


def test_resize_with_pad_early_return():
    img = np.zeros((224, 224, 3), np.uint8)
    assert O.resize_with_pad(img, 224, 224) is img


def test_prepare_images_from_split():
    rng = np.random.default_rng(16)
    msg = make_msg(rng, n=1, head=(72, 72), wrist=(48, 48))
    env = O.split_batch(roundtrip(msg))[0]
    out = O.prepare_images(env, 32)
    for role, key in (("head", C.HEAD_RGB_KEY), ("left_wrist", C.LEFT_RGB_KEY), ("right_wrist", C.RIGHT_RGB_KEY)):
        np.testing.assert_array_equal(out[role], ref_resize_with_pad(msg[key][0, ..., :3], 32, 32))
    same = O.prepare_images(env, 72)["head"]  # already the right size: a copy, not an alias
    np.testing.assert_array_equal(same, env.rgb["head"])
    assert not np.shares_memory(same, env.rgb["head"])


# ------------------------------------------------------------------------------------------------------------
# state23 / hold
# ------------------------------------------------------------------------------------------------------------
def test_constants_indices_match_eval_utils_table():
    for k, s in R1PRO_61.items():
        assert C.PROPRIO_INDICES_2026[k] == s


def test_state23_matches_rlc_formula():
    rng = np.random.default_rng(17)
    p = make_proprio(rng, 5)
    got = O.state23_action_order(p)
    ref = rlc_extract_state_from_proprio(p.astype(np.float32))
    assert got.shape == (5, 23) and got.dtype == np.float32
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(O.state23_action_order(p[0]), ref[0], rtol=0, atol=1e-6)


def test_state23_index_mapping():
    p = np.arange(61, dtype=np.float32) / 1000.0
    s = O.state23_action_order(p)
    expected_idx = [0, 1, 2, 53, 54, 55, 56, *range(3, 10), None, *range(28, 35), None]
    for j, i in enumerate(expected_idx):
        if i is not None:
            assert s[j] == pytest.approx(p[i]), j
    assert s[14] == pytest.approx(2 * (p[24] + p[25]) / 0.1 - 1)
    assert s[22] == pytest.approx(2 * (p[49] + p[50]) / 0.1 - 1)
    # The 23-D state lines up with the action layout.
    assert C.ACTION_SLICES["torso"] == slice(3, 7) and C.ACTION_SLICES["left_arm"] == slice(7, 14)
    assert C.ACTION_SLICES["right_arm"] == slice(15, 22)


def test_hold_action():
    rng = np.random.default_rng(18)
    p = make_proprio(rng)
    p[0:3] = [0.3, -0.2, 0.5]  # moving base
    p[24:26] = 0.06  # over-wide reading -> clipped to +1
    p[49:51] = 0.0  # fully closed -> -1
    h = O.hold_action(p)
    assert h.shape == (23,) and h.dtype == np.float32
    assert np.all(h[0:3] == 0)
    np.testing.assert_array_equal(h[3:7], p[53:57])
    np.testing.assert_array_equal(h[7:14], p[3:10])
    np.testing.assert_array_equal(h[15:22], p[28:35])
    assert h[14] == 1.0 and h[22] == -1.0
    assert np.any(h != 0)


def test_hold_action_nonfinite_proprio_is_finite():
    p = np.full(61, np.nan, dtype=np.float32)
    p[53:57] = [1.0, -1.0, 0.5, 0.0]
    h = O.hold_action(p)
    assert np.all(np.isfinite(h))
    np.testing.assert_array_equal(h[3:7], [1.0, -1.0, 0.5, 0.0])
