"""CPU tests for the openpi-family backends (no JAX, no openpi).

The openpi policy and, for the loading tests, the whole fork namespace are replaced by small fakes that follow the
fork APIs the backends use (``get_config``/``_CONFIGS_DICT``, ``create_trained_policy``, ``load_norm_stats``,
``Policy`` private attributes for the batched path).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.backends import base as backend_base
from b1k26.backends import openpi_b1k as ob
from b1k26.backends import openpi_comet as oc
from b1k26.backends.base import ChunkOut, InferItem

REPO = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------------------
# Fixtures and fakes
# ------------------------------------------------------------------------------------------------------------
def make_proprio(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=C.PROPRIO_DIM).astype(np.float32)
    p[24:26] = (0.02, 0.025)  # left width 0.045
    p[49:51] = (0.05, 0.049)  # right width 0.099
    return p


def make_images(seed: int = 0, size: int = 224) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {r: rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for r in ob.ROLES}


def make_item(task_id: int = 1, prompt: str = "do it", seed: int = 0, **kw: Any) -> InferItem:
    return InferItem(task_id=task_id, prompt=prompt, proprio=make_proprio(seed), images=make_images(seed), **kw)


class RecordingPolicy:
    """Stands in for openpi ``Policy``: records inputs, returns (T, out_dim) float64 actions."""

    def __init__(self, horizon: int = 32, out_dim: int = 23, value: float = 0.25):
        self.calls: list[dict[str, Any]] = []
        self.horizon, self.out_dim, self.value = horizon, out_dim, value

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(obs)
        actions = np.full((self.horizon, self.out_dim), self.value, dtype=np.float64)
        actions[:, 0] = float(obs["observation/state"][0])  # ties the output to the input item
        return {"actions": actions, "policy_timing": {"infer_ms": 1.0}}


@pytest.fixture
def clean_omnigibson_modules():
    names = ["omnigibson", "omnigibson.learning", "omnigibson.learning.utils", oc.EVAL_UTILS_MODULE]
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    yield
    for n, mod in saved.items():
        if mod is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = mod


# ------------------------------------------------------------------------------------------------------------
# Registry / import hygiene
# ------------------------------------------------------------------------------------------------------------
def test_registry_points_at_these_classes():
    assert backend_base._REGISTRY["openpi_comet"] == "b1k26.backends.openpi_comet:CometBackend"
    assert backend_base._REGISTRY["openpi_b1k"] == "b1k26.backends.openpi_b1k:OpenPIB1KBackend"


def test_importing_modules_does_not_import_jax_or_openpi():
    code = (
        "import sys; import b1k26.backends.openpi_comet, b1k26.backends.openpi_b1k; "
        "bad = [m for m in ('jax', 'openpi', 'torch', 'omnigibson') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_create_backend_with_injected_policy():
    be = backend_base.create_backend("openpi_comet", policy=RecordingPolicy(), action_horizon=32)
    assert isinstance(be, oc.CometBackend)
    be = backend_base.create_backend("openpi_b1k", policy=RecordingPolicy(), action_horizon=32)
    assert isinstance(be, ob.OpenPIB1KBackend)


def test_constructor_validation():
    with pytest.raises(ValueError, match="action_horizon"):
        oc.CometBackend(policy=RecordingPolicy())
    with pytest.raises(ValueError, match="checkpoint"):
        oc.CometBackend()
    with pytest.raises(ValueError, match="dtype"):
        oc.CometBackend(policy=RecordingPolicy(), action_horizon=32, dtype="float16")
    with pytest.raises(ValueError, match="num_steps"):
        oc.CometBackend(policy=RecordingPolicy(), action_horizon=32, num_steps=0)
    with pytest.raises(ValueError, match="prompt"):
        oc.CometBackend(policy=RecordingPolicy(), action_horizon=32, default_prompt_mode="nope")
    with pytest.raises(ValueError, match="gripper_state"):
        ob.OpenPIB1KBackend(policy=RecordingPolicy(), action_horizon=32, gripper_state="raw")
    with pytest.raises(ValueError):
        ob.OpenPIB1KBackend(policy=RecordingPolicy(), action_horizon=32, max_batch=0)


# ------------------------------------------------------------------------------------------------------------
# info()
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls,flavor", [(oc.CometBackend, "openpi_comet"), (ob.OpenPIB1KBackend, "openpi_b1k")])
def test_info_contract(cls, flavor):
    info = cls(policy=RecordingPolicy(), action_horizon=32).info()
    assert info == {
        "flavor": flavor,
        "action_horizon": 32,
        "image_size": 224,
        "num_stages": None,
        "supports_inpaint": False,
        "supports_stage": False,
    }


# ------------------------------------------------------------------------------------------------------------
# Input dict construction
# ------------------------------------------------------------------------------------------------------------
def test_comet_input_dict_exact():
    pol = RecordingPolicy()
    be = oc.CometBackend(policy=pol, action_horizon=32)
    item = make_item(prompt="Turn on the radio")
    be.infer([item])
    (ex,) = pol.calls
    assert set(ex) == {
        "observation/egocentric_camera",
        "observation/wrist_image_left",
        "observation/wrist_image_right",
        "observation/state",
        "prompt",
    }
    np.testing.assert_array_equal(ex["observation/egocentric_camera"], item.images["head"])
    np.testing.assert_array_equal(ex["observation/wrist_image_left"], item.images["left_wrist"])
    np.testing.assert_array_equal(ex["observation/wrist_image_right"], item.images["right_wrist"])
    for k in ("observation/egocentric_camera", "observation/wrist_image_left", "observation/wrist_image_right"):
        assert ex[k].dtype == np.uint8 and ex[k].shape == (224, 224, 3)
    state = ex["observation/state"]
    assert state.dtype == np.float32 and state.shape == (61,)
    np.testing.assert_array_equal(state, item.proprio)  # raw 61-D: the fork extracts the 23-D state itself
    assert state is not item.proprio  # a copy; the fork's transforms may not mutate our input
    assert ex["prompt"] == "Turn on the radio"


@pytest.mark.parametrize("mode", ["width", "pm1"])
def test_b1k_input_dict_exact(mode):
    pol = RecordingPolicy()
    be = ob.OpenPIB1KBackend(policy=pol, action_horizon=32, gripper_state=mode)
    item = make_item(prompt="turning_on_radio")
    be.infer([item])
    (ex,) = pol.calls
    expected = {"observation/image_0", "observation/image_1", "observation/image_2", "observation/state", "prompt"}
    assert set(ex) == expected
    np.testing.assert_array_equal(ex["observation/image_0"], item.images["head"])
    np.testing.assert_array_equal(ex["observation/image_1"], item.images["left_wrist"])
    np.testing.assert_array_equal(ex["observation/image_2"], item.images["right_wrist"])
    state = ex["observation/state"]
    assert state.dtype == np.float32 and state.shape == (61,)
    untouched = np.ones(61, dtype=bool)
    untouched[[24, 25, 49, 50]] = False
    np.testing.assert_array_equal(state[untouched], item.proprio[untouched])
    if mode == "width":
        np.testing.assert_array_equal(state, item.proprio)
    else:
        # The fork sums proprio[24:26] / [49:51]; the sums must now be the [-1, 1] gripper state.
        assert state[24:26].sum() == pytest.approx(2 * 0.045 / 0.1 - 1, abs=1e-6)
        assert state[49:51].sum() == pytest.approx(2 * 0.099 / 0.1 - 1, abs=1e-6)
    assert ex["prompt"] == "turning_on_radio"


def test_reference_state_layouts():
    p = np.arange(61, dtype=np.float32)
    comet = oc.comet_state_from_proprio(p)
    expect = np.concatenate([p[0:3], p[53:57], p[3:10], p[28:35], [p[24] + p[25]], [p[49] + p[50]]])
    np.testing.assert_array_equal(comet, expect)
    b1k = ob.b1k_state_from_proprio(p, "width")
    expect = np.concatenate([p[0:3], p[53:57], p[3:10], [p[24] + p[25]], p[28:35], [p[49] + p[50]]])
    np.testing.assert_array_equal(b1k, expect)
    assert comet.shape == b1k.shape == (23,)
    pm1 = ob.b1k_state_from_proprio(make_proprio(), "pm1")
    assert pm1[14] == pytest.approx(-0.1, abs=1e-6) and pm1[22] == pytest.approx(0.98, abs=1e-6)
    with pytest.raises(ValueError):
        ob.b1k_state_from_proprio(p, "raw")


def test_apply_gripper_pm1_is_a_copy_and_consistent():
    p = make_proprio()
    q = ob.apply_gripper_pm1(p)
    assert q is not p and np.shares_memory(q, p) is False
    np.testing.assert_array_equal(p, make_proprio())  # input untouched
    # Summing fingers of the rewritten proprio (the fork's extraction) == pm1 reference state.
    np.testing.assert_allclose(ob.b1k_state_from_proprio(q, "width"), ob.b1k_state_from_proprio(p, "pm1"), atol=1e-6)


def test_readonly_inputs_are_accepted_and_not_mutated():
    item = make_item()
    item.proprio.setflags(write=False)
    for img in item.images.values():
        img.setflags(write=False)
    pol = RecordingPolicy()
    ob.OpenPIB1KBackend(policy=pol, action_horizon=32, gripper_state="pm1").infer([item])
    np.testing.assert_array_equal(item.proprio, make_proprio())
    assert pol.calls[0]["observation/state"].flags.writeable


def test_item_validation_errors():
    be = oc.CometBackend(policy=RecordingPolicy(), action_horizon=32)
    item = make_item()
    del item.images["right_wrist"]
    with pytest.raises(ValueError, match="right_wrist"):
        be.infer([item])
    item = make_item()
    item.images["head"] = item.images["head"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        be.infer([item])
    item = make_item()
    item.images["head"] = np.zeros((224, 224, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="H, W, 3"):
        be.infer([item])
    item = make_item()
    item.proprio = np.zeros(60, dtype=np.float32)
    with pytest.raises(ValueError, match="61"):
        be.infer([item])


def test_nonfinite_proprio_is_zeroed():
    pol = RecordingPolicy()
    item = make_item()
    item.proprio[5] = np.nan
    item.proprio[6] = np.inf
    oc.CometBackend(policy=pol, action_horizon=32).infer([item])
    state = pol.calls[0]["observation/state"]
    assert np.all(np.isfinite(state)) and state[5] == 0 and state[6] == 0


def test_batched_proprio_row_is_accepted():
    pol = RecordingPolicy()
    item = make_item()
    item.proprio = item.proprio[None]  # (1, 61): flattened defensively
    oc.CometBackend(policy=pol, action_horizon=32).infer([item])
    assert pol.calls[0]["observation/state"].shape == (61,)


# ------------------------------------------------------------------------------------------------------------
# Base-velocity masking expectations
# ------------------------------------------------------------------------------------------------------------
def test_backends_pass_base_qvel_through_unchanged():
    # Masking is the front server's job (profile.mask_base_qvel); backends must neither mask nor un-mask.
    item = make_item()
    assert np.any(item.proprio[0:3] != 0)
    for cls in (oc.CometBackend, ob.OpenPIB1KBackend):
        pol = RecordingPolicy()
        cls(policy=pol, action_horizon=32).infer([item])
        np.testing.assert_array_equal(pol.calls[0]["observation/state"][0:3], item.proprio[0:3])


def test_masked_proprio_gives_zero_base_state():
    p = make_proprio()
    p[0:3] = 0.0  # what the front server sends with mask_base_qvel: true
    np.testing.assert_array_equal(oc.comet_state_from_proprio(p)[0:3], 0.0)
    np.testing.assert_array_equal(ob.b1k_state_from_proprio(p)[0:3], 0.0)


def test_recommended_masking_flags():
    assert oc.CometBackend.recommended_mask_base_qvel is True  # 2025 demos: base_qvel ~0
    assert ob.OpenPIB1KBackend.recommended_mask_base_qvel is False  # 2026 robot-frame velocity


# ------------------------------------------------------------------------------------------------------------
# Prompt mapping
# ------------------------------------------------------------------------------------------------------------
def test_prompt_passthrough_is_verbatim():
    assert ob.resolve_prompt("  Keep  spacing ", 3, "snake_case") == "  Keep  spacing "
    assert ob.resolve_prompt(b"bytes prompt", 3, "snake_case") == "bytes prompt"


def test_prompt_fallbacks():
    t1 = C.task(1)
    assert ob.resolve_prompt("", 1, "comet2025") == t1.instruction_comet2025
    assert "tash can" in ob.resolve_prompt(None, 1, "comet2025")  # Comet's 2025 text keeps its typos
    assert ob.resolve_prompt("", 1, "instruction") == t1.instruction
    assert ob.resolve_prompt("   ", 1, "snake_case") == "picking_up_trash"
    t60 = C.task(60)
    assert t60.instruction_comet2025 is None
    assert ob.resolve_prompt("", 60, "comet2025") == t60.instruction  # new 2026 task: no Comet text
    with pytest.raises(ValueError):
        ob.resolve_prompt("", 1, "bogus")
    with pytest.raises(IndexError):
        ob.resolve_prompt("", 100, "instruction")


def test_comet2025_prompts_cover_old_tasks():
    for t in C.tasks():
        if t.task_id < 50:
            assert t.instruction_comet2025, t.name
        assert ob.resolve_prompt("", t.task_id, "comet2025")


def test_backend_default_prompt_modes():
    item = make_item(task_id=0, prompt="")
    pol = RecordingPolicy()
    oc.CometBackend(policy=pol, action_horizon=32).infer([item])
    assert pol.calls[-1]["prompt"] == C.task(0).instruction_comet2025
    ob.OpenPIB1KBackend(policy=pol, action_horizon=32).infer([item])
    assert pol.calls[-1]["prompt"] == "turning_on_radio"
    ob.OpenPIB1KBackend(policy=pol, action_horizon=32, default_prompt_mode="instruction").infer([item])
    assert pol.calls[-1]["prompt"] == C.task(0).instruction


# ------------------------------------------------------------------------------------------------------------
# Output post-processing
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("out_dim", [23, 32])
def test_outputs_are_T_by_23_float32(out_dim):
    pol = RecordingPolicy(horizon=32, out_dim=out_dim)
    chunks = oc.CometBackend(policy=pol, action_horizon=32).infer([make_item(seed=1), make_item(seed=2)])
    assert len(chunks) == 2 and all(isinstance(c, ChunkOut) for c in chunks)
    for c, seed in zip(chunks, (1, 2)):
        assert c.actions.shape == (32, 23) and c.actions.dtype == np.float32
        assert c.actions.flags.c_contiguous
        assert c.subtask_logits is None
        assert c.actions[0, 0] == pytest.approx(make_proprio(seed)[0], abs=1e-6)  # order preserved


def test_postprocess_actions_cases():
    a = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
    out = ob.postprocess_actions(a)
    assert out.shape == (32, 23) and out.dtype == np.float32
    np.testing.assert_array_equal(out, a[:, :23].astype(np.float32))
    assert not np.shares_memory(out, a)
    b = np.ones((16, 23), dtype=np.float32)
    out = ob.postprocess_actions(b)
    assert not np.shares_memory(out, b)
    np.testing.assert_array_equal(ob.postprocess_actions(a[None]), a[:, :23].astype(np.float32))
    with pytest.raises(ValueError):
        ob.postprocess_actions(np.ones((32, 22)))
    with pytest.raises(ValueError):
        ob.postprocess_actions(np.ones((2, 32, 23)))
    with pytest.raises(ValueError):
        ob.postprocess_actions(np.ones((0, 23)))
    nan = np.ones((4, 23))
    nan[1, 2] = np.nan
    assert np.isnan(ob.postprocess_actions(nan)[1, 2])  # left for control.sanitize


def test_postprocess_accepts_torch_like():
    class FakeTensor:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    out = ob.postprocess_actions(FakeTensor(np.zeros((8, 32))))
    assert out.shape == (8, 23)


def test_infer_empty_and_extras():
    be = oc.CometBackend(policy=RecordingPolicy(), action_horizon=32)
    assert be.infer([]) == []
    item = make_item(stage=3, initial_actions=np.zeros((4, 23), dtype=np.float32))
    assert be.infer([item])[0].actions.shape == (32, 23)  # stage / inpainting ignored


def test_policy_errors_propagate():
    class Boom:
        def infer(self, obs):
            raise RuntimeError("xla oom")

    with pytest.raises(RuntimeError, match="xla oom"):
        oc.CometBackend(policy=Boom(), action_horizon=32).infer([make_item()])


def test_warmup_runs_one_dummy_inference():
    pol = RecordingPolicy()
    be = ob.OpenPIB1KBackend(policy=pol, action_horizon=32, warmup_task_id=5)
    ms = be.warmup()
    assert ms >= 0 and len(pol.calls) == 1
    ex = pol.calls[0]
    assert ex["prompt"] == C.task(5).name
    np.testing.assert_array_equal(ex["observation/state"][53:57], np.float32([1.025, -1.45, -0.47, 0.0]))
    assert ex["observation/image_0"].shape == (224, 224, 3)


# ------------------------------------------------------------------------------------------------------------
# Comet eval_utils stub
# ------------------------------------------------------------------------------------------------------------
def test_eval_utils_stub(clean_omnigibson_modules):
    mod = oc.install_eval_utils_stub()
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES  # noqa: PLC0415

    assert PROPRIOCEPTION_INDICES is mod.PROPRIOCEPTION_INDICES
    r1 = PROPRIOCEPTION_INDICES["R1Pro"]
    assert dict(r1) == C.PROPRIO_INDICES_2026
    assert list(r1) == list(C.PROPRIO_INDICES_2026)
    assert r1["trunk_qpos"] == slice(53, 57) and r1["gripper_right_qpos"] == slice(49, 51)
    import omnigibson.learning.utils as u  # noqa: PLC0415

    assert u.eval_utils is mod
    assert oc.install_eval_utils_stub().B1K26_STUB  # idempotent


def test_eval_utils_stub_replaces_stale_layout(clean_omnigibson_modules):
    stale = types.ModuleType(oc.EVAL_UTILS_MODULE)
    stale.PROPRIOCEPTION_INDICES = {"R1Pro": {"base_qvel": slice(253, 256)}}  # 2025 256-D layout
    sys.modules[oc.EVAL_UTILS_MODULE] = stale
    mod = oc.install_eval_utils_stub()
    assert sys.modules[oc.EVAL_UTILS_MODULE] is mod
    assert mod.PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"] == slice(0, 3)


def _fake_comet_b1k_policy(indices: dict) -> types.SimpleNamespace:
    """Copy of the fork's extract_state_from_proprio bound to ``indices`` (as the fork binds the import)."""

    def extract_state_from_proprio(proprio_data):
        r = indices["R1Pro"]
        return np.concatenate(
            [proprio_data[..., r["base_qvel"]], proprio_data[..., r["trunk_qpos"]],
             proprio_data[..., r["arm_left_qpos"]], proprio_data[..., r["arm_right_qpos"]],
             proprio_data[..., r["gripper_left_qpos"]].sum(axis=-1, keepdims=True),
             proprio_data[..., r["gripper_right_qpos"]].sum(axis=-1, keepdims=True)],
            axis=-1,
        )

    return types.SimpleNamespace(PROPRIOCEPTION_INDICES=indices, extract_state_from_proprio=extract_state_from_proprio)


def test_comet_state_self_check():
    oc.check_comet_state_extraction(_fake_comet_b1k_policy(oc.comet_proprio_indices()))
    stale = {"R1Pro": {**C.PROPRIO_INDICES_2026, "trunk_qpos": slice(236, 240)}}
    with pytest.raises(RuntimeError, match="61-D"):
        oc.check_comet_state_extraction(_fake_comet_b1k_policy(stale))


# ------------------------------------------------------------------------------------------------------------
# Norm-stat based gripper state detection
# ------------------------------------------------------------------------------------------------------------
RADIO_STATE_Q99 = [0.089, 0.115, 0.199, 1.287, -0.399, -0.086, 0.001, 0.219, 0.345, 0.656, 0.077, 1.351, 1.042,
                   0.631, 0.1, 0.092, 0.174, 1.474, 0.072, 1.521, 1.045, 1.303, 0.1]
HOSHIPU_STATE_Q99 = [0.248, 0.244, 0.3, 1.674, -0.399, 0.107, 0.001, 0.377, 0.538, 0.986, 0.061, 2.036, 1.046,
                     1.149, 1.0, 0.284, 0.174, 1.232, 0.079, 1.167, 1.046, 1.438, 1.0]


def test_detect_gripper_state_mode():
    assert ob.detect_gripper_state_mode({"q99": RADIO_STATE_Q99}) == "width"
    assert ob.detect_gripper_state_mode(types.SimpleNamespace(q99=HOSHIPU_STATE_Q99)) == "pm1"
    mixed = list(RADIO_STATE_Q99)
    mixed[22] = 1.0
    assert ob.detect_gripper_state_mode({"q99": mixed}) is None
    assert ob.detect_gripper_state_mode({"mean": [0.0] * 23, "std": [0.01] * 23}) == "width"
    assert ob.detect_gripper_state_mode({"mean": [0.0] * 10}) is None
    assert ob.detect_gripper_state_mode({"q99": [0.1] * 10}) is None


# ------------------------------------------------------------------------------------------------------------
# Batched-path helpers
# ------------------------------------------------------------------------------------------------------------
def test_bucket_helpers():
    assert ob.bucket_sizes(1) == [1]
    assert ob.bucket_sizes(8) == [1, 2, 4, 8]
    assert ob.bucket_sizes(6) == [1, 2, 4, 6]
    assert [ob.bucket_for(n, 8) for n in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]
    with pytest.raises(ValueError):
        ob.bucket_for(9, 8)


def test_stack_and_copy_trees():
    a = {"image": {"x": np.zeros((2, 2)), "y": np.ones(3)}, "mask": np.True_}
    b = {"image": {"x": np.ones((2, 2)), "y": np.zeros(3)}, "mask": np.False_}
    s = ob.stack_trees([a, b])
    assert s["image"]["x"].shape == (2, 2, 2) and s["mask"].tolist() == [True, False]
    c = ob.copy_tree(a)
    assert c is not a and c["image"] is not a["image"] and c["image"]["x"] is a["image"]["x"]
    with pytest.raises(ValueError):
        ob.stack_trees([a, {"other": 1}])


# ------------------------------------------------------------------------------------------------------------
# Loading through a fake fork (config validation, overrides, norm stats, self-checks, batched path)
# ------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FakeAssets:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class FakeGroup:
    inputs: tuple = ()
    outputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class FakeDataConfig:
    repo_id: str | None
    asset_id: str | None
    use_quantile_norm: bool
    data_transforms: FakeGroup = FakeGroup(inputs=("b1k_inputs",), outputs=("b1k_outputs",))
    model_transforms: FakeGroup = FakeGroup(inputs=("tokenize",), outputs=())


@dataclasses.dataclass(frozen=True)
class FakeB1KFactory:  # stands in for LeRobotB1KDataConfig
    repo_id: str = "turning_on_radio"
    assets: FakeAssets = FakeAssets()
    robot_config_name: str = "b1k/R1Pro"
    quantile: bool = False

    def create(self, assets_dirs, model_config):
        return FakeDataConfig(self.repo_id, self.assets.asset_id or self.repo_id, self.quantile)


@dataclasses.dataclass(frozen=True)
class FakeRGBDFactory(FakeB1KFactory):
    pass


@dataclasses.dataclass(frozen=True)
class FakeModelConfig:
    action_horizon: int = 32
    dtype: str = "bfloat16"

    def load(self, params):
        return ("model", params)


@dataclasses.dataclass(frozen=True)
class FakeTrainConfig:
    name: str
    model: FakeModelConfig = FakeModelConfig()
    data: Any = FakeB1KFactory()
    policy_metadata: dict | None = None

    @property
    def assets_dirs(self):
        return pathlib.Path("/nonexistent/assets") / self.name


@dataclasses.dataclass
class FakeNormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def _load_stats_dir(directory) -> dict[str, FakeNormStats]:
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text())["norm_stats"]
    return {k: FakeNormStats(**{f: np.asarray(v) for f, v in s.items()}) for k, s in raw.items()}


def write_stats(directory: pathlib.Path, state_q99: list[float], dim: int = 23) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stats = {
        "state": {"mean": [0.0] * 23, "std": [1.0] * 23, "q01": [-1.0] * 23, "q99": state_q99},
        "actions": {"mean": [0.0] * dim, "std": [1.0] * dim, "q01": [-1.0] * dim, "q99": [1.0] * dim},
    }
    (directory / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))


class FakeJaxPolicy:
    """Has the private attributes of the forks' Policy so the batched path can run on numpy fakes."""

    def __init__(self, horizon=32):
        self.horizon = horizon
        self._rng = 0
        self._sample_kwargs = {"num_steps": 5}
        self._is_pytorch_model = False
        self.sample_batches: list[int] = []
        self.infer_calls = 0

    def _input_transform(self, data):
        state = np.zeros(32, dtype=np.float32)
        state[:23] = ob.b1k_state_from_proprio(data["observation/state"])
        return {"state": state, "image": {"base_0_rgb": data["observation/image_0"]},
                "image_mask": {"base_0_rgb": np.True_}, "tokenized_prompt": np.zeros(8, np.int32)}

    def _sample_actions(self, rng, observation, **kw):
        assert kw == {"num_steps": 5}
        self.sample_batches.append(observation["state"].shape[0])
        return np.repeat(observation["state"][:, None, :], self.horizon, axis=1)

    def _output_transform(self, data):
        assert data["state"].shape == (32,) and data["actions"].shape == (self.horizon, 32)
        return {"actions": data["actions"][:, :23]}

    def infer(self, obs):
        self.infer_calls += 1
        out = self._sample_actions(0, {"state": self._input_transform(obs)["state"][None]}, **self._sample_kwargs)
        return self._output_transform({"state": np.zeros(32), "actions": out[0]})


def _tree_map(f, tree):
    if isinstance(tree, dict):
        return {k: _tree_map(f, v) for k, v in tree.items()}
    return f(tree)


def make_fake_fork(record: dict, *, comet: bool = False) -> types.SimpleNamespace:
    def get_config(name):
        return configs[name]

    configs = {
        "pi05_b1k": FakeTrainConfig("pi05_b1k"),
        "pi05_b1k-base": FakeTrainConfig(
            "pi05_b1k-base", data=FakeB1KFactory(repo_id="behavior-1k/2025-challenge-demos")
        ),
        "rgbd": FakeTrainConfig("rgbd", data=FakeRGBDFactory()),
    }
    config = types.SimpleNamespace(_CONFIGS_DICT=configs, get_config=get_config, LeRobotB1KDataConfig=FakeB1KFactory,
                                   LeRobotB1KRGBDDataConfig=FakeRGBDFactory)

    def create_trained_policy(train_config, ckpt, *, sample_kwargs=None, default_prompt=None, norm_stats=None):
        record.update(path="upstream", train_config=train_config, ckpt=ckpt, sample_kwargs=sample_kwargs,
                      default_prompt=default_prompt, norm_stats=norm_stats,
                      data_config=train_config.data.create(train_config.assets_dirs, train_config.model))
        return FakeJaxPolicy(train_config.model.action_horizon)

    def restore_params(path, *, dtype=None):
        record.update(restore_path=path, restore_dtype=dtype)
        return {"w": 1}

    class FakePolicyCls(FakeJaxPolicy):
        def __init__(self, model, *, transforms, output_transforms, sample_kwargs=None, metadata=None):
            super().__init__()
            record.update(path="float32", model=model, transforms=transforms, output_transforms=output_transforms)

    def transform(name):
        return lambda *a, **k: (name, a, k)

    robot = types.SimpleNamespace(observations={
        "image_0": types.SimpleNamespace(name="head"),
        "image_1": types.SimpleNamespace(name="left_wrist"),
        "image_2": types.SimpleNamespace(name="right_wrist"),
    })

    def b1k_extract(proprio, robot_config=None):
        if comet:
            return oc.comet_state_from_proprio(proprio)
        return ob.b1k_state_from_proprio(proprio, "width")  # the fork sums finger qpos

    jax = types.SimpleNamespace(
        __version__="fake", devices=lambda: [types.SimpleNamespace(platform="cpu")],
        tree=types.SimpleNamespace(map=_tree_map), random=types.SimpleNamespace(split=lambda r: (r + 1, r)),
    )
    return types.SimpleNamespace(
        jax=jax,
        jnp=types.SimpleNamespace(asarray=np.asarray, float32=np.float32),
        transforms=types.SimpleNamespace(InjectDefaultPrompt=transform("inject"), Normalize=transform("norm"),
                                         Unnormalize=transform("unnorm")),
        robot_registry={"b1k/R1Pro": robot},
        model=types.SimpleNamespace(
            restore_params=restore_params, Observation=types.SimpleNamespace(from_dict=lambda d: d)
        ),
        b1k_policy=types.SimpleNamespace(extract_state_from_proprio=b1k_extract),
        policy=types.SimpleNamespace(Policy=FakePolicyCls),
        policy_config=types.SimpleNamespace(create_trained_policy=create_trained_policy),
        download=types.SimpleNamespace(maybe_download=lambda p: pathlib.Path(p)),
        normalize=types.SimpleNamespace(load=_load_stats_dir),
        checkpoints=types.SimpleNamespace(load_norm_stats=lambda d, a: _load_stats_dir(pathlib.Path(d) / a)),
        config=config,
    )


@pytest.fixture
def ckpt(tmp_path):
    (tmp_path / "params").mkdir()
    return tmp_path


def load_b1k(monkeypatch, record, **kwargs):
    monkeypatch.setattr(ob.OpenPIB1KBackend, "_import_fork", lambda self: make_fake_fork(record))
    return ob.OpenPIB1KBackend(**kwargs)


def test_b1k_load_radio_layout(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    record: dict = {}
    be = load_b1k(monkeypatch, record, checkpoint=str(ckpt), num_steps=5)
    assert record["path"] == "upstream"
    assert record["sample_kwargs"] == {"num_steps": 5}
    assert record["default_prompt"] is None
    assert record["norm_stats"] is be.norm_stats
    assert be.gripper_state == "width"
    assert be.info()["action_horizon"] == 32 and be.info()["flavor"] == "openpi_b1k"
    assert be.batched is False


def test_b1k_load_hoshipu_layout(monkeypatch, ckpt):
    write_stats(ckpt / "assets", HOSHIPU_STATE_Q99)  # norm_stats.json directly under assets/
    record: dict = {}
    with pytest.raises(FileNotFoundError, match="assets/norm_stats.json"):
        load_b1k(monkeypatch, record, checkpoint=str(ckpt))
    be = load_b1k(monkeypatch, record, checkpoint=str(ckpt), norm_stats_dir="assets", action_horizon=50,
                  use_quantile_norm=True)
    assert be.gripper_state == "pm1"
    assert be.info()["action_horizon"] == 50
    assert record["train_config"].model.action_horizon == 50
    assert record["data_config"].use_quantile_norm is True  # the factory override reaches create()
    assert record["train_config"].data.robot_config_name == "b1k/R1Pro"  # delegated attribute


def test_b1k_explicit_gripper_state_wins(monkeypatch, ckpt, caplog):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    be = load_b1k(monkeypatch, {}, checkpoint=str(ckpt), gripper_state="pm1")
    assert be.gripper_state == "pm1"
    assert "norm stats look like 'width'" in caplog.text


def test_b1k_ambiguous_stats_need_explicit_mode(monkeypatch, ckpt):
    q = list(RADIO_STATE_Q99)
    q[22] = 1.0
    write_stats(ckpt / "assets" / "turning_on_radio", q)
    with pytest.raises(ValueError, match="gripper state"):
        load_b1k(monkeypatch, {}, checkpoint=str(ckpt))
    be = load_b1k(monkeypatch, {}, checkpoint=str(ckpt), gripper_state="width")
    assert be.gripper_state == "width"


def test_b1k_repo_id_and_asset_id_overrides(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "behavior-1k" / "2026-challenge-demos", RADIO_STATE_Q99)
    record: dict = {}
    load_b1k(monkeypatch, record, checkpoint=str(ckpt), repo_id="behavior-1k/2026-challenge-demos")
    assert record["data_config"].asset_id == "behavior-1k/2026-challenge-demos"
    write_stats(ckpt / "assets" / "custom", RADIO_STATE_Q99)
    load_b1k(monkeypatch, record, checkpoint=str(ckpt), asset_id="custom")
    assert record["data_config"].asset_id == "custom"


def test_unknown_config_and_missing_checkpoint(monkeypatch, ckpt, tmp_path_factory):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    with pytest.raises(ValueError, match="not found"):
        load_b1k(monkeypatch, {}, checkpoint=str(ckpt), config_name="pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40")
    empty = tmp_path_factory.mktemp("empty")
    with pytest.raises(FileNotFoundError, match="params"):
        load_b1k(monkeypatch, {}, checkpoint=str(empty))
    with pytest.raises(ValueError, match="LeRobotB1KDataConfig"):
        _raise_non_b1k(monkeypatch, ckpt)


def _raise_non_b1k(monkeypatch, ckpt):
    record: dict = {}
    fork = make_fake_fork(record)
    fork.config._CONFIGS_DICT["plain"] = FakeTrainConfig("plain", data=types.SimpleNamespace(repo_id="x"))
    monkeypatch.setattr(ob.OpenPIB1KBackend, "_import_fork", lambda self: fork)
    ob.OpenPIB1KBackend(checkpoint=str(ckpt), config_name="plain")


def test_float32_restore_path(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    record: dict = {}
    be = load_b1k(monkeypatch, record, checkpoint=str(ckpt), dtype="float32", num_steps=5)
    assert record["path"] == "float32"
    assert record["restore_dtype"] is np.float32
    assert record["restore_path"] == ckpt / "params"
    assert be.train_config.model.dtype == "float32"
    # Same order as create_trained_policy: inject prompt, data inputs, normalize, model inputs; inverse on output.
    tr = record["transforms"]
    assert [tr[0][0], tr[1], tr[2][0], tr[3]] == ["inject", "b1k_inputs", "norm", "tokenize"]
    assert tr[0][1] == (None,) and tr[2][2] == {"use_quantiles": False}
    out = record["output_transforms"]
    assert [out[0][0], out[1]] == ["unnorm", "b1k_outputs"]


def test_float32_ignored_for_pytorch_checkpoints(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    write_stats(tmp_path / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    record: dict = {}
    load_b1k(monkeypatch, record, checkpoint=str(tmp_path), dtype="float32")
    assert record["path"] == "upstream"


def test_state_self_check_catches_fork_mismatch(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    fork = make_fake_fork({}, comet=True)  # Comet order != b1k order
    monkeypatch.setattr(ob.OpenPIB1KBackend, "_import_fork", lambda self: fork)
    with pytest.raises(RuntimeError, match="state extraction mismatch"):
        ob.OpenPIB1KBackend(checkpoint=str(ckpt))


def test_comet_load_with_fake_fork(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "behavior-1k" / "2025-challenge-demos", [0.1] * 23)
    record: dict = {}
    monkeypatch.setattr(oc.CometBackend, "_import_fork", lambda self: make_fake_fork(record, comet=True))
    be = oc.CometBackend(checkpoint=str(ckpt))
    assert record["train_config"].name == "pi05_b1k-base"
    assert record["data_config"].asset_id == "behavior-1k/2025-challenge-demos"
    assert be.info()["flavor"] == "openpi_comet"
    with pytest.raises(ValueError, match="depth"):
        oc.CometBackend(checkpoint=str(ckpt), config_name="rgbd")


def test_batched_path_matches_loop(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    record: dict = {}
    be = load_b1k(monkeypatch, record, checkpoint=str(ckpt), batched=True, max_batch=4)
    assert be.batched is True
    items = [make_item(seed=s) for s in range(6)]
    batched = be.infer(items)
    assert be.policy.sample_batches == [4, 2]  # 4 + 2 (bucketed); no per-item infer calls
    assert be.policy.infer_calls == 0
    be.batched = False
    looped = be.infer(items)
    assert be.policy.infer_calls == 6
    for b, l, item in zip(batched, looped, items):
        assert b.actions.shape == (32, 23) and b.actions.dtype == np.float32
        np.testing.assert_allclose(b.actions, l.actions)
        np.testing.assert_allclose(b.actions[0], ob.b1k_state_from_proprio(item.proprio))
    # Single items always take the upstream Policy.infer path.
    be.batched = True
    be.infer(items[:1])
    assert be.policy.infer_calls == 7


def test_batched_warmup_compiles_every_bucket(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    be = load_b1k(monkeypatch, {}, checkpoint=str(ckpt), batched=True, max_batch=4)
    be.warmup()
    # Size 1 goes through Policy.infer (which samples a batch of 1 itself); 2 and 4 through the batched path.
    assert be.policy.infer_calls == 1
    assert be.policy.sample_batches == [1, 2, 4]


def test_batched_runtime_failure_falls_back_to_loop(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    be = load_b1k(monkeypatch, {}, checkpoint=str(ckpt), batched=True, max_batch=4)

    def oom(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    monkeypatch.setattr(be, "_infer_batched", oom)
    items = [make_item(seed=s) for s in range(3)]
    for attempt in range(ob.MAX_BATCHED_FAILURES):
        assert be.batched is True
        chunks = be.infer(items)  # served by the loop
        assert len(chunks) == 3 and all(c.actions.shape == (32, 23) for c in chunks)
    assert be.batched is False  # disabled after repeated failures
    assert be.policy.infer_calls == 3 * ob.MAX_BATCHED_FAILURES


def test_batched_falls_back_without_private_api(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    fork = make_fake_fork({})
    fork.policy_config.create_trained_policy = lambda *a, **k: RecordingPolicy()
    monkeypatch.setattr(ob.OpenPIB1KBackend, "_import_fork", lambda self: fork)
    be = ob.OpenPIB1KBackend(checkpoint=str(ckpt), batched=True)
    assert be.batched is False


def test_mem_fraction_sets_env(monkeypatch, ckpt):
    write_stats(ckpt / "assets" / "turning_on_radio", RADIO_STATE_Q99)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    load_b1k(monkeypatch, {}, checkpoint=str(ckpt), mem_fraction=0.4)
    import os  # noqa: PLC0415

    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.400"
    with pytest.raises(ValueError):
        load_b1k(monkeypatch, {}, checkpoint=str(ckpt), mem_fraction=2.0)


# ------------------------------------------------------------------------------------------------------------
# Env scripts and example configs
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,commit", [("openpi_comet", oc.OPENPI_COMET_COMMIT), ("openpi_b1k", ob.OPENPI_B1K_COMMIT)]
)
def test_env_scripts(name, commit):
    script = REPO / "scripts" / "envs" / f"{name}.sh"
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert commit in text and len(commit) == 40
    out = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=True)
    assert "--prefix" in out.stdout
    bad = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert bad.returncode != 0  # --prefix is required


@pytest.mark.parametrize("name", ["comet_pt50.example.yaml", "openpi_b1k.example.yaml"])
def test_example_configs_parse(name):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "configs" / name).read_text())
    assert {"server", "workers", "profiles", "routing"} <= set(cfg)
    assert cfg["routing"]["default"] in cfg["profiles"]
    for prof in cfg["profiles"].values():
        assert prof["worker"] in cfg["workers"]
        assert prof["image_size"] == 224
        assert prof["prompt"] in ob.PROMPT_MODES
        ex = prof["execution"]
        assert 1 <= ex["execute_steps"] <= ex["predicted_steps_to_use"] <= 32
    for worker in cfg["workers"].values():
        argv = worker["launch"]
        assert "b1k26.worker" in argv and "--backend" in argv
        backend = argv[argv.index("--backend") + 1]
        assert backend in ("openpi_comet", "openpi_b1k")
        if "--backend-kwargs" in argv:
            kwargs = json.loads(argv[argv.index("--backend-kwargs") + 1])
            cls = oc.CometBackend if backend == "openpi_comet" else ob.OpenPIB1KBackend
            import inspect  # noqa: PLC0415

            accepted = set(inspect.signature(ob.OpenPIBackendBase.__init__).parameters) | set(
                inspect.signature(cls.__init__).parameters)
            assert set(kwargs) <= accepted, set(kwargs) - accepted
