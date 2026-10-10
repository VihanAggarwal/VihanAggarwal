"""CPU tests for the PiBehavior (RLC / JackLiu) and GR00T backends (no JAX, no fork, no GPU).

Policies (and, for the loading tests, the fork namespace / gr00t module) are replaced by small fakes that follow the
APIs the backends use. The real-fork checks (tiny random PiBehavior through both forks; Gr00tPolicy's own validation
and decode_action with the kmy statistics) are described in docs/BACKENDS.md.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import pathlib
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.backends import base as backend_base
from b1k26.backends import gr00t as g
from b1k26.backends import openpi_comet as oc
from b1k26.backends import pibehavior as pb
from b1k26.backends.base import ChunkOut, InferItem
from b1k26.obs import state23_action_order
from b1k26.stage import StageTracker

REPO = pathlib.Path(__file__).resolve().parents[1]
ROLES = ("head", "left_wrist", "right_wrist")

# RLC 2025 / JackLiu 2026 stage tables (b1k/models/pi_behavior_config.py at ca556f7 / 7146d7b).
RLC_STAGES = (5, 6, 15, 15, 14, 12, 9, 15, 10, 15, 7, 13, 10, 15, 15, 15, 15, 11, 13, 12, 14, 15, 9, 15, 15, 15, 15,
              15, 15, 15, 11, 10, 10, 13, 5, 5, 14, 6, 8, 10, 5, 15, 8, 15, 12, 11, 9, 14, 15, 15)
JACKLIU_STAGES = RLC_STAGES + (15, 9, 12, 14, 13, 11, 6, 5, 15, 7, 5, 13, 8, 5, 15, 13, 12, 8, 14, 5, 10, 15, 10, 15,
                               13, 13, 11, 5, 5, 12, 10, 10, 9, 8, 15, 14, 11, 12, 15, 5, 5, 11, 5, 5, 15, 15, 6, 15,
                               15, 9)


def test_stage_tables_match_the_forks():
    assert len(RLC_STAGES) == 50 and sum(RLC_STAGES) == 596
    assert len(JACKLIU_STAGES) == 100 and sum(JACKLIU_STAGES) == 1120


# ------------------------------------------------------------------------------------------------------------
# Fixtures and fakes
# ------------------------------------------------------------------------------------------------------------
def make_proprio(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=C.PROPRIO_DIM).astype(np.float32)
    p[24:26] = (0.02, 0.025)
    p[49:51] = (0.05, 0.049)
    return p


def make_images(seed: int = 0, size: int = 224) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {r: rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for r in ROLES}


def make_item(task_id: int = 1, seed: int = 0, size: int = 224, **kw: Any) -> InferItem:
    kw.setdefault("prompt", "some prompt")
    return InferItem(task_id=task_id, proprio=make_proprio(seed), images=make_images(seed, size), **kw)


class FakePiBehaviorPolicy:
    """Stands in for b1k ``PiBehaviorPolicy``: same infer signature, records calls, returns (30, 23) + logits."""

    def __init__(self, horizon: int = 30, stages: tuple[int, ...] = JACKLIU_STAGES, name: str = "p"):
        self.calls: list[tuple[dict[str, Any], Any]] = []
        self.horizon, self.stages, self.name = horizon, stages, name
        self._model = object()
        self._sample_actions = object()

    def infer(self, obs: dict, *, noise: Any = None, initial_actions: Any = None) -> dict:
        self.calls.append((obs, initial_actions))
        task = int(obs["tokenized_prompt"][0])
        actions = np.zeros((self.horizon, 23), dtype=np.float64)
        actions[:, 0] = float(obs["observation/state"][3])  # ties the output to the item
        actions[:, 1] = task
        actions[:, 2] = float(obs["tokenized_prompt"][1])
        logits = np.arange(15, dtype=np.float32)
        logits[self.stages[task]:] = -np.inf  # as PiBehavior.sample_actions masks invalid stages
        return {"actions": actions, "subtask_logits": logits, "predicted_stage": int(np.argmax(logits)),
                "policy_timing": {"infer_ms": 1.0}}


def pib(**kw: Any) -> pb.PiBehaviorBackend:
    kw.setdefault("policy", FakePiBehaviorPolicy())
    kw.setdefault("task_num_stages", JACKLIU_STAGES)
    kw.setdefault("action_horizon", 30)
    return pb.PiBehaviorBackend(**kw)


# RLC 2025 wrapper (b1k/shared/eval_b1k_wrapper.py), copied verbatim except that resize_with_pad is the identity
# for 224 inputs (openpi_client returns the image unchanged when it already has the target size).
def rlc_wrapper_batch(obs: dict, task_id: int, current_stage: int) -> dict:
    def process_obs(obs):
        prop_state = obs["robot_r1::proprio"]
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        return {
            "observation/egocentric_camera": head_original,
            "observation/wrist_image_left": left_original,
            "observation/wrist_image_right": right_original,
            "observation/state": prop_state,
            "prompt": "PI_BEHAVIOR model (task-conditioned)",
        }

    def prepare_batch_for_pi_behavior(batch):
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]
        batch_copy["tokenized_prompt"] = np.array([task_id, current_stage], dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
        batch_copy["subtask_state"] = np.array(current_stage, dtype=np.int32)
        return batch_copy

    return prepare_batch_for_pi_behavior(process_obs(obs))


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


@pytest.fixture
def restore_flash_attn():
    saved = {n: sys.modules[n] for n in list(sys.modules) if n == "flash_attn" or n.startswith("flash_attn.")}
    yield
    for n in [m for m in sys.modules if m == "flash_attn" or m.startswith("flash_attn.")]:
        del sys.modules[n]
    sys.modules.update(saved)


# ------------------------------------------------------------------------------------------------------------
# Registry / import hygiene
# ------------------------------------------------------------------------------------------------------------
def test_registry_points_at_these_classes():
    assert backend_base._REGISTRY["pibehavior"] == "b1k26.backends.pibehavior:PiBehaviorBackend"
    assert backend_base._REGISTRY["gr00t"] == "b1k26.backends.gr00t:Gr00tBackend"


def test_importing_modules_does_not_import_heavy_deps():
    code = (
        "import sys; import b1k26.backends.pibehavior, b1k26.backends.gr00t; "
        "bad = [m for m in ('jax', 'flax', 'openpi', 'b1k', 'torch', 'gr00t', 'transformers', 'omnigibson') "
        "if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_create_backend_with_injected_policies():
    be = backend_base.create_backend("pibehavior", policy=FakePiBehaviorPolicy(), task_num_stages=JACKLIU_STAGES,
                                     action_horizon=30)
    assert isinstance(be, pb.PiBehaviorBackend)
    be = backend_base.create_backend("gr00t", policy=FakeGr00tPolicy())
    assert isinstance(be, g.Gr00tBackend)


# ============================================================================================================
# PiBehavior
# ============================================================================================================
def test_pibehavior_constructor_validation(tmp_path):
    with pytest.raises(ValueError, match="checkpoint or task_checkpoint_mapping"):
        pb.PiBehaviorBackend()
    with pytest.raises(ValueError, match="task_num_stages and action_horizon"):
        pb.PiBehaviorBackend(policy=FakePiBehaviorPolicy())
    with pytest.raises(ValueError, match="num_steps"):
        pib(num_steps=0)
    with pytest.raises(ValueError, match="max_resident"):
        pib(max_resident=0)
    with pytest.raises(ValueError, match="warmup_inpaint_steps"):
        pib(warmup_inpaint_steps=-1)
    with pytest.raises(ValueError, match="num_tasks"):
        pib(num_tasks=101)
    with pytest.raises(ValueError, match="time_threshold_inpaint"):
        pib(time_threshold_inpaint=1.5)
    with pytest.raises(ValueError, match="replaces"):
        pib(checkpoint=str(tmp_path))
    with pytest.raises(ValueError, match="tasks restricts"):
        pb.PiBehaviorBackend(task_checkpoint_mapping={"checkpoints": {"a": {"path": "x", "tasks": [0]}}}, tasks=[0])
    with pytest.raises(ValueError, match="warmup_task_id"):
        pib(warmup_task_id=99, task_num_stages=RLC_STAGES)


def test_pibehavior_info_contract_2026_table():
    info = pib().info()
    assert info["flavor"] == "pibehavior"
    assert info["action_horizon"] == 30 and info["image_size"] == 224
    assert info["supports_inpaint"] is True and info["supports_stage"] is True
    assert info["num_stages"] == list(JACKLIU_STAGES)
    assert info["supported_tasks"] == list(range(100))
    json.dumps(info)  # plain types only (the worker msgpacks it)


def test_pibehavior_info_2025_table_flags_new_tasks():
    be = pib(task_num_stages=RLC_STAGES)
    info = be.info()
    assert len(info["num_stages"]) == C.NUM_TASKS
    assert info["num_stages"][:50] == list(RLC_STAGES)
    assert info["num_stages"][50:] == [0] * 50  # 0 = not served by this worker
    assert info["supported_tasks"] == list(range(50))
    with pytest.raises(ValueError, match=r"task 60 .* not served .*50-task table"):
        be.infer([make_item(task_id=60)])


def test_pibehavior_tasks_restriction():
    be = pib(tasks=[0, "dispose_of_glass", 77])
    assert be.info()["supported_tasks"] == [0, 76, 77]
    assert be.info()["num_stages"][76] == JACKLIU_STAGES[76] and be.info()["num_stages"][5] == 0
    with pytest.raises(ValueError, match="not served"):
        be.infer([make_item(task_id=5)])


def test_pibehavior_example_dict_matches_rlc_wrapper():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol)
    item = make_item(task_id=7, stage=3)
    be.infer([item])
    ours, initial = pol.calls[-1]
    assert initial is None
    obs = {C.PROPRIO_KEY: item.proprio}
    for role in ROLES:
        obs[C.rgb_key(role)] = np.concatenate([item.images[role], np.full((224, 224, 1), 255, np.uint8)], axis=-1)
    theirs = rlc_wrapper_batch(obs, task_id=7, current_stage=3)
    assert set(ours) == set(theirs)
    for k in ours:
        a, b = np.asarray(ours[k]), np.asarray(theirs[k])
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), k
    assert "prompt" not in ours


def test_pibehavior_inpainting_is_passed_like_the_wrapper():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol)
    prefix = np.linspace(0, 1, 4 * 23, dtype=np.float64).reshape(4, 23)
    be.infer([make_item(task_id=2, stage=1, initial_actions=prefix)])
    obs, initial = pol.calls[-1]
    assert initial is not None and initial.dtype == np.float32 and initial.shape == (4, 23)
    assert np.array_equal(initial, prefix.astype(np.float32))
    assert obs["initial_actions"] is initial  # the wrapper puts it in the dict and passes it as the kwarg
    # (1, k, 23) is accepted, k is capped at the horizon, wider rows are cut to 23
    be.infer([make_item(task_id=2, initial_actions=np.zeros((1, 40, 32), np.float32))])
    assert pol.calls[-1][1].shape == (30, 23)


def test_pibehavior_bad_initial_actions():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol)
    bad = np.zeros((4, 23), np.float32)
    bad[1, 3] = np.nan
    be.infer([make_item(initial_actions=bad)])
    assert pol.calls[-1][1] is None and "initial_actions" not in pol.calls[-1][0]
    be.infer([make_item(initial_actions=np.zeros((0, 23), np.float32))])
    assert pol.calls[-1][1] is None
    with pytest.raises(ValueError, match="initial_actions"):
        be.infer([make_item(initial_actions=np.zeros((4, 7), np.float32))])


def test_pibehavior_stage_handling():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol)
    be.infer([make_item(task_id=0)])  # stage None -> 0
    assert pol.calls[-1][0]["tokenized_prompt"].tolist() == [0, 0]
    be.infer([make_item(task_id=0, stage=4)])
    assert pol.calls[-1][0]["tokenized_prompt"].tolist() == [0, 4]
    assert int(pol.calls[-1][0]["subtask_state"]) == 4
    be.infer([make_item(task_id=0, stage=9)])  # task 0 has 5 stages: clamp, never index another task's rows
    assert pol.calls[-1][0]["tokenized_prompt"].tolist() == [0, 4]
    be.infer([make_item(task_id=0, stage=-2)])
    assert pol.calls[-1][0]["tokenized_prompt"].tolist() == [0, 0]
    assert pb.clamp_stage(None, 5) == 0 and pb.clamp_stage(7, 5) == 4 and pb.clamp_stage(3, 5) == 3


def test_pibehavior_outputs_and_logits():
    be = pib()
    items = [make_item(task_id=t, seed=t, stage=1) for t in (0, 76, 3)]
    out = be.infer(items)
    assert len(out) == 3
    for item, o in zip(items, out):
        assert isinstance(o, ChunkOut)
        assert o.actions.shape == (30, 23) and o.actions.dtype == np.float32
        assert o.actions[0, 0] == pytest.approx(float(item.proprio[3]))
        assert o.actions[0, 1] == item.task_id
        lg = o.subtask_logits
        n = JACKLIU_STAGES[item.task_id]
        assert lg.shape == (15,) and np.all(np.isfinite(lg))
        assert np.all(lg[n:] == pb.MASKED_LOGIT) and np.array_equal(lg[:n], np.arange(n, dtype=np.float32))
    assert be.infer([]) == []


def test_sanitized_logits_drive_the_stage_tracker():
    raw = np.array([0.0, 5.0, 1.0] + [-np.inf] * 12, dtype=np.float32)  # task with 3 stages predicting stage 1
    t_raw, t_ok = StageTracker(3), StageTracker(3)
    for _ in range(3):
        t_raw.update(raw)
        t_ok.update(pb.sanitize_subtask_logits(raw))
    assert t_raw.stage == 0  # the tracker ignores non-finite logits: raw model output would freeze the stage
    assert t_ok.stage == 1


def test_sanitize_subtask_logits_cases():
    assert pb.sanitize_subtask_logits(None) is None
    x = np.array([1.0, -np.inf, 2.0], np.float32)
    out = pb.sanitize_subtask_logits(x)
    assert out.tolist() == [1.0, pb.MASKED_LOGIT, 2.0] and x[1] == -np.inf  # input not mutated
    assert pb.sanitize_subtask_logits(np.array([1.0, np.nan])) is None
    assert pb.sanitize_subtask_logits(np.array([1.0, np.inf])) is None
    assert pb.sanitize_subtask_logits(np.array([-np.inf, -np.inf])) is None
    assert pb.sanitize_subtask_logits(np.array([])) is None

    class TorchLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[0.5, -np.inf]], np.float32)

    assert pb.sanitize_subtask_logits(TorchLike()).tolist() == [0.5, pb.MASKED_LOGIT]


def test_pibehavior_validation_happens_before_any_model_call():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol, task_num_stages=RLC_STAGES)
    bad = make_item(task_id=1)
    bad.images = {k: v for k, v in bad.images.items() if k != "left_wrist"}
    with pytest.raises(ValueError, match="left_wrist"):
        be.infer([make_item(task_id=0), bad])
    with pytest.raises(ValueError, match="not served"):
        be.infer([make_item(task_id=0), make_item(task_id=99)])
    assert pol.calls == []


def test_pibehavior_nonfinite_proprio_and_readonly_inputs():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol)
    item = make_item()
    item.proprio[5] = np.nan
    item.proprio.setflags(write=False)
    for img in item.images.values():
        img.setflags(write=False)
    be.infer([item])
    state = pol.calls[-1][0]["observation/state"]
    assert np.isfinite(state).all() and state[5] == 0.0 and np.isnan(item.proprio[5])


def test_pibehavior_warmup_compiles_plain_and_inpaint_paths():
    pol = FakePiBehaviorPolicy()
    be = pib(policy=pol, warmup_task_id=77)
    ms = be.warmup()
    assert ms >= 0 and len(pol.calls) == 2
    (obs0, ia0), (obs1, ia1) = pol.calls
    assert ia0 is None and ia1.shape == (4, 23)
    assert obs0["tokenized_prompt"].tolist() == [77, 0]
    # the inpainting prefix is a hold of the reset pose (never zeros: zero torso = stand upright)
    assert np.allclose(ia1[0, 3:7], (1.025, -1.45, -0.47, 0.0))
    pol2 = FakePiBehaviorPolicy()
    pib(policy=pol2, warmup_inpaint_steps=0).warmup()
    assert len(pol2.calls) == 1


def test_base_qvel_stats_convention():
    assert pb.base_qvel_stats_convention([0.0094, 0.0087, 0.0137, 0.5]) == "2025"
    assert pb.base_qvel_stats_convention([0.109, 0.061, 0.133]) == "2026"
    assert pb.base_qvel_stats_convention([0.0844, 0.0798, 0.1869]) == "2026"
    assert pb.base_qvel_stats_convention([0.01, 0.2, 0.1]) == "unknown"
    assert pb.base_qvel_stats_convention([np.nan, 1, 1]) == "unknown"
    assert pb.base_qvel_stats_convention(None) == "unknown"


def test_num_stages_table():
    t = pb.num_stages_table(RLC_STAGES, frozenset({0, 3, 49, 60}))
    assert len(t) == 100 and t[0] == 5 and t[3] == 15 and t[49] == 15 and t[60] == 0 and t[1] == 0


# ---- task -> checkpoint mapping ------------------------------------------------------------------------------
def test_parse_mapping_file(tmp_path):
    (tmp_path / "ck1").mkdir()
    doc = {"checkpoints": {
        "c1": {"path": "ck1", "tasks": [0, "picking_up_trash"], "norm_stats_dir": "stats"},
        "c2": {"path": "~/models/ck2", "tasks": ["40"]},
    }}
    f = tmp_path / "mapping.json"
    f.write_text(json.dumps(doc))
    specs = pb.parse_task_checkpoint_mapping(f)
    assert specs["c1"].path == str(tmp_path / "ck1") and specs["c1"].tasks == (0, 1)
    assert specs["c1"].norm_stats_dir == str(tmp_path / "ck1" / "stats")
    assert specs["c2"].path == str(pathlib.Path("~/models/ck2").expanduser()) and specs["c2"].tasks == (40,)


@pytest.mark.parametrize("doc,match", [
    ({}, "checkpoints"),
    ({"checkpoints": {}}, "checkpoints"),
    ({"checkpoints": {"a": {"path": "x", "tasks": [0]}}, "default": "a"}, "unknown top-level"),
    ({"checkpoints": {"a": {"path": "x"}}}, "required"),
    ({"checkpoints": {"a": {"path": "x", "tasks": []}}}, "non-empty"),
    ({"checkpoints": {"a": {"path": "x", "tasks": [0], "foo": 1}}}, "unknown keys"),
    ({"checkpoints": {"a": {"path": "x", "tasks": [0]}, "b": {"path": "y", "tasks": [0]}}}, "both"),
    ({"checkpoints": {"a": {"path": "x", "tasks": [100]}}}, "outside"),
    ({"checkpoints": {"a": {"path": "x", "tasks": ["no_such_task"]}}}, "unknown task"),
    ({"checkpoints": {"a": {"path": "x", "tasks": [True]}}}, "invalid task"),
])
def test_parse_mapping_errors(doc, match):
    with pytest.raises(ValueError, match=match):
        pb.parse_task_checkpoint_mapping(doc)


class Factory:
    """policy_factory: one FakePiBehaviorPolicy per load, with a load log."""

    def __init__(self):
        self.loads: list[str] = []
        self.policies: list[FakePiBehaviorPolicy] = []

    def __call__(self, spec: pb.CheckpointSpec) -> FakePiBehaviorPolicy:
        self.loads.append(spec.name)
        p = FakePiBehaviorPolicy(name=spec.name)
        self.policies.append(p)
        return p


MAPPING = {"checkpoints": {"a": {"path": "/x/a", "tasks": [0, 1]}, "b": {"path": "/x/b", "tasks": [2]},
                           "c": {"path": "/x/c", "tasks": [3]}}}


def mapped(**kw: Any) -> tuple[pb.PiBehaviorBackend, Factory]:
    f = Factory()
    be = pb.PiBehaviorBackend(task_checkpoint_mapping=MAPPING, policy_factory=f, task_num_stages=RLC_STAGES,
                              action_horizon=30, **kw)
    return be, f


def test_lru_single_resident_swaps_and_releases():
    be, f = mapped()
    assert be.info()["supported_tasks"] == [0, 1, 2, 3]
    assert be.info()["num_stages"][4] == 0 and be.info()["num_stages"][2] == RLC_STAGES[2]
    for t in (0, 1, 2, 0):
        be.infer([make_item(task_id=t)])
    assert f.loads == ["a", "b", "a"] and be.resident == ["a"]
    released = f.policies[0]  # the first "a" was unloaded: device references dropped
    assert not hasattr(released, "_model") and not hasattr(released, "_sample_actions")
    with pytest.raises(ValueError, match="no checkpoint of this worker is mapped"):
        be.infer([make_item(task_id=4)])


def test_lru_mixed_batch_loads_each_checkpoint_once_and_keeps_order():
    be, f = mapped()
    be.infer([make_item(task_id=2)])
    out = be.infer([make_item(task_id=0, seed=1), make_item(task_id=2, seed=2), make_item(task_id=1, seed=3),
                    make_item(task_id=2, seed=4)])
    assert f.loads == ["b", "a"]  # b (resident) served first, then one swap to a
    assert [int(o.actions[0, 1]) for o in out] == [0, 2, 1, 2]
    assert [o.actions[0, 0] for o in out] == pytest.approx([make_proprio(s)[3] for s in (1, 2, 3, 4)])


def test_lru_two_resident():
    be, f = mapped(max_resident=2)
    for t in (0, 2, 0, 2, 3, 0, 2):
        be.infer([make_item(task_id=t)])
    # a, b loaded and hit (b most recent); c evicts a; a evicts b; b evicts c
    assert f.loads == ["a", "b", "c", "a", "b"] and be.resident == ["a", "b"]
    be.infer([make_item(task_id=0)])
    be.infer([make_item(task_id=3)])  # a was used last, so c evicts b
    assert f.loads[-1] == "c" and be.resident == ["a", "c"]


def test_mapping_with_fallback_checkpoint(tmp_path):
    f = Factory()
    be = pb.PiBehaviorBackend(str(tmp_path), task_checkpoint_mapping=MAPPING, policy_factory=f,
                              task_num_stages=RLC_STAGES, action_horizon=30)
    assert be.info()["supported_tasks"] == list(range(50))
    be.infer([make_item(task_id=10)])
    be.infer([make_item(task_id=2)])
    assert f.loads == ["default", "b"]


def test_warmup_all_leaves_the_warmup_checkpoint_resident():
    be, f = mapped(warmup_all=True, warmup_task_id=2, warmup_inpaint_steps=4)
    be.warmup()
    assert f.loads == ["a", "c", "b"] and be.resident == ["b"]
    assert all(len(p.calls) == 2 for p in f.policies)


def test_mapping_rejects_tasks_beyond_the_fork_table():
    doc = {"checkpoints": {"a": {"path": "/x/a", "tasks": [0, 55]}}}
    with pytest.raises(ValueError, match="2025 RLC checkpoints know tasks 0-49"):
        pb.PiBehaviorBackend(task_checkpoint_mapping=doc, policy_factory=Factory(), task_num_stages=RLC_STAGES,
                             action_horizon=30)


# ---- param-tree helpers -----------------------------------------------------------------------------------------
class ArrayMeta:
    def __init__(self, *shape: int):
        self.shape = shape


def test_flatten_param_shapes_strips_value_suffix_and_handles_lists():
    tree = {"task_embeddings": {"embedding": {"value": ArrayMeta(100, 2048)}},
            "layers": [{"w": {"value": np.zeros((2, 3))}}, {"w": {"value": np.zeros((4,))}}]}
    flat = pb.flatten_param_shapes(tree)
    assert flat == {("task_embeddings", "embedding"): (100, 2048), ("layers", "0", "w"): (2, 3),
                    ("layers", "1", "w"): (4,)}
    assert pb.detect_num_tasks(flat) == 100
    mixed = pb.flatten_param_shapes({"a": {"value": ArrayMeta(1)}, "b": ArrayMeta(2)})
    assert mixed == {("a", "value"): (1,), ("b",): (2,)}  # suffix only stripped when every path has it


def test_detect_rows_and_compare():
    shapes = {("task_embeddings", "embedding"): (50, 64), ("task_stage_embeddings", "embedding"): (596, 32),
              ("x", "kernel"): (3, 3)}
    assert pb.detect_num_tasks(shapes) == 50 and pb.detect_stage_rows(shapes) == 596
    assert pb.detect_num_tasks({("x",): (1,)}) is None
    expected = {("x", "kernel"): (3, 3), ("y",): (2,), ("task_embeddings", "embedding"): (100, 64)}
    diff = pb.compare_param_shapes(expected, shapes)
    assert diff.missing == [("y",)] and diff.extra == [("task_stage_embeddings", "embedding")]
    assert diff.mismatched == [(("task_embeddings", "embedding"), (100, 64), (50, 64))]
    assert diff.common == 2 and not diff.ok
    assert pb.compare_param_shapes(expected, expected).ok


# ---- state self-check and omnigibson stub ---------------------------------------------------------------------
RLC_B1K_POLICY_SNIPPET = '''
import numpy as np
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES


def extract_state_from_proprio(proprio_data):
    """Verbatim from b1k/policies/b1k_policy.py (RLC ca556f7)."""
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    MAX_GRIPPER_WIDTH = 0.1  # From statistics q99 values
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,
        arm_right_qpos,
        right_gripper_width,
    ], axis=-1)
'''


def test_rlc_extraction_through_the_stub(clean_omnigibson_modules, tmp_path, monkeypatch):
    pb.install_eval_utils_stub()
    (tmp_path / "fake_rlc_b1k_policy.py").write_text(RLC_B1K_POLICY_SNIPPET)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_rlc_b1k_policy", None)
    import fake_rlc_b1k_policy  # noqa: PLC0415

    pb.check_rlc_state_extraction(fake_rlc_b1k_policy)
    p = make_proprio(3)
    assert np.allclose(fake_rlc_b1k_policy.extract_state_from_proprio(p), state23_action_order(p))
    sys.modules.pop("fake_rlc_b1k_policy", None)


def test_rlc_extraction_check_rejects_wrong_layouts():
    with pytest.raises(RuntimeError, match="61-D"):
        pb.check_rlc_state_extraction(types.SimpleNamespace(PROPRIOCEPTION_INDICES=None))
    wrong = dict(C.PROPRIO_INDICES_2026)
    wrong["trunk_qpos"] = slice(236, 240)  # the 2025 256-D layout
    with pytest.raises(RuntimeError, match="61-D"):
        pb.check_rlc_state_extraction(types.SimpleNamespace(PROPRIOCEPTION_INDICES={"R1Pro": wrong}))
    comet_order = types.SimpleNamespace(
        PROPRIOCEPTION_INDICES={"R1Pro": dict(C.PROPRIO_INDICES_2026)},
        extract_state_from_proprio=oc.comet_state_from_proprio,  # grippers last: not RLC's order
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        pb.check_rlc_state_extraction(comet_order)


# ---- loading with a fake fork ------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FakeAssets:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class FakeData:
    repo_id: str = "IliaLarchenko/behavior_224_rgb"
    assets: FakeAssets = dataclasses.field(default_factory=FakeAssets)


@dataclasses.dataclass(frozen=True)
class FakePiBehaviorConfig:
    num_tasks: int = 50
    action_horizon: int = 30
    time_threshold_inpaint: float = 0.3
    use_correlated_noise: bool = True
    da3: Any = None
    use_spatial_action_cross_attention: bool = False


@dataclasses.dataclass(frozen=True)
class FakeTrainConfig:
    name: str
    model: Any
    data: FakeData = dataclasses.field(default_factory=FakeData)


@dataclasses.dataclass
class FakeNormStats:
    mean: Any
    std: Any
    action_correlation_cholesky: Any = None


def write_ckpt(root: pathlib.Path, num_tasks: int, stage_rows: int, extra: dict | None = None,
               state_std=(0.109, 0.061, 0.133), cholesky: bool = True, asset="IliaLarchenko/behavior_224_rgb"
               ) -> pathlib.Path:
    (root / "params").mkdir(parents=True)
    shapes = {"task_embeddings/embedding": [num_tasks, 64], "task_stage_embeddings/embedding": [stage_rows, 32],
              "x/kernel": [3, 3]}
    shapes.update(extra or {})
    (root / "params" / "shapes.json").write_text(json.dumps(shapes))
    stats_dir = root / "assets" / asset
    stats_dir.mkdir(parents=True)
    ns = {"state": {"mean": [0.0] * 32, "std": list(state_std) + [1.0] * 29},
          "actions": {"mean": [0.0] * 32, "std": [1.0] * 32, "cholesky": cholesky}}
    (stats_dir / "norm_stats.json").write_text(json.dumps(ns))
    return root


class FakeForkPolicy(FakePiBehaviorPolicy):
    pass


class NoInpaintPolicy:
    def infer(self, obs: dict) -> dict:
        return {}


def make_fake_fork(stages: tuple[int, ...], configs: dict[str, Any] | None = None) -> types.SimpleNamespace:
    created: list[dict[str, Any]] = []

    def get_config(name):
        return cfg_dict[name]

    cfg_dict = configs or {
        "pi_behavior_b1k_fast": FakeTrainConfig("pi_behavior_b1k_fast", FakePiBehaviorConfig()),
        "pi_behavior_b1k_stage_only": FakeTrainConfig("pi_behavior_b1k_stage_only",
                                                      FakePiBehaviorConfig(use_correlated_noise=False)),
    }

    def load_stats(directory):
        path = pathlib.Path(directory) / "norm_stats.json"
        if not path.exists():
            raise FileNotFoundError(path)
        d = json.loads(path.read_text())
        a = d["actions"]
        return {"state": FakeNormStats(d["state"]["mean"], d["state"]["std"]),
                "actions": FakeNormStats(a["mean"], a["std"], np.eye(2) if a["cholesky"] else None)}

    def create_trained_policy(train_config, checkpoint_dir, *, sample_kwargs=None, norm_stats=None, **kw):
        created.append({"config": train_config, "ckpt": str(checkpoint_dir), "sample_kwargs": sample_kwargs,
                        "norm_stats": norm_stats})
        cls = fork.policy_class
        return cls() if cls is NoInpaintPolicy else cls(stages=stages)

    def checkpoint_param_shapes(params_dir):
        raw = json.loads((pathlib.Path(params_dir) / "shapes.json").read_text())
        return {tuple(k.split("/")): tuple(v) for k, v in raw.items()}

    def model_param_shapes(model_config):
        return {("task_embeddings", "embedding"): (model_config.num_tasks, 64),
                ("task_stage_embeddings", "embedding"): (sum(stages), 32), ("x", "kernel"): (3, 3)}

    fork = types.SimpleNamespace(
        pi_behavior_config=types.SimpleNamespace(TASK_NUM_STAGES=stages, PiBehaviorConfig=FakePiBehaviorConfig),
        config=types.SimpleNamespace(_CONFIGS_DICT=cfg_dict, get_config=get_config),
        normalize=types.SimpleNamespace(load=load_stats),
        policy_config=types.SimpleNamespace(create_trained_policy=create_trained_policy),
        checkpoint_param_shapes=checkpoint_param_shapes,
        model_param_shapes=model_param_shapes,
        clear_caches=lambda: fork.cleared.append(1),
        cleared=[],
        created=created,
        policy_class=FakeForkPolicy,
    )
    return fork


def load(monkeypatch, fork, *args, **kw) -> pb.PiBehaviorBackend:
    monkeypatch.setattr(pb.PiBehaviorBackend, "_import_fork", lambda self: fork)
    return pb.PiBehaviorBackend(*args, **kw)


def test_load_jackliu_checkpoint_detects_100_tasks(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "meta100", 100, 1120)
    be = load(monkeypatch, fork, str(ck), num_steps=7, time_threshold_inpaint=0.25)
    assert be.fork_variant == "jackliu2026" and be.specs["default"].num_tasks == 100
    assert be.info()["supported_tasks"] == list(range(100)) and be.info()["num_stages"] == list(JACKLIU_STAGES)
    be.warmup()
    made = fork.created[0]
    assert made["config"].model.num_tasks == 100  # the fork's named config says 50; the checkpoint has 100 rows
    assert made["config"].model.time_threshold_inpaint == 0.25
    assert made["sample_kwargs"] == {"num_steps": 7}
    assert made["ckpt"] == str(ck)
    assert made["norm_stats"]["state"].std[:3] == [0.109, 0.061, 0.133]
    assert be.base_qvel_convention["default"] == "2026"
    out = be.infer([make_item(task_id=77, stage=2)])[0]
    assert out.actions.shape == (30, 23) and out.subtask_logits.shape == (15,)


def test_load_rlc_checkpoint_on_jackliu_fork_is_rejected(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "ckpt1", 50, 596)
    with pytest.raises(ValueError, match="596 stage-embedding rows .* 2025 RLC fork"):
        load(monkeypatch, fork, str(ck))


def test_load_jackliu_checkpoint_on_rlc_fork_is_rejected(monkeypatch, tmp_path):
    fork = make_fake_fork(RLC_STAGES)
    ck = write_ckpt(tmp_path / "meta100", 100, 1120)
    with pytest.raises(ValueError, match="1120 stage-embedding rows .* 2026 JackLiu fork"):
        load(monkeypatch, fork, str(ck))


def test_load_rlc_mapping_and_2025_stats_warning(monkeypatch, tmp_path, caplog):
    fork = make_fake_fork(RLC_STAGES)
    ck1 = write_ckpt(tmp_path / "checkpoint_1", 50, 596, state_std=(0.0094, 0.0087, 0.0137))
    ck4 = write_ckpt(tmp_path / "checkpoint_4", 50, 596, state_std=(0.0094, 0.0087, 0.0137))
    mapping = tmp_path / "task_checkpoint_mapping.json"
    mapping.write_text(json.dumps({"checkpoints": {"checkpoint_1": {"path": "checkpoint_1", "tasks": [2, 3]},
                                                   "checkpoint_4": {"path": "checkpoint_4", "tasks": [40]}}}))
    be = load(monkeypatch, fork, task_checkpoint_mapping=str(mapping))
    assert be.fork_variant == "rlc2025" and be.info()["supported_tasks"] == [2, 3, 40]
    with caplog.at_level(logging.WARNING, logger="b1k26.backends.pibehavior"):
        be.infer([make_item(task_id=40)])
        be.infer([make_item(task_id=2)])
    assert any("mask_base_qvel: true" in r.message for r in caplog.records)
    assert [c["ckpt"] for c in fork.created] == [str(ck4), str(ck1)]
    assert fork.cleared == [1]  # jax.clear_caches after unloading checkpoint_4
    assert be.base_qvel_convention == {"checkpoint_4": "2025", "checkpoint_1": "2025"}


def test_load_unknown_config_and_missing_params(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    with pytest.raises(ValueError, match="not found in this fork; known"):
        load(monkeypatch, fork, str(ck), config_name="pi05_b1k")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="params"):
        load(monkeypatch, fork, str(tmp_path / "empty"))


def test_load_rejects_da3_configs_and_non_pibehavior_models(monkeypatch, tmp_path):
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    da3 = {"pi_behavior_b1k_fast": FakeTrainConfig("x", FakePiBehaviorConfig(da3=types.SimpleNamespace(enabled=True)))}
    with pytest.raises(ValueError, match="DA3"):
        load(monkeypatch, make_fake_fork(JACKLIU_STAGES, da3), str(ck))
    other = {"pi_behavior_b1k_fast": FakeTrainConfig("x", object())}
    with pytest.raises(ValueError, match="not a PiBehaviorConfig"):
        load(monkeypatch, make_fake_fork(JACKLIU_STAGES, other), str(ck))


def test_load_rejects_extra_and_mismatched_params(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "da3", 100, 1120, extra={"da3_bank/proj/kernel": [4, 4]})
    be = load(monkeypatch, fork, str(ck))
    with pytest.raises(ValueError, match="DA3/spatial"):
        be.infer([make_item(task_id=0)])
    be = load(monkeypatch, fork, str(ck), allow_extra_params=True)  # spatial extras are never allowed
    with pytest.raises(ValueError, match="DA3/spatial"):
        be.infer([make_item(task_id=0)])
    ck2 = write_ckpt(tmp_path / "extra", 100, 1120, extra={"aux_head/kernel": [4, 4]})
    with pytest.raises(ValueError, match="unexpected checkpoint parameters"):
        load(monkeypatch, fork, str(ck2)).infer([make_item(task_id=0)])
    assert load(monkeypatch, fork, str(ck2), allow_extra_params=True).infer([make_item(task_id=0)])
    assert load(monkeypatch, fork, str(ck2), strict_params=False).infer([make_item(task_id=0)])
    ck3 = write_ckpt(tmp_path / "shape", 100, 1120, extra={"x/kernel": [3, 4]})
    with pytest.raises(ValueError, match="shape mismatches: x/kernel model"):
        load(monkeypatch, fork, str(ck3)).infer([make_item(task_id=0)])


def test_load_num_tasks_override_must_match_checkpoint(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    with pytest.raises(ValueError, match="100 task rows but num_tasks=50"):
        load(monkeypatch, fork, str(ck), num_tasks=50)
    ck50 = write_ckpt(tmp_path / "c50", 50, 1120)  # a 50-row table trained with the 2026 fork
    be = load(monkeypatch, fork, str(ck50))
    assert be.info()["supported_tasks"] == list(range(50))


def test_load_metadata_failure_falls_back_to_the_fork_table(monkeypatch, tmp_path, caplog):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    (ck / "params" / "shapes.json").unlink()
    with caplog.at_level(logging.WARNING, logger="b1k26.backends.pibehavior"):
        be = load(monkeypatch, fork, str(ck))
        be.infer([make_item(task_id=99)])
    assert be.specs["default"].num_tasks == 100
    assert any("parameter check skipped" in r.message for r in caplog.records)


def test_load_norm_stats_resolution(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120, asset="other/asset")
    with pytest.raises(FileNotFoundError, match="found in checkpoint: .*other/asset/norm_stats.json"):
        load(monkeypatch, fork, str(ck)).infer([make_item()])
    load(monkeypatch, fork, str(ck), asset_id="other/asset").infer([make_item()])
    assert fork.created[-1]["config"].data.assets.asset_id == "other/asset"
    fixed = tmp_path / "norm-stats-fixed"
    fixed.mkdir()
    (fixed / "norm_stats.json").write_text((ck / "assets/other/asset/norm_stats.json").read_text())
    load(monkeypatch, fork, str(ck), norm_stats_dir=str(fixed)).infer([make_item()])
    load(monkeypatch, fork, str(ck), norm_stats_dir="assets/other/asset").infer([make_item()])  # relative to ckpt


def test_load_requires_correlation_for_correlated_noise(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120, cholesky=False)
    with pytest.raises(ValueError, match="action_correlation_cholesky"):
        load(monkeypatch, fork, str(ck)).infer([make_item()])
    # the stage-only ablation config has use_correlated_noise=False
    load(monkeypatch, fork, str(ck), config_name="pi_behavior_b1k_stage_only").infer([make_item()])


def test_load_rejects_plain_openpi_policies(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    fork.policy_class = NoInpaintPolicy
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    with pytest.raises(RuntimeError, match="not PiBehaviorPolicy"):
        load(monkeypatch, fork, str(ck)).infer([make_item()])


def test_load_sets_xla_env(monkeypatch, tmp_path):
    fork = make_fake_fork(JACKLIU_STAGES)
    ck = write_ckpt(tmp_path / "c", 100, 1120)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    load(monkeypatch, fork, str(ck), mem_fraction=0.45, xla_allocator="platform")
    import os  # noqa: PLC0415

    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.450"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    with pytest.raises(ValueError, match="mem_fraction"):
        load(monkeypatch, fork, str(ck), mem_fraction=1.5)
    with pytest.raises(ValueError, match="xla_allocator"):
        load(monkeypatch, fork, str(ck), xla_allocator="cuda_async")


# ============================================================================================================
# GR00T
# ============================================================================================================
KMY_MODALITY = {  # processor_config.json "new_embodiment" of kmy17518/gr00t-n1.7-b1k-multitask (and the radio zip)
    "video": {"delta_indices": [0], "modality_keys": ["head", "left_wrist", "right_wrist"]},
    "state": {"delta_indices": [0], "modality_keys": ["base_qvel", "torso", "left_arm", "left_gripper", "right_arm",
                                                      "right_gripper"]},
    "action": {"delta_indices": list(range(16)), "modality_keys": ["base", "torso", "left_arm", "left_gripper",
                                                                   "right_arm", "right_gripper"]},
    "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
}

R1PRO_JSON = {  # Isaac-GR00T@behavior examples/b1k/r1pro.json
    "state": {"base_qvel": {"start": 0, "end": 3}, "torso": {"start": 53, "end": 57},
              "left_arm": {"start": 3, "end": 10}, "left_gripper": {"start": 24, "end": 26},
              "right_arm": {"start": 28, "end": 35}, "right_gripper": {"start": 49, "end": 51}},
    "action": {"base": {"start": 0, "end": 3}, "torso": {"start": 3, "end": 7}, "left_arm": {"start": 7, "end": 14},
               "left_gripper": {"start": 14, "end": 15}, "right_arm": {"start": 15, "end": 22},
               "right_gripper": {"start": 22, "end": 23}},
    "video": {"head": {"original_key": "observation.rgb.zed_link_camera_0"},
              "left_wrist": {"original_key": "observation.rgb.left_realsense_link_camera_0"},
              "right_wrist": {"original_key": "observation.rgb.right_realsense_link_camera_0"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


class FakeGr00tPolicy:
    """Stands in for ``Gr00tPolicy``: the same validation as its ``check_observation`` (strict mode) and a
    decode that returns absolute groups whose values identify the item and the group."""

    def __init__(self, horizon: int = 16, fail: int = 0):
        self.modality_configs = KMY_MODALITY
        self.language_key = "annotation.human.task_description"
        self.calls: list[dict[str, Any]] = []
        self.horizon = horizon
        self.fail = fail

    def check_observation(self, obs: dict) -> None:
        bs = -1
        for k in self.modality_configs["video"]["modality_keys"]:
            v = obs["video"][k]
            assert isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 5 and v.shape[1] == 1
            assert v.shape[-1] == 3
            bs = len(v) if bs < 0 else bs
            assert len(v) == bs
        for k in self.modality_configs["state"]["modality_keys"]:
            v = obs["state"][k]
            assert isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim == 3 and v.shape[1] == 1
            assert len(v) == bs
        lang = obs["language"][self.language_key]
        assert isinstance(lang, list) and len(lang) == bs
        for item in lang:
            assert isinstance(item, list) and len(item) == 1 and isinstance(item[0], str)

    def get_action(self, obs: dict, options: Any = None) -> tuple[dict, dict]:
        self.check_observation(obs)
        self.calls.append(obs)
        if self.fail:
            self.fail -= 1
            if len(obs["language"][self.language_key]) > 1:
                raise RuntimeError("CUDA out of memory (fake)")
        b = len(obs["language"][self.language_key])
        out = {}
        for gi, key in enumerate(g.ACTION_KEYS):
            d = g.ACTION_DIMS[key]
            arr = np.zeros((b, self.horizon, d), np.float32)
            arr += 10.0 * (gi + 1)
            arr += obs["state"]["base_qvel"][:, :, :1]  # item identity
            arr += np.arange(d, dtype=np.float32) * 0.01
            out[key] = arr
        return out, {}


def test_gr00t_constructor_validation():
    for kw, match in [({"prompt_style": "x"}, "prompt_style"), ({"attn_implementation": "flash"}, "attn"),
                      ({"dtype": "float16"}, "dtype"), ({"num_steps": 0}, "num_steps"), ({"image_size": 8}, "image"),
                      ({"max_batch": 0}, "max_batch"), ({"warmup_task_id": 100}, "warmup_task_id")]:
        with pytest.raises(ValueError, match=match):
            g.Gr00tBackend(policy=FakeGr00tPolicy(), **kw)
    with pytest.raises(ValueError, match="checkpoint"):
        g.Gr00tBackend()
    plain = types.SimpleNamespace(get_action=lambda obs: None)
    with pytest.raises(ValueError, match="action_horizon"):
        g.Gr00tBackend(policy=plain)
    assert g.Gr00tBackend(policy=plain, action_horizon=16).action_horizon == 16
    with pytest.raises(ValueError, match="!= policy horizon"):
        g.Gr00tBackend(policy=FakeGr00tPolicy(), action_horizon=8)


def test_gr00t_info_contract():
    info = g.Gr00tBackend(policy=FakeGr00tPolicy()).info()
    assert info == {"flavor": "gr00t", "action_horizon": 16, "image_size": 224, "num_stages": None,
                    "supports_inpaint": False, "supports_stage": False}
    assert g.Gr00tBackend(policy=FakeGr00tPolicy(), image_size=256).info()["image_size"] == 256


def test_gr00t_observation_matches_the_wrapper_layout():
    items = [make_item(task_id=t, seed=t) for t in (0, 77)]
    prepared = [(i.proprio, i.images, C.task(i.task_id).name) for i in items]
    obs = g.build_observation(prepared)
    assert sorted(obs) == ["language", "state", "video"]
    for role in ROLES:
        v = obs["video"][role]
        assert v.shape == (2, 1, 224, 224, 3) and v.dtype == np.uint8
        assert np.array_equal(v[1, 0], items[1].images[role])
    wrapper_slices = {k: (e["start"], e["end"]) for k, e in R1PRO_JSON["state"].items()}  # process_input slicing
    for key, (start, end) in wrapper_slices.items():
        s = obs["state"][key]
        assert s.shape == (2, 1, end - start) and s.dtype == np.float32
        assert np.array_equal(s[0, 0], items[0].proprio[start:end])
    assert obs["state"]["left_gripper"].shape[-1] == 2  # both finger positions, not their sum
    assert obs["language"] == {"annotation.human.task_description": [["turning_on_radio"], ["installing_a_modem"]]}
    FakeGr00tPolicy().check_observation(obs)


def test_gr00t_infer_maps_groups_by_name():
    pol = FakeGr00tPolicy()
    be = g.Gr00tBackend(policy=pol)
    items = [make_item(task_id=t, seed=t) for t in (3, 4, 5)]
    out = be.infer(items)
    assert len(pol.calls) == 1  # one batched get_action
    for item, o in zip(items, out):
        a = o.actions
        assert a.shape == (16, 23) and a.dtype == np.float32
        for gi, key in enumerate(g.ACTION_KEYS):
            sl = C.ACTION_SLICES[key]
            want = 10.0 * (gi + 1) + item.proprio[0] + np.arange(sl.stop - sl.start) * 0.01
            assert np.allclose(a[:, sl], want, atol=1e-5), key
        assert o.subtask_logits is None
    assert be.infer([]) == []


def test_gr00t_batching_controls():
    pol = FakeGr00tPolicy()
    be = g.Gr00tBackend(policy=pol, max_batch=2)
    be.infer([make_item(seed=s) for s in range(5)])
    assert [len(c["language"][pol.language_key]) for c in pol.calls] == [2, 2, 1]
    pol2 = FakeGr00tPolicy()
    g.Gr00tBackend(policy=pol2, batched=False).infer([make_item(seed=s) for s in range(3)])
    assert [len(c["language"][pol2.language_key]) for c in pol2.calls] == [1, 1, 1]
    pol3 = FakeGr00tPolicy()
    be3 = g.Gr00tBackend(policy=pol3)
    be3.infer([make_item(seed=0), make_item(seed=1, size=256)])  # different image sizes never share a batch
    assert [c["video"]["head"].shape[2] for c in pol3.calls] == [224, 256]


def test_gr00t_batched_failure_falls_back_then_disables():
    pol = FakeGr00tPolicy(fail=10)
    be = g.Gr00tBackend(policy=pol)
    for n in range(3):
        out = be.infer([make_item(seed=0), make_item(seed=1)])
        assert len(out) == 2 and out[1].actions[0, 0] == pytest.approx(10.0 + make_proprio(1)[0])
    assert be.batched is False


def test_gr00t_prompt_styles():
    pol = FakeGr00tPolicy()
    be = g.Gr00tBackend(policy=pol)
    be.infer([make_item(task_id=0, prompt="Turn on the radio please.")])
    assert pol.calls[-1]["language"][pol.language_key] == [["turning_on_radio"]]
    be = g.Gr00tBackend(policy=pol, prompt_style="instruction")
    be.infer([make_item(task_id=0)])
    assert pol.calls[-1]["language"][pol.language_key] == [[C.task(0).instruction]]
    be = g.Gr00tBackend(policy=pol, prompt_style="item")
    be.infer([make_item(task_id=0, prompt="custom"), make_item(task_id=1, prompt="")])
    assert pol.calls[-1]["language"][pol.language_key] == [["custom"], [C.task(1).name]]
    assert g.gr00t_prompt("item", 2, b"bytes prompt") == "bytes prompt"


def test_gr00t_item_validation():
    be = g.Gr00tBackend(policy=FakeGr00tPolicy())
    bad = make_item()
    bad.images["head"] = bad.images["head"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        be.infer([bad])
    bad = make_item()
    bad.proprio = np.zeros(60, np.float32)
    with pytest.raises(ValueError, match="61"):
        be.infer([bad])


def test_gr00t_actions_to_23_errors_and_tensors():
    good = {k: np.zeros((2, 16, g.ACTION_DIMS[k]), np.float32) for k in g.ACTION_KEYS}
    assert g.actions_to_23(good, 2).shape == (2, 16, 23)
    with pytest.raises(ValueError, match="lacks"):
        g.actions_to_23({k: v for k, v in good.items() if k != "torso"}, 2)
    with pytest.raises(ValueError, match="shape"):
        g.actions_to_23({**good, "left_gripper": np.zeros((2, 16, 2), np.float32)}, 2)
    with pytest.raises(ValueError, match="horizon"):
        g.actions_to_23({**good, "base": np.zeros((2, 8, 3), np.float32)}, 2)
    single = {k: np.ones((16, g.ACTION_DIMS[k]), np.float32) for k in g.ACTION_KEYS}
    assert g.actions_to_23(single, 1).shape == (1, 16, 23)

    class T:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    assert g.actions_to_23({k: T(v) for k, v in good.items()}, 2).shape == (2, 16, 23)


def test_gr00t_nonfinite_actions_are_left_for_the_front_server(caplog):
    class NaNPolicy(FakeGr00tPolicy):
        def get_action(self, obs, options=None):
            out, info = super().get_action(obs, options)
            out["torso"][:, 3, :] = np.nan
            return out, info

    with caplog.at_level(logging.WARNING):
        out = g.Gr00tBackend(policy=NaNPolicy()).infer([make_item()])[0]
    assert np.isnan(out.actions[3, 3:7]).all() and np.isfinite(out.actions[4]).all()


def test_gr00t_warmup():
    pol = FakeGr00tPolicy()
    ms = g.Gr00tBackend(policy=pol, warmup_task_id=12).warmup()
    assert ms >= 0 and len(pol.calls) == 2
    assert pol.calls[0]["language"][pol.language_key] == [[C.task(12).name]]
    assert np.allclose(pol.calls[0]["state"]["torso"][0, 0], (1.025, -1.45, -0.47, 0.0))


@pytest.mark.parametrize("requested,cc,flash,want", [
    ("auto", (7, 5), True, "sdpa"),           # TITAN RTX: flash-attn 2 refuses sm_75
    ("auto", (8, 6), True, "flash_attention_2"),
    ("auto", (8, 9), False, "sdpa"),          # flash-attn not installed
    ("auto", None, True, "sdpa"),             # CPU
    ("sdpa", (9, 0), True, "sdpa"),
    ("flash_attention_2", (8, 0), True, "flash_attention_2"),
])
def test_decide_attention(requested, cc, flash, want):
    assert g.decide_attention(requested, cc, flash) == want


def test_decide_attention_errors_and_dtype():
    with pytest.raises(ValueError, match="sm_80"):
        g.decide_attention("flash_attention_2", (7, 5), True)
    with pytest.raises(ValueError, match="not installed"):
        g.decide_attention("flash_attention_2", (8, 6), False)
    assert g.decide_dtype("auto", (7, 5)) == "float32" and g.decide_dtype("auto", None) == "float32"
    assert g.decide_dtype("auto", (8, 9)) == "bfloat16" and g.decide_dtype("bfloat16", (7, 5)) == "bfloat16"
    assert g.decide_dtype("float32", (9, 0)) == "float32"
    with pytest.raises(ValueError):
        g.decide_dtype("fp16", None)


def test_block_flash_attn(restore_flash_attn):
    g.block_flash_attn()
    with pytest.raises(ImportError):
        import flash_attn  # noqa: F401, PLC0415
    assert g.flash_attn_available() is False
    g.block_flash_attn()  # idempotent


def test_check_modality_json(tmp_path):
    f = tmp_path / "r1pro.json"
    f.write_text(json.dumps(R1PRO_JSON))
    g.check_modality_json(f)
    bad = json.loads(json.dumps(R1PRO_JSON))
    bad["state"]["torso"] = {"start": 236, "end": 240}
    f.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="torso"):
        g.check_modality_json(f)
    bad = json.loads(json.dumps(R1PRO_JSON))
    bad["action"]["left_gripper"] = {"start": 14, "end": 16}
    f.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="left_gripper"):
        g.check_modality_json(f)


def test_check_policy_modalities():
    assert g.check_policy_modalities(KMY_MODALITY) == 16
    for mutate, match in [
        (lambda m: m["video"].update(modality_keys=["head"]), "video"),
        (lambda m: m["state"].update(modality_keys=["base_qvel"]), "state"),
        (lambda m: m["action"].update(modality_keys=list(reversed(g.ACTION_KEYS))), "action"),
        (lambda m: m["state"].update(delta_indices=[-1, 0]), "timesteps"),
        (lambda m: m.pop("language"), "language"),
    ]:
        m = json.loads(json.dumps(KMY_MODALITY))
        mutate(m)
        with pytest.raises(ValueError, match=match):
            g.check_policy_modalities(m)
    objs = {k: types.SimpleNamespace(**v) for k, v in KMY_MODALITY.items()}  # ModalityConfig-like objects
    assert g.check_policy_modalities(objs) == 16


# ---- loading with fake torch / gr00t -----------------------------------------------------------------------------
class FakeModel:
    def __init__(self):
        self.dtype_calls: list[Any] = []
        self.action_head = types.SimpleNamespace(num_inference_timesteps=4)
        self.backbone = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(
            _attn_implementation=None)))

    def to(self, device=None, dtype=None):
        self.dtype_calls.append(dtype)
        return self


def install_fake_gr00t(monkeypatch, *, flash_in_config: bool = True, raise_oserror: bool = False) -> types.ModuleType:
    """A fake ``gr00t.policy.gr00t_policy`` whose Gr00tPolicy mimics the backbone's flash-attn fallback."""
    mod = types.ModuleType("gr00t.policy.gr00t_policy")
    mod.constructed = []

    def rec_to_dtype(x, dtype):
        return ("cast", dtype, x)

    class Gr00tPolicy(FakeGr00tPolicy):
        def __init__(self, embodiment_tag, model_path, *, device, strict=True):
            super().__init__()
            if raise_oserror:
                raise OSError("nvidia/Cosmos-Reason2-2B is a gated repo")
            mod.constructed.append({"tag": embodiment_tag, "path": model_path, "device": device, "strict": strict})
            self.model = FakeModel()
            try:  # qwen3_backbone.py: flash only if use_flash_attention and `import flash_attn` works
                if not flash_in_config:
                    raise ImportError
                import flash_attn  # noqa: F401, PLC0415

                impl = "flash_attention_2"
            except ImportError:
                impl = "sdpa"
            self.model.backbone.model.config._attn_implementation = impl

    mod.Gr00tPolicy = Gr00tPolicy
    mod._rec_to_dtype = rec_to_dtype
    pkg = types.ModuleType("gr00t")
    pkg.__path__ = []
    sub = types.ModuleType("gr00t.policy")
    sub.__path__ = []
    monkeypatch.setitem(sys.modules, "gr00t", pkg)
    monkeypatch.setitem(sys.modules, "gr00t.policy", sub)
    monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", mod)
    return mod


def fake_torch(monkeypatch, capability: tuple[int, int] | None) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"
    torch.seeds = []
    torch.manual_seed = torch.seeds.append
    torch.device = lambda d: d
    torch.cuda = types.SimpleNamespace(is_available=lambda: capability is not None,
                                       get_device_capability=lambda d: capability)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def gr00t_ckpt(tmp_path: pathlib.Path) -> pathlib.Path:
    ck = tmp_path / "checkpoint-238000"
    ck.mkdir()
    for name in ("config.json", "processor_config.json", "model-00001-of-00002.safetensors"):
        (ck / name).write_text("{}")
    return ck


def test_gr00t_load_on_turing_forces_sdpa_and_float32(monkeypatch, tmp_path, restore_flash_attn):
    torch = fake_torch(monkeypatch, (7, 5))
    mod = install_fake_gr00t(monkeypatch)
    monkeypatch.setattr(g, "flash_attn_available", lambda: True)
    ck = gr00t_ckpt(tmp_path)
    be = g.Gr00tBackend(str(ck), device="cuda", dtype="auto", num_steps=8, seed=3)
    assert be.capability == (7, 5) and be.attention == "sdpa" and be.dtype == "float32"
    assert sys.modules["flash_attn"] is None  # blocked before the model was built
    pol = be.policy
    assert pol.model.backbone.model.config._attn_implementation == "sdpa"
    assert pol.model.dtype_calls == ["float32"] and pol.model.action_head.num_inference_timesteps == 8
    assert mod._rec_to_dtype({"x": 1}, dtype="bfloat16") == ("cast", "float32", {"x": 1})
    assert mod.constructed == [{"tag": "NEW_EMBODIMENT", "path": str(ck), "device": "cuda", "strict": True}]
    assert torch.seeds == [3]
    assert be.info()["action_horizon"] == 16
    assert be.infer([make_item()])[0].actions.shape == (16, 23)


def test_gr00t_load_on_ampere_keeps_flash_and_bf16(monkeypatch, tmp_path, restore_flash_attn):
    fake_torch(monkeypatch, (8, 6))
    mod = install_fake_gr00t(monkeypatch)
    monkeypatch.setattr(g, "flash_attn_available", lambda: True)
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    be = g.Gr00tBackend(str(gr00t_ckpt(tmp_path)))
    assert be.attention == "flash_attention_2" and be.dtype == "bfloat16"
    assert be.policy.model.dtype_calls == [] and mod._rec_to_dtype(1, dtype="bfloat16") == ("cast", "bfloat16", 1)
    assert be.policy.model.backbone.model.config._attn_implementation == "flash_attention_2"


def test_gr00t_load_detects_flash_on_turing(monkeypatch, tmp_path, restore_flash_attn):
    """If something still selects flash-attn below sm_80, the load fails instead of the first inference."""
    fake_torch(monkeypatch, (7, 5))
    install_fake_gr00t(monkeypatch)
    monkeypatch.setattr(g, "flash_attn_available", lambda: True)
    monkeypatch.setattr(g, "block_flash_attn", lambda: None)  # simulate a block that did not take
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    with pytest.raises(RuntimeError, match="flash_attention_2"):
        g.Gr00tBackend(str(gr00t_ckpt(tmp_path)))


def test_gr00t_load_errors(monkeypatch, tmp_path):
    fake_torch(monkeypatch, None)
    install_fake_gr00t(monkeypatch, raise_oserror=True)
    ck = gr00t_ckpt(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        g.Gr00tBackend(str(ck), device="cuda")
    with pytest.raises(OSError, match="HF_HUB_OFFLINE"):
        g.Gr00tBackend(str(ck), device="cpu", attn_implementation="sdpa")
    with pytest.raises(ValueError, match="sm_80"):
        g.Gr00tBackend(str(ck), device="cpu", attn_implementation="flash_attention_2")
    with pytest.raises(FileNotFoundError, match="not found"):
        g.Gr00tBackend(str(tmp_path / "nope"))
    (ck / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        g.Gr00tBackend(str(ck))


def test_gr00t_patch_input_dtype_is_idempotent():
    mod = types.SimpleNamespace(_rec_to_dtype=lambda x, dtype: (dtype, x))
    g.Gr00tBackend._patch_input_dtype(mod, "float32")
    g.Gr00tBackend._patch_input_dtype(mod, "float64")
    assert mod._rec_to_dtype(1, dtype="bfloat16") == ("float64", 1)
    with pytest.raises(RuntimeError, match="_rec_to_dtype"):
        g.Gr00tBackend._patch_input_dtype(types.SimpleNamespace(), "float32")


# ============================================================================================================
# Env scripts and example configs
# ============================================================================================================
@pytest.mark.parametrize("name,commits", [
    ("pibehavior", (pb.RLC_COMMIT, pb.RLC_OPENPI_COMMIT, pb.JACKLIU_COMMIT)),
    ("gr00t", (g.GR00T_COMMIT,)),
])
def test_env_scripts(name, commits):
    script = REPO / "scripts" / "envs" / f"{name}.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for commit in commits:
        assert commit in text and len(commit) == 40
    out = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=True)
    assert "--prefix" in out.stdout
    assert subprocess.run(["bash", str(script)], capture_output=True, text=True).returncode != 0


def test_pibehavior_env_script_requires_a_fork(tmp_path):
    script = REPO / "scripts" / "envs" / "pibehavior.sh"
    r = subprocess.run(["bash", str(script), "--prefix", str(tmp_path / "env")], capture_output=True, text=True)
    assert r.returncode != 0 and "--fork must be rlc2025 or jackliu2026" in r.stderr


@pytest.mark.parametrize("name,backend,cls", [
    ("pibehavior.example.yaml", "pibehavior", pb.PiBehaviorBackend),
    ("gr00t.example.yaml", "gr00t", g.Gr00tBackend),
])
def test_example_configs(name, backend, cls):
    from b1k26.config import load_config  # noqa: PLC0415

    cfg = load_config(REPO / "configs" / name)
    accepted = set(inspect.signature(cls.__init__).parameters)
    for worker in cfg.workers.values():
        argv = worker.launch
        assert argv[argv.index("--backend") + 1] == backend
        kwargs = json.loads(argv[argv.index("--backend-kwargs") + 1])
        assert set(kwargs) <= accepted, set(kwargs) - accepted
        if "task_checkpoint_mapping" in kwargs:
            specs = pb.parse_task_checkpoint_mapping(kwargs["task_checkpoint_mapping"])
            tasks = sorted(t for s in specs.values() for t in s.tasks)
            assert tasks == list(range(50))  # RLC's four checkpoints cover exactly the 50 2025 tasks
    for prof in cfg.profiles.values():
        assert prof.image_size == 224
        if backend == "pibehavior":
            ex = prof.execution
            assert (ex.execute_steps, ex.predicted_steps_to_use, ex.keep_for_inpaint) == (20, 26, 4)
            assert ex.predicted_steps_to_use + ex.keep_for_inpaint <= 30 and prof.use_stage
        else:
            assert prof.execution.predicted_steps_to_use <= 16 and not prof.use_stage
    assert cfg.profiles["rlc2025" if backend == "pibehavior" else "gr00t"].mask_base_qvel is (backend == "pibehavior")


# ============================================================================================================
# Through the real worker protocol (b1k26.worker), in-process
# ============================================================================================================
def _worker_roundtrip(factory, items: list[dict]) -> tuple[dict, dict]:
    import asyncio  # noqa: PLC0415
    import threading  # noqa: PLC0415

    from websockets.sync.client import connect  # noqa: PLC0415

    from b1k26.engine import no_proxy_kwargs  # noqa: PLC0415
    from b1k26.protocol import packb, unpackb  # noqa: PLC0415
    from b1k26.worker import WorkerServer  # noqa: PLC0415

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    ws = WorkerServer(factory, host="127.0.0.1", port=0, name="t")
    try:
        asyncio.run_coroutine_threadsafe(ws.start(), loop).result(20)
        asyncio.run_coroutine_threadsafe(ws.load(), loop).result(60)
        replies = []
        with connect(f"ws://127.0.0.1:{ws.bound_port}", compression=None, max_size=None,
                     **no_proxy_kwargs(connect)) as c:
            for msg in ({"op": "info"}, {"op": "infer", "items": items}):
                c.send(packb(msg))
                raw = c.recv(timeout=30)
                assert isinstance(raw, bytes)
                replies.append(unpackb(raw))
        return replies[0], replies[1]
    finally:
        asyncio.run_coroutine_threadsafe(ws.stop(), loop).result(20)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def _wire_item(task_id: int, seed: int, **extra: Any) -> dict:
    it = make_item(task_id=task_id, seed=seed)
    return {"task_id": task_id, "prompt": "p", "proprio": it.proprio, "images": it.images, "stage": None,
            "initial_actions": None, **extra}


def test_pibehavior_through_the_worker_protocol():
    pol = FakePiBehaviorPolicy()
    info, reply = _worker_roundtrip(
        lambda: pib(policy=pol, task_num_stages=RLC_STAGES),
        [_wire_item(0, 1, stage=2, initial_actions=np.ones((4, 23), np.float32)), _wire_item(40, 2)])
    assert info["num_stages"][:50] == list(RLC_STAGES) and info["num_stages"][50:] == [0] * 50
    assert info["supports_inpaint"] and info["supports_stage"] and info["supported_tasks"] == list(range(50))
    assert "error" not in reply, reply
    c0, c1 = reply["chunks"]
    assert c0["actions"].shape == (30, 23) and c0["actions"].dtype == np.float32
    assert c0["subtask_logits"].shape == (15,) and np.isfinite(c0["subtask_logits"]).all()
    assert c0["actions"][0, 2] == 2  # stage reached the model
    obs, initial = pol.calls[-2]  # warmup ran first; the two items follow
    assert initial.shape == (4, 23) and obs["tokenized_prompt"].tolist() == [0, 2]
    assert pol.calls[-1][0]["tokenized_prompt"].tolist() == [40, 0]


def test_pibehavior_worker_reports_unserved_task_as_error():
    _, reply = _worker_roundtrip(lambda: pib(task_num_stages=RLC_STAGES), [_wire_item(77, 0)])
    assert "not served" in reply["error"]


def test_gr00t_through_the_worker_protocol():
    info, reply = _worker_roundtrip(lambda: g.Gr00tBackend(policy=FakeGr00tPolicy()),
                                    [_wire_item(3, 1), _wire_item(98, 2)])
    assert info["action_horizon"] == 16 and info["num_stages"] is None and not info["supports_inpaint"]
    c0, c1 = reply["chunks"]
    assert c0["actions"].shape == (16, 23) and c0["subtask_logits"] is None
    assert c1["actions"][0, 0] == pytest.approx(10.0 + make_proprio(2)[0])


def test_lru_thrashing_is_reported(caplog):
    be, f = mapped()
    with caplog.at_level(logging.WARNING, logger="b1k26.backends.pibehavior"):
        for _ in range(5):
            be.infer([make_item(task_id=0), make_item(task_id=2)])  # two concurrent rollouts, two checkpoints
    assert f.loads == ["a", "b", "a", "b", "a", "b"]  # one swap per request (resident checkpoint served first)
    warnings = [r.message for r in caplog.records if "thrashing" in r.message]
    assert len(warnings) == 1 and "max_resident=1" in warnings[0]  # rate-limited to one per window
