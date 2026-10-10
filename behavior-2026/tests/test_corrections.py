"""Tests for b1k26.corrections and data/gripper_rules.json."""

from __future__ import annotations

import json
from importlib import resources

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.corrections import GripperRules, SideRule, TaskRule, task_progress
from b1k26.obs import state23_action_order

# ------------------------------------------------------------------------------------------------------------
# Verbatim RLC 2025 reference (shared/correction_rules.py), minus logging.
# ------------------------------------------------------------------------------------------------------------
OPEN_THRESHOLD = 0.90
CLOSED_THRESHOLD = -0.98
LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
ALWAYS_OPEN_LEFT_GRIPPER_TASKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 48}  # noqa: E501
ALWAYS_OPEN_RIGHT_GRIPPER_TASKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 47, 48, 49}  # noqa: E501
MIN_STAGE_FOR_CLOSURE = {
    0: {'left': 2, 'right': 2},
    30: {'right': 6},
    31: {'right': 8},
    32: {'right': 5},
    33: {'right': 11},
    40: {'right': 4},
    41: {'left': 14, 'right': 14},
    45: {'right': 10},
    46: {'left': 8, 'right': 8},
    49: {'left': 14},
}
RIGHT_GRIPPER_ALWAYS_ALLOWED = {38, 39}


def task0_stage4_reset_to_stage2(task_id, stage, state, actions):
    if task_id != 0 or stage < 2:
        return None
    if stage == 4:
        corrected_stage = 2
    else:
        corrected_stage = stage
    left_gripper = state[LEFT_GRIPPER_IDX]
    right_gripper = state[RIGHT_GRIPPER_IDX]
    left_closed = left_gripper < CLOSED_THRESHOLD
    right_closed = right_gripper < CLOSED_THRESHOLD
    left_open = left_gripper > OPEN_THRESHOLD
    right_open = right_gripper > OPEN_THRESHOLD
    left_middle = not (left_open or left_closed)
    right_middle = not (right_open or right_closed)
    change_action = False
    corrected_actions = np.tile(state, (actions.shape[0], 1))
    if left_closed and not right_middle:
        corrected_actions[:, LEFT_GRIPPER_IDX] = 1.0
        change_action = True
    if right_closed and not left_middle:
        corrected_actions[:, RIGHT_GRIPPER_IDX] = 1.0
        change_action = True
    if not change_action:
        if stage == corrected_stage:
            return None
        corrected_actions = actions
    return corrected_actions, corrected_stage


def general_gripper_correction(task_id, stage, state, actions):
    left_gripper = state[LEFT_GRIPPER_IDX]
    right_gripper = state[RIGHT_GRIPPER_IDX]
    left_closed = left_gripper < CLOSED_THRESHOLD
    right_closed = right_gripper < CLOSED_THRESHOLD
    left_needs_opening = False
    right_needs_opening = False
    if left_closed:
        if task_id in ALWAYS_OPEN_LEFT_GRIPPER_TASKS:
            left_needs_opening = True
        elif task_id in MIN_STAGE_FOR_CLOSURE:
            min_stage_left = MIN_STAGE_FOR_CLOSURE[task_id].get('left')
            if min_stage_left is not None and stage < min_stage_left:
                left_needs_opening = True
    if right_closed:
        if task_id in RIGHT_GRIPPER_ALWAYS_ALLOWED:
            pass
        elif task_id in ALWAYS_OPEN_RIGHT_GRIPPER_TASKS:
            right_needs_opening = True
        elif task_id in MIN_STAGE_FOR_CLOSURE:
            min_stage_right = MIN_STAGE_FOR_CLOSURE[task_id].get('right')
            if min_stage_right is not None and stage < min_stage_right:
                right_needs_opening = True
    if left_needs_opening or right_needs_opening:
        corrected_actions = np.tile(state, (actions.shape[0], 1))
        if left_needs_opening:
            corrected_actions[:, LEFT_GRIPPER_IDX] = 1.0
        if right_needs_opening:
            corrected_actions[:, RIGHT_GRIPPER_IDX] = 1.0
        return corrected_actions, stage
    return None


TASK_SPECIFIC_RULES = {0: [task0_stage4_reset_to_stage2]}
GENERAL_RULES = [general_gripper_correction]


def apply_correction_rules(task_id, stage, state, actions):
    for rule in TASK_SPECIFIC_RULES.get(task_id, []):
        result = rule(task_id, stage, state, actions)
        if result is not None:
            return result
    for rule in GENERAL_RULES:
        result = rule(task_id, stage, state, actions)
        if result is not None:
            return result
    return actions, stage


# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rules() -> GripperRules:
    return GripperRules.load()


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(resources.files("b1k26.data").joinpath("gripper_rules.json").read_text())


def random_state23(rng, left=None, right=None):
    s = rng.normal(scale=0.5, size=23).astype(np.float32)
    s[0:3] = rng.uniform(-0.4, 0.4, size=3)  # a moving base
    s[14] = rng.uniform(-1, 1) if left is None else left
    s[22] = rng.uniform(-1, 1) if right is None else right
    return s


def test_json_encodes_rlc_tables_verbatim(doc):
    assert doc["closed_threshold"] == pytest.approx(CLOSED_THRESHOLD)
    tasks = doc["tasks"]
    for tid in range(50):
        entry = tasks[str(tid)]
        assert entry["source"] == "rlc2025"
        assert entry.get("exempt_right", False) == (tid in RIGHT_GRIPPER_ALWAYS_ALLOWED)
        for side, always in (("left", ALWAYS_OPEN_LEFT_GRIPPER_TASKS), ("right", ALWAYS_OPEN_RIGHT_GRIPPER_TASKS)):
            rule = entry.get(side)
            min_stage = MIN_STAGE_FOR_CLOSURE.get(tid, {}).get(side)
            if tid in always:
                assert rule["always_open"] is True, (tid, side)
            elif min_stage is not None:
                assert rule["always_open"] is False and rule["min_stage"] == min_stage, (tid, side)
            else:
                assert rule is None or (not rule["always_open"] and rule.get("min_stage") is None), (tid, side)
    assert tasks["0"].get("rlc_task0_rule") is True


def test_json_demo_rules_are_consistent(doc):
    params = doc["params"]
    for tid in range(50, 100):
        entry = doc["tasks"].get(str(tid))
        if entry is None:
            continue  # no statistics -> no rule
        assert entry["source"] == "demos2026" and entry["exempt_right"] is False
        for side in ("left", "right"):
            rule, st = entry[side], entry["demo_stats"][side]
            assert rule["min_stage"] is None
            rare = st["close_frac"] <= params["always_open_max_frac"]
            late = st.get("first_close_progress_p00", 0.0) >= params["late_progress"]
            assert rule["always_open"] == (rare and not late)
            if rule["min_progress"] is not None:
                assert not rule["always_open"]
                assert 0 < rule["min_progress"] <= st["first_close_progress_p50"]
                assert rule["min_progress"] <= st["first_close_progress_p01"] or (rare and late)
    GripperRules.from_dict(doc)  # loads without error


@pytest.mark.parametrize("tid", range(50))
def test_matches_rlc_on_tasks_0_49(rules, tid):
    rng = np.random.default_rng(1000 + tid)
    grip_values = [-1.0, -0.99, -0.981, -0.979, -0.5, 0.0, 0.89, 0.91, 1.0]
    for _ in range(60):
        state = random_state23(rng, left=float(rng.choice(grip_values)), right=float(rng.choice(grip_values)))
        stage = int(rng.integers(0, 16))
        actions = rng.normal(size=(30, 23)).astype(np.float32)
        ref_actions, ref_stage = apply_correction_rules(tid, stage, state.copy(), actions)
        out, changed, new_stage = rules.apply_with_stage(tid, stage, state, actions)
        ref_changed = ref_actions is not actions
        assert changed == ref_changed, (tid, stage, state[14], state[22])
        assert (new_stage if new_stage is not None else stage) == ref_stage
        if changed:
            assert out.shape == (30, 23) and out.dtype == np.float32
            np.testing.assert_array_equal(out[:, 0:3], 0.0)  # 2026: base velocity 0, not base_qvel
            np.testing.assert_allclose(out[:, 3:], ref_actions[:, 3:], atol=1e-6)
        else:
            assert out is actions
        out2, changed2 = rules.apply(tid, stage, state, actions)
        assert changed2 == changed


def test_hold_base_is_zero_and_other_gripper_kept(rules):
    rng = np.random.default_rng(1)
    state = random_state23(rng, left=-0.995, right=-0.4)
    state[0:3] = [0.3, -0.2, 0.5]
    actions = rng.normal(size=(32, 23)).astype(np.float32)
    out, changed = rules.apply(5, None, state, actions)  # task 5: both always-open in RLC
    assert changed and out.shape == (32, 23)
    assert np.all(out[:, 0:3] == 0)
    assert np.all(out[:, 14] == 1.0)
    np.testing.assert_allclose(out[:, 22], -0.4, atol=1e-6)  # right not closed -> keeps current width
    np.testing.assert_allclose(out[:, 3:14], np.tile(state[3:14], (32, 1)), atol=1e-6)
    np.testing.assert_allclose(out[:, 15:22], np.tile(state[15:22], (32, 1)), atol=1e-6)


def test_wide_actions_and_unknown_task():
    r = GripperRules({7: TaskRule(left=SideRule(always_open=True))})
    rng = np.random.default_rng(2)
    state = random_state23(rng, left=-1.0)
    actions = rng.normal(size=(10, 32)).astype(np.float32)
    out, changed = r.apply(7, None, state, actions)
    assert changed and out.shape == (10, 23)
    out, changed = r.apply(8, None, state, actions)
    assert not changed and out is actions
    out, changed = r.apply(7, None, state, actions[None])  # (1, T, D) batch of one
    assert changed and out.shape == (10, 23)


def test_stage_rules_only_with_stage():
    r = GripperRules({60: TaskRule(right=SideRule(min_stage=3))})
    rng = np.random.default_rng(3)
    state = random_state23(rng, right=-1.0)
    acts = np.zeros((5, 23), np.float32)
    assert r.apply(60, 2, state, acts)[1]
    assert not r.apply(60, 3, state, acts)[1]
    assert not r.apply(60, None, state, acts)[1]
    assert not r.apply(60, None, state, acts, progress=0.0)[1]


def test_progress_rules():
    r = GripperRules({61: TaskRule(right=SideRule(min_progress=0.4), left=SideRule(min_stage=5, min_progress=0.3))})
    rng = np.random.default_rng(4)
    acts = np.zeros((5, 23), np.float32)
    closed_r = random_state23(rng, left=0.5, right=-1.0)
    assert r.apply(61, None, closed_r, acts, progress=0.39)[1]
    assert not r.apply(61, None, closed_r, acts, progress=0.41)[1]
    assert not r.apply(61, None, closed_r, acts)[1]  # no progress, no rule
    assert r.apply(61, 7, closed_r, acts, progress=0.39)[1]  # no min_stage on this side: progress decides
    closed_l = random_state23(rng, left=-1.0, right=0.5)
    assert r.apply(61, None, closed_l, acts, progress=0.2)[1]
    assert not r.apply(61, 6, closed_l, acts, progress=0.2)[1]  # known stage decides alone
    assert r.apply(61, 4, closed_l, acts, progress=0.9)[1]


def test_task_progress():
    t = C.task(51)
    assert task_progress(51, 0) == 0.0
    assert task_progress(51, int(t.human_mean_len)) == pytest.approx(int(t.human_mean_len) / t.human_mean_len)


def test_task0_rule_needs_stage(rules):
    rng = np.random.default_rng(5)
    acts = rng.normal(size=(8, 23)).astype(np.float32)
    # Stage 4 with no gripper issue: stage reset to 2, actions untouched.
    state = random_state23(rng, left=0.95, right=0.95)
    out, changed, new_stage = rules.apply_with_stage(0, 4, state, acts)
    assert out is acts and not changed and new_stage == 2
    # Without a stage the task-0 rule is off and the general min_stage rule cannot apply either.
    state = random_state23(rng, left=-1.0, right=0.95)
    out, changed, new_stage = rules.apply_with_stage(0, None, state, acts)
    assert not changed and new_stage is None
    out, changed, new_stage = rules.apply_with_stage(0, 3, state, acts)
    assert changed and new_stage is None and np.all(out[:, 14] == 1.0) and np.all(out[:, 0:3] == 0)


def test_with_real_proprio(rules):
    p = np.zeros(61, np.float32)
    p[53:57] = [1.0, -1.5, -0.5, 0.0]
    p[24:26] = 0.0004  # left sum 0.0008 m < 0.001 m: fully closed
    p[49:51] = 0.02
    p[0:3] = [0.2, 0.1, -0.3]
    s23 = state23_action_order(p)
    out, changed = rules.apply(3, None, s23, np.zeros((4, 23), np.float32))
    assert changed
    np.testing.assert_allclose(out[0, 3:7], p[53:57])
    assert out[0, 14] == 1.0 and out[0, 22] == pytest.approx(-0.2, abs=1e-6)
    assert np.all(out[:, :3] == 0)


def test_load_validation(tmp_path):
    good = {"closed_threshold": -0.98, "tasks": {"3": {"left": {"always_open": True}}}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(good))
    r = GripperRules.load(str(p))
    assert r.tasks[3].left.always_open and r.tasks[3].right is None
    for bad in (
        {"tasks": {"x": {}}},
        {"tasks": {"100": {}}},
        {"tasks": {"3": {"left": {"min_stage": -1}}}},
        {"tasks": {"3": {"left": {"min_progress": "soon"}}}},
        {"tasks": {"3": {"left": {"always_open": "yes"}}}},
        {"closed_threshold": 2.0, "tasks": {}},
        {"tasks": []},
    ):
        with pytest.raises(ValueError):
            GripperRules.from_dict(bad)
    assert GripperRules().apply(3, None, np.full(23, -1.0), np.zeros((2, 23)))[1] is False


def test_exempt_right(rules):
    rng = np.random.default_rng(6)
    acts = np.zeros((4, 23), np.float32)
    for tid in (38, 39):
        assert not rules.apply(tid, 0, random_state23(rng, left=0.5, right=-1.0), acts)[1]
        assert rules.apply(tid, 0, random_state23(rng, left=-1.0, right=-1.0), acts)[1]


# ------------------------------------------------------------------------------------------------------------
# scripts/compute_gripper_rules.py (pure parts; the network phase is exercised against a local parquet file)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cgr():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "compute_gripper_rules.py"
    spec = importlib.util.spec_from_file_location("compute_gripper_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_rlc_tables_match_reference(cgr):
    assert cgr.RLC_ALWAYS_OPEN_LEFT == ALWAYS_OPEN_LEFT_GRIPPER_TASKS
    assert cgr.RLC_ALWAYS_OPEN_RIGHT == ALWAYS_OPEN_RIGHT_GRIPPER_TASKS
    assert cgr.RLC_MIN_STAGE_FOR_CLOSURE == MIN_STAGE_FOR_CLOSURE
    assert cgr.RLC_RIGHT_GRIPPER_ALWAYS_ALLOWED == RIGHT_GRIPPER_ALWAYS_ALLOWED


def test_script_parse_tasks_and_runs(cgr):
    assert cgr.parse_tasks("50-52,0,51") == [0, 50, 51, 52]
    with pytest.raises(SystemExit):
        cgr.parse_tasks("99-100")
    assert cgr.closed_runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 2), (4, 1), (7, 1)]
    assert cgr.closed_runs(np.array([], bool)) == []


def _synthetic_frames(rng, n_eps=10, length=100, close_eps=(), close_at=50, close_len=5):
    ep, fr, left, right = [], [], [], []
    for e in range(n_eps):
        ep.append(np.full(length, e))
        fr.append(np.arange(length))
        lsum = np.full(length, 0.08)
        if e in close_eps:
            lsum[close_at: close_at + close_len] = 0.0005
        left.append(lsum)
        right.append(np.full(length, 0.0995))
    order = rng.permutation(n_eps * length)  # rows need not be sorted
    cat = lambda xs: np.concatenate(xs)[order]  # noqa: E731
    return {"episode": cat(ep), "frame": cat(fr), "left": cat(left), "right": cat(right)}


def test_script_episode_stats_and_rules(cgr):
    import argparse

    rng = np.random.default_rng(0)
    per_frame = _synthetic_frames(rng, close_eps=(2, 5), close_at=40, close_len=5)
    per_frame["left"][(per_frame["episode"] == 7) & (per_frame["frame"] == 10)] = 0.0  # 1-frame glitch
    eps = cgr.episode_stats(per_frame, 0.001)
    assert [e["episode_index"] for e in eps] == list(range(10))
    assert all(e["length"] == 100 and e["contiguous"] for e in eps)
    assert eps[2]["left"]["runs"] == [(40, 5)] and eps[2]["left"]["closed_frames"] == 5
    assert eps[7]["left"]["runs"] == [(10, 1)]
    assert eps[0]["right"]["closed_frames"] == 0
    st = {"episodes": eps, "n_frames": 1000}
    s = cgr.side_summary(st, "left", human_mean_len=100.0, min_run_frames=3)
    assert s["n_closing"] == 2 and s["n_any_closed_frame"] == 3 and s["close_frac"] == 0.2
    assert s["first_close_progress_p00"] == pytest.approx(0.4)
    args = argparse.Namespace(always_open_max_frac=0.03, late_progress=0.9, percentile=1, progress_safety=0.6,
                              min_useful_progress=0.05)
    r = cgr.derive_side_rule(s, args)
    assert r == {"always_open": False, "min_stage": None, "min_progress": pytest.approx(0.24)}
    assert cgr.derive_side_rule(cgr.side_summary(st, "right", 100.0, 3), args)["always_open"] is True
    rare = dict(s, close_frac=0.02)  # rare and early -> always open; rare and late -> gate at the earliest
    assert cgr.derive_side_rule(rare, args)["always_open"] is True
    late = dict(rare, first_close_progress_p00=0.95)
    assert cgr.derive_side_rule(late, args) == {"always_open": False, "min_stage": None,
                                                "min_progress": pytest.approx(0.57)}
    early_common = dict(s, first_close_progress_p01=0.05)  # 0.6 * 0.05 < min_useful_progress -> no gate
    assert cgr.derive_side_rule(early_common, args)["min_progress"] is None
    # Progress uses the smaller of f / length and f / human_mean_len.
    s2 = cgr.side_summary(st, "left", human_mean_len=200.0, min_run_frames=3)
    assert s2["first_close_progress_p00"] == pytest.approx(0.2)


def test_script_remote_column_fetch_roundtrip(cgr, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(1)
    n = 5000
    state = rng.normal(size=(n, 61)).astype(np.float32)
    table = pa.table({
        "action": pa.FixedSizeListArray.from_arrays(rng.normal(size=n * 23).astype(np.float32), 23).cast(
            pa.list_(pa.float32())),
        "observation.state": pa.FixedSizeListArray.from_arrays(state.reshape(-1), 61).cast(pa.list_(pa.float32())),
        "episode_index": pa.array(np.repeat(np.arange(5), n // 5), pa.int64()),
        "frame_index": pa.array(np.tile(np.arange(n // 5), 5), pa.int64()),
        "noise": pa.array(rng.normal(size=n)),
    })
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, compression="zstd", row_group_size=1500)
    blob = src.read_bytes()

    rp = object.__new__(cgr.RemoteParquet)
    rp.size, rp.workers, rp.retries, rp.url = len(blob), 4, 1, "local"
    requested = []

    def get_range(a, b):
        requested.append((a, b))
        return blob[a: b + 1]

    rp.get_range = get_range
    sparse = tmp_path / "sparse.parquet"
    rp.fetch_columns(sparse, cgr.COLUMNS)
    # Column projection: the big "action" column chunks outside the 1 MiB footer read are never requested.
    md = pq.ParquetFile(src).metadata
    tail_start = len(blob) - min(len(blob), 1 << 20)
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(rg).column(c)
            if not col.path_in_schema.startswith("action"):
                continue
            lo = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset is not None:
                lo = min(lo, col.dictionary_page_offset)
            hi = lo + col.total_compressed_size - 1
            if hi < tail_start:
                assert not any(a <= hi and b >= lo for a, b in requested[1:]), (lo, hi)
    got = cgr.finger_sums(pq.read_table(str(sparse), columns=list(cgr.COLUMNS)))
    np.testing.assert_array_equal(got["episode"], table.column("episode_index").to_numpy())
    np.testing.assert_array_equal(got["frame"], table.column("frame_index").to_numpy())
    np.testing.assert_allclose(got["left"], state[:, 24:26].astype(np.float64).sum(1))
    np.testing.assert_allclose(got["right"], state[:, 49:51].astype(np.float64).sum(1))
