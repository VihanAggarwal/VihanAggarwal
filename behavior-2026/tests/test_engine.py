"""PolicyEngine / InferenceScheduler semantics with an in-process stand-in for the worker connection."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut
from b1k26.backends.fake import HoldBackend, SineBackend
from b1k26.config import parse_config, resolve_prompt
from b1k26.corrections import GripperRules
from b1k26.engine import InferenceScheduler, PolicyEngine, WorkerClient, WorkerError
from b1k26.obs import hold_action, split_batch
from b1k26.protocol import packb, unpackb
from b1k26.session import RolloutSession
from b1k26.worker import decode_item, encode_chunk

from runtime_helpers import base_doc, batched_obs, proprio_at

_P = C.PROPRIO_INDICES_2026


class StubClient:
    """WorkerClient stand-in: runs a backend in-process, with the same msgpack round trip as the wire."""

    def __init__(self, backend: Backend, name: str = "w", delay_s: float = 0.0):
        self.backend = backend
        self.name = name
        self.state = "ready"
        self.info = WorkerClient._validate_info(backend.info())
        self.calls: list[list[dict[str, Any]]] = []
        self.delay_s = delay_s
        self.fail_next = 0
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        return self.state == "failed"

    async def start(self) -> None:
        if self.state == "failed":
            raise WorkerError("failed")

    async def close(self) -> None:
        self.state = "stopped"

    def status(self) -> dict[str, Any]:
        return {"state": self.state}

    serves = WorkerClient.serves  # same task filter as the real client (reads self.info)

    async def infer(self, items: list[dict[str, Any]], timeout: float) -> list[dict[str, Any]]:
        if self.state != "ready":
            raise WorkerError(f"worker {self.name} is {self.state}")
        wire_items = unpackb(packb(items))
        self.calls.append(wire_items)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise WorkerError("injected")
        chunks = self.backend.infer([decode_item(it, i) for i, it in enumerate(wire_items)])
        return unpackb(packb([encode_chunk(c, i) for i, c in enumerate(chunks)]))


def make_engine(doc: dict[str, Any], clients: dict[str, StubClient], rules: GripperRules | None = None) -> PolicyEngine:
    cfg = parse_config(doc)
    return PolicyEngine(cfg, worker_clients=clients, rules=rules)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def envs_at(step: int, task_ids: list[int]):
    return split_batch(batched_obs(step, task_ids, res=(48, 48), depth=False))


async def drive(eng: PolicyEngine, sessions: list[RolloutSession], tasks: list[int], steps: int, k: int,
                obs_fn=envs_at) -> list[tuple[np.ndarray, np.ndarray | None]]:
    out = []
    t = 0
    while t < steps:
        action, chunk = await eng.step(sessions, obs_fn(t, tasks), k)
        out.append((action, chunk))
        t += max(k, 1)
    return out


# --------------------------------------------------------------------------------------------------------------
def test_chunk_queue_semantics_and_bit_exact_first_row() -> None:
    client = StubClient(SineBackend(horizon=32))
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> None:
        res = await drive(eng, [s], [3], 64, 8)
        for action, chunk in res:
            assert action.dtype == np.float32 and chunk.dtype == np.float32
            assert action.shape == (1, 23) and chunk.shape == (1, 8, 23)
            assert np.array_equal(chunk[:, 0], action)
            assert action.flags["C_CONTIGUOUS"] and chunk.flags["C_CONTIGUOUS"]
        # execute 20: 8 + 8 consumed, the 4-action leftover is discarded -> a plan every 16 steps.
        assert len(client.calls) == 4
        assert s.step == 64 and s.stats.plans == 4 and s.stats.hold_steps == 0
        # No chunk requested: one action per call, a plan every execute_steps.
        s.reset()
        client.calls.clear()
        res = await drive(eng, [s], [3], 40, 0)
        assert all(c is None and a.shape == (1, 23) for a, c in res)
        assert len(client.calls) == 2 and s.step == 40

    run(main())


def test_plan_output_follows_corrections_compression_sanitize() -> None:
    """The executed actions are exactly plan_execution(raw) of the backend's chunk (no correction applies)."""
    from b1k26.control import ExecutionConfig, plan_execution, sanitize

    backend = SineBackend(horizon=32)
    client = StubClient(backend)
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> np.ndarray:
        res = await drive(eng, [s], [3], 20, 20)
        return res[0][1][0]

    executed = run(main())
    env = envs_at(0, [3])[0]
    item = decode_item(client.calls[0][0], 0)
    raw = backend.infer([item])[0].actions
    planned = plan_execution(raw, ExecutionConfig(execute_steps=20, predicted_steps_to_use=26, keep_for_inpaint=4))
    assert planned.compressed
    np.testing.assert_array_equal(executed, sanitize(planned.actions, env.proprio))


def test_inpaint_tail_is_passed_and_cleared_on_discard() -> None:
    backend = SineBackend(horizon=32)
    client = StubClient(backend)
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> None:
        await drive(eng, [s], [3], 40, 0)  # plans at steps 0 and 20
        assert client.calls[0][0]["initial_actions"] is None
        first_raw = backend.infer([decode_item(client.calls[0][0], 0)])[0].actions
        np.testing.assert_array_equal(client.calls[1][0]["initial_actions"], first_raw[26:30])
        # With K=8, the leftover is discarded, so the tail is not aligned any more and must not be sent.
        s.reset()
        client.calls.clear()
        await drive(eng, [s], [3], 32, 8)
        assert all(c[0]["initial_actions"] is None for c in client.calls[1:])

    run(main())


def test_no_inpaint_for_workers_without_support() -> None:
    client = StubClient(SineBackend(horizon=32, supports_inpaint=False))
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))
    run(drive(eng, [s], [3], 40, 0))
    assert all(c[0]["initial_actions"] is None for c in client.calls)


def test_item_contents_mask_prompt_images() -> None:
    client = StubClient(HoldBackend(horizon=16))
    doc = base_doc(mask_base_qvel=True, prompt="comet2025", image_size=40, cameras=["head", "right_wrist"])
    eng = make_engine(doc, {"w": client})
    s = RolloutSession((0, 0, 0))
    env = envs_at(5, [0])[0]
    assert np.any(env.proprio[0:3] != 0)
    run(eng.step([s], [env], 0))
    item = client.calls[0][0]
    assert item["task_id"] == 0
    assert item["prompt"] == C.task(0).instruction_comet2025
    np.testing.assert_array_equal(item["proprio"][0:3], 0.0)
    np.testing.assert_array_equal(item["proprio"][3:], env.proprio[3:])
    assert sorted(item["images"]) == ["head", "right_wrist"]
    assert item["images"]["head"].shape == (40, 40, 3) and item["images"]["head"].dtype == np.uint8
    assert item["stage"] is None


def test_prompt_styles() -> None:
    assert resolve_prompt("snake_case", 0) == "turning_on_radio"
    assert resolve_prompt("instruction", 60) == C.task(60).instruction
    assert resolve_prompt("comet2025", 60) == C.task(60).instruction  # new task: no 2025 text
    assert resolve_prompt("comet2025", 1) == C.task(1).instruction_comet2025
    with pytest.raises(ValueError):
        resolve_prompt("nope", 0)


def _closed_left_obs(step: int, tasks: list[int]):
    envs = envs_at(step, tasks)
    for e in envs:
        e.proprio[_P["gripper_left_qpos"]] = 0.0  # fully closed: normalized width -1 < -0.98
    return envs


def _always_open_left_task(rules: GripperRules) -> int:
    for tid in range(C.NUM_TASKS):
        r = rules.tasks.get(tid)
        if r is not None and r.left is not None and r.left.always_open:
            return tid
    raise AssertionError("no always-open left rule")


@pytest.mark.parametrize("corrections", [True, False])
def test_gripper_correction_reopens_closed_gripper(corrections: bool) -> None:
    rules = GripperRules.load()
    tid = _always_open_left_task(rules)
    client = StubClient(SineBackend(horizon=32))  # keeps grippers at the current (closed) width
    eng = make_engine(base_doc(corrections=corrections), {"w": client}, rules=rules)
    s = RolloutSession((0, 0, 0))
    action, chunk = run(eng.step([s], _closed_left_obs(0, [tid]), 20))
    if corrections:
        assert s.stats.corrections == 1
        np.testing.assert_array_equal(chunk[0, :, C.LEFT_GRIPPER_ACTION_IDX], 1.0)
        np.testing.assert_array_equal(chunk[0, :, 0:3], 0.0)  # hold chunk: base stopped
    else:
        assert s.stats.corrections == 0
        np.testing.assert_array_equal(chunk[0, :, C.LEFT_GRIPPER_ACTION_IDX], -1.0)


def test_stage_tracking_votes_and_is_sent_to_worker() -> None:
    client = StubClient(SineBackend(horizon=32, num_stages=4, stage_step=1))
    eng = make_engine(base_doc(use_stage=True), {"w": client})
    s = RolloutSession((0, 0, 0))
    run(drive(eng, [s], [3], 20 * 10, 20))
    stages = [c[0]["stage"] for c in client.calls]
    # The fake predicts stage + 1. RLC voting decides only on a full history of 3: promote after 3 plans.
    assert stages == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert s.stage is not None and s.stage.stage == 3


def test_stage_disabled_without_use_stage() -> None:
    client = StubClient(SineBackend(horizon=32, num_stages=4))
    eng = make_engine(base_doc(use_stage=False), {"w": client})
    s = RolloutSession((0, 0, 0))
    run(drive(eng, [s], [3], 60, 20))
    assert all(c[0]["stage"] is None for c in client.calls) and s.stage is None


def test_task_change_resets_plan_state() -> None:
    client = StubClient(SineBackend(horizon=32))
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> None:
        await eng.step([s], envs_at(0, [3]), 0)
        assert len(s.queue) == 19 and s.inpaint_tail is not None and s.step == 1
        await eng.step([s], envs_at(1, [4]), 0)  # task changed without a reset: replan immediately
        assert s.task_id == 4 and s.step == 1 and len(client.calls) == 2
        assert client.calls[1][0]["initial_actions"] is None

    run(main())


def test_worker_failure_gives_hold_and_retries_next_step() -> None:
    client = StubClient(SineBackend(horizon=32))
    client.fail_next = 1
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> None:
        env0 = envs_at(0, [3])
        action, chunk = await eng.step([s], env0, 8)
        np.testing.assert_array_equal(chunk[0], np.tile(hold_action(env0[0].proprio), (8, 1)))
        assert s.stats.plan_failures == 1 and s.stats.hold_steps == 8 and len(s.queue) == 0
        env1 = envs_at(8, [3])
        action, chunk = await eng.step([s], env1, 8)
        assert not np.array_equal(action[0], hold_action(env1[0].proprio))
        assert s.stats.plans == 1 and s.step == 16

    run(main())


def test_plan_timeout_gives_hold_quickly() -> None:
    doc = base_doc()
    doc["engine"]["plan_timeout_s"] = 0.05
    client = StubClient(SineBackend(horizon=32), delay_s=0.5)
    eng = make_engine(doc, {"w": client})
    s = RolloutSession((0, 0, 0))

    async def main() -> float:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        env = envs_at(0, [3])
        action, _ = await eng.step([s], env, 0)
        np.testing.assert_array_equal(action[0], hold_action(env[0].proprio))
        return loop.time() - t0

    assert run(main()) < 0.4
    assert s.stats.plan_failures == 1


def test_worker_not_ready_fails_fast_with_hold() -> None:
    client = StubClient(SineBackend(horizon=32))
    client.state = "restarting"
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))
    env = envs_at(0, [3])
    action, _ = run(eng.step([s], env, 0))
    np.testing.assert_array_equal(action[0], hold_action(env[0].proprio))
    assert client.calls == []


def test_padding_when_plan_is_shorter_than_chunk() -> None:
    doc = base_doc(execution={"execute_steps": 16, "predicted_steps_to_use": 16, "keep_for_inpaint": 0})
    client = StubClient(SineBackend(horizon=32))
    eng = make_engine(doc, {"w": client})
    s = RolloutSession((0, 0, 0))
    _, chunk = run(eng.step([s], envs_at(0, [3]), 20))
    c = chunk[0]
    for i in range(16, 20):
        np.testing.assert_array_equal(c[i, 3:], c[15, 3:])
        np.testing.assert_array_equal(c[i, 0:3], 0.0)
    assert s.stats.padded == 4 and s.inpaint_tail is None


def test_routing_and_fallback_to_default_when_routed_worker_failed() -> None:
    doc = base_doc()
    doc["workers"]["w2"] = {"endpoint": "ws://127.0.0.1:2"}
    doc["profiles"]["q"] = copy.deepcopy(doc["profiles"]["p"]) | {"worker": "w2", "prompt": "snake_case"}
    doc["routing"]["per_task"] = {"turning_on_radio": "q"}
    c1, c2 = StubClient(SineBackend(horizon=32)), StubClient(SineBackend(horizon=32), name="w2")
    eng = make_engine(doc, {"w": c1, "w2": c2})
    s0, s1 = RolloutSession((0, 0, 0)), RolloutSession((0, 0, 1))
    run(eng.step([s0, s1], envs_at(0, [0, 3]), 0))
    assert s0.profile_name == "q" and s1.profile_name == "p"
    assert len(c2.calls) == 1 and c2.calls[0][0]["prompt"] == "turning_on_radio"
    assert len(c1.calls) == 1 and c1.calls[0][0]["task_id"] == 3
    c2.state = "failed"
    s0.reset()
    action, _ = run(eng.step([s0], envs_at(0, [0]), 0))
    assert s0.profile_name == "p" and s0.fallback_note is not None
    assert len(c1.calls) == 2 and s0.stats.plans == 1


def test_scheduler_micro_batches_concurrent_plans() -> None:
    doc = base_doc()
    doc["engine"]["max_batch"] = 4
    doc["engine"]["batch_wait_ms"] = 0  # one message: all plans are submitted together, no waiting needed
    client = StubClient(SineBackend(horizon=32))
    eng = make_engine(doc, {"w": client})
    sessions = [RolloutSession((0, 0, b)) for b in range(10)]
    run(eng.step(sessions, envs_at(0, list(range(10))), 0))
    assert [len(c) for c in client.calls] == [4, 4, 2]
    assert all(s.stats.plans == 1 for s in sessions)

    # Separate step() calls (one per port, as with --policy-endpoints) arriving together share one batch.
    doc["engine"]["max_batch"] = 8
    doc["engine"]["batch_wait_ms"] = 100
    client = StubClient(SineBackend(horizon=32))
    eng = make_engine(doc, {"w": client})
    sessions = [RolloutSession((p, 0, 0)) for p in range(6)]

    async def main() -> None:
        await asyncio.gather(*(eng.step([s], envs_at(0, [i]), 0) for i, s in enumerate(sessions)))

    run(main())
    # Timing-dependent under heavy load, so only require that the six requests were actually batched.
    assert sum(len(c) for c in client.calls) == 6 and len(client.calls) <= 2


def test_scheduler_skips_cancelled_requests() -> None:
    class SlowClient(StubClient):
        async def infer(self, items, timeout):
            self.calls.append(items)
            await asyncio.sleep(0.05)
            return [{"actions": np.zeros((4, 23), np.float32)} for _ in items]

    client = SlowClient(HoldBackend())

    async def main() -> None:
        sched = InferenceScheduler(client, max_batch=8, batch_wait_s=0.0)
        f1 = sched.submit({"n": 1})
        await asyncio.sleep(0.01)  # f1 is in flight
        f2 = sched.submit({"n": 2})
        f3 = sched.submit({"n": 3})
        f2.cancel()
        await f1
        await f3
        await sched.close()

    run(main())
    assert [[i["n"] for i in c] for c in client.calls] == [[1], [3]]


def test_engine_never_raises_and_sanitizes_bad_chunks() -> None:
    class NaNBackend(SineBackend):
        def _chunk(self, item):
            out = super()._chunk(item)
            a = out.actions.copy()
            a[3, 5] = np.nan
            return ChunkOut(actions=a)

    client = StubClient(NaNBackend(horizon=32))
    eng = make_engine(base_doc(), {"w": client})
    s = RolloutSession((0, 0, 0))
    env = envs_at(0, [3])
    _, chunk = run(eng.step([s], env, 20))
    assert np.all(np.isfinite(chunk))
    # A NaN in the consumed segment disables compression; only that row becomes a hold.
    np.testing.assert_array_equal(chunk[0, 3], hold_action(env[0].proprio))
    assert not np.array_equal(chunk[0, 4], hold_action(env[0].proprio))
    # Mismatched sessions/observations: still a valid reply (hold) instead of an exception.
    action, chunk = run(eng.step([s], envs_at(0, [3, 4]), 4))
    assert action.shape == (2, 23) and chunk.shape == (2, 4, 23)


def test_malformed_worker_chunk_gives_hold() -> None:
    class BadClient(StubClient):
        async def infer(self, items, timeout):
            return [{"actions": np.zeros((5, 7), np.float32)} for _ in items]

    eng = make_engine(base_doc(), {"w": BadClient(HoldBackend())})
    s = RolloutSession((0, 0, 0))
    env = envs_at(0, [3])
    action, _ = run(eng.step([s], env, 0))
    np.testing.assert_array_equal(action[0], hold_action(env[0].proprio))
    assert s.stats.plan_failures == 1


def test_engine_start_and_warm() -> None:
    doc = base_doc()
    doc["workers"]["w2"] = {"endpoint": "ws://127.0.0.1:2"}
    doc["profiles"]["q"] = copy.deepcopy(doc["profiles"]["p"]) | {"worker": "w2"}
    doc["routing"]["per_task"] = {5: "q"}
    c1, c2 = StubClient(HoldBackend()), StubClient(HoldBackend(), name="w2")
    c2.state = "failed"
    eng = make_engine(doc, {"w": c1, "w2": c2})
    assert not eng.warm
    run(eng.start())  # a failed non-default worker does not block warm-up
    assert eng.warm
    c1.state = "failed"
    assert not eng.warm
    eng2 = make_engine(doc, {"w": StubClient(HoldBackend()), "w2": StubClient(HoldBackend(), name="w2")})
    eng2.clients["w"].state = "failed"
    with pytest.raises(WorkerError):
        run(eng2.start())
    assert not eng2.warm


def test_hold_actions_are_never_zero() -> None:
    p = proprio_at(3)
    h = hold_action(p)
    np.testing.assert_array_equal(h[3:7], p[_P["trunk_qpos"]])
    assert h[0:3].tolist() == [0, 0, 0]


# --------------------------------------------------------------------------------------------------------------
# Profile / worker consistency (integration checks)
# --------------------------------------------------------------------------------------------------------------
class ServedSine(SineBackend):
    """A sine backend that serves only some tasks (like a pibehavior worker) and rejects a batch with any other."""

    def __init__(self, served: list[int], **kwargs: Any):
        super().__init__(**kwargs)
        self.served = set(served)

    def info(self) -> dict[str, Any]:
        info = super().info()
        info["supported_tasks"] = sorted(self.served)
        return info

    def infer(self, items):
        bad = [it.task_id for it in items if it.task_id not in self.served]
        if bad:
            raise ValueError(f"tasks {bad} are not served by this worker")
        return super().infer(items)


def test_check_profiles_reports_mismatches() -> None:
    doc = base_doc(image_size=64, use_stage=True)
    eng = make_engine(doc, {"w": StubClient(SineBackend(horizon=24, image_size=224))})
    joined = "\n".join(eng.check_profiles())
    assert "image_size 64 but worker w expects 224" in joined
    assert "use_stage is on but worker w reports no stage support" in joined
    assert "predicted_steps_to_use 26 > worker horizon 24" in joined
    assert "keep_for_inpaint 4 does not fit in the worker horizon 24" in joined
    short = make_engine(base_doc(image_size=224), {"w": StubClient(SineBackend(horizon=16, image_size=224))})
    assert any("execute_steps 20 > worker horizon 16" in p for p in short.check_profiles())
    staged = make_engine(base_doc(image_size=224), {"w": StubClient(SineBackend(horizon=32, num_stages=3))})
    assert any("stage-conditioned but use_stage is off" in p for p in staged.check_profiles())
    # A consistent profile reports nothing, and start() publishes the result in status().
    ok = make_engine(base_doc(image_size=224), {"w": StubClient(SineBackend(horizon=32, image_size=224))})
    run(ok.start())
    assert ok.check_profiles() == [] and ok.status()["config_problems"] == []
    assert ok.status()["warm"] is True


def test_served_tasks_from_info() -> None:
    c = StubClient(ServedSine([0, 5], horizon=32))
    assert c.serves(0) and c.serves(5) and not c.serves(1)
    # num_stages == 0 marks an unserved task for stage-tracking workers (pibehavior); a plain worker serves all.
    s = StubClient(SineBackend(horizon=32))
    s.info = WorkerClient._validate_info({**s.info, "num_stages": [0] * 50 + [3] * 50, "supports_stage": True})
    assert not s.serves(10) and s.serves(60)
    assert StubClient(SineBackend(horizon=32)).serves(99)
    with pytest.raises(WorkerError):
        WorkerClient._validate_info({"action_horizon": 8, "supported_tasks": ["x"]})


def test_unserved_tasks_fall_back_to_default_and_never_poison_a_batch() -> None:
    doc = base_doc()
    doc["workers"]["w2"] = {"endpoint": "ws://127.0.0.1:2"}
    doc["profiles"]["q"] = copy.deepcopy(doc["profiles"]["p"]) | {"worker": "w2"}
    doc["routing"]["per_task"] = {0: "q", 1: "q", 60: "q"}
    c1, c2 = StubClient(SineBackend(horizon=32)), StubClient(ServedSine([0, 1], horizon=32), name="w2")
    eng = make_engine(doc, {"w": c1, "w2": c2})
    problems = eng.check_profiles()
    assert any("does not serve 1 routed task(s) [60]" in p and "fall back to the default profile p" in p
               for p in problems), problems
    sessions = [RolloutSession((0, 0, b)) for b in range(3)]
    run(eng.step(sessions, envs_at(0, [0, 60, 1]), 0))
    assert [s.profile_name for s in sessions] == ["q", "p", "q"]
    assert sessions[1].fallback_note == "worker w2 does not serve task 60"
    assert all(s.stats.plans == 1 and s.stats.plan_failures == 0 for s in sessions)
    assert sorted(it["task_id"] for call in c2.calls for it in call) == [0, 1]
    assert [it["task_id"] for call in c1.calls for it in call] == [60]

    # Without fallback the unserved task keeps its profile: its plan fails and it holds its pose.
    doc["engine"]["fallback_to_default"] = False
    c1, c2 = StubClient(SineBackend(horizon=32)), StubClient(ServedSine([0, 1], horizon=32), name="w2")
    eng = make_engine(doc, {"w": c1, "w2": c2})
    assert any("their plans fail" in p for p in eng.check_profiles())
    sessions = [RolloutSession((0, 0, b)) for b in range(3)]
    action, _ = run(eng.step(sessions, envs_at(0, [0, 60, 1]), 0))
    assert sessions[1].profile_name == "q" and sessions[1].stats.plan_failures == 1
    np.testing.assert_array_equal(action[1], hold_action(proprio_at(0, 1)))
    # ... but it is never sent, so the served tasks of the same message still plan normally.
    assert sessions[0].stats.plans == 1 and sessions[2].stats.plans == 1
    assert sorted(it["task_id"] for call in c2.calls for it in call) == [0, 1]


def test_restart_window_forgets_old_restarts() -> None:
    from b1k26.config import WorkerConfig

    c = WorkerClient(WorkerConfig(name="x", endpoint="ws://127.0.0.1:1", max_restarts=2, restart_window_s=100.0))
    assert c._restart_allowed(now=0.0)
    c._record_restart(now=0.0)
    c._record_restart(now=50.0)
    assert not c._restart_allowed(now=99.0)
    assert c._restart_allowed(now=100.5)  # the first restart has left the window
    c._record_restart(now=101.0)
    assert not c._restart_allowed(now=120.0)
    assert c.restarts == 3
    zero = WorkerClient(WorkerConfig(name="z", endpoint="ws://127.0.0.1:1", max_restarts=0))
    assert not zero._restart_allowed(now=0.0)
