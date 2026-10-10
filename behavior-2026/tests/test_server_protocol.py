"""End-to-end protocol tests: the real evaluator clients (vendored verbatim, see tests/vendor) against b1k26-serve.

Covers the v3.9.3-post2 WebsocketClientPolicy (single port, batched, chunk replay), the 2026/eval client
(reconnect with resend, per-query deadlines) and MultiWebsocketPolicy (--policy-endpoints, one env per port),
health gating, worker failures and v3.9.2-style unbatched observations.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import requests

torch = pytest.importorskip("torch")

from b1k26 import constants as C  # noqa: E402
from b1k26.obs import hold_action  # noqa: E402

from runtime_helpers import Harness, base_doc, batched_obs, env_obs_dict, proprio_at, to_torch  # noqa: E402
from vendor import load_2026eval, load_post2  # noqa: E402

SINE = ("fake_sine", {"horizon": 32, "period": 40.0})
STEPS = 200


def run_post2(port: int, task_ids: list[int], steps: int, chunk: int, start_step: int = 0) -> np.ndarray:
    nu = load_post2()
    client = nu.WebsocketClientPolicy(host="127.0.0.1", port=port, action_chunk_size=chunk)
    try:
        client.reset()
        out = []
        for t in range(start_step, start_step + steps):
            a = client.act(to_torch(batched_obs(t, task_ids)))
            assert isinstance(a, torch.Tensor) and a.dtype == torch.float32
            out.append(a.numpy().copy())
        return np.stack(out)
    finally:
        if client._ws is not None:
            client._ws.close()


def sessions_of(h: Harness, port: int) -> list:
    return h.run(lambda: [s for g in h.server.groups.get(port, []) for s in g.sessions.values()])


@pytest.fixture(scope="module")
def sine_server():
    h = Harness(base_doc(), workers={"w": SINE}, n_ports=3).start()
    yield h
    h.stop()


# --------------------------------------------------------------------------------------------------------------
# (1) v3.9.3-post2 client: reset, 200 steps, N = 1 and 3, with and without chunk replay
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("chunk", [0, 8, 20])
def test_post2_client_rollout(sine_server: Harness, n: int, chunk: int) -> None:
    port = sine_server.ports[0]
    tasks = [0, 5, 60][:n]
    q0 = sine_server.run(lambda: sine_server.server.counters["queries"])
    actions = run_post2(port, tasks, STEPS, chunk)
    assert actions.shape == (STEPS, n, C.ACTION_DIM)
    assert np.all(np.isfinite(actions))
    assert not np.any(np.all(actions[..., 3:7] == 0, axis=-1)), "zero torso command"
    assert np.all(np.abs(actions[..., 0:3]) <= 1.0)
    queries = sine_server.run(lambda: sine_server.server.counters["queries"]) - q0
    assert queries == (STEPS if chunk <= 1 else math.ceil(STEPS / chunk))
    sess = sorted(sessions_of(sine_server, port), key=lambda s: s.key)
    active = [s for s in sess if s.active]
    assert len(active) == n
    for s, t in zip(active, tasks):
        assert s.task_id == t
        assert s.step == (STEPS if chunk <= 1 else math.ceil(STEPS / chunk) * chunk)
        assert s.stats.plan_failures == 0 and s.stats.hold_steps == 0
        # execute_steps 20: K=8 replans every 16 steps (a 4-action leftover is discarded); otherwise every 20.
        expected_plans = math.ceil(STEPS / 16) if chunk == 8 else math.ceil(STEPS / 20)
        assert s.stats.plans == expected_plans


def test_chunk_replay_matches_per_step_serving(sine_server: Harness) -> None:
    """With K == execute_steps, client-side replay executes exactly the actions per-step serving would send."""
    port = sine_server.ports[0]
    per_step = run_post2(port, [0, 33, 77], 100, 0)
    chunked = run_post2(port, [0, 33, 77], 100, 20)
    np.testing.assert_array_equal(per_step, chunked)
    chunk8 = run_post2(port, [0, 33, 77], 100, 8)
    assert not np.array_equal(per_step, chunk8)  # K=8 replans every 16 steps instead of 20


def test_concurrent_connections_on_one_port_get_separate_slot_groups(sine_server: Harness) -> None:
    import concurrent.futures

    port = sine_server.ports[1]
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        fa = ex.submit(run_post2, port, [1], 60, 0)
        fb = ex.submit(run_post2, port, [2], 60, 0)
        a, b = fa.result(30), fb.result(30)
    ref_a = run_post2(port, [1], 60, 0)
    ref_b = run_post2(port, [2], 60, 0)
    np.testing.assert_array_equal(a, ref_a)
    np.testing.assert_array_equal(b, ref_b)
    assert len(sine_server.run(lambda: sine_server.server.groups[port])) >= 2


# --------------------------------------------------------------------------------------------------------------
# (2) 2026/eval client: reconnect after a server-side drop resends the observation; no double advance
# --------------------------------------------------------------------------------------------------------------
def run_2026(port: int, task_id: int, steps: int, chunk: int, deadline_s: float | None = 120.0) -> np.ndarray:
    nu, _ = load_2026eval()
    client = nu.WebsocketClientPolicy(host="127.0.0.1", port=port, allow_reconnect=True, action_chunk_size=chunk)
    try:
        client.set_deadline(None if deadline_s is None else time.monotonic() + deadline_s)
        client.reset()
        out = []
        for t in range(steps):
            out.append(client.act(to_torch(batched_obs(t, [task_id]))).numpy().copy())
        return np.stack(out)
    finally:
        client.close()


class DropOnce:
    """Server fault hook: abort the TCP connection at the n-th occurrence of ``stage``."""

    def __init__(self, stage: str, n: int, action: str = "drop"):
        self.stage, self.n, self.count, self.fired, self.action = stage, n, 0, False, action

    def __call__(self, stage: str, port: int, group) -> str | None:
        if stage != self.stage or self.fired:
            return None
        self.count += 1
        if self.count == self.n:
            self.fired = True
            return self.action
        return None


@pytest.mark.parametrize("chunk", [0, 20])
@pytest.mark.parametrize("stage", ["send", "recv"])
def test_2026_reconnect_resend_does_not_double_advance(sine_server: Harness, stage: str, chunk: int) -> None:
    port = sine_server.ports[2]
    steps = 120
    reference = run_2026(port, 11, steps, chunk)
    replays0 = sine_server.run(lambda: sine_server.server.counters["replays"])
    conns0 = sine_server.run(lambda: sine_server.server.counters["connections"])
    hook = DropOnce(stage, 3 if chunk else 50)
    sine_server.server.fault_hook = hook
    try:
        dropped = run_2026(port, 11, steps, chunk)
    finally:
        sine_server.server.fault_hook = None
    assert hook.fired
    np.testing.assert_array_equal(dropped, reference)
    replays = sine_server.run(lambda: sine_server.server.counters["replays"]) - replays0
    # Dropped after stepping ("send"): the resent observation must be answered from the cache. Dropped before
    # stepping ("recv"): the server never saw it, so it is stepped normally.
    assert replays == (1 if stage == "send" else 0)
    assert sine_server.run(lambda: sine_server.server.counters["connections"]) - conns0 == 2
    s = [s for s in sessions_of(sine_server, port) if s.task_id == 11]
    assert len(s) == 1 and s[0].step == (steps if not chunk else math.ceil(steps / chunk) * chunk)


def test_2026_reconnect_while_step_is_in_flight() -> None:
    """The connection drops while a (slow) plan is running; the client reconnects and resends before the step
    finishes. The new connection must wait for that step and replay its reply, not step a second time."""
    worker = ("fake_sine", {"horizon": 32, "slow_task_ids": [11], "slow_ms": 300})
    with Harness(base_doc(), workers={"w": worker}) as h:
        port = h.ports[0]
        reference = run_2026(port, 11, 100, 20)
        hook = DropOnce("recv", 3, action="abort")
        h.server.fault_hook = hook
        dropped = run_2026(port, 11, 100, 20)
        h.server.fault_hook = None
        assert hook.fired
        np.testing.assert_array_equal(dropped, reference)
        assert h.run(lambda: h.server.counters["replays"]) == 1
        s = [s for s in sessions_of(h, port) if s.active]
        assert len(s) == 1 and s[0].step == 100 and s[0].stats.plans == 5


# --------------------------------------------------------------------------------------------------------------
# (3) 2026/eval MultiWebsocketPolicy: one env per port, per-env observations unsqueezed to N = 1
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [0, 20])
def test_2026_multiport_policy(sine_server: Harness, chunk: int) -> None:
    _, pol = load_2026eval()
    ports = sine_server.ports
    tasks = [4, 42, 88]
    policy = pol.MultiWebsocketPolicy([{"host": "127.0.0.1", "port": p} for p in ports], allow_reconnect=True,
                                      action_chunk_size=chunk)
    try:
        policy.set_action_dim(C.ACTION_DIM)
        policy.set_time_budget(600.0)
        policy.reset()
        assert policy.failures == {}
        out = []
        for t in range(100):
            obs = [{k: torch.from_numpy(v) for k, v in env_obs_dict(t, task, env=i).items()}
                   for i, task in enumerate(tasks)]
            res = policy.forward(obs, active_env_indices=[0, 1, 2])
            assert res.failures == {}, res.failures
            assert tuple(res.actions.shape) == (3, C.ACTION_DIM)
            out.append(res.actions.numpy().copy())
    finally:
        policy.close()
    multi = np.stack(out)
    for p, task in zip(ports, tasks):
        active = [s for s in sessions_of(sine_server, p) if s.task_id == task]
        assert len(active) == 1 and active[0].key[0] == p
    # Each env's actions only depend on its own observations: same as one batched single-port connection.
    single = run_post2(ports[0], tasks, 100, chunk)
    np.testing.assert_array_equal(multi, single)


# --------------------------------------------------------------------------------------------------------------
# (4) Health gating: 503 until the default worker is warm
# --------------------------------------------------------------------------------------------------------------
def test_health_gating_until_worker_is_up() -> None:
    h = Harness(base_doc(), workers={"w": SINE}, start_workers=False)
    h.start(wait_warm=False)
    try:
        url = f"http://127.0.0.1:{h.ports[0]}/healthz"
        r = requests.get(url, timeout=2)
        assert r.status_code == 503
        status = requests.get(f"http://127.0.0.1:{h.ports[0]}/status", timeout=2).json()
        assert status["healthy"] is False and status["engine"]["workers"]["w"]["state"] == "starting"
        h.start_worker("w")
        t0 = time.monotonic()
        while requests.get(url, timeout=2).status_code != 200:
            assert time.monotonic() - t0 < 10, "server did not become healthy after the worker came up"
            time.sleep(0.05)
        r = requests.get(url, timeout=2)
        assert r.status_code == 200 and r.text == "OK\n"
        actions = run_post2(h.ports[0], [3], 5, 0)
        assert actions.shape == (5, 1, C.ACTION_DIM)
    finally:
        h.stop()


def test_health_not_gated_when_disabled() -> None:
    doc = base_doc()
    doc["server"]["health_requires_warm"] = False
    h = Harness(doc, workers={"w": SINE}, start_workers=False)
    h.start(wait_warm=False)
    try:
        assert requests.get(f"http://127.0.0.1:{h.ports[0]}/healthz", timeout=2).status_code == 200
        # The worker is not up: queries still get (hold) actions immediately.
        nu = load_post2()
        client = nu.WebsocketClientPolicy(host="127.0.0.1", port=h.ports[0])
        client.reset()
        a = client.act(to_torch(batched_obs(0, [3]))).numpy()
        np.testing.assert_array_equal(a[0], hold_action(proprio_at(0)))
        client._ws.close()
    finally:
        h.stop()


# --------------------------------------------------------------------------------------------------------------
# (5) Worker errors and timeouts: hold actions, the connection stays open, planning retries next step
# --------------------------------------------------------------------------------------------------------------
def test_worker_error_gives_hold_actions_and_keeps_connection() -> None:
    worker = ("fake_sine", {"horizon": 32, "fail_task_ids": [7]})
    with Harness(base_doc(), workers={"w": worker}) as h:
        nu = load_post2()
        client = nu.WebsocketClientPolicy(host="127.0.0.1", port=h.ports[0])
        client.reset()
        for t in range(30):
            a = client.act(to_torch(batched_obs(t, [7]))).numpy()
            np.testing.assert_array_equal(a[0], hold_action(proprio_at(t)))
        # Same connection, next rollout on a healthy task: real (non-hold) actions again.
        client.reset()
        a = client.act(to_torch(batched_obs(0, [3]))).numpy()
        assert not np.array_equal(a[0], hold_action(proprio_at(0)))
        s = sessions_of(h, h.ports[0])[0]
        assert s.stats.plans == 1
        client._ws.close()
        failures = h.run(lambda: h.server.engine.schedulers["w"].batches)
        assert failures >= 31  # a plan was attempted on every failing step


def test_worker_timeout_gives_hold_then_recovers() -> None:
    doc = base_doc()
    doc["engine"]["plan_timeout_s"] = 0.3
    worker = ("fake_sine", {"horizon": 32, "slow_task_ids": [8], "slow_ms": 1200})
    with Harness(doc, workers={"w": worker}) as h:
        nu = load_post2()
        client = nu.WebsocketClientPolicy(host="127.0.0.1", port=h.ports[0])
        client.reset()
        t0 = time.monotonic()
        a = client.act(to_torch(batched_obs(0, [8]))).numpy()
        assert time.monotonic() - t0 < 1.0
        np.testing.assert_array_equal(a[0], hold_action(proprio_at(0)))
        time.sleep(1.5)  # let the worker finish the slow request; its late reply must be discarded
        client.reset()
        a = client.act(to_torch(batched_obs(0, [3]))).numpy()
        assert not np.array_equal(a[0], hold_action(proprio_at(0)))
        client._ws.close()


def test_bad_observation_gets_fallback_reply_not_disconnect(sine_server: Harness) -> None:
    nu = load_post2()
    port = sine_server.ports[0]
    client = nu.WebsocketClientPolicy(host="127.0.0.1", port=port)
    client.reset()
    obs = to_torch(batched_obs(0, [3]))
    del obs[C.HEAD_RGB_KEY]  # split_batch rejects a missing head camera
    a = client.act(obs).numpy()
    np.testing.assert_array_equal(a[0], hold_action(proprio_at(0)))
    a = client.act(to_torch(batched_obs(1, [3]))).numpy()  # same connection still serves normally
    assert a.shape == (1, C.ACTION_DIM)
    client._ws.close()


# --------------------------------------------------------------------------------------------------------------
# (6) v3.9.2-style unbatched observations
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [0, 8])
def test_unbatched_v392_observations(sine_server: Harness, chunk: int) -> None:
    nu = load_post2()
    client = nu.WebsocketClientPolicy(host="127.0.0.1", port=sine_server.ports[1], action_chunk_size=chunk)
    client.reset()
    out = []
    for t in range(40):
        obs = env_obs_dict(t, 9)
        if t % 2:
            obs[C.TASK_ID_KEY] = np.int64(9)  # scalar task id
        a = client.act({k: torch.as_tensor(v) for k, v in obs.items()})
        assert tuple(a.shape) == (1, C.ACTION_DIM)
        out.append(a.numpy())
    client._ws.close()
    batched = run_post2(sine_server.ports[1], [9], 40, chunk)
    np.testing.assert_array_equal(np.stack(out), batched)
