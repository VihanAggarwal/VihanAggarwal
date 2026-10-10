"""Launched-worker lifecycle with configs/fake.yaml: subprocess start, restart after a crash, clean shutdown on
SIGTERM, worker exit when the front server is killed, and a non-zero exit when the default worker cannot start."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import requests
import yaml

from b1k26 import constants as C
from b1k26.client import run_probe
from b1k26.obs import hold_action

from runtime_helpers import Harness, batched_obs, free_port, proprio_at, to_torch

REPO = pathlib.Path(__file__).resolve().parents[1]
FAKE_YAML = REPO / "configs" / "fake.yaml"


def fake_doc(worker_port: int, front_ports: list[int] | None = None) -> dict:
    doc = yaml.safe_load(FAKE_YAML.read_text())
    doc["workers"]["fake"]["port"] = worker_port
    doc["server"]["host"] = "127.0.0.1"
    doc["server"]["ports"] = front_ports or [0]
    return doc


def pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover - non-Linux
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def wait_for(cond, timeout: float, what: str, poll: float = 0.05) -> None:
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(what)
        time.sleep(poll)


def test_fake_yaml_is_valid_and_launches_a_worker() -> None:
    from b1k26.config import load_config

    cfg = load_config(FAKE_YAML)
    w = cfg.workers["fake"]
    assert w.launch is not None and w.launch[0] == sys.executable and "--port" in w.launch
    assert w.launch[w.launch.index("--port") + 1] == str(w.port)
    assert cfg.server.ports == list(range(8000, 8050))


def test_launched_worker_restarts_after_crash_and_stops_with_server() -> None:
    """With engine.restart_wait_s 0, queries get hold actions while the worker restarts (see the next test for the
    default, waiting)."""
    pytest.importorskip("torch")
    from vendor import load_post2

    doc = fake_doc(free_port())
    doc["workers"]["fake"]["launch"] += ["--backend-arg", "warmup_ms=500"]  # a restart window we can observe
    doc["engine"]["restart_wait_s"] = 0
    h = Harness(doc).start(timeout=60)
    client = h.server.engine.clients["fake"]
    try:
        pid1 = client.pid
        assert pid1 is not None and pid_alive(pid1)
        res = run_probe("127.0.0.1", h.ports[0], steps=60, batch=2, chunk=20, res="224", task_ids=[1, 2],
                        health_timeout_s=5)
        assert res.ok, res.violations

        nu = load_post2()
        pol = nu.WebsocketClientPolicy(host="127.0.0.1", port=h.ports[0])
        pol.reset()
        pol.act(to_torch(batched_obs(0, [3])))
        os.kill(pid1, signal.SIGKILL)
        wait_for(lambda: h.run(lambda: client.state) != "ready", 5, "supervisor did not notice the crash")
        pol.reset()  # new rollout: the next query must plan while the worker is down
        a = pol.act(to_torch(batched_obs(0, [3]))).numpy()
        np.testing.assert_array_equal(a[0], hold_action(proprio_at(0)))
        t, holds = 1, 1
        t0 = time.monotonic()
        # Keep stepping through the crash: every reply is an action (holds while the worker restarts).
        while True:
            a = pol.act(to_torch(batched_obs(t, [3]))).numpy()
            assert a.shape == (1, C.ACTION_DIM) and np.all(np.isfinite(a))
            if np.array_equal(a[0], hold_action(proprio_at(t))):
                holds += 1
            t += 1
            if h.run(lambda: client.ready and client.pid != pid1) and t > 60:
                break
            assert time.monotonic() - t0 < 30, "worker was not restarted"
            time.sleep(0.01)
        assert holds >= 1
        assert client.restarts == 1
        pid2 = client.pid
        assert pid2 != pid1 and pid_alive(pid2)
        pol.reset()
        a = pol.act(to_torch(batched_obs(0, [3]))).numpy()
        assert not np.array_equal(a[0], hold_action(proprio_at(0)))
        pol._ws.close()
    finally:
        h.stop()
    wait_for(lambda: not pid_alive(pid2), 15, "launched worker still running after server close")


def test_queries_wait_for_a_restarting_worker_instead_of_holding() -> None:
    """robustness-4: while a crashed worker is relaunched, a query waits for it (engine.restart_wait_s) and gets a
    real plan, instead of an immediate hold that uses up one of the episode's steps."""
    pytest.importorskip("torch")
    from vendor import load_post2

    doc = fake_doc(free_port())
    doc["workers"]["fake"]["launch"] += ["--backend-arg", "warmup_ms=1500"]
    h = Harness(doc).start(timeout=60)
    client = h.server.engine.clients["fake"]
    try:
        pid1 = client.pid
        nu = load_post2()
        pol = nu.WebsocketClientPolicy(host="127.0.0.1", port=h.ports[0])
        pol.reset()
        pol.act(to_torch(batched_obs(0, [3])))
        os.kill(pid1, signal.SIGKILL)
        wait_for(lambda: h.run(lambda: client.state) != "ready", 5, "supervisor did not notice the crash")
        pol.reset()  # new rollout: its first query must plan while the worker is down
        t0 = time.monotonic()
        a = pol.act(to_torch(batched_obs(0, [3]))).numpy()
        waited = time.monotonic() - t0
        assert waited > 0.5, waited  # it waited for the restart (warmup alone is 1.5 s) ...
        assert not np.array_equal(a[0], hold_action(proprio_at(0)))  # ... and got a real plan, not a hold
        assert h.run(lambda: client.ready and client.pid != pid1 and client.restarts == 1)
        session = h.run(lambda: h.server.groups[h.ports[0]][0].sessions[0])
        assert session.stats.hold_steps == 0 and session.stats.plan_failures == 0
        assert session.stats.restart_wait_ms > 500
        pol._ws.close()
    finally:
        h.stop()


def _start_cli(tmp_path: pathlib.Path, doc: dict) -> tuple[subprocess.Popen, int]:
    front = free_port()
    doc["server"]["ports"] = [front]
    cfg = tmp_path / "serve.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    proc = subprocess.Popen([sys.executable, "-m", "b1k26.server", "--config", str(cfg), "--log-level", "WARNING"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, front


def _wait_healthy(proc: subprocess.Popen, port: int, timeout: float = 60) -> None:
    def ok() -> bool:
        assert proc.poll() is None, f"server exited early with {proc.returncode}"
        try:
            return requests.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200
        except requests.RequestException:
            return False
    wait_for(ok, timeout, "server not healthy", poll=0.1)


def test_serve_cli_sigterm_stops_launched_worker(tmp_path: pathlib.Path) -> None:
    proc, port = _start_cli(tmp_path, fake_doc(free_port()))
    try:
        _wait_healthy(proc, port)
        res = run_probe("127.0.0.1", port, steps=40, chunk=20, res="full", health_timeout_s=5)
        assert res.ok, res.violations
        status = requests.get(f"http://127.0.0.1:{port}/status", timeout=2).json()
        wpid = status["engine"]["workers"]["fake"]["pid"]
        assert pid_alive(wpid)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
        wait_for(lambda: not pid_alive(wpid), 15, "worker survived the front server's SIGTERM")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)


def test_worker_exits_when_front_server_is_killed(tmp_path: pathlib.Path) -> None:
    proc, port = _start_cli(tmp_path, fake_doc(free_port()))
    try:
        _wait_healthy(proc, port)
        wpid = requests.get(f"http://127.0.0.1:{port}/status", timeout=2).json()["engine"]["workers"]["fake"]["pid"]
        proc.kill()
        proc.wait(10)
        wait_for(lambda: not pid_alive(wpid), 15, "orphaned worker kept running after SIGKILL of the front server")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_serve_cli_exits_nonzero_when_default_worker_cannot_start(tmp_path: pathlib.Path) -> None:
    doc = fake_doc(free_port())
    w = doc["workers"]["fake"]
    w["launch"] = ["{python}", "-m", "b1k26.worker", "--backend", "no_such_backend", "--port", "{port}"]
    w["max_restarts"] = 0
    proc, port = _start_cli(tmp_path, doc)
    try:
        rc = proc.wait(timeout=60)
        assert rc == 1
    finally:
        if proc.poll() is None:
            proc.kill()


def test_serve_cli_rejects_bad_config(tmp_path: pathlib.Path) -> None:
    doc = fake_doc(free_port())
    doc["profiles"]["fake"]["typo_key"] = 1
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    out = subprocess.run([sys.executable, "-m", "b1k26.server", "--config", str(cfg)], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 2
    assert "typo_key" in out.stderr


def test_status_endpoint_is_json() -> None:
    h = Harness(fake_doc(free_port())).start(timeout=60)
    try:
        st = requests.get(f"http://127.0.0.1:{h.ports[0]}/status", timeout=2).json()
        assert st["healthy"] is True and st["engine"]["workers"]["fake"]["state"] == "ready"
        json.dumps(st)
    finally:
        h.stop()


def test_restart_budget_is_a_sliding_window_then_backoff() -> None:
    """max_restarts counts only restarts within restart_window_s: a later crash is restarted at once again. A crash
    loop inside the window is not given up (robustness-3): the worker waits restart_backoff_s (state backoff,
    /healthz 503) and is then relaunched, and /healthz returns to 200."""
    doc = fake_doc(free_port())
    doc["workers"]["fake"]["max_restarts"] = 1
    doc["workers"]["fake"]["restart_window_s"] = 2.0
    doc["workers"]["fake"]["restart_backoff_s"] = 1.5
    h = Harness(doc).start(timeout=60)
    client = h.server.engine.clients["fake"]
    pids: list[int] = []

    def healthz() -> int:
        return requests.get(f"http://127.0.0.1:{h.ports[0]}/healthz", timeout=2).status_code

    try:
        def crash_and_wait_restart() -> None:
            old = h.run(lambda: client.pid)
            pids.append(old)
            os.kill(old, signal.SIGKILL)
            wait_for(lambda: h.run(lambda: client.ready and client.pid != old), 30, "worker was not restarted")

        crash_and_wait_restart()
        time.sleep(2.3)  # the first restart leaves the 2 s window
        crash_and_wait_restart()
        assert h.run(lambda: client.restarts) == 2
        last = h.run(lambda: client.pid)
        pids.append(last)
        os.kill(last, signal.SIGKILL)  # a second crash within 2 s of the last restart: back off
        wait_for(lambda: h.run(lambda: client.state) == "backoff", 15, "worker should be in backoff")
        assert h.run(lambda: client.failed) and h.run(lambda: h.server.engine.warm) is False
        assert healthz() == 503
        wait_for(lambda: h.run(lambda: client.ready and client.pid != last), 30, "worker not relaunched after backoff")
        assert h.run(lambda: h.server.engine.warm) is True and healthz() == 200
        assert h.run(lambda: client.restarts) == 3
        pids.append(h.run(lambda: client.pid))
    finally:
        h.stop()
    for pid in pids:
        wait_for(lambda: not pid_alive(pid), 15, f"worker {pid} still running")


def test_slow_loading_default_worker_is_not_killed_at_startup_timeout(tmp_path: pathlib.Path) -> None:
    """protocol-3: a worker that still answers /healthz 503 (loading) at startup_timeout_s keeps loading (up to 3x)
    instead of being killed, and the server becomes healthy."""
    doc = fake_doc(free_port())
    doc["workers"]["fake"]["launch"] += ["--backend-arg", "warmup_ms=3500"]
    doc["workers"]["fake"]["startup_timeout_s"] = 2.0
    proc, port = _start_cli(tmp_path, doc)
    try:
        _wait_healthy(proc, port, timeout=30)
        st = requests.get(f"http://127.0.0.1:{port}/status", timeout=2).json()["engine"]["workers"]["fake"]
        assert st["state"] == "ready" and st["restarts"] == 0
    finally:
        proc.terminate()
        proc.wait(30)


def test_default_worker_with_transient_load_failures_is_retried_until_ready(tmp_path: pathlib.Path) -> None:
    """protocol-3: a default worker that keeps failing to load (exit status 3) is relaunched with backoff, not given
    up: the front server stays up with /healthz 503 and turns healthy once the worker loads."""
    flag = tmp_path / "fail_load"
    flag.write_text("")
    doc = fake_doc(free_port())
    doc["workers"]["fake"]["launch"] += ["--backend-arg", f"fail_load_if_exists={flag}"]
    doc["workers"]["fake"]["max_restarts"] = 1
    doc["workers"]["fake"]["restart_backoff_s"] = 0.5
    doc["workers"]["fake"]["restart_backoff_max_s"] = 1.0
    proc, port = _start_cli(tmp_path, doc)
    try:
        time.sleep(6.0)  # several failed loads and backoffs
        assert proc.poll() is None, "front server exited although the failure is transient"
        assert requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).status_code == 503
        st = requests.get(f"http://127.0.0.1:{port}/status", timeout=2).json()["engine"]["workers"]["fake"]
        assert st["restarts"] >= 2 and st["state"] in ("backoff", "starting", "restarting"), st
        flag.unlink()
        _wait_healthy(proc, port, timeout=30)
    finally:
        proc.terminate()
        proc.wait(30)


def test_two_servers_from_one_config_on_one_host(tmp_path: pathlib.Path) -> None:
    """protocol-1 / robustness-6: two front servers started from the same config (same launched worker port) on
    different front ports both turn healthy: the second one's worker moves to a free loopback port, and each front
    server talks to its own worker."""
    worker_port = free_port()
    procs = []
    try:
        for i in range(2):
            d = tmp_path / f"s{i}"
            d.mkdir()
            procs.append(_start_cli(d, fake_doc(worker_port)))
            if i == 0:
                _wait_healthy(*procs[0])
        for proc, port in procs:
            _wait_healthy(proc, port, timeout=60)
        workers = [requests.get(f"http://127.0.0.1:{port}/status", timeout=2).json()["engine"]["workers"]["fake"]
                   for _, port in procs]
        assert workers[0]["endpoint"] == f"ws://127.0.0.1:{worker_port}"
        assert workers[1]["endpoint"] != workers[0]["endpoint"] and workers[0]["pid"] != workers[1]["pid"]
        for _, port in procs:
            res = run_probe("127.0.0.1", port, steps=40, chunk=20, res="224", health_timeout_s=5)
            assert res.ok, res.violations
    finally:
        for proc, _ in procs:
            proc.terminate()
        for proc, _ in procs:
            proc.wait(30)


def test_routed_worker_gets_one_start_attempt_then_background_retries(tmp_path: pathlib.Path) -> None:
    """protocol-4: /healthz does not wait for a routed (non-default) worker's relaunch loop: it gets one start
    attempt; while it is retried in the background (backoff) its tasks fall back to the default, and once it is up
    new rollouts are routed to it again."""
    flag = tmp_path / "fail_load"
    flag.write_text("")
    doc = fake_doc(free_port())
    routed = copy.deepcopy(doc["workers"]["fake"])
    routed["port"] = free_port()
    routed["launch"] = routed["launch"] + ["--backend-arg", f"fail_load_if_exists={flag}"]
    routed["restart_backoff_s"] = 1.0
    routed["restart_backoff_max_s"] = 1.0
    doc["workers"]["routed"] = routed
    doc["profiles"]["routed"] = dict(doc["profiles"]["fake"], worker="routed")
    doc["routing"]["per_task"] = {"turning_on_radio": "routed"}
    h = Harness(doc).start(timeout=30)
    try:
        client = h.server.engine.clients["routed"]
        assert h.run(lambda: client.restarts) == 0 and h.run(lambda: client.failed)  # one attempt, then backoff
        res = run_probe("127.0.0.1", h.ports[0], steps=20, chunk=0, res="224", task_ids=[0], health_timeout_s=5)
        assert res.ok, res.violations
        s = h.run(lambda: h.server.groups[h.ports[0]][0].sessions[0])
        assert s.profile_name == "fake" and s.fallback_note and s.stats.hold_steps == 0
        flag.unlink()
        wait_for(lambda: h.run(lambda: client.ready), 30, "routed worker not brought back in the background")
        res = run_probe("127.0.0.1", h.ports[0], steps=20, chunk=0, res="224", task_ids=[0], health_timeout_s=5)
        assert res.ok, res.violations
        assert h.run(lambda: h.server.groups[h.ports[0]][0].sessions[0].profile_name) == "routed"
    finally:
        h.stop()
