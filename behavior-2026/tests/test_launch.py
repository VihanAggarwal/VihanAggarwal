"""Launched-worker lifecycle with configs/fake.yaml: subprocess start, restart after a crash, clean shutdown on
SIGTERM, worker exit when the front server is killed, and a non-zero exit when the default worker cannot start."""

from __future__ import annotations

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
    pytest.importorskip("torch")
    from vendor import load_post2

    doc = fake_doc(free_port())
    doc["workers"]["fake"]["launch"] += ["--backend-arg", "warmup_ms=500"]  # a restart window we can observe
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


def test_restart_budget_is_a_sliding_window() -> None:
    """max_restarts counts only restarts within restart_window_s: a later crash is restarted again, a crash loop
    inside the window gives up (worker failed, /healthz 503)."""
    doc = fake_doc(free_port())
    doc["workers"]["fake"]["max_restarts"] = 1
    doc["workers"]["fake"]["restart_window_s"] = 2.0
    h = Harness(doc).start(timeout=60)
    client = h.server.engine.clients["fake"]
    pids: list[int] = []
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
        os.kill(last, signal.SIGKILL)  # a second crash within 2 s of the last restart: give up
        wait_for(lambda: h.run(lambda: client.failed), 15, "worker should be marked failed")
        assert h.run(lambda: h.server.engine.warm) is False
        assert requests.get(f"http://127.0.0.1:{h.ports[0]}/healthz", timeout=2).status_code == 503
    finally:
        h.stop()
    for pid in pids:
        wait_for(lambda: not pid_alive(pid), 15, f"worker {pid} still running")
