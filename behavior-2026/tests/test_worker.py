"""Worker protocol (b1k26.worker) and its async client (engine.WorkerClient)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import requests

from b1k26 import constants as C
from b1k26.backends.fake import SineBackend
from b1k26.config import WorkerConfig
from b1k26.engine import WorkerClient, WorkerError, WorkerTimeout, http_status, no_proxy_kwargs
from b1k26.protocol import packb, unpackb
from b1k26.worker import WorkerServer, parse_backend_args

from runtime_helpers import free_port, proprio_at


class LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def call(self, coro, timeout: float = 20.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


def item(task_id: int = 3, size: int = 32, **extra) -> dict:
    img = np.full((size, size, 3), 7, np.uint8)
    return {"task_id": task_id, "prompt": "do it", "proprio": proprio_at(0),
            "images": {"head": img, "left_wrist": img, "right_wrist": img}, "stage": None,
            "initial_actions": None, **extra}


@pytest.fixture()
def worker():
    lt = LoopThread()
    gate = threading.Event()

    def factory():
        gate.wait(10)
        return SineBackend(horizon=16, num_stages=3, fail_task_ids=[9], slow_task_ids=[8], slow_ms=600)

    ws = WorkerServer(factory, host="127.0.0.1", port=0, name="sine")
    lt.call(ws.start())
    load = asyncio.run_coroutine_threadsafe(ws.load(), lt.loop)
    yield lt, ws, gate, load
    gate.set()
    try:
        load.result(10)
    except Exception:
        pass
    lt.call(ws.stop())
    lt.stop()


def sync_request(port: int, msg, timeout: float = 10.0):
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, **no_proxy_kwargs(connect)) as c:
        c.send(msg if isinstance(msg, (bytes, str)) else packb(msg))
        raw = c.recv(timeout=timeout)
        assert isinstance(raw, bytes), "worker must never send text frames"
        return unpackb(raw)


def test_health_503_while_loading_then_200(worker) -> None:
    lt, ws, gate, load = worker
    url = f"http://127.0.0.1:{ws.bound_port}/healthz"
    assert requests.get(url, timeout=2).status_code == 503
    assert "error" in sync_request(ws.bound_port, {"op": "info"})  # not loaded yet, but answered
    gate.set()
    load.result(10)
    assert requests.get(url, timeout=2).status_code == 200
    assert lt.call(http_status("127.0.0.1", ws.bound_port)) == 200


def test_protocol_ops_and_errors(worker) -> None:
    lt, ws, gate, load = worker
    gate.set()
    load.result(10)
    port = ws.bound_port
    info = sync_request(port, {"op": "info", "id": 5})
    assert info["id"] == 5 and info["warm"] is True and info["action_horizon"] == 16
    assert info["num_stages"] == [3] * C.NUM_TASKS and info["supports_stage"] is True
    assert sync_request(port, {"op": "warmup"})["ok"] is True

    rep = sync_request(port, {"op": "infer", "items": [item(3), item(4, stage=1)]})
    assert len(rep["chunks"]) == 2 and rep["ms"] >= 0
    a = rep["chunks"][0]["actions"]
    assert a.dtype == np.float32 and a.shape == (16, 23)
    assert rep["chunks"][1]["subtask_logits"].shape == (3,)
    assert int(np.argmax(rep["chunks"][1]["subtask_logits"])) == 2

    for bad in (
        {"op": "infer", "items": []},
        {"op": "infer", "items": [item(3) | {"proprio": np.zeros(5, np.float32)}]},
        {"op": "infer", "items": [item(3) | {"task_id": 100}]},
        {"op": "infer", "items": [item(3) | {"images": {"head": np.zeros((4, 4), np.uint8)}}]},
        {"op": "infer", "items": [item(3) | {"initial_actions": np.zeros((2, 5), np.float32)}]},
        {"op": "infer", "items": [item(9)]},  # backend raises
        {"op": "bogus"},
        [1, 2, 3],
    ):
        rep = sync_request(port, bad)
        assert isinstance(rep, dict) and isinstance(rep.get("error"), str), bad
    assert "error" in sync_request(port, b"\xc1not msgpack")
    assert "text frames" in sync_request(port, "hello")["error"]
    # The connection survives errors: several requests on one connection.
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, **no_proxy_kwargs(connect)) as c:
        for msg in ({"op": "bogus"}, {"op": "info"}, {"op": "infer", "items": [item(9)]}, {"op": "info"}):
            c.send(packb(msg))
            assert isinstance(unpackb(c.recv(timeout=5)), dict)


def test_parse_backend_args() -> None:
    assert parse_backend_args(["a=1", "b=2.5", "c=true", "d=null", "e=[1,2]", "f=hello", "g={\"x\": 1}", "h="]) == {
        "a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2], "f": "hello", "g": {"x": 1}, "h": ""}
    with pytest.raises(ValueError):
        parse_backend_args(["novalue"])
    with pytest.raises(ValueError):
        parse_backend_args(["bad key=1"])


def test_worker_client_roundtrip_timeout_and_stale_reply(worker) -> None:
    lt, ws, gate, load = worker
    gate.set()
    load.result(10)
    cfg = WorkerConfig(name="sine", endpoint=f"ws://127.0.0.1:{ws.bound_port}", startup_timeout_s=5)
    client = WorkerClient(cfg)
    lt.call(client.start())
    assert client.ready and client.info["action_horizon"] == 16 and client.info["supports_inpaint"] is True
    chunks = lt.call(client.infer([item(3)], timeout=5))
    assert chunks[0]["actions"].shape == (16, 23)
    with pytest.raises(WorkerError, match="injected failure"):
        lt.call(client.infer([item(9)], timeout=5))
    with pytest.raises(WorkerTimeout):
        lt.call(client.infer([item(8)], timeout=0.1))
    time.sleep(0.7)  # the late reply to the timed-out request arrives and must be skipped by id
    chunks = lt.call(client.infer([item(4)], timeout=5))
    ref = SineBackend(horizon=16, num_stages=3)
    from b1k26.worker import decode_item

    np.testing.assert_array_equal(chunks[0]["actions"], ref.infer([decode_item(item(4), 0)])[0].actions)
    lt.call(client.close())
    assert client.state == "stopped"


def test_worker_client_start_fails_when_unreachable() -> None:
    lt = LoopThread()
    try:
        cfg = WorkerConfig(name="x", endpoint=f"ws://127.0.0.1:{free_port()}", startup_timeout_s=0.5)
        client = WorkerClient(cfg)
        t0 = time.monotonic()
        with pytest.raises(WorkerError, match="not healthy"):
            lt.call(client.start())
        assert time.monotonic() - t0 < 5 and client.failed
        with pytest.raises(WorkerError):
            lt.call(client.infer([item(3)], timeout=1))
    finally:
        lt.stop()


def test_worker_cli_load_failure_exit_status(tmp_path) -> None:
    """Exit status 2 for configuration errors a relaunch cannot fix (the front server gives up), 3 for other load
    failures (the front server relaunches)."""
    def run(*args: str) -> int:
        return subprocess.run([sys.executable, "-m", "b1k26.worker", "--port", str(free_port()), *args],
                              capture_output=True, timeout=60).returncode

    assert run("--backend", "fake_replay", "--backend-arg", "path=/nonexistent/actions.npy") == 2  # missing file
    assert run("--backend", "no_such_backend") == 2
    assert run("--backend", "fake_sine", "--backend-kwargs", "{not json") == 2
    assert run("--backend", "fake_sine", "--backend-arg", "no_such_kwarg=1") == 2  # TypeError in the constructor
    flag = tmp_path / "fail"
    flag.write_text("")
    assert run("--backend", "fake_sine", "--backend-arg", f"fail_load_if_exists={flag}") == 3  # transient


def test_worker_cli_replay_backend(tmp_path) -> None:
    path = tmp_path / "acts.npy"
    data = np.tile(np.arange(23, dtype=np.float32) * 0.01, (50, 1))
    data[:, 0] = np.linspace(-0.5, 0.5, 50)
    np.save(path, data)
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "b1k26.worker", "--backend", "fake_replay", "--port", str(port),
                             "--checkpoint", str(path), "--backend-kwargs", '{"horizon": 20}', "--log-level", "WARNING"])
    try:
        t0 = time.monotonic()
        while True:
            try:
                if requests.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            assert proc.poll() is None and time.monotonic() - t0 < 30
            time.sleep(0.05)
        rep = sync_request(port, {"op": "infer", "items": [item(3)]})
        np.testing.assert_array_equal(rep["chunks"][0]["actions"], data[:20])
        rep = sync_request(port, {"op": "infer", "items": [item(3)]})
        np.testing.assert_array_equal(rep["chunks"][0]["actions"], data[20:40])
    finally:
        proc.terminate()
        assert proc.wait(10) == 0


def test_client_refuses_another_servers_worker(worker, monkeypatch) -> None:
    """protocol-1: when two front servers race for one worker port, the loser's worker cannot bind but the winner's
    answers /healthz. The launch token (echoed in info) keeps the loser from attaching to the winner's worker."""
    from b1k26.engine import _ProcessExited
    from b1k26.worker import LAUNCH_TOKEN_ENV

    lt, ws, gate, load = worker
    gate.set()
    load.result(10)
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, "theirs")
    client = WorkerClient(WorkerConfig(name="x", endpoint=f"ws://127.0.0.1:{ws.bound_port}", startup_timeout_s=5))
    client._launch_token = "ours"  # as if we had launched a worker on this port
    with pytest.raises(_ProcessExited, match="another server's worker"):
        lt.call(client._bring_up())
    assert not client.ready
    client._launch_token = "theirs"
    lt.call(client._bring_up())
    assert client.ready
    lt.call(client.close())
