"""b1k26-probe (b1k26.client): passes against b1k26-serve and flags protocol violations of a broken server."""

from __future__ import annotations

import http
import threading

import numpy as np
import pytest

from b1k26 import client as probe
from b1k26 import constants as C
from b1k26.protocol import packb, unpackb

from runtime_helpers import Harness, base_doc

SINE = ("fake_sine", {"horizon": 32})


@pytest.fixture(scope="module")
def server():
    h = Harness(base_doc(), workers={"w": SINE}).start()
    yield h
    h.stop()


@pytest.mark.parametrize("res,batch,chunk", [("full", 1, 20), ("224", 3, 0), ("224", 2, 8)])
def test_probe_passes(server: Harness, res: str, batch: int, chunk: int) -> None:
    r = probe.run_probe("127.0.0.1", server.ports[0], steps=48, batch=batch, chunk=chunk, res=res,
                        task_ids=[0, 50, 99][:batch], health_timeout_s=5, record_actions=True)
    assert r.ok, r.violations
    assert r.steps == 48 and len(r.actions) == 48
    assert r.requests == (48 if chunk <= 1 else -(-48 // chunk))
    assert r.metadata["server"] == "b1k26"
    s = r.summary()
    assert s["ok"] and s["latency_ms"]["p50"] > 0


def test_probe_unbatched_and_cli(server: Harness, capsys) -> None:
    r = probe.run_probe("127.0.0.1", server.ports[0], steps=10, unbatched=True, res="224", health_timeout_s=5)
    assert r.ok, r.violations
    rc = probe.main(["--port", str(server.ports[0]), "--steps", "40", "--chunk", "20", "--res", "224",
                     "--task-id", "turning_on_radio", "--json"])
    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out


def test_probe_cli_unreachable_server_exits_2() -> None:
    from runtime_helpers import free_port

    assert probe.main(["--port", str(free_port()), "--health-timeout", "0.2", "--steps", "1"]) == 2


# --------------------------------------------------------------------------------------------------------------
# A deliberately broken server
# --------------------------------------------------------------------------------------------------------------
class BadServer:
    def __init__(self, mode: str):
        from websockets.sync.server import serve

        self.mode = mode
        self.server = serve(self.handler, "127.0.0.1", 0, compression=None, max_size=None,
                            process_request=self.health)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def handler(self, ws) -> None:
        if self.mode != "no_metadata":
            ws.send(packb({"server": "bad"}))
        for raw in ws:
            msg = unpackb(raw)
            if "reset" in msg:
                if self.mode == "reply_to_reset":
                    ws.send(packb({"ok": True}))
                continue
            n = np.asarray(msg[C.PROPRIO_KEY]).reshape(-1, C.PROPRIO_DIM).shape[0]
            k = int(msg.get(C.ACTION_CHUNK_REQUEST_KEY, 0))
            act = np.zeros((n, 23), np.float32)
            act[:, 3:7] = 0.5
            chunk = np.repeat(act[:, None], max(k, 1), axis=1)
            if self.mode == "text":
                ws.send("Traceback: boom")
                continue
            if self.mode == "zero":
                act[:] = 0
                chunk[:] = 0
            if self.mode == "chunk_mismatch":
                chunk = chunk.copy()
                chunk[:, 0, 5] += 1e-6
            if self.mode == "chunk_shape":
                chunk = chunk[:, :-1]
            if self.mode == "nan":
                act[0, 9] = np.nan
                chunk[0, 0, 9] = np.nan
            if self.mode == "no_action":
                ws.send(packb({"actions": act}))
                continue
            reply = {"action": act}
            if k > 1 and self.mode != "no_chunk":
                reply["action_chunk"] = chunk
            ws.send(packb(reply))

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(5)


@pytest.mark.parametrize("mode,chunk,needle", [
    ("ok", 8, None),
    ("reply_to_reset", 0, "replied to reset"),
    ("text", 0, "text frame"),
    ("zero", 0, "zero torso"),
    ("chunk_mismatch", 8, "exactly equal"),
    ("chunk_shape", 8, "action_chunk shape"),
    ("no_chunk", 8, "no 'action_chunk'"),
    ("nan", 0, "non-finite"),
    ("no_action", 0, "no 'action'"),
    ("no_metadata", 0, "metadata"),
])
def test_probe_detects_violations(mode: str, chunk: int, needle: str | None) -> None:
    bad = BadServer(mode)
    try:
        r = probe.run_probe("127.0.0.1", bad.port, steps=16, chunk=chunk, res="224", health_timeout_s=5,
                            reset_check_s=0.2, metadata_timeout_s=0.5)
        if needle is None:
            assert r.ok, r.violations
        else:
            assert not r.ok and needle in r.violations[0], r.violations
    finally:
        bad.close()
