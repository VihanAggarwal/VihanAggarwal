"""Front server details below the evaluator clients: raw frames, bad input, slot-group bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from b1k26 import constants as C
from b1k26.engine import no_proxy_kwargs
from b1k26.obs import hold_action
from b1k26.protocol import Packer, packb, unpackb
from b1k26.server import MAX_FREE_GROUPS_PER_PORT, _parse_chunk_k, _ports_str

from runtime_helpers import Harness, base_doc, batched_obs, proprio_at

SINE = ("fake_sine", {"horizon": 32})


@pytest.fixture(scope="module")
def h():
    harness = Harness(base_doc(), workers={"w": SINE}, n_ports=2).start()
    yield harness
    harness.stop()


def connect(port: int):
    from websockets.sync.client import connect as ws_connect

    return ws_connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, **no_proxy_kwargs(ws_connect))


def recv_map(ws) -> dict:
    raw = ws.recv(timeout=10)
    assert isinstance(raw, bytes), "server must never send text frames"
    out = unpackb(raw)
    assert isinstance(out, dict)
    return out


def test_metadata_and_reset_without_reply(h: Harness) -> None:
    with connect(h.ports[0]) as ws:
        meta = recv_map(ws)
        assert meta["server"] == "b1k26" and meta["port"] == h.ports[0] and meta["action_dim"] == 23
        ws.send(packb({"reset": True}))
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.3)
        ws.send(Packer().pack(batched_obs(0, [3])))
        rep = recv_map(ws)
        assert rep["action"].shape == (1, 23) and rep["action"].dtype == np.float32
        assert "action_chunk" not in rep and "infer_ms" in rep["server_timing"]


def test_bad_frames_get_binary_errors_and_connection_survives(h: Harness) -> None:
    with connect(h.ports[0]) as ws:
        recv_map(ws)
        ws.send(packb({"reset": True}))  # forget the last actions of an inherited slot group
        ws.send("hello")
        assert "text frames" in recv_map(ws)["error"]
        ws.send(b"\xc1\xc1")
        assert "decode" in recv_map(ws)["error"]
        ws.send(packb([1, 2]))
        assert "map" in recv_map(ws)["error"]
        ws.send(packb({"something": 1}))  # no proprio: nothing to hold from, no previous action
        assert "error" in recv_map(ws)
        ws.send(packb(batched_obs(0, [3])))
        good = recv_map(ws)["action"]
        obs = batched_obs(1, [3])
        obs[C.PROPRIO_KEY] = obs[C.PROPRIO_KEY][:, :10]  # malformed proprio: no hold can be computed
        ws.send(packb(obs))
        rep = recv_map(ws)
        np.testing.assert_array_equal(rep["action"], good)  # falls back to the slot's last action
        obs = batched_obs(1, [3])
        del obs[C.HEAD_RGB_KEY]  # rejected by split_batch, but the proprio is usable: hold the pose
        obs[C.ACTION_CHUNK_REQUEST_KEY] = 4
        ws.send(packb(obs))
        rep = recv_map(ws)
        np.testing.assert_array_equal(rep["action"][0], hold_action(proprio_at(1)))
        assert rep["action_chunk"].shape == (1, 4, 23)
        np.testing.assert_array_equal(rep["action_chunk"][:, 0], rep["action"])
        ws.send(packb(batched_obs(1, [3])))
        assert recv_map(ws)["action"].shape == (1, 23)


def test_missing_task_id_uses_last_known_task(h: Harness) -> None:
    with connect(h.ports[0]) as ws:
        recv_map(ws)
        ws.send(packb({"reset": True}))
        ws.send(packb(batched_obs(0, [42])))
        recv_map(ws)
        obs = batched_obs(1, [42])
        del obs[C.TASK_ID_KEY]
        ws.send(packb(obs))
        rep = recv_map(ws)
        assert not np.array_equal(rep["action"][0], hold_action(proprio_at(1)))
    sessions = h.run(lambda: [s for g in h.server.groups[h.ports[0]] for s in g.sessions.values() if s.active])
    assert any(s.task_id == 42 and s.step == 2 for s in sessions)


@pytest.mark.parametrize("value,expected", [(None, 0), (0, 0), (1, 0), (2, 2), (20, 20), (np.int64(8), 8),
                                            ("x", 0), (10**6, 0), (-3, 0)])
def test_parse_chunk_k(value, expected) -> None:
    assert _parse_chunk_k(value) == expected


def test_chunk_k_one_and_huge_get_no_chunk(h: Harness) -> None:
    with connect(h.ports[1]) as ws:
        recv_map(ws)
        for k in (1, 10**6):
            obs = batched_obs(0, [3])
            obs[C.ACTION_CHUNK_REQUEST_KEY] = k
            ws.send(packb(obs))
            rep = recv_map(ws)
            assert "action_chunk" not in rep and rep["action"].shape == (1, 23)


def test_half_open_connection_is_taken_over_by_resend(h: Harness) -> None:
    port = h.ports[1]
    a = connect(port)
    recv_map(a)
    a.send(packb({"reset": True}))
    obs = batched_obs(0, [5])
    a.send(packb(obs))
    first = recv_map(a)["action"]
    # Connection A still looks alive to the server; the client resends the same observation on B.
    with connect(port) as b:
        recv_map(b)
        b.send(packb(obs))
        again = recv_map(b)["action"]
        np.testing.assert_array_equal(again, first)
    assert h.run(lambda: h.server.counters["replays"]) >= 1
    a.close()


def test_idle_groups_are_bounded(h: Harness) -> None:
    import concurrent.futures

    port = h.ports[1]
    n = MAX_FREE_GROUPS_PER_PORT + 4
    conns = [connect(port) for _ in range(n)]  # concurrent: each gets its own group
    try:
        for c in conns:
            recv_map(c)
            c.send(packb({"reset": True}))
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda c: (c.send(packb(batched_obs(0, [1]))), recv_map(c)), conns))
    finally:
        for c in conns:
            c.close()
    with connect(port) as c:  # the next connection triggers eviction of the oldest idle groups
        recv_map(c)
        c.send(packb({"reset": True}))
        c.send(packb(batched_obs(0, [1])))
        recv_map(c)
    groups = h.run(lambda: list(h.server.groups[port]))
    assert len(groups) <= MAX_FREE_GROUPS_PER_PORT + 1


def test_ports_str() -> None:
    assert _ports_str(list(range(8000, 8050))) == "8000-8049"
    assert _ports_str([1, 3]) == "1,3"
