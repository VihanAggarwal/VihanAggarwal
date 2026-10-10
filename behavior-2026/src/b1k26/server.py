"""Front server: the evaluator-facing websocket policy server, on many ports in one event loop.

    b1k26-serve --config configs/fake.yaml [--ports 8000-8049] [--host 0.0.0.0] [--log-level INFO]

Protocol (OmniGibson eval/utils/network_utils.py, v3.9.3-post1/post2 and the 2026/eval branch):
- ``GET /healthz``: 200 "OK" once the engine is warm (503 before, unless ``server.health_requires_warm: false``):
  every routed worker has finished its first start attempt and the default profile's worker is ready. It turns 503
  again while the default worker is down (backoff) and back to 200 when it is relaunched.
  ``GET /status`` returns a JSON summary. Every other request is a websocket upgrade.
- On connect the server first sends a msgpack map (metadata); the client blocks on it.
- ``{"reset": True}`` resets every session of the connection's slot group and gets no reply.
- An observation (batched (N, ...) or unbatched v3.9.2 style) gets ``{"action": (B, 23) float32,
  "action_chunk"?: (B, K, 23), "server_timing": {...}}``; ``action_chunk`` is sent when the request carries
  ``"__action_chunk_size__": K > 1``, and ``action_chunk[:, 0]`` equals ``action`` bit for bit.
- Failures never close the connection and never produce text frames: the reply carries hold actions.

Slot groups: the sessions of a rollout belong to a slot group of the port, not to the connection. A new
connection on a port takes over the free group (no live connection) whose last observation fingerprints match its
first observation, else the most recently released free group, else a fresh group. A second concurrent connection
on a port therefore gets its own fresh group. If the first observation on a new connection equals the group's last
observation (2026/eval resends the same observation after a reconnect), the cached reply is re-sent instead of
stepping again, so a reconnect never advances the rollout twice.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import signal
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

from b1k26 import __version__
from b1k26 import constants as C
from b1k26.config import Config, ConfigError, load_config, parse_ports
from b1k26.engine import PolicyEngine, hold_chunk
from b1k26.obs import EnvObs, split_batch
from b1k26.protocol import packb, unpackb
from b1k26.session import RolloutSession

logger = logging.getLogger("b1k26.server")

MAX_CHUNK_K = 4096  # larger chunk requests are answered without action_chunk (the client falls back to 1/step)
MAX_FREE_GROUPS_PER_PORT = 8


class SlotGroup:
    """The sessions served through one connection at a time on one port (batch index -> session)."""

    def __init__(self, port: int, index: int):
        self.port = port
        self.index = index
        self.sessions: dict[int, RolloutSession] = {}
        self.conn: Any = None
        self.lock = asyncio.Lock()
        # (fingerprints, chunk_k, packed reply) of the last answered observation: reconnect replay
        self.cache: tuple[tuple[bytes, ...], int, bytes] | None = None
        self.released_at = time.monotonic()
        self.connections = 0
        self.last_task_id: int | None = None

    def session(self, b: int) -> RolloutSession:
        s = self.sessions.get(b)
        if s is None:
            s = self.sessions[b] = RolloutSession((self.port, self.index, b))
        return s

    def is_free(self) -> bool:
        if self.conn is None:
            return True
        state = getattr(self.conn, "state", None)
        return getattr(state, "name", "") in ("CLOSING", "CLOSED")

    def log_and_reset(self, reason: str) -> None:
        for b in sorted(self.sessions):
            s = self.sessions[b]
            if s.active:
                log_rollout_end(s, reason)
            s.reset()
        self.cache = None


def log_rollout_end(s: RolloutSession, reason: str) -> None:
    st = s.stats.summary()
    logger.info(
        "rollout end (%s) port=%s group=%s slot=%s task=%s profile=%s steps=%d queries=%d plans=%d "
        "plan_failures=%d hold_steps=%d corrections=%d compressed=%d replays=%d restart_wait=%.0fs "
        "query_ms[mean=%.1f p50=%.1f max=%.1f] plan_ms[mean=%.1f max=%.1f] wall=%.0fs", reason, s.key[0], s.key[1],
        s.key[2], s.task_id, s.profile_name, s.step, st["queries"], st["plans"], st["plan_failures"], st["hold_steps"],
        st["corrections"], st["compressed"], st["replays"], st["restart_wait_s"], st["query_ms_mean"],
        st["query_ms_p50"], st["query_ms_max"], st["plan_ms_mean"], st["plan_ms_max"], st["wall_s"],
    )


def _parse_chunk_k(value: Any) -> int:
    """__action_chunk_size__ -> K (0 = no chunk requested)."""
    if value is None:
        return 0
    try:
        k = int(np.asarray(value).reshape(-1)[0])
    except Exception:
        logger.warning("ignoring invalid %s=%r", C.ACTION_CHUNK_REQUEST_KEY, value)
        return 0
    if k <= 1:
        return 0
    if k > MAX_CHUNK_K:
        logger.warning("%s=%d exceeds %d; replying without action_chunk", C.ACTION_CHUNK_REQUEST_KEY, k, MAX_CHUNK_K)
        return 0
    return k


class PolicyServer:
    """Listens on all configured ports; owns slot groups and forwards observations to the engine."""

    def __init__(self, config: Config, engine: PolicyEngine | None = None, host: str | None = None,
                 ports: list[int] | None = None):
        self.config = config
        self.engine = engine if engine is not None else PolicyEngine(config)
        self.host = host if host is not None else config.server.host
        self.ports = list(ports) if ports is not None else list(config.server.ports)
        self.bound_ports: list[int] = []
        self.groups: dict[int, list[SlotGroup]] = {}
        self._group_counter: dict[int, int] = {}
        self._servers: list[Any] = []
        self._engine_task: asyncio.Task | None = None
        self.engine_failed = asyncio.Event()
        self.counters = {"connections": 0, "queries": 0, "resets": 0, "replays": 0, "bad_requests": 0}
        self.started_at = time.monotonic()
        # Test hook: called as hook(stage, port, group) with stage "recv" (observation decoded, before stepping)
        # or "send" (reply ready). Returning "drop" aborts the TCP connection and stops handling the message;
        # "abort" aborts the TCP connection but still steps (a network drop while inference is in flight).
        self.fault_hook: Callable[[str, int, SlotGroup], str | None] | None = None

    # ---- lifecycle ---------------------------------------------------------------------------------------
    async def start(self, start_engine: bool = True) -> None:
        """Bind every port (health answers 503 until warm), then start the engine in the background."""
        from websockets.asyncio.server import serve

        for i, port in enumerate(self.ports):
            srv = await serve(
                functools.partial(self._handler, listen_index=i), self.host, port,
                compression=None, max_size=None, ping_interval=None, process_request=self._process_request,
            )
            self._servers.append(srv)
            self.bound_ports.append(srv.sockets[0].getsockname()[1])
        logger.info("listening on %s ports %s", self.host, _ports_str(self.bound_ports))
        if start_engine:
            self._engine_task = asyncio.get_running_loop().create_task(self._start_engine())

    async def _start_engine(self) -> None:
        try:
            await self.engine.start()
            logger.info("engine warm: /healthz is 200 on %d port(s)", len(self.bound_ports))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.critical("engine failed to start: %s", e)
            self.engine_failed.set()

    async def close(self) -> None:
        if self._engine_task is not None and not self._engine_task.done():
            self._engine_task.cancel()
            try:
                await self._engine_task
            except (asyncio.CancelledError, Exception):
                pass
        for srv in self._servers:
            srv.close()
        for srv in self._servers:
            try:
                await asyncio.wait_for(srv.wait_closed(), 5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        for groups in self.groups.values():
            for g in groups:
                for s in g.sessions.values():
                    if s.active:
                        log_rollout_end(s, "shutdown")
        await self.engine.close()

    # ---- HTTP --------------------------------------------------------------------------------------------
    @property
    def healthy(self) -> bool:
        return self.engine.warm or not self.config.server.health_requires_warm

    def _process_request(self, connection: Any, request: Any) -> Any:
        path = getattr(request, "path", "/").split("?", 1)[0]
        if path == "/healthz":
            if self.healthy:
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return connection.respond(http.HTTPStatus.SERVICE_UNAVAILABLE, "warming up\n")
        if path == "/status":
            try:
                body = json.dumps(self.status(), default=str)
            except Exception as e:  # pragma: no cover
                body = json.dumps({"error": str(e)})
            return connection.respond(http.HTTPStatus.OK, body + "\n")
        return None

    def status(self) -> dict[str, Any]:
        groups = []
        for port, gs in self.groups.items():
            for g in gs:
                groups.append({"port": port, "group": g.index, "connected": not g.is_free(),
                               "sessions": [s.describe() for s in g.sessions.values()]})
        return {"healthy": self.healthy, "uptime_s": round(time.monotonic() - self.started_at, 1),
                "ports": self.bound_ports, "counters": self.counters, "engine": self.engine.status(),
                "groups": groups}

    def metadata(self, port: int) -> dict[str, Any]:
        return {
            "server": "b1k26",
            "version": __version__,
            "port": int(port),
            "action_dim": C.ACTION_DIM,
            "action_chunk": True,
            "default_profile": self.config.routing.default,
        }

    # ---- slot groups -------------------------------------------------------------------------------------
    def _claim_group(self, port: int, ws: Any, fingerprints: tuple[bytes, ...] | None) -> SlotGroup:
        groups = self.groups.setdefault(port, [])
        free = [g for g in groups if g.is_free()]
        chosen = None
        if fingerprints is not None:
            for g in free:
                if g.cache is not None and g.cache[0] == fingerprints:
                    chosen = g
                    break
        if chosen is None and fingerprints is not None:
            # The client resent the last observation of a group whose connection still looks alive: that
            # connection is half-open (the client already gave up on it). Take the group over.
            for g in groups:
                if g not in free and g.cache is not None and g.cache[0] == fingerprints:
                    logger.warning("port %d: observation matches live slot group %d; taking it over from a stale "
                                   "connection", port, g.index)
                    try:
                        g.conn.transport.abort()
                    except Exception:
                        pass
                    chosen = g
                    break
        if chosen is None and free:
            # Prefer a group whose (closed) connection has not been released yet: its handler is still finishing
            # a step for the observation this client is about to resend (it waits for group.lock, then replays).
            chosen = max(free, key=lambda g: float("inf") if g.conn is not None else g.released_at)
        if chosen is None:
            idx = self._group_counter.get(port, 0)
            self._group_counter[port] = idx + 1
            chosen = SlotGroup(port, idx)
            groups.append(chosen)
            if idx > 0:
                logger.info("port %d: concurrent connection gets a fresh slot group %d", port, idx)
        elif chosen.connections > 0:
            logger.info("port %d: new connection inherits slot group %d (%d session(s))", port, chosen.index,
                        sum(1 for s in chosen.sessions.values() if s.active))
        chosen.conn = ws
        chosen.connections += 1
        # Bound the number of idle groups kept per port (they only hold small queues).
        idle = sorted((g for g in groups if g.is_free() and g is not chosen), key=lambda g: g.released_at)
        for g in idle[: max(0, len(idle) - MAX_FREE_GROUPS_PER_PORT)]:
            groups.remove(g)
            g.log_and_reset("evicted idle slot group")
        return chosen

    # ---- websocket handler -------------------------------------------------------------------------------
    async def _handler(self, ws: Any, listen_index: int) -> None:
        from websockets.exceptions import ConnectionClosed

        port = self.bound_ports[listen_index] if listen_index < len(self.bound_ports) else self.ports[listen_index]
        self.counters["connections"] += 1
        peer = getattr(ws, "remote_address", None)
        logger.info("port %d: connection from %s", port, peer)
        group: SlotGroup | None = None
        first_obs = True
        try:
            await ws.send(packb(self.metadata(port)))
            async for message in ws:
                t_recv = time.monotonic()
                if isinstance(message, str):
                    self.counters["bad_requests"] += 1
                    await ws.send(packb({"error": "text frames are not supported; send msgpack binary frames"}))
                    continue
                try:
                    msg = unpackb(message)
                except Exception as e:
                    self.counters["bad_requests"] += 1
                    logger.error("port %d: undecodable message (%d bytes): %s", port, len(message), e)
                    await ws.send(packb({"error": f"could not decode msgpack: {e}"}))
                    continue
                if not isinstance(msg, dict):
                    self.counters["bad_requests"] += 1
                    await ws.send(packb({"error": "message must be a msgpack map"}))
                    continue

                if C.RESET_KEY in msg:
                    if group is None:
                        group = self._claim_group(port, ws, None)
                    async with group.lock:
                        self.counters["resets"] += 1
                        group.log_and_reset("reset")
                    first_obs = False
                    continue  # a reset never gets a reply

                chunk_k = _parse_chunk_k(msg.get(C.ACTION_CHUNK_REQUEST_KEY))
                try:
                    envs, parse_error = await self._split(msg, group)
                    fps = tuple(e.fingerprint for e in envs) if envs is not None else None
                    if group is None:
                        group = self._claim_group(port, ws, fps)
                    if self._fault("recv", port, group, ws):
                        return
                    async with group.lock:
                        reply = await self._answer(group, msg, envs, parse_error, fps, chunk_k, first_obs, t_recv)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # defensive: one bad message must never end the connection
                    logger.error("port %d: error handling an observation:\n%s", port, traceback.format_exc())
                    if group is None:
                        group = self._claim_group(port, ws, None)
                    reply = packb(self._emergency_reply(group, msg, chunk_k, f"{type(e).__name__}: {e}"))
                first_obs = False
                if self._fault("send", port, group, ws):
                    return
                await ws.send(reply)
        except ConnectionClosed:
            pass
        except Exception:
            logger.error("port %d: connection handler error:\n%s", port, traceback.format_exc())
        finally:
            if group is not None and group.conn is ws:
                group.conn = None
                group.released_at = time.monotonic()
            logger.info("port %d: connection from %s closed", port, peer)

    def _fault(self, stage: str, port: int, group: SlotGroup, ws: Any) -> bool:
        hook = self.fault_hook
        if hook is None:
            return False
        try:
            action = hook(stage, port, group)
        except Exception:  # pragma: no cover
            return False
        if action in ("drop", "abort"):
            logger.warning("fault hook: aborting connection on port %d at %s (%s)", port, stage, action)
            ws.transport.abort()
            return action == "drop"
        return False

    async def _split(self, msg: dict[str, Any], group: SlotGroup | None) -> tuple[list[EnvObs] | None, str | None]:
        # copy_images=False: camera images stay read-only views of the received message (no ~7.8 MB copy per
        # step at full resolution); the engine copies and resizes them only when it plans.
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, functools.partial(split_batch, msg, copy_images=False)), None
        except Exception as e:
            first_error = f"{type(e).__name__}: {e}"
        if C.TASK_ID_KEY not in msg:
            default_tid = group.last_task_id if group is not None and group.last_task_id is not None else 0
            try:
                envs = await loop.run_in_executor(
                    None, functools.partial(split_batch, msg, default_task_id=default_tid, copy_images=False))
                logger.error("observation has no task_id; assuming task %d", default_tid)
                return envs, None
            except Exception:
                pass
        return None, first_error

    async def _answer(self, group: SlotGroup, msg: dict[str, Any], envs: list[EnvObs] | None,
                      parse_error: str | None, fps: tuple[bytes, ...] | None, chunk_k: int, first_obs: bool,
                      t_recv: float) -> bytes:
        self.counters["queries"] += 1
        if envs is None:
            self.counters["bad_requests"] += 1
            logger.error("port %d group %d: bad observation (%s); replying with a fallback action", group.port,
                         group.index, parse_error)
            return packb(self._emergency_reply(group, msg, chunk_k, parse_error or "bad observation"))

        if first_obs and group.cache is not None and fps is not None and group.cache[0] == fps \
                and group.cache[1] == chunk_k:
            self.counters["replays"] += 1
            for b in range(len(envs)):
                group.session(b).stats.replays += 1
            logger.warning("port %d group %d: observation re-sent after a reconnect; replaying the cached reply",
                           group.port, group.index)
            return group.cache[2]

        sessions = [group.session(b) for b in range(len(envs))]
        group.last_task_id = int(envs[0].task_id)
        t0 = time.monotonic()
        action, chunk = await self.engine.step(sessions, envs, chunk_k)
        step_ms = (time.monotonic() - t0) * 1e3
        reply: dict[str, Any] = {"action": action}
        if chunk_k > 1 and chunk is not None:
            reply["action_chunk"] = chunk
        total_ms = (time.monotonic() - t_recv) * 1e3
        reply["server_timing"] = {"infer_ms": step_ms, "total_ms": total_ms}
        packed = packb(reply)
        group.cache = (fps, chunk_k, packed) if fps is not None else None
        for b, s in enumerate(sessions):
            s.last_fingerprint = fps[b] if fps is not None else None
            s.last_response = (action[b], None if chunk is None else chunk[b])
            s.stats.query_ms.append(total_ms)
        return packed

    def _emergency_reply(self, group: SlotGroup, msg: dict[str, Any], chunk_k: int, error: str) -> dict[str, Any]:
        """Best-effort reply for an observation that split_batch rejected: hold from the raw proprio if it is
        usable, else the slot group's last actions, else an error map (the client will retry or fail)."""
        k = max(chunk_k, 1)
        rows: list[np.ndarray] | None = None
        try:
            p = np.asarray(msg.get(C.PROPRIO_KEY), dtype=np.float32)
            if p.ndim == 1:
                p = p[None]
            if p.ndim == 2 and p.shape[1] == C.PROPRIO_DIM and p.shape[0] >= 1:
                rows = [hold_chunk(p[b], k) for b in range(p.shape[0])]
        except Exception:
            rows = None
        if rows is None:
            last = [group.sessions[b].last_action for b in sorted(group.sessions)]
            if last and all(a is not None for a in last):
                rows = [np.tile(a, (k, 1)).astype(np.float32) for a in last]
        if rows is None:
            return {"error": f"bad observation: {error}"}
        arr = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        reply: dict[str, Any] = {"action": np.ascontiguousarray(arr[:, 0, :])}
        if chunk_k > 1:
            reply["action_chunk"] = arr
        reply["server_timing"] = {"infer_ms": 0.0, "error": error}
        return reply


def _ports_str(ports: list[int]) -> str:
    if not ports:
        return "[]"
    if len(ports) > 4 and ports == list(range(ports[0], ports[0] + len(ports))):
        return f"{ports[0]}-{ports[-1]}"
    return ",".join(map(str, ports))


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="b1k26-serve", description="b1k26 front policy server")
    p.add_argument("--config", required=True, help="serving config YAML (see b1k26.config)")
    p.add_argument("--ports", default=os.environ.get("B1K26_PORTS"),
                   help="override server.ports, e.g. 8000 or 8000-8049 (env B1K26_PORTS)")
    p.add_argument("--host", default=os.environ.get("B1K26_HOST"), help="override server.host (env B1K26_HOST)")
    p.add_argument("--log-level", default=os.environ.get("B1K26_LOG_LEVEL", "INFO"))
    p.add_argument("--check", action="store_true",
                   help="validate the config and the launched workers' interpreters, print a summary and exit "
                        "(0 = ok, 2 = problem); nothing is started")
    return p


def check_config(cfg: Config, ports: list[int] | None = None) -> list[str]:
    """Static pre-flight checks beyond schema validation (used by ``--check`` and the Docker build): every worker
    that routing can reach and that is launched must have an executable interpreter / program."""
    import shutil

    problems: list[str] = []
    for name in cfg.used_workers():
        w = cfg.workers[name]
        if not w.launch:
            continue
        prog = w.launch[0]
        found = prog if os.path.isabs(prog) else shutil.which(prog)
        if not found or not os.path.isfile(found) or not os.access(found, os.X_OK):
            problems.append(f"worker {name}: launch program {prog!r} does not exist or is not executable")
        if "b1k26.worker" in w.launch and "--backend" not in w.launch:
            problems.append(f"worker {name}: launch runs b1k26.worker without --backend")
    if ports is not None:
        clash = [w.name for w in cfg.workers.values() if w.launch and w.port in ports]
        if clash:
            problems.append(f"server ports overlap the port of launched worker(s) {clash}")
    return problems


async def serve_until_stopped(server: PolicyServer, stop: asyncio.Event) -> int:
    await server.start()
    stop_task = asyncio.create_task(stop.wait())
    fail_task = asyncio.create_task(server.engine_failed.wait())
    done, pending = await asyncio.wait({stop_task, fail_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    rc = 1 if fail_task in done and not stop.is_set() else 0
    logger.info("shutting down (rc=%d)", rc)
    await server.close()
    return rc


async def _amain(cfg: Config, host: str | None, ports: list[int] | None) -> int:
    server = PolicyServer(cfg, host=host, ports=ports)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-POSIX
            pass
    return await serve_until_stopped(server, stop)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [front] %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)  # one INFO line per health poll otherwise
    try:
        cfg = load_config(args.config)
        ports = parse_ports(args.ports, "--ports") if args.ports else None
    except (ConfigError, OSError) as e:
        print(f"b1k26-serve: {e}", file=sys.stderr)
        return 2
    if ports is not None:
        clash = [w.name for w in cfg.workers.values() if w.launch and w.port in ports]
        if clash:
            print(f"b1k26-serve: --ports overlaps the port of launched worker(s) {clash}", file=sys.stderr)
            return 2
    if args.check:
        problems = check_config(cfg, ports)
        print(f"config {args.config}: ports {_ports_str(ports or cfg.server.ports)}, default profile "
              f"{cfg.routing.default}, {len(cfg.routing.per_task)} per-task route(s), workers {cfg.used_workers()}")
        for prob in problems:
            print(f"b1k26-serve: {prob}", file=sys.stderr)
        return 2 if problems else 0
    try:
        return asyncio.run(_amain(cfg, args.host, ports))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
