"""Planning engine of the front server: worker clients, micro-batching scheduler and per-session planning.

- ``WorkerClient``: async websocket client for one worker (b1k26.worker). Optionally launches the worker
  process (moving it to a free loopback port if its port is taken), waits for its ``/healthz``, fetches ``info``,
  warms it up if needed, supervises it (relaunched at once while its restart budget allows, then with backoff; given
  up only on a configuration error) and turns every failure into ``WorkerError``. One request in flight per
  connection.
- ``InferenceScheduler``: one per worker. Collects concurrent plan requests and sends them as micro-batches of
  up to ``max_batch`` items, waiting at most ``batch_wait_ms`` for stragglers when the worker was idle.
- ``PolicyEngine``: maps every rollout session to a profile, plans when the session's queue cannot serve the
  requested chunk, post-processes (gripper corrections -> compression -> sanitize), and returns exactly the
  actions the evaluator executes. A query whose worker is (re)starting waits for it (``restart_wait_s``). It never
  raises: any failure yields hold actions (never zeros) and the session plans again on its next query.

Chunk replay semantics (docs/ARCHITECTURE.md, engine.py): with ``chunk_k = K > 1`` the evaluator executes K
returned actions open loop. A session plans whenever its queue holds fewer than ``max(K, 1)`` actions; a partial
leftover queue is discarded first, so a returned chunk always comes from one plan. ``action`` is a copy of
``chunk[:, 0]`` of the same float32 array, so the two are bit-identical.
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import importlib
import itertools
import logging
import os
import secrets
import signal
import socket
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from b1k26 import constants as C
from b1k26.config import Config, Profile, WorkerConfig, resolve_prompt
from b1k26.control import as_chunk, plan_execution, sanitize
from b1k26.corrections import GripperRules, task_progress
from b1k26.obs import EnvObs, hold_action, prepare_images, state23_action_order
from b1k26.protocol import packb, unpackb
from b1k26.session import RolloutSession
from b1k26.stage import StageTracker

logger = logging.getLogger("b1k26.engine")

PARENT_PID_ENV = "B1K26_PARENT_PID"  # same name as b1k26.worker.PARENT_PID_ENV (not imported: keep worker light)
LAUNCH_TOKEN_ENV = "B1K26_LAUNCH_TOKEN"  # b1k26.worker echoes it in info: proves the worker on the port is ours
EXIT_CONFIG_ERROR = 2  # b1k26.worker exit status for configuration errors (never relaunched)
HANG_TIMEOUTS_BEFORE_RESTART = 3  # consecutive request timeouts after which a launched worker is restarted
STARTUP_LOADING_GRACE = 3.0  # a worker still loading (/healthz 503) gets up to this x startup_timeout_s
_TERMINATE_GRACE_S = 10.0


class WorkerError(RuntimeError):
    """A worker request failed (error reply, timeout, lost connection, worker not ready)."""


class WorkerTimeout(WorkerError):
    """A worker request exceeded its timeout."""


class _ProcessExited(WorkerError):
    """The launched worker process exited during start-up."""


class WorkerConfigError(WorkerError):
    """A static problem that relaunching cannot fix: the launch program is missing, or the worker exited with
    status 2 (bad arguments, unknown backend, missing checkpoint files)."""


# ----------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------
async def http_status(host: str, port: int, path: str = "/healthz", timeout: float = 2.0) -> int | None:
    """Minimal async HTTP GET returning the status code, or None if the server is unreachable."""
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        parts = line.decode("latin-1").split()
        if len(parts) >= 2 and parts[0].startswith("HTTP/"):
            return int(parts[1])
        return None
    except (OSError, asyncio.TimeoutError, ValueError):
        return None
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 1.0)
            except Exception:
                pass


def no_proxy_kwargs(connect_fn: Any) -> dict[str, Any]:
    """``{"proxy": None}`` for websockets versions whose clients honor HTTP(S)_PROXY by default (>= 15), so a
    proxy configured in the environment never sits between the front server and a local worker."""
    try:
        import inspect

        params = inspect.signature(connect_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    return {"proxy": None} if "proxy" in params else {}


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _free_loopback_port(host: str = "127.0.0.1") -> int:
    """A port the OS reports free on the loopback interface (bound and released at once)."""
    family, addr = (socket.AF_INET6, "::1") if ":" in host else (socket.AF_INET, "127.0.0.1")
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((addr, 0))
        return int(s.getsockname()[1])


def _make_preexec() -> Callable[[], None] | None:
    """On Linux, make a launched worker receive SIGTERM when the front server dies (PR_SET_PDEATHSIG)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        prctl = libc.prctl
    except (OSError, AttributeError):
        return None
    pr_set_pdeathsig = 1

    def preexec() -> None:  # runs in the child between fork and exec: keep it trivial
        prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)

    return preexec


class _RateLimitedLog:
    """Log a recurring warning at most once per ``period_s`` per key, with a count of suppressed repeats."""

    def __init__(self, period_s: float = 5.0):
        self.period_s = period_s
        self._last: dict[str, float] = {}
        self._suppressed: collections.Counter[str] = collections.Counter()

    def warning(self, key: str, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last.get(key, -1e9) >= self.period_s:
            n = self._suppressed.pop(key, 0)
            suffix = f" (+{n} similar suppressed)" if n else ""
            logger.warning(msg + suffix, *args)
            self._last[key] = now
        else:
            self._suppressed[key] += 1


# ----------------------------------------------------------------------------------------------------------
# Worker client
# ----------------------------------------------------------------------------------------------------------
class WorkerClient:
    """Async client of one worker; launches and supervises it when the config has ``launch``.

    States: init -> starting -> ready; ready -> restarting -> ready (a launched worker that died is relaunched at
    once while ``max_restarts`` per ``restart_window_s`` allows); -> backoff -> restarting (beyond that budget, or
    after a failed start: relaunch after a capped exponential backoff, forever); -> failed only on a static error
    (``WorkerConfigError``: missing program, worker exit status 2) or for a non-persistent endpoint-only worker that
    could not be reached; stopped after ``close()``. ``failed`` (the property) is true in backoff and failed: the
    worker is down and not expected back soon, so its tasks fall back to the default profile.

    ``persistent`` (set by PolicyEngine for the default profile's worker): ``start()`` keeps relaunching until the
    worker is ready (at once after a crash while the budget allows, else with backoff) and raises only on a static
    error. A non-persistent worker gets exactly one start attempt, so /healthz never waits for a routed worker's
    relaunch loop; if it fails, ``start()`` raises and a launched worker is retried in the background with backoff.
    """

    def __init__(self, cfg: WorkerConfig, connect_timeout_s: float = 10.0, health_poll_s: float = 0.25,
                 persistent: bool = False):
        self.cfg = cfg
        self.name = cfg.name
        self._state = "init"
        self._state_waiters: list[asyncio.Future] = []
        self.persistent = persistent
        self.info: dict[str, Any] = {}
        self.restarts = 0
        self.last_error: str | None = None
        self.consecutive_timeouts = 0
        self.connect_timeout_s = connect_timeout_s
        self.health_poll_s = health_poll_s
        self._ws: Any = None
        self._proc: asyncio.subprocess.Process | None = None
        self._launch_token: str | None = None
        self._lock = asyncio.Lock()
        self._req_ids = itertools.count(1)
        self._supervisor: asyncio.Task | None = None
        self._closing = False
        self._restart_times: collections.deque[float] = collections.deque()  # monotonic times of recent restarts
        self._backoff_s = float(cfg.restart_backoff_s)

    # ---- state -------------------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value
        waiters, self._state_waiters = self._state_waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        """Down and not expected back soon: in backoff (retried later) or failed for good."""
        return self.state in ("backoff", "failed")

    @property
    def coming_up(self) -> bool:
        """Being (re)started right now: a query may wait for it (PolicyEngine restart_wait_s)."""
        return self.state in ("starting", "restarting")

    async def wait_ready(self, timeout: float) -> bool:
        """Wait up to ``timeout`` s while the worker is starting/restarting; True if it is ready."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(float(timeout), 0.0)
        while self.coming_up:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            fut = loop.create_future()
            self._state_waiters.append(fut)
            try:
                await asyncio.wait_for(fut, remaining)
            except asyncio.TimeoutError:
                break
        return self.ready

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "endpoint": self.cfg.endpoint, "pid": self.pid, "restarts": self.restarts,
                "last_error": self.last_error, "flavor": self.info.get("flavor"),
                "action_horizon": self.info.get("action_horizon")}

    def serves(self, task_id: int) -> bool:
        """Whether the worker's ``info`` says it can serve ``task_id`` (True when it does not say).

        A worker restricts its tasks with ``supported_tasks`` (list of ids) or, for stage-tracking backends, with
        ``num_stages[task] == 0`` (b1k26.backends.pibehavior). Sending it another task fails the whole micro-batch.
        """
        served = self.info.get("supported_tasks")
        if served is not None:
            return int(task_id) in served
        ns = self.info.get("num_stages")
        if ns is not None and 0 <= int(task_id) < len(ns):
            return int(ns[int(task_id)]) >= 1
        return True

    def _restart_allowed(self, now: float | None = None) -> bool:
        """At most ``max_restarts`` immediate restarts within any ``restart_window_s`` (a sliding window), so one
        crash a day is always restarted at once while a crash loop is slowed down to the backoff schedule."""
        now = time.monotonic() if now is None else now
        window = float(self.cfg.restart_window_s)
        while self._restart_times and now - self._restart_times[0] > window:
            self._restart_times.popleft()
        return len(self._restart_times) < self.cfg.max_restarts

    def _record_restart(self, now: float | None = None) -> None:
        self._restart_times.append(time.monotonic() if now is None else now)
        self.restarts += 1

    async def _backoff_sleep(self) -> None:
        """State backoff for the current delay (logged), then double the delay up to restart_backoff_max_s."""
        delay = self._backoff_s
        self.state = "backoff"
        logger.error("worker %s is down (%s); %d restart(s) within %.0f s already; relaunching in %.0f s",
                     self.name, self.last_error, len(self._restart_times), self.cfg.restart_window_s, delay)
        await asyncio.sleep(delay)
        self._backoff_s = min(self._backoff_s * 2.0, float(self.cfg.restart_backoff_max_s))

    # ---- lifecycle ---------------------------------------------------------------------------------------
    async def start(self) -> None:
        """Bring the worker up (launch, health, connect, info, warmup); see the class docstring for retries.

        Raises WorkerConfigError on a static error, WorkerError when a non-persistent worker's start failed."""
        while True:
            try:
                await self._bring_up()
                break
            except asyncio.CancelledError:
                await self._kill()
                raise
            except WorkerConfigError as e:
                self.last_error = str(e)
                self.state = "failed"
                await self._kill()
                logger.error("worker %s: %s (not retried)", self.name, e)
                raise
            except Exception as e:
                self.last_error = str(e) if isinstance(e, WorkerError) else f"{type(e).__name__}: {e}"
                await self._kill()
                if self._closing:
                    self.state = "failed"
                    if isinstance(e, WorkerError):
                        raise
                    raise WorkerError(self.last_error) from e
                if self.persistent and self.cfg.launch and isinstance(e, _ProcessExited) and self._restart_allowed():
                    self._record_restart()
                    logger.error("worker %s: %s; relaunching (%d/%d)", self.name, e, len(self._restart_times),
                                 self.cfg.max_restarts)
                    continue
                if self.persistent:
                    await self._backoff_sleep()
                    if self.cfg.launch:
                        self._record_restart()
                    continue
                if self.cfg.launch:
                    # Retried in the background (backoff first); its tasks fall back to the default meanwhile.
                    self.state = "backoff"
                    self._ensure_supervisor()
                else:
                    self.state = "failed"
                raise WorkerError(f"worker {self.name} failed to start: {self.last_error}") from None
        self._backoff_s = float(self.cfg.restart_backoff_s)
        if self.cfg.launch and not self._closing:
            self._ensure_supervisor()

    def _ensure_supervisor(self) -> None:
        if self._supervisor is None or self._supervisor.done():
            self._supervisor = asyncio.get_running_loop().create_task(self._supervise())

    async def _bring_up(self) -> None:
        self.state = "starting"
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        await self._close_ws()
        if self.cfg.launch:
            await self._launch()
        await self._wait_healthy(loop.time() + self.cfg.startup_timeout_s)
        await self._connect(self.connect_timeout_s)
        info = await self._request({"op": "info"}, 30.0)
        token = info.get("launch_token")
        if self._launch_token is not None and token is not None and token != self._launch_token:
            # Another front server's worker answers on this port (two servers raced for it): ours could not bind.
            raise _ProcessExited(f"worker {self.name}: {self.cfg.endpoint} is answered by another server's worker "
                                 f"(pid {info.get('pid')})")
        info = self._validate_info(info)
        if not info.get("warm", True):
            reply = await self._request({"op": "warmup"}, max(self.cfg.startup_timeout_s, 30.0))
            logger.info("worker %s warmed up in %.0f ms", self.name, float(reply.get("ms", 0.0)))
        self.info = info
        self.consecutive_timeouts = 0
        self.state = "ready"
        logger.info("worker %s ready in %.1f s (pid %s): flavor=%s horizon=%s stages=%s inpaint=%s",
                    self.name, time.monotonic() - t0, self.pid, info.get("flavor"), info.get("action_horizon"),
                    info.get("supports_stage"), info.get("supports_inpaint"))

    async def _launch(self) -> None:
        assert self.cfg.launch is not None and self.cfg.port is not None
        await self._kill()  # never leave a previous instance of ours running
        host = self.cfg.host
        for _ in range(50):  # a previous instance may still be shutting down
            if not _port_in_use(host, self.cfg.port):
                break
            await asyncio.sleep(0.1)
        else:
            if not self.cfg.relocatable:
                raise WorkerError(f"worker {self.name}: port {self.cfg.port} is already in use (stale worker process "
                                  "or another server on this host?)")
            new_port = _free_loopback_port(host)
            logger.warning("worker %s: port %d is in use (another server on this host, or a stale worker); using "
                           "free port %d instead", self.name, self.cfg.port, new_port)
            self.cfg = self.cfg.relocated(new_port)
        env = dict(os.environ)
        env.update(self.cfg.env)
        env[PARENT_PID_ENV] = str(os.getpid())
        self._launch_token = secrets.token_hex(8)
        env[LAUNCH_TOKEN_ENV] = self._launch_token
        logger.info("launching worker %s: %s", self.name, " ".join(self.cfg.launch))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.cfg.launch, env=env, cwd=self.cfg.cwd, stdin=asyncio.subprocess.DEVNULL,
                preexec_fn=_make_preexec(),
            )
        except OSError as e:  # missing or non-executable program, bad cwd: relaunching cannot fix it
            raise WorkerConfigError(f"worker {self.name}: cannot launch {self.cfg.launch[0]!r}: {e}") from None

    async def _wait_healthy(self, deadline: float) -> None:
        """Poll /healthz until 200. A worker that still answers 503 (loading) at ``deadline`` gets up to
        STARTUP_LOADING_GRACE x startup_timeout_s in total; one that does not answer at all is given up on."""
        loop = asyncio.get_running_loop()
        host, port = self.cfg.host, self.cfg.endpoint_port
        hard_deadline = deadline + (STARTUP_LOADING_GRACE - 1.0) * self.cfg.startup_timeout_s
        last_log = loop.time()
        while True:
            if self._proc is not None and self._proc.returncode is not None:
                rc = self._proc.returncode
                if rc == EXIT_CONFIG_ERROR:
                    raise WorkerConfigError(f"worker {self.name} exited with status {rc} (configuration error: bad "
                                            "arguments, unknown backend or missing files; see its log)")
                raise _ProcessExited(f"worker {self.name} exited with status {rc} during start-up")
            status = await http_status(host, port, "/healthz", timeout=2.0)
            if status == 200:
                return
            now = loop.time()
            if now >= deadline:
                if status != 503 or now >= hard_deadline:
                    waited = now - deadline + self.cfg.startup_timeout_s
                    raise WorkerError(f"worker {self.name} not healthy after {waited:.0f} s (last /healthz status: "
                                      f"{status})")
                if now - last_log > 60.0 or last_log < deadline:
                    logger.warning("worker %s still loading after startup_timeout_s %.0f s (/healthz 503); waiting up "
                                   "to %.0f s in total", self.name, self.cfg.startup_timeout_s,
                                   STARTUP_LOADING_GRACE * self.cfg.startup_timeout_s)
                    last_log = now
            elif now - last_log > 30.0:
                logger.info("waiting for worker %s at %s:%d (/healthz: %s)", self.name, host, port, status)
                last_log = now
            await asyncio.sleep(self.health_poll_s)

    async def _connect(self, timeout: float) -> None:
        from websockets.asyncio.client import connect

        self._ws = await asyncio.wait_for(
            connect(self.cfg.endpoint, compression=None, max_size=None, ping_interval=None,
                    open_timeout=timeout, close_timeout=2, **no_proxy_kwargs(connect)),
            timeout + 1.0,
        )

    async def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), 2.0)
            except Exception:
                pass

    async def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), _TERMINATE_GRACE_S)
        except asyncio.TimeoutError:
            logger.warning("worker %s did not exit after SIGTERM; killing", self.name)
            try:
                proc.kill()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except asyncio.TimeoutError:  # pragma: no cover
                pass

    async def _supervise(self) -> None:
        """Keep a launched worker up: whenever it is down, relaunch it at once while the restart budget allows,
        otherwise after the backoff delay. Gives up only on a static error (WorkerConfigError)."""
        while not self._closing:
            if self.state == "ready":
                proc = self._proc
                if proc is None:
                    return
                rc = await proc.wait()
                if self._closing:
                    return
                self.state = "restarting"  # before any await: queries must not use the dead worker
                self.last_error = f"process exited with status {rc}"
                logger.error("worker %s exited with status %s", self.name, rc)
            await self._close_ws()
            await self._kill()
            if self.state != "backoff" and self._restart_allowed():
                logger.error("worker %s: restarting (%d in the last %.0f s, limit %d; %d total)", self.name,
                             len(self._restart_times) + 1, self.cfg.restart_window_s, self.cfg.max_restarts,
                             self.restarts + 1)
            else:
                await self._backoff_sleep()
                if self._closing:
                    return
            self._record_restart()
            self.state = "restarting"
            try:
                await self._bring_up()
                self._backoff_s = float(self.cfg.restart_backoff_s)
            except asyncio.CancelledError:
                raise
            except WorkerConfigError as e:
                self.last_error = str(e)
                logger.error("worker %s: %s; giving up", self.name, e)
                self.state = "failed"
                await self._kill()
                return
            except Exception as e:
                self.last_error = str(e) if isinstance(e, WorkerError) else f"{type(e).__name__}: {e}"
                logger.error("worker %s restart failed: %s", self.name, self.last_error)
                self.state = "restarting"  # the loop relaunches it (at once or after a backoff)

    async def close(self) -> None:
        self._closing = True
        if self._supervisor is not None:
            self._supervisor.cancel()
            try:
                await self._supervisor
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_ws()
        await self._kill()
        self.state = "stopped"

    # ---- requests ----------------------------------------------------------------------------------------
    @staticmethod
    def _validate_info(info: dict[str, Any]) -> dict[str, Any]:
        out = dict(info)
        try:
            horizon = int(out.get("action_horizon") or 0)
        except (TypeError, ValueError):
            horizon = 0
        if horizon < 1:
            raise WorkerError(f"worker info has no valid action_horizon: {info!r}")
        out["action_horizon"] = horizon
        ns = out.get("num_stages")
        if ns is not None:
            if not isinstance(ns, (list, tuple)) or len(ns) != C.NUM_TASKS:
                raise WorkerError(f"worker info num_stages must be None or a list of {C.NUM_TASKS} ints")
            out["num_stages"] = [int(x) for x in ns]
        out["supports_inpaint"] = bool(out.get("supports_inpaint", False))
        out["supports_stage"] = bool(out.get("supports_stage", False)) and out.get("num_stages") is not None
        st = out.get("supported_tasks")
        if st is not None:
            try:
                out["supported_tasks"] = frozenset(int(t) for t in st)
            except (TypeError, ValueError):
                raise WorkerError(f"worker info supported_tasks must be a list of task ids: {st!r}") from None
        try:
            out["image_size"] = None if out.get("image_size") is None else int(out["image_size"])
        except (TypeError, ValueError):
            out["image_size"] = None
        return out

    async def _request(self, msg: dict[str, Any], timeout: float) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        async with self._lock:
            ws = self._ws
            if ws is None:
                raise WorkerError(f"worker {self.name} is not connected")
            rid = next(self._req_ids)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            try:
                await ws.send(packb({**msg, "id": rid}))
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    raw = await asyncio.wait_for(ws.recv(), remaining)
                    if isinstance(raw, str):
                        raise WorkerError(f"worker {self.name} sent a text frame: {raw[:200]!r}")
                    reply = unpackb(raw)
                    if not isinstance(reply, dict):
                        raise WorkerError(f"worker {self.name} sent a non-map reply")
                    if reply.get("id") != rid:
                        continue  # late reply to an earlier request that timed out
                    break
            except asyncio.TimeoutError:
                self.consecutive_timeouts += 1
                self._maybe_restart_hung()
                raise WorkerTimeout(f"worker {self.name}: {msg.get('op')} timed out after {timeout:.1f} s") from None
            except ConnectionClosed as e:
                self._ws = None
                raise WorkerError(f"worker {self.name}: connection lost ({e})") from None
            except OSError as e:
                self._ws = None
                raise WorkerError(f"worker {self.name}: {e}") from None
            self.consecutive_timeouts = 0
            if "error" in reply:
                raise WorkerError(f"worker {self.name}: {reply['error']}")
            return reply

    def _maybe_restart_hung(self) -> None:
        if (self.cfg.launch and self.consecutive_timeouts >= HANG_TIMEOUTS_BEFORE_RESTART and self._proc is not None
                and self._proc.returncode is None and self.state == "ready"):
            logger.error("worker %s timed out %d times in a row; killing it so the supervisor restarts it",
                         self.name, self.consecutive_timeouts)
            self.consecutive_timeouts = 0
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    async def infer(self, items: list[dict[str, Any]], timeout: float) -> list[dict[str, Any]]:
        """Send one micro-batch; returns one chunk map per item. Raises WorkerError."""
        if self.state != "ready":
            raise WorkerError(f"worker {self.name} is {self.state}")
        if self._ws is None:
            # The connection was lost but the worker is not known to be down: reconnect lazily.
            try:
                await self._connect(min(self.connect_timeout_s, timeout))
            except Exception as e:
                raise WorkerError(f"worker {self.name}: reconnect failed: {e}") from None
        reply = await self._request({"op": "infer", "items": items}, timeout)
        chunks = reply.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(items):
            raise WorkerError(f"worker {self.name} returned {len(chunks) if isinstance(chunks, list) else chunks!r}"
                              f" chunks for {len(items)} items")
        return chunks


# ----------------------------------------------------------------------------------------------------------
# Micro-batching scheduler
# ----------------------------------------------------------------------------------------------------------
class InferenceScheduler:
    """Batches concurrent plan requests for one worker; exactly one worker request in flight at a time."""

    def __init__(self, client: WorkerClient, max_batch: int = 8, batch_wait_s: float = 0.005,
                 request_timeout_s: float = 120.0):
        self.client = client
        self.max_batch = max(1, int(max_batch))
        self.batch_wait_s = max(0.0, float(batch_wait_s))
        self.request_timeout_s = float(request_timeout_s)
        self._queue: collections.deque[tuple[dict[str, Any], asyncio.Future]] = collections.deque()
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self.batches = 0
        self.items = 0
        self.batch_sizes: collections.Counter[int] = collections.Counter()

    def submit(self, item: dict[str, Any]) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.append((item, fut))
        if self._task is None or self._task.done():
            self._wakeup = asyncio.Event()  # bound to the running loop, like the task
            self._task = loop.create_task(self._run())
        assert self._wakeup is not None
        self._wakeup.set()
        return fut

    def _pending(self) -> int:
        return sum(1 for _, f in self._queue if not f.done())

    async def _run(self) -> None:
        assert self._wakeup is not None
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            idle = True
            while self._queue:
                try:
                    if idle and self.batch_wait_s > 0 and self._pending() < self.max_batch:
                        await asyncio.sleep(self.batch_wait_s)  # let concurrent requests join this batch
                    idle = False
                    batch: list[tuple[dict[str, Any], asyncio.Future]] = []
                    while self._queue and len(batch) < self.max_batch:
                        item, fut = self._queue.popleft()
                        if not fut.done():  # skip requests whose caller timed out
                            batch.append((item, fut))
                    if not batch:
                        continue
                    self.batches += 1
                    self.items += len(batch)
                    self.batch_sizes[len(batch)] += 1
                    try:
                        chunks = await self.client.infer([it for it, _ in batch], self.request_timeout_s)
                    except Exception as e:
                        err = e if isinstance(e, WorkerError) else WorkerError(f"{type(e).__name__}: {e}")
                        for _, fut in batch:
                            if not fut.done():
                                fut.set_exception(err)
                        continue
                    for (_, fut), chunk in zip(batch, chunks):
                        if not fut.done():
                            fut.set_result(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - the loop must never die
                    logger.error("scheduler for %s: unexpected error:\n%s", self.client.name, traceback.format_exc())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        while self._queue:
            _, fut = self._queue.popleft()
            if not fut.done():
                fut.set_exception(WorkerError("scheduler closed"))


# ----------------------------------------------------------------------------------------------------------
# Policy engine
# ----------------------------------------------------------------------------------------------------------
def hold_chunk(proprio: np.ndarray, k: int) -> np.ndarray:
    """(k, 23) float32 rows of hold_action(proprio)."""
    return np.tile(hold_action(proprio), (max(int(k), 1), 1)).astype(np.float32)


class PolicyEngine:
    """Routes sessions to profiles and plans action chunks through the workers (see the module docstring)."""

    def __init__(self, config: Config, worker_clients: dict[str, WorkerClient] | None = None,
                 rules: GripperRules | None = None):
        self.config = config
        used = config.used_workers()
        default_worker = config.default_profile.worker
        if worker_clients is None:
            worker_clients = {name: WorkerClient(config.workers[name], persistent=(name == default_worker))
                              for name in used}
        missing = [n for n in used if n not in worker_clients]
        if missing:
            raise ValueError(f"no WorkerClient for worker(s) {missing}")
        if isinstance(worker_clients[default_worker], WorkerClient):
            # The default profile's worker is never given up (except on a static error): without it nothing plans.
            worker_clients[default_worker].persistent = True
        self.clients = worker_clients
        self.rules = rules if rules is not None else GripperRules.load(config.engine.gripper_rules)
        self.plan_timeout_s = float(config.engine.plan_timeout_s)
        self.restart_wait_s = float(config.engine.restart_wait_s)
        self.schedulers: dict[str, InferenceScheduler] = {}
        for name in used:
            wcfg = config.workers.get(name)
            mb = config.engine.max_batch
            if wcfg is not None and wcfg.max_batch is not None:
                mb = min(mb, wcfg.max_batch)
            self.schedulers[name] = InferenceScheduler(
                self.clients[name], max_batch=mb, batch_wait_s=config.engine.batch_wait_ms / 1e3,
                request_timeout_s=self.plan_timeout_s,
            )
        self._warm = False
        self.start_error: str | None = None
        self.config_problems: list[str] = []  # profile/worker mismatches found by check_profiles()
        self._log = _RateLimitedLog()
        self._warned: set[str] = set()
        try:  # import scipy once now so the first plan does not pay for it
            importlib.import_module("scipy.interpolate")
        except ImportError:  # pragma: no cover
            logger.warning("scipy is missing: compression falls back to linear interpolation")

    # ---- lifecycle -----------------------------------------------------------------------------------------
    @property
    def default_worker(self) -> str:
        return self.config.default_profile.worker

    @property
    def warm(self) -> bool:
        """True once start() finished, while the default profile's worker is not down (backoff/failed)."""
        return self._warm and not self.clients[self.default_worker].failed

    async def start(self) -> None:
        """Start every worker that routing can reach (concurrently) and return when each has finished its start:
        the default profile's worker keeps relaunching until it is ready, the others get one attempt (failed ones
        are retried in the background; their tasks fall back to the default profile meanwhile). /healthz therefore
        waits for every routed worker's first start attempt. Raises WorkerError only if the default worker cannot
        start for a static reason (WorkerConfigError: missing program, configuration error)."""
        names = self.config.used_workers()
        t0 = time.monotonic()
        results = await asyncio.gather(*(self.clients[n].start() for n in names), return_exceptions=True)
        for n, r in zip(names, results):
            if isinstance(r, BaseException):
                logger.error("worker %s failed to start: %s", n, r)
        default = self.clients[self.default_worker]
        default_result = results[names.index(self.default_worker)]
        if isinstance(default_result, BaseException) or default.state == "failed":
            self.start_error = f"default worker {self.default_worker} failed: {getattr(default, 'last_error', None)}"
            raise WorkerError(self.start_error)
        self.config_problems = self.check_profiles()
        self._warm = True
        logger.info("engine warm in %.1f s: %s", time.monotonic() - t0,
                    {n: self.clients[n].state for n in names})

    def routed_tasks(self, profile_name: str) -> list[int]:
        """Task ids that routing sends to ``profile_name``."""
        r = self.config.routing
        return [t for t in range(C.NUM_TASKS) if r.profile_for(t) == profile_name]

    def check_profiles(self) -> list[str]:
        """Compare every routed profile with its (ready) worker's ``info`` and log each mismatch.

        Checked: image size, stage support, chunk horizon vs the execution settings, and routed tasks the worker
        does not serve (those are re-routed to the default profile at plan time when it can serve them, see
        ``_effective_profile``). Returns the problems as strings (also exposed in ``status()``).
        """
        problems: list[str] = []
        names = [self.config.routing.default, *sorted(set(self.config.routing.per_task.values()))]
        for pname in dict.fromkeys(names):
            prof = self.config.profiles[pname]
            client = self.clients[prof.worker]
            if not client.ready:
                continue
            info = client.info
            size = info.get("image_size")
            if size is not None and int(size) != prof.image_size:
                problems.append(f"profile {pname}: image_size {prof.image_size} but worker {prof.worker} expects "
                                f"{size}")
            if prof.use_stage and not info.get("supports_stage"):
                problems.append(f"profile {pname}: use_stage is on but worker {prof.worker} reports no stage "
                                "support; stage tracking is disabled")
            elif info.get("supports_stage") and not prof.use_stage:
                problems.append(f"profile {pname}: worker {prof.worker} is stage-conditioned but use_stage is off; "
                                "the model is never told a stage (it sees its default stage 0)")
            horizon = int(info.get("action_horizon") or 0)
            ex = prof.execution
            if horizon and ex.execute_steps > horizon:
                problems.append(f"profile {pname}: execute_steps {ex.execute_steps} > worker horizon {horizon}; "
                                "plans are shorter than configured")
            elif horizon and ex.predicted_steps_to_use > horizon:
                problems.append(f"profile {pname}: predicted_steps_to_use {ex.predicted_steps_to_use} > worker "
                                f"horizon {horizon}; compression uses only {horizon} predicted actions")
            if (horizon and info.get("supports_inpaint") and ex.keep_for_inpaint > 0
                    and min(ex.predicted_steps_to_use, horizon) + ex.keep_for_inpaint > horizon):
                problems.append(f"profile {pname}: keep_for_inpaint {ex.keep_for_inpaint} does not fit in the worker "
                                f"horizon {horizon}; the inpainting tail is never available")
            unserved = [t for t in self.routed_tasks(pname) if not client.serves(t)]
            if unserved:
                default = self.config.default_profile
                dclient = self.clients[default.worker]
                can_fallback = (pname != default.name and self.config.engine.fallback_to_default and dclient.ready)
                rescued = [t for t in unserved if can_fallback and dclient.serves(t)]
                lost = [t for t in unserved if t not in rescued]
                parts = []
                if rescued:
                    parts.append(f"{_tasks_str(rescued)} fall back to the default profile {default.name}")
                if lost:
                    parts.append(f"{_tasks_str(lost)} cannot be served: their plans fail and the robot holds its pose")
                problems.append(f"profile {pname}: worker {prof.worker} does not serve {len(unserved)} routed task(s) "
                                f"{_tasks_str(unserved)}; " + "; ".join(parts))
        for msg in problems:
            logger.error("config check: %s", msg)
        return problems

    async def close(self) -> None:
        for s in self.schedulers.values():
            await s.close()
        await asyncio.gather(*(c.close() for c in self.clients.values()), return_exceptions=True)

    def status(self) -> dict[str, Any]:
        return {
            "warm": self.warm,
            "start_error": self.start_error,
            "config_problems": list(self.config_problems),
            "workers": {n: c.status() for n, c in self.clients.items()},
            "batches": {n: {"batches": s.batches, "items": s.items, "sizes": dict(s.batch_sizes)}
                        for n, s in self.schedulers.items()},
        }

    # ---- stepping ------------------------------------------------------------------------------------------
    async def step(self, sessions: Sequence[RolloutSession], envs: Sequence[EnvObs], chunk_k: int
                   ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return action (B, 23) float32 and, if chunk_k > 1, action_chunk (B, chunk_k, 23) with
        ``action_chunk[:, 0] == action`` exactly. Never raises (falls back to hold actions)."""
        try:
            k = max(int(chunk_k), 1)
        except (TypeError, ValueError):
            k = 1
        want_chunk = k > 1
        if not envs:
            empty = np.zeros((0, k, C.ACTION_DIM), dtype=np.float32)
            return empty[:, 0, :].copy(), (empty if want_chunk else None)
        try:
            if len(sessions) != len(envs):
                raise ValueError(f"{len(sessions)} sessions for {len(envs)} observations")
            return await self._step(sessions, envs, k, want_chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("engine.step failed; sending hold actions:\n%s", traceback.format_exc())
            return self._fallback(sessions, envs, k, want_chunk)

    async def _step(self, sessions: Sequence[RolloutSession], envs: Sequence[EnvObs], k: int, want_chunk: bool
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        to_plan: list[tuple[RolloutSession, EnvObs]] = []
        for s, env in zip(sessions, envs):
            s.stats.queries += 1
            self._sync_session(s, env)
            if len(s.queue) < k:
                if s.queue:
                    # A partial leftover cannot start a contiguous open-loop chunk: drop it and its inpaint tail
                    # (the tail assumed every planned action would be executed before the next plan).
                    note = f"discard-{s.profile_name}-{k}"
                    if note not in self._warned:
                        self._warned.add(note)
                        logger.info("profile %s: the evaluator replays chunks of %d, so %d leftover planned "
                                    "action(s) are dropped before each new plan (execute_steps is not a multiple "
                                    "of the chunk size)", s.profile_name, k, len(s.queue))
                    s.queue.clear()
                    s.inpaint_tail = None
                to_plan.append((s, env))
        if to_plan:
            await self._plan_many(to_plan, k)
        chunks = []
        for s, env in zip(sessions, envs):
            if len(s.queue) >= k:
                chunk = np.stack([s.queue.popleft() for _ in range(k)]).astype(np.float32, copy=False)
            else:
                chunk = hold_chunk(env.proprio, k)
                s.stats.hold_steps += k
            s.step += k
            s.last_action = chunk[0].copy()
            chunks.append(chunk)
        return self._pack(chunks, want_chunk)

    @staticmethod
    def _pack(chunks: list[np.ndarray], want_chunk: bool) -> tuple[np.ndarray, np.ndarray | None]:
        arr = np.ascontiguousarray(np.stack(chunks), dtype=np.float32)  # (B, k, 23)
        action = np.ascontiguousarray(arr[:, 0, :])
        return action, (arr if want_chunk else None)

    def _fallback(self, sessions: Sequence[RolloutSession], envs: Sequence[EnvObs], k: int, want_chunk: bool
                  ) -> tuple[np.ndarray, np.ndarray | None]:
        chunks = []
        for i, env in enumerate(envs):
            chunk = hold_chunk(env.proprio, k)
            chunks.append(chunk)
            if i < len(sessions):
                s = sessions[i]
                s.queue.clear()
                s.inpaint_tail = None
                s.step += k
                s.stats.hold_steps += k
                s.last_action = chunk[0].copy()
        return self._pack(chunks, want_chunk)

    def _sync_session(self, s: RolloutSession, env: EnvObs) -> None:
        if s.task_id == env.task_id and s.profile_name is not None:
            return
        if s.task_id is not None and s.task_id != env.task_id:
            logger.warning("session %s: task changed %s -> %s without a reset; starting a new plan", s.key,
                           s.task_id, env.task_id)
            s.reset_plan_state()
            s.step = 0
        s.task_id = int(env.task_id)
        s.profile_name = self.config.routing.profile_for(s.task_id)
        s.stage = None
        s.fallback_note = None

    def _effective_profile(self, s: RolloutSession) -> Profile:
        assert s.profile_name is not None and s.task_id is not None
        prof = self.config.profiles[s.profile_name]
        if not self.config.engine.fallback_to_default or prof.name == self.config.routing.default:
            return prof
        client = self.clients[prof.worker]
        why = None
        if client.failed:
            why = f"worker {prof.worker} failed"
        elif client.ready and not client.serves(s.task_id):
            why = f"worker {prof.worker} does not serve task {s.task_id}"
        if why is None:
            return prof
        default = self.config.default_profile
        if self.clients[default.worker].ready and not self.clients[default.worker].serves(s.task_id):
            return prof  # the default cannot serve it either: keep the routed profile (its plans fail -> hold)
        s.fallback_note = why
        logger.warning("session %s task %s: profile %s unavailable (%s); using default profile %s", s.key,
                       s.task_id, prof.name, why, default.name)
        s.profile_name = default.name
        s.stage = None
        s.inpaint_tail = None
        return default

    async def _plan_many(self, pairs: list[tuple[RolloutSession, EnvObs]], k: int) -> None:
        """Plan one chunk for each (session, observation) and fill the queues. Never raises; a failed plan leaves
        its queue empty (the caller sends a hold). Every item is prepared before any is submitted, so the plans of
        one evaluator message reach the scheduler together and form full micro-batches without relying on
        batch_wait timing."""
        prepared = await asyncio.gather(*(self._prepare(s, env) for s, env in pairs))
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        deadline = loop.time() + self.plan_timeout_s
        futures: list[asyncio.Future | None] = []
        for (s, env), prep in zip(pairs, prepared):
            if prep is None:
                futures.append(None)
                continue
            prof, item = prep
            futures.append(self.schedulers[prof.worker].submit(item))
        await asyncio.gather(*(self._finish(s, env, prep, fut, k, deadline, t0)
                               for (s, env), prep, fut in zip(pairs, prepared, futures)
                               if prep is not None and fut is not None))

    def _plan_failed(self, s: RolloutSession, task_id: int, e: BaseException) -> None:
        s.queue.clear()
        s.inpaint_tail = None
        s.stats.plan_failures += 1
        what = "timed out" if isinstance(e, (asyncio.TimeoutError, WorkerTimeout)) else "failed"
        self._log.warning(f"plan-{s.profile_name}-{type(e).__name__}",
                          "plan %s for session %s task %s (profile %s): %s; sending hold actions",
                          what, s.key, task_id, s.profile_name, f"{type(e).__name__}: {e}")

    async def _prepare(self, s: RolloutSession, env: EnvObs) -> tuple[Profile, dict[str, Any]] | None:
        """Build the worker item for one session (None after logging a failure)."""
        task_id = int(env.task_id)
        try:
            prof = self._effective_profile(s)
            client = self.clients[prof.worker]
            if not client.ready and getattr(client, "coming_up", False) and self.restart_wait_s > 0:
                # The worker is being (re)started: wait for it instead of holding. A hold costs one of the
                # episode's max_steps per step; waiting only costs wall time, of which the evaluator allows plenty
                # (post2: no limit; 2026/eval: max_steps seconds against ~0.1-0.2 s of simulation per step).
                self._log.warning(f"wait-{prof.worker}", "worker %s is %s: queries wait up to %.0f s for it",
                                  prof.worker, client.state, self.restart_wait_s)
                t_wait = time.monotonic()
                await client.wait_ready(self.restart_wait_s)
                s.stats.restart_wait_ms += (time.monotonic() - t_wait) * 1e3
                prof = self._effective_profile(s)  # it may have gone to backoff: fall back to the default
                client = self.clients[prof.worker]
            if not client.ready:
                raise WorkerError(f"worker {prof.worker} is {client.state}")
            if not client.serves(task_id):
                # Never send it: the worker would reject the whole micro-batch, failing other rollouts' plans too.
                raise WorkerError(f"worker {prof.worker} does not serve task {task_id} (route it to another profile)")
            info = client.info
            if prof.use_stage and s.stage is None and info.get("supports_stage") and info.get("num_stages"):
                n = int(info["num_stages"][task_id])
                if n >= 1:
                    s.stage = StageTracker(n)
            stage = s.stage.stage if (s.stage is not None and info.get("supports_stage")) else None
            initial = s.inpaint_tail if info.get("supports_inpaint") else None
            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(None, prepare_images, env, prof.image_size, prof.resize,
                                                prof.cameras)
            proprio = np.array(env.proprio, dtype=np.float32, copy=True)
            if prof.mask_base_qvel:
                proprio[C.PROPRIO_INDICES_2026["base_qvel"]] = 0.0
            return prof, {
                "task_id": task_id,
                "prompt": resolve_prompt(prof.prompt, task_id),
                "proprio": proprio,
                "images": images,
                "stage": stage,
                "initial_actions": initial,
            }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._plan_failed(s, task_id, e)
            return None

    async def _finish(self, s: RolloutSession, env: EnvObs, prep: tuple[Profile, dict[str, Any]],
                      fut: asyncio.Future, k: int, deadline: float, t0: float) -> None:
        """Await one submitted plan and post-process it into the session queue."""
        prof, _ = prep
        task_id = int(env.task_id)
        try:
            remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
            out = await asyncio.wait_for(fut, remaining)
            plan_ms = (time.monotonic() - t0) * 1e3
            self._apply_plan(s, env, prof, out, k)
            s.stats.plans += 1
            s.stats.plan_ms.append(plan_ms)
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            raise
        except Exception as e:
            self._plan_failed(s, task_id, e)

    def _apply_plan(self, s: RolloutSession, env: EnvObs, prof: Profile, out: Any, k: int) -> None:
        if not isinstance(out, dict) or "actions" not in out:
            raise WorkerError("worker chunk has no 'actions'")
        raw = as_chunk(out["actions"])  # (T, 23) float64, validated
        logits = out.get("subtask_logits")
        task_id = int(env.task_id)
        if prof.corrections:
            s23 = state23_action_order(env.proprio)
            stage = s.stage.stage if s.stage is not None else None
            raw, changed, new_stage = self.rules.apply_with_stage(
                task_id, stage, s23, raw, progress=task_progress(task_id, s.step))
            if changed:
                s.stats.corrections += 1
            if new_stage is not None and s.stage is not None:
                s.stage.set_stage(new_stage)
        planned = plan_execution(raw, prof.execution)
        actions = sanitize(planned.actions, env.proprio, clip_base=prof.execution.clip_base)
        if planned.compressed:
            s.stats.compressed += 1
        s.inpaint_tail = planned.inpaint_tail
        if s.stage is not None and logits is not None:
            s.stage.update(np.asarray(logits))
        rows = [np.array(a, dtype=np.float32, copy=True) for a in actions]
        if len(rows) < k:
            # The chunk the evaluator asked for is longer than this plan: hold the final pose (base stopped) for
            # the rest. The inpainting tail no longer lines up with the next plan.
            pad = rows[-1].copy()
            pad[C.ACTION_SLICES["base"]] = 0.0
            if prof.name not in self._warned:
                self._warned.add(prof.name)
                logger.warning("profile %s plans %d actions but the evaluator replays chunks of %d; padding with "
                               "a hold of the last pose (set execute_steps >= the replay chunk size)",
                               prof.name, len(rows), k)
            s.stats.padded += k - len(rows)
            rows.extend(pad.copy() for _ in range(k - len(rows)))
            s.inpaint_tail = None
        s.queue.extend(rows)


def _tasks_str(tasks: Sequence[int], limit: int = 12) -> str:
    shown = ",".join(str(t) for t in tasks[:limit])
    return f"[{shown}{',...' if len(tasks) > limit else ''}]"


def make_engine(config: Config) -> PolicyEngine:
    """PolicyEngine with one WorkerClient per used worker (launching those that have ``launch``)."""
    return PolicyEngine(config)


__all__: Iterable[str] = (
    "InferenceScheduler", "PolicyEngine", "WorkerClient", "WorkerError", "WorkerTimeout", "hold_chunk",
    "http_status", "make_engine",
)
