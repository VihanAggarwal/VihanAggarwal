"""Worker process: hosts one Backend behind the worker protocol (docs/ARCHITECTURE.md, "Worker protocol").

    b1k26-worker --backend fake_sine --port 9101 [--host 127.0.0.1] [--checkpoint PATH]
                 [--backend-arg key=value ...] [--backend-kwargs '{"key": value}'] [--no-warmup]

Messages are msgpack maps (b1k26.protocol) in binary websocket frames:
- ``{"op": "info"}`` -> backend.info() plus ``"warm"``, ``"backend"``, ``"pid"``
- ``{"op": "warmup"}`` -> ``{"ok": True, "ms": float}``
- ``{"op": "infer", "items": [...]}`` -> ``{"chunks": [{"actions": (T, 23) float32, "subtask_logits": ...}], "ms"}``
- a request may carry ``"id"``; the reply echoes it (the front server uses it to drop late replies).
- any failure -> ``{"error": str}`` (binary, never a text frame); the connection stays open.

The server starts listening immediately: ``/healthz`` answers 503 while the backend loads and warms up, then 200.
Inference runs on one dedicated thread (one model, one GPU), so pings and health checks stay responsive. A load
failure exits the process so the front server's supervisor sees it: status 2 (``EXIT_CONFIG_ERROR``) for a
configuration error that a relaunch cannot fix (bad arguments, unknown backend, missing files, bad constructor
arguments: ``CONFIG_ERRORS``), status 3 for anything else (CUDA/XLA errors, out of memory, ...), which the front
server retries.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import http
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut, InferItem, create_backend
from b1k26.protocol import packb, unpackb

logger = logging.getLogger("b1k26.worker")

PARENT_PID_ENV = "B1K26_PARENT_PID"  # set by the front server: exit when that process goes away
LAUNCH_TOKEN_ENV = "B1K26_LAUNCH_TOKEN"  # set by the front server; echoed in info so it can tell its worker apart
EXIT_CONFIG_ERROR = 2  # same value as b1k26.engine.EXIT_CONFIG_ERROR (argparse also exits 2)
EXIT_LOAD_ERROR = 3
# Load exceptions that are deterministic for a given command line: relaunching cannot help.
CONFIG_ERRORS: tuple[type[BaseException], ...] = (FileNotFoundError, NotADirectoryError, IsADirectoryError,
                                                  ImportError, TypeError, ValueError, KeyError)
ROLES = ("head", "left_wrist", "right_wrist")


class RequestError(ValueError):
    """A malformed request (reported to the caller as {"error": ...})."""


# ----------------------------------------------------------------------------------------------------------
# Request decoding / reply encoding
# ----------------------------------------------------------------------------------------------------------
def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def decode_item(raw: Any, index: int) -> InferItem:
    """Validate one wire item and build an InferItem (arrays are copied, so they are writeable)."""
    if not isinstance(raw, dict):
        raise RequestError(f"items[{index}] must be a map")
    try:
        task_id = int(np.asarray(raw["task_id"]).reshape(-1)[0])
    except Exception:
        raise RequestError(f"items[{index}].task_id missing or not an int") from None
    if not 0 <= task_id < C.NUM_TASKS:
        raise RequestError(f"items[{index}].task_id {task_id} outside [0, {C.NUM_TASKS})")
    proprio = raw.get("proprio")
    if not isinstance(proprio, np.ndarray) or proprio.size != C.PROPRIO_DIM:
        raise RequestError(f"items[{index}].proprio must be a ({C.PROPRIO_DIM},) array")
    images_raw = raw.get("images")
    if not isinstance(images_raw, dict) or not images_raw:
        raise RequestError(f"items[{index}].images must be a non-empty map role -> (S, S, 3) uint8")
    images: dict[str, np.ndarray] = {}
    for role, img in images_raw.items():
        role = _text(role)
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
            raise RequestError(f"items[{index}].images[{role!r}] must be (S, S, 3) uint8, got "
                               f"{getattr(img, 'shape', type(img))} {getattr(img, 'dtype', '')}")
        images[role] = np.array(img, copy=True)
    stage = raw.get("stage")
    if stage is not None:
        try:
            stage = int(stage)
        except Exception:
            raise RequestError(f"items[{index}].stage must be an int or nil") from None
    init = raw.get("initial_actions")
    if init is not None:
        if not isinstance(init, np.ndarray) or init.ndim != 2 or init.shape[1] < C.ACTION_DIM:
            raise RequestError(f"items[{index}].initial_actions must be (k, >= {C.ACTION_DIM}) or nil")
        init = np.array(init, dtype=np.float32, copy=True)
    prompt = raw.get("prompt")
    return InferItem(
        task_id=task_id,
        prompt="" if prompt is None else _text(prompt),
        proprio=np.array(proprio, dtype=np.float32, copy=True).reshape(C.PROPRIO_DIM),
        images=images,
        stage=stage,
        initial_actions=init,
    )


def encode_chunk(chunk: ChunkOut, index: int) -> dict[str, Any]:
    if not isinstance(chunk, ChunkOut):
        raise RuntimeError(f"backend returned {type(chunk).__name__} instead of ChunkOut for item {index}")
    actions = np.asarray(chunk.actions)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < C.ACTION_DIM:
        raise RuntimeError(f"backend returned actions of shape {actions.shape} for item {index}; "
                           f"expected (T, >= {C.ACTION_DIM})")
    out: dict[str, Any] = {"actions": np.ascontiguousarray(actions[:, : C.ACTION_DIM], dtype=np.float32)}
    logits = chunk.subtask_logits
    out["subtask_logits"] = None if logits is None else np.ascontiguousarray(np.asarray(logits, np.float32).reshape(-1))
    return out


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    out = dict(info)
    ns = out.get("num_stages")
    if ns is not None:
        out["num_stages"] = [int(x) for x in ns]
    return out


# ----------------------------------------------------------------------------------------------------------
# Worker server
# ----------------------------------------------------------------------------------------------------------
class WorkerServer:
    """Serves one backend. ``backend_factory`` runs on the inference thread (CUDA contexts stay on one thread)."""

    def __init__(
        self,
        backend_factory: Callable[[], Backend],
        host: str = "127.0.0.1",
        port: int = 9100,
        warmup: bool = True,
        name: str = "backend",
    ):
        self.backend_factory = backend_factory
        self.host = host
        self.port = port
        self.do_warmup = warmup
        self.name = name
        self.backend: Backend | None = None
        self.loaded = False
        self.warm = False
        self.load_error: str | None = None
        self.bound_port: int | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="b1k26-infer")
        self._server: Any = None
        self._stop: asyncio.Event | None = None

    # ---- health ----------------------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.loaded and (self.warm or not self.do_warmup)

    def _process_request(self, connection: Any, request: Any) -> Any:
        path = getattr(request, "path", "").split("?", 1)[0]
        if path == "/healthz":
            if self.ready:
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            msg = "load failed\n" if self.load_error else "loading\n"
            return connection.respond(http.HTTPStatus.SERVICE_UNAVAILABLE, msg)
        return None

    # ---- lifecycle -------------------------------------------------------------------------------------------
    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._stop = asyncio.Event()
        self._server = await serve(
            self._handler, self.host, self.port, compression=None, max_size=None, ping_interval=None,
            process_request=self._process_request,
        )
        self.bound_port = self._server.sockets[0].getsockname()[1]
        logger.info("worker %s listening on %s:%d (loading backend)", self.name, self.host, self.bound_port)

    async def load(self) -> None:
        """Construct and warm the backend on the inference thread. Raises on failure (load_error is set)."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            self.backend = await loop.run_in_executor(self._executor, self.backend_factory)
            self.loaded = True
            logger.info("backend %s loaded in %.1f s: %s", self.name, time.monotonic() - t0,
                        _jsonable_info(self.backend.info()))
            if self.do_warmup:
                ms = await loop.run_in_executor(self._executor, self.backend.warmup)
                self.warm = True
                logger.info("backend %s warm (warmup %.0f ms)", self.name, ms)
        except BaseException:
            self.load_error = traceback.format_exc()
            logger.error("backend %s failed to load:\n%s", self.name, self.load_error)
            raise

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 5.0)
            except asyncio.TimeoutError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---- request handling ------------------------------------------------------------------------------------
    async def _handler(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        peer = getattr(ws, "remote_address", None)
        logger.info("connection from %s", peer)
        try:
            async for message in ws:
                reply = await self.handle_message(message)
                await ws.send(packb(reply))
        except ConnectionClosed:
            pass
        except Exception:  # pragma: no cover - defensive: never let a handler crash silently
            logger.error("connection handler error:\n%s", traceback.format_exc())
        logger.info("connection from %s closed", peer)

    async def handle_message(self, message: Any) -> dict[str, Any]:
        """Decode one request and produce its reply map. Never raises."""
        req_id = None
        try:
            if isinstance(message, str):
                raise RequestError("text frames are not supported; send msgpack in binary frames")
            try:
                req = unpackb(message)
            except Exception as e:
                raise RequestError(f"could not decode msgpack request: {e}") from None
            if not isinstance(req, dict):
                raise RequestError("request must be a msgpack map")
            req_id = req.get("id")
            op = _text(req.get("op", ""))
            reply = await self._dispatch(op, req)
        except RequestError as e:
            reply = {"error": f"bad request: {e}"}
        except Exception as e:
            logger.error("request failed:\n%s", traceback.format_exc())
            reply = {"error": f"{type(e).__name__}: {e}"}
        if req_id is not None:
            reply["id"] = req_id
        return reply

    async def _dispatch(self, op: str, req: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        if op == "info":
            if not self.loaded or self.backend is None:
                raise RuntimeError("backend is not loaded yet" + (" (load failed)" if self.load_error else ""))
            info = _jsonable_info(self.backend.info())
            info.update({"warm": self.warm, "backend": self.name, "pid": os.getpid()})
            token = os.environ.get(LAUNCH_TOKEN_ENV)
            if token:
                info["launch_token"] = token
            return info
        if op == "warmup":
            if not self.loaded or self.backend is None:
                raise RuntimeError("backend is not loaded yet")
            ms = await loop.run_in_executor(self._executor, self.backend.warmup)
            self.warm = True
            return {"ok": True, "ms": float(ms)}
        if op == "infer":
            if not self.loaded or self.backend is None:
                raise RuntimeError("backend is not loaded yet")
            raw_items = req.get("items")
            if not isinstance(raw_items, list) or not raw_items:
                raise RequestError("infer needs a non-empty 'items' list")
            items = [decode_item(it, i) for i, it in enumerate(raw_items)]
            t0 = time.monotonic()
            chunks = await loop.run_in_executor(self._executor, self.backend.infer, items)
            ms = (time.monotonic() - t0) * 1e3
            if not isinstance(chunks, (list, tuple)) or len(chunks) != len(items):
                raise RuntimeError(f"backend returned {len(chunks) if isinstance(chunks, (list, tuple)) else type(chunks)}"
                                   f" chunks for {len(items)} items")
            return {"chunks": [encode_chunk(c, i) for i, c in enumerate(chunks)], "ms": ms}
        raise RequestError(f"unknown op {op!r} (expected info | warmup | infer)")


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def parse_backend_args(pairs: list[str]) -> dict[str, Any]:
    """``key=value`` pairs -> kwargs. Values are parsed as JSON when possible (numbers, true/false, null,
    lists, objects), otherwise kept as strings."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--backend-arg expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key.isidentifier():
            raise ValueError(f"--backend-arg key {key!r} is not a valid identifier")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="b1k26-worker", description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", required=True, help="registry name (backends.base._REGISTRY), e.g. fake_sine")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--checkpoint", default=None, help="passed to the backend as checkpoint=")
    p.add_argument("--backend-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="backend constructor argument (repeatable; value parsed as JSON when possible)")
    p.add_argument("--backend-kwargs", default=None, metavar="JSON",
                   help="backend constructor arguments as one JSON object (merged before --backend-arg)")
    p.add_argument("--no-warmup", action="store_true", help="report healthy without a warmup inference")
    p.add_argument("--log-level", default=os.environ.get("B1K26_LOG_LEVEL", "INFO"))
    return p


def _start_parent_watchdog() -> None:
    """Exit when the launching front server dies (also covered by PR_SET_PDEATHSIG on Linux)."""
    raw = os.environ.get(PARENT_PID_ENV)
    if not raw:
        return
    try:
        parent = int(raw)
    except ValueError:
        return

    def watch() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                logger.warning("parent process %d is gone; worker exiting", parent)
                os._exit(0)

    threading.Thread(target=watch, name="b1k26-parent-watchdog", daemon=True).start()


def backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Constructor kwargs from --backend-kwargs, --backend-arg and --checkpoint. Raises ValueError."""
    kwargs: dict[str, Any] = {}
    if args.backend_kwargs:
        try:
            parsed = json.loads(args.backend_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(f"--backend-kwargs is not valid JSON: {e}") from None
        if not isinstance(parsed, dict):
            raise ValueError("--backend-kwargs must be a JSON object")
        kwargs.update(parsed)
    kwargs.update(parse_backend_args(args.backend_arg))
    if args.checkpoint is not None:
        kwargs["checkpoint"] = args.checkpoint
    return kwargs


async def _amain(args: argparse.Namespace, kwargs: dict[str, Any]) -> int:
    server = WorkerServer(lambda: create_backend(args.backend, **kwargs), host=args.host, port=args.port,
                          warmup=not args.no_warmup, name=args.backend)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-POSIX
            pass
    await server.start()
    load_task = asyncio.create_task(server.load())
    stop_task = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({load_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    rc = 0
    if load_task in done and load_task.exception() is not None:
        exc = load_task.exception()
        rc = EXIT_CONFIG_ERROR if isinstance(exc, CONFIG_ERRORS) else EXIT_LOAD_ERROR
        if rc == EXIT_CONFIG_ERROR:
            logger.error("configuration error (%s): relaunching this worker cannot fix it", type(exc).__name__)
    elif stop_task not in done:
        await stop_task
    else:
        load_task.cancel()
    logger.info("worker %s shutting down (rc=%d)", args.backend, rc)
    await server.stop()
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from b1k26.backends.base import _REGISTRY

    if args.backend not in _REGISTRY:  # exit status 2: a configuration error the front server does not retry
        parser.error(f"unknown backend {args.backend!r}; known: {sorted(_REGISTRY)}")
    try:
        kwargs = backend_kwargs(args)
    except ValueError as e:
        parser.error(str(e))
    logging.basicConfig(level=args.log_level.upper(),
                        format=f"%(asctime)s [worker:{args.backend}:{args.port}] %(levelname)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    _start_parent_watchdog()
    try:
        rc = asyncio.run(_amain(args, kwargs))
    except KeyboardInterrupt:
        rc = 130
    # Backends may leave non-daemon threads (JAX/torch); do not hang on interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
