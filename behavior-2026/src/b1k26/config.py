"""Serving configuration: YAML schema, dataclasses, loading and validation.

A config has five sections (only ``workers``, ``profiles`` and ``routing`` are required)::

    server:                       # front server (b1k26.server)
      host: 0.0.0.0
      ports: "8000-8049"          # int, "8000", "8000-8049", "8000,8002", or a list of ints/ranges
      health_requires_warm: true  # /healthz is 503 until every routed worker has finished its first start attempt
                                  # (ready, or failed and handed to background retries) and the default one is ready
    engine:                       # planning (b1k26.engine); every key is optional
      plan_timeout_s: 120         # one plan request (queueing + inference); on timeout the step gets a hold action
      restart_wait_s: 420         # a query whose worker is (re)starting waits this long for it before holding;
                                  # restart_wait_s + plan_timeout_s must stay under the 2026/eval 600 s query cap
      max_batch: 8                # micro-batch size per worker request (a worker may lower it)
      batch_wait_ms: 5            # how long an idle worker waits for more plan requests before sending a batch
      gripper_rules: null         # path to a gripper_rules.json (default: the packaged one)
      fallback_to_default: true   # use the default profile while a routed profile's worker is down (backoff/failed)
    workers:
      comet_pt50:
        endpoint: ws://127.0.0.1:9101   # connect to this worker, and/or
        launch: ["{python}", "-m", "b1k26.worker", "--backend", "fake_sine", "--port", "{port}"]
        port: 9101                       # required with launch unless the endpoint gives it; if it is taken
                                         # (another server on this host), a free loopback port is used instead
        startup_timeout_s: 900           # no /healthz answer by then -> relaunch; still loading (503) -> up to 3x
        env: {CUDA_VISIBLE_DEVICES: "0"} # extra environment for the launched process
        cwd: null
        max_restarts: 3                  # immediate relaunches of a worker that died, at most this many ...
        restart_window_s: 3600           # ... within any window of this length; beyond that, relaunch with backoff
        restart_backoff_s: 10            # first backoff delay, doubled after each failed relaunch ...
        restart_backoff_max_s: 300       # ... up to this (a launched worker is never given up, except on a
                                         # configuration error: missing program, worker exit status 2)
        max_batch: null                  # per-worker override of engine.max_batch
    profiles:
      comet:
        worker: comet_pt50
        image_size: 224
        resize: bilinear_pad
        mask_base_qvel: true
        prompt: comet2025                # comet2025 | instruction | snake_case
        execution: {execute_steps: 32, predicted_steps_to_use: 32, keep_for_inpaint: 0}
        corrections: true
        use_stage: false
        cameras: [head, left_wrist, right_wrist]
    routing:
      default: comet
      per_task: {turning_on_radio: comet, 12: comet}   # task name or id -> profile

In ``launch`` argv entries, ``{python}`` expands to the front server's interpreter and ``{port}`` to the worker's
port (a literal ``--port N`` equal to the worker port is treated like ``--port {port}``, so the port can move).
Unknown keys anywhere are errors, so a typo never silently falls back to a default.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

from b1k26 import constants as C
from b1k26.control import ExecutionConfig

PROMPT_STYLES = ("comet2025", "instruction", "snake_case")
CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
RESIZE_METHODS = ("bilinear", "nearest", "lanczos", "bicubic")
MAX_PORTS = 1024
QUERY_CAP_S = 600.0  # 2026/eval POLICY_RESPONSE_TIMEOUT: one query must be answered within this


class ConfigError(ValueError):
    """Invalid serving configuration (the message names the offending key)."""


# ----------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    ports: list[int] = field(default_factory=lambda: [8000])
    health_requires_warm: bool = True


@dataclass
class EngineConfig:
    plan_timeout_s: float = 120.0
    restart_wait_s: float = 420.0
    max_batch: int = 8
    batch_wait_ms: float = 5.0
    gripper_rules: str | None = None
    fallback_to_default: bool = True


@dataclass
class WorkerConfig:
    name: str
    endpoint: str  # ws://host:port, always set after validation
    launch: list[str] | None = None  # argv with placeholders expanded
    port: int | None = None
    startup_timeout_s: float = 900.0
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    max_restarts: int = 3  # immediate relaunches allowed within any restart_window_s (start-up relaunches count)
    restart_window_s: float = 3600.0
    restart_backoff_s: float = 10.0  # beyond max_restarts: relaunch after this delay, doubled each time ...
    restart_backoff_max_s: float = 300.0  # ... up to this
    max_batch: int | None = None
    # launch argv with "{port}" kept as a placeholder (also where a literal --port value was), for relocated()
    launch_template: list[str] | None = None

    @property
    def host(self) -> str:
        return urlparse(self.endpoint).hostname or "127.0.0.1"

    @property
    def relocatable(self) -> bool:
        """Whether the launched worker can be moved to another port: it listens on loopback and its argv carries
        the port (``{port}`` or ``--port N``)."""
        if not self.launch or not self.launch_template or not any("{port}" in a for a in self.launch_template):
            return False
        host = self.host
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def relocated(self, port: int) -> "WorkerConfig":
        """A copy of this launched worker's config on another port (argv, port and endpoint updated)."""
        if not self.relocatable or self.launch_template is None:
            raise ValueError(f"worker {self.name} cannot be relocated")
        u = urlparse(self.endpoint)
        host = u.hostname or "127.0.0.1"
        netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        endpoint = u._replace(netloc=netloc).geturl()
        launch = [a.replace("{port}", str(port)) for a in self.launch_template]
        return dataclasses.replace(self, endpoint=endpoint, launch=launch, port=int(port))

    @property
    def endpoint_port(self) -> int:
        port = urlparse(self.endpoint).port
        assert port is not None
        return port


@dataclass
class Profile:
    name: str
    worker: str
    image_size: int = 224
    resize: str = "bilinear_pad"
    mask_base_qvel: bool = False
    prompt: str = "instruction"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    corrections: bool = True
    use_stage: bool = False
    cameras: tuple[str, ...] = CAMERA_ROLES


@dataclass
class RoutingTable:
    default: str
    per_task: dict[int, str] = field(default_factory=dict)  # task id -> profile name

    def profile_for(self, task_id: int) -> str:
        return self.per_task.get(int(task_id), self.default)


@dataclass
class Config:
    server: ServerConfig
    engine: EngineConfig
    workers: dict[str, WorkerConfig]
    profiles: dict[str, Profile]
    routing: RoutingTable
    source: str | None = None

    def profile_for(self, task_id: int) -> Profile:
        return self.profiles[self.routing.profile_for(task_id)]

    @property
    def default_profile(self) -> Profile:
        return self.profiles[self.routing.default]

    def used_workers(self) -> list[str]:
        """Workers referenced by a profile that routing can reach (default first)."""
        names = [self.routing.default, *self.routing.per_task.values()]
        out: list[str] = []
        for p in names:
            w = self.profiles[p].worker
            if w not in out:
                out.append(w)
        return out


# ----------------------------------------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------------------------------------
def _check_keys(where: str, raw: Mapping[str, Any], allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> None:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    unknown = sorted(str(k) for k in raw if k not in allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {list(allowed)}")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")


def _bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    return value


def _int(where: str, value: Any, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        else:
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if lo is not None and value < lo:
        raise ConfigError(f"{where}: must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ConfigError(f"{where}: must be <= {hi}, got {value}")
    return int(value)


def _float(where: str, value: Any, lo: float | None = None, lo_inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    v = float(value)
    if v != v:
        raise ConfigError(f"{where}: NaN is not allowed")
    if lo is not None and (v < lo or (not lo_inclusive and v == lo)):
        raise ConfigError(f"{where}: must be {'>=' if lo_inclusive else '>'} {lo}, got {v}")
    return v


def _str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def parse_ports(spec: Any, where: str = "ports") -> list[int]:
    """Port spec -> sorted unique list. Accepts 8000, "8000", "8000-8049", "8000,8002-8003" or a list of those.

    Port 0 (an ephemeral port chosen by the OS) is accepted for tests.
    """
    items: list[Any]
    if isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        items = [spec]
    out: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ConfigError(f"{where}: invalid port {item!r}")
        if isinstance(item, int):
            out.append(_int(where, item, 0, 65535))
            continue
        if not isinstance(item, str):
            raise ConfigError(f"{where}: invalid port {item!r}")
        for part in item.split(","):
            part = part.strip()
            if not part:
                raise ConfigError(f"{where}: empty entry in {item!r}")
            if "-" in part:
                lo_s, _, hi_s = part.partition("-")
                lo, hi = _int(where, lo_s.strip(), 1, 65535), _int(where, hi_s.strip(), 1, 65535)
                if hi < lo:
                    raise ConfigError(f"{where}: range {part!r} is reversed")
                out.extend(range(lo, hi + 1))
            else:
                out.append(_int(where, part, 0, 65535))
    if not out:
        raise ConfigError(f"{where}: no ports given")
    zeros = [p for p in out if p == 0]
    nonzero = sorted(set(p for p in out if p != 0))
    if len(nonzero) + len(zeros) > MAX_PORTS:
        raise ConfigError(f"{where}: more than {MAX_PORTS} ports")
    return nonzero + zeros


def _expand_argv(argv: list[str], port: int | None) -> list[str]:
    out = []
    for a in argv:
        a = a.replace("{python}", sys.executable)
        if "{port}" in a:
            if port is None:
                raise ConfigError("launch uses {port} but the worker has no port")
            a = a.replace("{port}", str(port))
        out.append(a)
    return out


def _launch_template(argv: list[str], port: int) -> list[str]:
    """argv with {python} expanded and the port kept as "{port}" (a literal value after --port included)."""
    out = [a.replace("{python}", sys.executable) for a in argv]
    for i in range(1, len(out)):
        if out[i - 1] == "--port" and out[i] == str(port):
            out[i] = "{port}"
    return out


def _parse_server(raw: Any) -> ServerConfig:
    raw = raw or {}
    _check_keys("server", raw, ("host", "ports", "health_requires_warm"))
    cfg = ServerConfig()
    if "host" in raw:
        cfg.host = _str("server.host", raw["host"])
    if "ports" in raw:
        cfg.ports = parse_ports(raw["ports"], "server.ports")
    if "health_requires_warm" in raw:
        cfg.health_requires_warm = _bool("server.health_requires_warm", raw["health_requires_warm"])
    return cfg


def _parse_engine(raw: Any, base_dir: str | None) -> EngineConfig:
    raw = raw or {}
    _check_keys("engine", raw, ("plan_timeout_s", "restart_wait_s", "max_batch", "batch_wait_ms", "gripper_rules",
                                "fallback_to_default"))
    cfg = EngineConfig()
    if "plan_timeout_s" in raw:
        cfg.plan_timeout_s = _float("engine.plan_timeout_s", raw["plan_timeout_s"], 0.0, lo_inclusive=False)
    if "restart_wait_s" in raw:
        cfg.restart_wait_s = _float("engine.restart_wait_s", raw["restart_wait_s"], 0.0)
    if cfg.restart_wait_s + cfg.plan_timeout_s >= QUERY_CAP_S - 10:
        raise ConfigError(f"engine: restart_wait_s ({cfg.restart_wait_s:g}) + plan_timeout_s ({cfg.plan_timeout_s:g}) "
                          f"must stay below {QUERY_CAP_S - 10:g} s: 2026/eval fails a rollout whose query takes "
                          f"{QUERY_CAP_S:g} s")
    if "max_batch" in raw:
        cfg.max_batch = _int("engine.max_batch", raw["max_batch"], 1, 256)
    if "batch_wait_ms" in raw:
        cfg.batch_wait_ms = _float("engine.batch_wait_ms", raw["batch_wait_ms"], 0.0)
    if raw.get("gripper_rules") is not None:
        path = _str("engine.gripper_rules", raw["gripper_rules"])
        if base_dir and not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        cfg.gripper_rules = path
    if "fallback_to_default" in raw:
        cfg.fallback_to_default = _bool("engine.fallback_to_default", raw["fallback_to_default"])
    return cfg


def _parse_endpoint(where: str, value: Any) -> tuple[str, int]:
    ep = _str(where, value)
    u = urlparse(ep)
    if u.scheme not in ("ws", "wss") or not u.hostname:
        raise ConfigError(f"{where}: expected ws://host:port, got {ep!r}")
    try:
        port = u.port
    except ValueError:
        port = None
    if port is None:
        raise ConfigError(f"{where}: endpoint {ep!r} has no port")
    return ep, port


def _parse_worker(name: str, raw: Any) -> WorkerConfig:
    where = f"workers.{name}"
    _check_keys(where, raw, ("endpoint", "launch", "port", "startup_timeout_s", "env", "cwd", "max_restarts",
                             "restart_window_s", "restart_backoff_s", "restart_backoff_max_s", "max_batch"))
    endpoint = None
    ep_port = None
    if raw.get("endpoint") is not None:
        endpoint, ep_port = _parse_endpoint(f"{where}.endpoint", raw["endpoint"])
    port = None
    if raw.get("port") is not None:
        port = _int(f"{where}.port", raw["port"], 1, 65535)
    if port is not None and ep_port is not None and port != ep_port:
        raise ConfigError(f"{where}: port {port} disagrees with endpoint {endpoint!r}")
    port = port if port is not None else ep_port

    launch = None
    template = None
    if raw.get("launch") is not None:
        argv = raw["launch"]
        if not isinstance(argv, list) or not argv or not all(isinstance(a, (str, int, float)) for a in argv):
            raise ConfigError(f"{where}.launch: expected a non-empty list of strings (argv)")
        if port is None:
            raise ConfigError(f"{where}: launch needs a port (set `port:` or an `endpoint:` with a port)")
        launch = _expand_argv([str(a) for a in argv], port)
        template = _launch_template([str(a) for a in argv], port)
        if "--port" in launch:
            i = launch.index("--port")
            if i + 1 < len(launch) and launch[i + 1].isdigit() and int(launch[i + 1]) != port:
                raise ConfigError(f"{where}: launch passes --port {launch[i + 1]} but the worker port is {port}")
    if endpoint is None:
        if port is None:
            raise ConfigError(f"{where}: needs `endpoint:` or `launch:` + `port:`")
        endpoint = f"ws://127.0.0.1:{port}"

    cfg = WorkerConfig(name=name, endpoint=endpoint, launch=launch, port=port, launch_template=template)
    if "startup_timeout_s" in raw:
        cfg.startup_timeout_s = _float(f"{where}.startup_timeout_s", raw["startup_timeout_s"], 0.0, lo_inclusive=False)
    if raw.get("env") is not None:
        env = raw["env"]
        if not isinstance(env, Mapping):
            raise ConfigError(f"{where}.env: expected a mapping")
        cfg.env = {str(k): str(v) for k, v in env.items()}
    if raw.get("cwd") is not None:
        cfg.cwd = _str(f"{where}.cwd", raw["cwd"])
    if "max_restarts" in raw:
        cfg.max_restarts = _int(f"{where}.max_restarts", raw["max_restarts"], 0, 100)
    if "restart_window_s" in raw:
        cfg.restart_window_s = _float(f"{where}.restart_window_s", raw["restart_window_s"], 0.0, lo_inclusive=False)
    if "restart_backoff_s" in raw:
        cfg.restart_backoff_s = _float(f"{where}.restart_backoff_s", raw["restart_backoff_s"], 0.0,
                                       lo_inclusive=False)
    if "restart_backoff_max_s" in raw:
        cfg.restart_backoff_max_s = _float(f"{where}.restart_backoff_max_s", raw["restart_backoff_max_s"], 0.0,
                                           lo_inclusive=False)
    if cfg.restart_backoff_max_s < cfg.restart_backoff_s:
        raise ConfigError(f"{where}: restart_backoff_max_s ({cfg.restart_backoff_max_s:g}) < restart_backoff_s "
                          f"({cfg.restart_backoff_s:g})")
    if raw.get("max_batch") is not None:
        cfg.max_batch = _int(f"{where}.max_batch", raw["max_batch"], 1, 256)
    return cfg


_EXEC_FIELDS = tuple(f.name for f in dataclasses.fields(ExecutionConfig))


def _parse_profile(name: str, raw: Any, workers: Mapping[str, WorkerConfig]) -> Profile:
    where = f"profiles.{name}"
    _check_keys(where, raw, ("worker", "image_size", "resize", "mask_base_qvel", "prompt", "execution",
                             "corrections", "use_stage", "cameras"), required=("worker",))
    worker = _str(f"{where}.worker", raw["worker"])
    if worker not in workers:
        raise ConfigError(f"{where}.worker: unknown worker {worker!r}; defined: {sorted(workers)}")
    p = Profile(name=name, worker=worker)
    if "image_size" in raw:
        p.image_size = _int(f"{where}.image_size", raw["image_size"], 16, 4096)
    if "resize" in raw:
        r = _str(f"{where}.resize", raw["resize"]).lower()
        base = r[: -len("_pad")] if r.endswith("_pad") else r
        if base not in RESIZE_METHODS:
            raise ConfigError(f"{where}.resize: unknown method {r!r}; known: {list(RESIZE_METHODS)} (+ '_pad')")
        p.resize = r
    if "mask_base_qvel" in raw:
        p.mask_base_qvel = _bool(f"{where}.mask_base_qvel", raw["mask_base_qvel"])
    if "prompt" in raw:
        pr = _str(f"{where}.prompt", raw["prompt"])
        if pr not in PROMPT_STYLES:
            raise ConfigError(f"{where}.prompt: unknown style {pr!r}; known: {list(PROMPT_STYLES)}")
        p.prompt = pr
    if raw.get("execution") is not None:
        ex = raw["execution"]
        _check_keys(f"{where}.execution", ex, _EXEC_FIELDS)
        try:
            p.execution = ExecutionConfig(**dict(ex))
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{where}.execution: {e}") from None
    if "corrections" in raw:
        p.corrections = _bool(f"{where}.corrections", raw["corrections"])
    if "use_stage" in raw:
        p.use_stage = _bool(f"{where}.use_stage", raw["use_stage"])
    if raw.get("cameras") is not None:
        cams = raw["cameras"]
        if not isinstance(cams, list) or not cams or any(c not in CAMERA_ROLES for c in cams) or len(set(cams)) != len(cams):
            raise ConfigError(f"{where}.cameras: expected a non-empty list of distinct roles from {list(CAMERA_ROLES)}")
        p.cameras = tuple(cams)
    return p


def _task_id_from_key(where: str, key: Any) -> int:
    if isinstance(key, bool):
        raise ConfigError(f"{where}: invalid task {key!r}")
    if isinstance(key, int) or (isinstance(key, str) and key.strip().isdigit()):
        tid = int(key)
        if not 0 <= tid < C.NUM_TASKS:
            raise ConfigError(f"{where}: task id {tid} outside [0, {C.NUM_TASKS})")
        return tid
    if isinstance(key, str) and key in C.task_by_name():
        return C.task_by_name()[key].task_id
    raise ConfigError(f"{where}: unknown task {key!r} (use a task name from constants or an id 0-99)")


def _parse_routing(raw: Any, profiles: Mapping[str, Profile]) -> RoutingTable:
    _check_keys("routing", raw, ("default", "per_task"), required=("default",))
    default = _str("routing.default", raw["default"])
    if default not in profiles:
        raise ConfigError(f"routing.default: unknown profile {default!r}; defined: {sorted(profiles)}")
    per_task: dict[int, str] = {}
    pt = raw.get("per_task") or {}
    if not isinstance(pt, Mapping):
        raise ConfigError("routing.per_task: expected a mapping task -> profile")
    for key, prof in pt.items():
        where = f"routing.per_task[{key!r}]"
        tid = _task_id_from_key(where, key)
        prof = _str(where, prof)
        if prof not in profiles:
            raise ConfigError(f"{where}: unknown profile {prof!r}; defined: {sorted(profiles)}")
        if tid in per_task and per_task[tid] != prof:
            raise ConfigError(f"{where}: task {tid} is routed twice ({per_task[tid]!r} and {prof!r})")
        per_task[tid] = prof
    return RoutingTable(default=default, per_task=per_task)


def parse_config(doc: Any, source: str | None = None) -> Config:
    """Validate a config document (the parsed YAML) and build a Config. Raises ConfigError."""
    if not isinstance(doc, Mapping):
        raise ConfigError("config: expected a mapping at the top level")
    _check_keys("config", doc, ("server", "engine", "workers", "profiles", "routing"),
                required=("workers", "profiles", "routing"))
    base_dir = os.path.dirname(os.path.abspath(source)) if source else None
    server = _parse_server(doc.get("server"))
    engine = _parse_engine(doc.get("engine"), base_dir)
    raw_workers = doc["workers"]
    if not isinstance(raw_workers, Mapping) or not raw_workers:
        raise ConfigError("workers: expected a non-empty mapping")
    workers = {str(n): _parse_worker(str(n), w) for n, w in raw_workers.items()}
    launched_ports: dict[int, str] = {}
    for w in workers.values():
        if w.launch is not None and w.port is not None:
            if w.port in launched_ports:
                raise ConfigError(f"workers.{w.name}: port {w.port} is also used by {launched_ports[w.port]!r}")
            launched_ports[w.port] = w.name
            if w.port in server.ports:
                raise ConfigError(f"workers.{w.name}: port {w.port} collides with a server port")
    raw_profiles = doc["profiles"]
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ConfigError("profiles: expected a non-empty mapping")
    profiles = {str(n): _parse_profile(str(n), p, workers) for n, p in raw_profiles.items()}
    routing = _parse_routing(doc["routing"], profiles)
    return Config(server=server, engine=engine, workers=workers, profiles=profiles, routing=routing, source=source)


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load and validate a YAML config file. Raises ConfigError (or OSError if the file is unreadable)."""
    path = os.fspath(path)
    with open(path) as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: invalid YAML: {e}") from None
    try:
        return parse_config(doc, source=path)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def resolve_prompt(style: str, task_id: int) -> str:
    """Prompt text for a task under a profile's prompt style (see PROMPT_STYLES)."""
    t = C.task(int(task_id))
    if style == "snake_case":
        return t.name
    if style == "comet2025":
        return t.instruction_comet2025 or t.instruction
    if style == "instruction":
        return t.instruction
    raise ValueError(f"unknown prompt style {style!r}")
