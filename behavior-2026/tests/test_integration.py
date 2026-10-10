"""Cross-module consistency of the whole package: console scripts, backend registry, shipped configs vs backends and
the Docker env layout, the serve --check pre-flight, and the end-to-end smoke script (scripts/smoke_local.sh)."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

from b1k26.backends import base
from b1k26.config import load_config

from runtime_helpers import free_port

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))

# Nominal chunk horizons of the model families (checkpoint configs; overridable with the action_horizon kwarg).
NOMINAL_HORIZON = {"openpi_comet": 32, "openpi_b1k": 32, "pibehavior": 30, "gr00t": 16}
# docker/install_envs.sh env names -> (env script, backend registry names it can host)
DOCKER_ENVS = {
    "openpi_comet": ("openpi_comet.sh", {"openpi_comet"}),
    "openpi_b1k": ("openpi_b1k.sh", {"openpi_b1k"}),
    "gr00t": ("gr00t.sh", {"gr00t"}),
    "pibehavior-2025": ("pibehavior.sh --fork rlc2025", {"pibehavior"}),
    "pibehavior-2026": ("pibehavior.sh --fork jackliu2026", {"pibehavior"}),
}


# ------------------------------------------------------------------------------------------------------------
# Console scripts and registry
# ------------------------------------------------------------------------------------------------------------
def _console_scripts() -> dict[str, str]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["scripts"]


@pytest.mark.parametrize("name", sorted(_console_scripts()))
def test_console_scripts_resolve_and_show_help(name: str, capsys: pytest.CaptureFixture) -> None:
    target = _console_scripts()[name]
    module, func = target.split(":")
    main = getattr(importlib.import_module(module), func)
    assert callable(main)
    params = inspect.signature(main).parameters
    assert "argv" in params, f"{target} must accept argv for testing"
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_registry_names_match_backend_classes() -> None:
    assert set(base._REGISTRY) == {"fake_hold", "fake_sine", "fake_replay", *NOMINAL_HORIZON}
    for name, target in base._REGISTRY.items():
        module, cls_name = target.split(":")
        cls = getattr(importlib.import_module(module), cls_name)  # light import: heavy deps load in __init__
        assert issubclass(cls, base.Backend), target
        assert cls.flavor == name, f"{target}.flavor is {cls.flavor!r}, registry name is {name!r}"
    with pytest.raises(KeyError):
        base.create_backend("no_such_backend")


def test_fake_backends_honor_the_info_contract() -> None:
    keys = {"flavor", "action_horizon", "image_size", "num_stages", "supports_inpaint", "supports_stage"}
    for name in ("fake_hold", "fake_sine"):
        info = base.create_backend(name, horizon=24).info()
        assert keys <= set(info) and info["flavor"] == name and info["action_horizon"] == 24
        json.dumps(info)  # travels over the worker protocol


# ------------------------------------------------------------------------------------------------------------
# Shipped configs vs backends and the Docker layout
# ------------------------------------------------------------------------------------------------------------
def _accepted_kwargs(cls: type) -> tuple[set[str], bool]:
    """Constructor keyword names over the class hierarchy, and whether **kwargs is accepted all the way up."""
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        params = inspect.signature(init).parameters
        names |= {n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return names - {"self"}, False
    return names - {"self"}, True


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_config_matches_backends_and_docker_layout(path: pathlib.Path) -> None:
    cfg = load_config(path)
    for wname in cfg.used_workers():
        w = cfg.workers[wname]
        assert w.launch, f"{path.name}: worker {wname} has no launch command"
        argv = w.launch
        assert argv[1:3] == ["-m", "b1k26.worker"], argv
        backend = argv[argv.index("--backend") + 1]
        assert backend in base._REGISTRY, backend
        module, cls_name = base._REGISTRY[backend].split(":")
        cls = getattr(importlib.import_module(module), cls_name)
        kwargs = json.loads(argv[argv.index("--backend-kwargs") + 1]) if "--backend-kwargs" in argv else {}
        for i, a in enumerate(argv):
            if a == "--backend-arg":
                key, _, value = argv[i + 1].partition("=")
                kwargs[key] = value
        accepted, open_ended = _accepted_kwargs(cls)
        if not open_ended:
            assert set(kwargs) <= accepted, f"{path.name}: unknown {backend} kwargs {set(kwargs) - accepted}"
        # Interpreter: the front server's own ({python}) for fake backends, else a Docker-layout env.
        if backend.startswith("fake_"):
            assert argv[0] == sys.executable
        else:
            m = re.fullmatch(r"/opt/envs/([^/]+)/venv/bin/python", argv[0])
            assert m, f"{path.name}: worker {wname} interpreter {argv[0]} is not /opt/envs/<name>/venv/bin/python"
            assert m.group(1) in DOCKER_ENVS, f"{m.group(1)} is not an env name of docker/install_envs.sh"
            assert backend in DOCKER_ENVS[m.group(1)][1], f"env {m.group(1)} cannot host backend {backend}"
        # Every profile on this worker fits the model's chunk.
        horizon = int(kwargs.get("action_horizon") or NOMINAL_HORIZON.get(backend) or int(kwargs.get("horizon", 32)))
        for prof in cfg.profiles.values():
            if prof.worker != wname:
                continue
            ex = prof.execution
            assert prof.image_size == 224
            assert ex.execute_steps <= horizon, (path.name, prof.name)
            assert min(ex.predicted_steps_to_use, horizon) + ex.keep_for_inpaint <= horizon, (path.name, prof.name)


def test_docker_env_names_map_to_env_scripts() -> None:
    out = subprocess.run(["bash", str(REPO / "docker" / "install_envs.sh"), "--dry-run", "--root", "/x",
                          *DOCKER_ENVS], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("[install_envs]")]
    assert len(lines) == len(DOCKER_ENVS)
    for line, (name, (script, _)) in zip(lines, DOCKER_ENVS.items()):
        script_name, *extra = script.split()
        assert f"scripts/envs/{script_name} --prefix /x/{name}" in line, line
        assert " ".join(extra) in line
        assert (REPO / "scripts" / "envs" / script_name).is_file()
    bad = subprocess.run(["bash", str(REPO / "docker" / "install_envs.sh"), "--dry-run", "pibehavior"],
                         capture_output=True, text=True, timeout=30)
    assert bad.returncode == 2 and "unknown env name" in bad.stderr


def test_dockerfile_layout() -> None:
    text = (REPO / "docker" / "Dockerfile").read_text()
    # The env scripts install <dir containing behavior-2026>/behavior-2026: the package must keep that name.
    assert "COPY context/b1k26 /opt/behavior-2026" in text
    assert "docker/install_envs.sh" in text
    # Nothing needed at run time may live under /tmp (it can be an empty tmpfs under enroot or --tmpfs).
    assert re.search(r"UV_PYTHON_INSTALL_DIR=/opt/", text)
    assert re.search(r"HF_HOME=/opt/", text)
    assert "b1k26-serve --config /config/serve.yaml --check" in text
    assert re.search(r'ENTRYPOINT \["/opt/envs/front/bin/b1k26-serve", "--config", "/config/serve.yaml"\]', text)


# ------------------------------------------------------------------------------------------------------------
# serve --check and the smoke script
# ------------------------------------------------------------------------------------------------------------
def _serve(*args: str, env: dict | None = None, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "b1k26.server", *args], capture_output=True, text=True,
                          timeout=timeout, env=env)


def test_serve_check(tmp_path: pathlib.Path) -> None:
    ok = _serve("--config", str(REPO / "configs" / "fake.yaml"), "--check")
    assert ok.returncode == 0, ok.stderr
    assert "default profile fake" in ok.stdout
    doc = yaml.safe_load((REPO / "configs" / "fake.yaml").read_text())
    doc["workers"]["fake"]["launch"][0] = "/nonexistent/venv/bin/python"
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    bad = _serve("--config", str(cfg), "--check")
    assert bad.returncode == 2 and "/nonexistent/venv/bin/python" in bad.stderr
    clash = _serve("--config", str(REPO / "configs" / "fake.yaml"), "--ports", "9100", "--check")
    assert clash.returncode == 2 and "overlaps" in clash.stderr


def test_smoke_local_script(tmp_path: pathlib.Path) -> None:
    """scripts/smoke_local.sh end to end (front server CLI + launched fake worker + probes), on free ports."""
    doc = yaml.safe_load((REPO / "configs" / "fake.yaml").read_text())
    doc["workers"]["fake"]["port"] = free_port()
    cfg = tmp_path / "fake.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    ports = ",".join(str(free_port()) for _ in range(3))
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHON"] = sys.executable
    out = subprocess.run(["bash", str(REPO / "scripts" / "smoke_local.sh"), "--config", str(cfg), "--ports", ports,
                          "--steps", "60", "--out", str(tmp_path / "smoke")], capture_output=True, text=True,
                         timeout=300, env=env)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "[smoke] PASSED" in out.stdout
    for name in ("full_chunk", "full_step", "multi_a", "multi_b", "batched_224"):
        summary = json.loads((tmp_path / "smoke" / f"{name}.json").read_text())
        assert summary["ok"] and summary["num_violations"] == 0, (name, summary)


# ------------------------------------------------------------------------------------------------------------
# A worker that serves only some tasks, end to end through the worker protocol
# ------------------------------------------------------------------------------------------------------------
def test_partial_worker_routes_unserved_tasks_to_default_over_the_wire() -> None:
    import copy

    import requests

    from b1k26.backends.fake import SineBackend
    from b1k26.client import run_probe

    from runtime_helpers import Harness, base_doc

    class FirstHalf(SineBackend):
        """Like a 2025 RLC pibehavior worker: serves tasks 0-49 and rejects any batch containing another task."""

        def info(self):
            return {**super().info(), "supported_tasks": list(range(50))}

        def infer(self, items):
            if any(it.task_id >= 50 for it in items):
                raise ValueError("task not served")
            return super().infer(items)

    doc = base_doc()
    doc["workers"]["half"] = {}
    doc["profiles"]["q"] = copy.deepcopy(doc["profiles"]["p"]) | {"worker": "half"}
    doc["routing"]["per_task"] = {10: "q", 60: "q"}  # 60 is a mistake the engine must absorb
    h = Harness(doc, workers={"w": ("fake_sine", {"horizon": 32}), "half": lambda: FirstHalf(horizon=32)}).start()
    try:
        status = requests.get(f"http://127.0.0.1:{h.ports[0]}/status", timeout=2).json()
        problems = status["engine"]["config_problems"]
        assert any("does not serve 1 routed task(s) [60]" in p for p in problems), problems
        res = run_probe("127.0.0.1", h.ports[0], steps=40, batch=2, chunk=20, res="224", task_ids=[10, 60],
                        health_timeout_s=5)
        assert res.ok, res.violations
        groups = h.run(lambda: h.server.groups[h.ports[0]])
        sessions = h.run(lambda: {b: (s.profile_name, s.stats.plans, s.stats.plan_failures, s.stats.hold_steps)
                                  for g in groups for b, s in g.sessions.items()})
        assert sessions[0] == ("q", 2, 0, 0) and sessions[1] == ("p", 2, 0, 0), sessions
        assert h.backend("half").items_seen == 2  # only the two plans of task 10 reached this worker
    finally:
        h.stop()
