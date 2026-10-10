"""b1k26.config: YAML schema, validation and the shipped configs."""

from __future__ import annotations

import copy
import pathlib
import sys

import pytest
import yaml

from b1k26.config import ConfigError, load_config, parse_config, parse_ports
from b1k26.control import ExecutionConfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def doc() -> dict:
    return {
        "server": {"host": "0.0.0.0", "ports": "8000-8003", "health_requires_warm": True},
        "workers": {
            "a": {"endpoint": "ws://127.0.0.1:9101"},
            "b": {"launch": ["{python}", "-m", "b1k26.worker", "--port", "{port}"], "port": 9102,
                  "env": {"X": 1}, "max_restarts": 2, "max_batch": 4},
        },
        "profiles": {
            "pa": {"worker": "a", "execution": {"execute_steps": 32, "predicted_steps_to_use": 32,
                                                "keep_for_inpaint": 0}, "prompt": "comet2025",
                   "mask_base_qvel": True},
            "pb": {"worker": "b", "use_stage": True, "cameras": ["head"]},
        },
        "routing": {"default": "pa", "per_task": {"turning_on_radio": "pb", 12: "pb", "13": "pa"}},
    }


def test_parse_full_document() -> None:
    cfg = parse_config(doc())
    assert cfg.server.ports == [8000, 8001, 8002, 8003]
    assert cfg.workers["a"].endpoint == "ws://127.0.0.1:9101" and cfg.workers["a"].launch is None
    b = cfg.workers["b"]
    assert b.launch == [sys.executable, "-m", "b1k26.worker", "--port", "9102"]
    assert b.endpoint == "ws://127.0.0.1:9102" and b.env == {"X": "1"} and b.max_restarts == 2 and b.max_batch == 4
    assert cfg.profiles["pa"].execution == ExecutionConfig(execute_steps=32, predicted_steps_to_use=32,
                                                          keep_for_inpaint=0)
    assert cfg.profiles["pb"].execution == ExecutionConfig()
    assert cfg.profiles["pb"].cameras == ("head",)
    assert cfg.routing.per_task == {0: "pb", 12: "pb", 13: "pa"}
    assert cfg.profile_for(0).name == "pb" and cfg.profile_for(50).name == "pa"
    assert cfg.used_workers() == ["a", "b"]
    assert cfg.engine.plan_timeout_s == 120.0 and cfg.engine.max_batch == 8


@pytest.mark.parametrize("spec,expected", [
    (8000, [8000]), ("8000", [8000]), ("8000-8002", [8000, 8001, 8002]), ("8000,8005", [8000, 8005]),
    ([8001, "8003-8004", 8001], [8001, 8003, 8004]), ("8000-8049", list(range(8000, 8050))), ([0, 0], [0, 0]),
])
def test_parse_ports(spec, expected) -> None:
    assert parse_ports(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "8002-8000", "70000", -1, True, None, "8000-", [3.5]])
def test_parse_ports_rejects(spec) -> None:
    with pytest.raises(ConfigError):
        parse_ports(spec)


def _bad(mutate) -> str:
    d = doc()
    mutate(d)
    with pytest.raises(ConfigError) as e:
        parse_config(d)
    return str(e.value)


def test_unknown_keys_are_errors_at_every_level() -> None:
    assert "colour" in _bad(lambda d: d.update(colour=1))
    assert "prots" in _bad(lambda d: d["server"].update(prots=1))
    assert "endpont" in _bad(lambda d: d["workers"]["a"].update(endpont="x"))
    assert "img_size" in _bad(lambda d: d["profiles"]["pa"].update(img_size=1))
    assert "exec_steps" in _bad(lambda d: d["profiles"]["pa"]["execution"].update(exec_steps=1))
    assert "tasks" in _bad(lambda d: d["routing"].update(tasks={}))
    assert "timeout" in _bad(lambda d: d.update(engine={"timeout": 3}))


def test_references_must_exist() -> None:
    assert "unknown worker" in _bad(lambda d: d["profiles"]["pa"].update(worker="zz"))
    assert "unknown profile" in _bad(lambda d: d["routing"].update(default="zz"))
    assert "unknown profile" in _bad(lambda d: d["routing"]["per_task"].update({5: "zz"}))
    assert "unknown task" in _bad(lambda d: d["routing"]["per_task"].update({"not_a_task": "pa"}))
    assert "outside" in _bad(lambda d: d["routing"]["per_task"].update({100: "pa"}))
    assert "routed twice" in _bad(lambda d: d["routing"]["per_task"].update({0: "pa"}))


def test_worker_validation() -> None:
    assert "needs `endpoint:`" in _bad(lambda d: d["workers"].update(c={"startup_timeout_s": 3}))
    assert "launch needs a port" in _bad(lambda d: d["workers"]["b"].pop("port"))
    assert "disagrees" in _bad(lambda d: d["workers"]["b"].update(endpoint="ws://127.0.0.1:9999"))
    assert "launch passes --port" in _bad(
        lambda d: d["workers"]["b"].update(launch=["python", "-m", "b1k26.worker", "--port", "1234"]))
    assert "ws://host:port" in _bad(lambda d: d["workers"]["a"].update(endpoint="http://127.0.0.1:1"))
    assert "no port" in _bad(lambda d: d["workers"]["a"].update(endpoint="ws://127.0.0.1"))
    assert "collides" in _bad(lambda d: d["workers"]["b"].update(port=8001))
    assert "also used" in _bad(lambda d: d["workers"].update(
        c={"launch": ["x", "--port", "{port}"], "port": 9102}))
    # A worker given by port only is an external worker on localhost.
    d = doc()
    d["workers"]["a"] = {"port": 9300}
    assert parse_config(d).workers["a"].endpoint == "ws://127.0.0.1:9300"
    # launch + endpoint (as in the example configs): the endpoint gives the port.
    d = doc()
    d["workers"]["b"] = {"launch": ["x", "--port", "{port}"], "endpoint": "ws://127.0.0.1:9400"}
    w = parse_config(d).workers["b"]
    assert w.port == 9400 and w.launch == ["x", "--port", "9400"]
    # Restart budget: max_restarts per sliding restart_window_s (defaults 3 per hour).
    assert (w.max_restarts, w.restart_window_s) == (3, 3600.0)
    d["workers"]["b"]["restart_window_s"] = 60
    assert parse_config(d).workers["b"].restart_window_s == 60.0
    assert "restart_window_s" in _bad(lambda d: d["workers"]["b"].update(restart_window_s=0))


def test_profile_value_validation() -> None:
    assert "resize" in _bad(lambda d: d["profiles"]["pa"].update(resize="cubic_spline"))
    assert "prompt" in _bad(lambda d: d["profiles"]["pa"].update(prompt="fancy"))
    assert "true/false" in _bad(lambda d: d["profiles"]["pa"].update(corrections="yes"))
    assert "image_size" in _bad(lambda d: d["profiles"]["pa"].update(image_size=4))
    assert "execute_steps" in _bad(lambda d: d["profiles"]["pa"]["execution"].update(execute_steps=0))
    assert "cameras" in _bad(lambda d: d["profiles"]["pa"].update(cameras=["head", "head"]))
    assert "cameras" in _bad(lambda d: d["profiles"]["pa"].update(cameras=["chest"]))
    d = doc()
    d["profiles"]["pa"]["resize"] = "LANCZOS_PAD"
    assert parse_config(d).profiles["pa"].resize == "lanczos_pad"


def test_engine_section(tmp_path: pathlib.Path) -> None:
    d = doc()
    d["engine"] = {"plan_timeout_s": 30, "max_batch": 4, "batch_wait_ms": 0, "gripper_rules": "rules.json",
                   "fallback_to_default": False}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    cfg = load_config(p)
    assert cfg.engine.plan_timeout_s == 30 and cfg.engine.max_batch == 4 and cfg.engine.batch_wait_ms == 0
    assert cfg.engine.gripper_rules == str(tmp_path / "rules.json") and cfg.engine.fallback_to_default is False
    assert "plan_timeout_s" in _bad(lambda d: d.update(engine={"plan_timeout_s": 0}))


def test_load_config_errors(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("server: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)
    p.write_text("- just a list")
    with pytest.raises(ConfigError, match="top level"):
        load_config(p)
    p.write_text(yaml.safe_dump({"workers": {}, "profiles": {}, "routing": {"default": "x"}}))
    with pytest.raises(ConfigError, match=str(p)):
        load_config(p)


@pytest.mark.parametrize("path", sorted((REPO / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: pathlib.Path) -> None:
    cfg = load_config(path)
    assert cfg.routing.default in cfg.profiles
    for w in cfg.workers.values():
        assert w.endpoint.startswith("ws://")


def test_config_is_not_mutated_by_parsing() -> None:
    d = doc()
    before = copy.deepcopy(d)
    parse_config(d)
    assert d == before
