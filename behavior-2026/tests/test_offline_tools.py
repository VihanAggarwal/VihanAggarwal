"""Tests for scoring, selection, orchestration and packaging (pure CPU, no simulator)."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import pytest

from b1k26 import constants, orchestrate, package, scoring, selection


def _doc(task: str, inst: int, q: float, success: bool | None = None, rollout_id: int = 0, steps: int = 100) -> dict:
    return {
        "task": task, "instance_id": inst, "rollout_id": rollout_id, "steps": steps,
        "success": bool(q >= 1.0) if success is None else success,
        "agent_distance": {"base": 1.0, "left": 1.0, "right": 1.0},
        "normalized_agent_distance": {"base": 1.0, "left": 1.0, "right": 1.0},
        "q_score": {"final": q},
        "time": {"simulator_steps": steps, "simulator_time": steps / 30, "normalized_time": 1.0},
    }


def _write(dirpath: Path, doc: dict) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"{doc['task']}_{doc['instance_id']}_{doc['rollout_id']}.json"
    p.write_text(json.dumps(doc))
    return p


# ------------------------------------------------------------------------------------------ scoring
def test_leaderboard_and_official_q_agree_on_complete_submission(tmp_path):
    rng = random.Random(0)
    for t in constants.tasks():
        for inst in constants.REPORTED_INSTANCE_IDS:
            _write(tmp_path / "json", _doc(t.name, inst, rng.random()))
    rollouts = scoring.load_rollouts([tmp_path])
    lq, lsr = scoring.leaderboard_q(rollouts)
    oq, osr = scoring.official_q(rollouts)
    assert len(rollouts) == 1000
    assert lq == pytest.approx(oq)
    assert lsr == pytest.approx(osr)
    assert scoring.validate_submission(tmp_path / "json") == []


def test_missing_instances_differ_between_formulas(tmp_path):
    # One task, 5 of 10 instances present, all q=1: extractor gives 1/100, the written rule gives 0.5/100.
    for inst in range(301, 306):
        _write(tmp_path, _doc("turning_on_radio", inst, 1.0))
    rollouts = scoring.load_rollouts([tmp_path])
    assert scoring.leaderboard_q(rollouts)[0] == pytest.approx(0.01)
    assert scoring.official_q(rollouts)[0] == pytest.approx(0.005)


def test_validation_catches_bad_submissions(tmp_path):
    d = tmp_path / "json"
    _write(d, _doc("turning_on_radio", 301, 0.5))
    _write(d, _doc("turning_on_radio", 301, 0.5, rollout_id=1))  # second rollout of same instance
    _write(d, _doc("turning_on_radio", 315, 0.5))  # held-out instance in a submission
    bad = _doc("picking_up_trash", 302, 1.5)  # q out of range
    _write(d, bad)
    (d / "notes.json").write_text("{}")
    problems = scoring.validate_submission(d)
    text = "\n".join(problems)
    assert "rollout_id 1" in text
    assert "315" in text
    assert "not in [0, 1]" in text
    assert "file name is not" in text
    assert "missing" in text


def test_zip_input_and_cli(tmp_path, capsys):
    d = tmp_path / "json"
    _write(d, _doc("turning_on_radio", 301, 1.0))
    zpath = tmp_path / "metrics.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in d.glob("*.json"):
            zf.write(p, arcname=p.name)
    assert len(scoring.load_rollouts([zpath])) == 1
    assert scoring.main([str(zpath), "--per-task"]) == 0
    assert "turning_on_radio" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ selection
def _synthetic(cands: dict[str, float], task_bonus: dict[tuple[str, str], float], ids, seed=0, sigma=0.3):
    rng = random.Random(seed)
    out = {c: [] for c in cands}
    names = [t.name for t in constants.tasks()[:40]]
    for ti, task in enumerate(names):
        base = 0.1 + 0.6 * (ti % 7) / 7
        for c, strength in cands.items():
            mu = min(1, max(0, base + strength + task_bonus.get((c, task), 0.0)))
            for inst in ids:
                q = min(1.0, max(0.0, rng.gauss(mu, sigma)))
                out[c].append(scoring.Rollout(task, inst, 0, q >= 1.0, q, 100, 1.0, {}, {}, None, "synthetic"))
    return out


def test_selection_does_not_route_on_noise():
    data = _synthetic({"a": 0.0, "b": -0.02, "c": -0.02}, {}, ids=range(311, 313))
    fitted = selection.fit(data)
    decisions = selection.route(fitted, "a", margin=0.05, all_tasks=sorted({r.task for r in data["a"]}))
    switched = [d for d in decisions if d.chosen != "a"]
    # With 2 rollouts per cell and no true interaction, almost nothing should move.
    assert len(switched) <= 2, [(d.task, d.chosen, d.gain) for d in switched]


def test_selection_finds_large_real_interaction():
    task = constants.task(3).name
    data = _synthetic({"a": 0.0, "b": -0.05}, {("b", task): 0.6}, ids=range(311, 321), seed=1, sigma=0.15)
    fitted = selection.fit(data)
    decisions = {d.task: d for d in selection.route(fitted, "a", margin=0.05,
                                                     all_tasks=sorted({r.task for r in data["a"]}))}
    assert decisions[task].chosen == "b"
    cv = selection.split_half_gain(data, "a", 0.05, 2, None)
    assert cv["available"]


def test_selection_respects_allowed_and_refuses_reported(tmp_path):
    data = _synthetic({"a": 0.0, "old": 0.3}, {}, ids=range(311, 313))
    fitted = selection.fit(data)
    allowed = {"old": set(range(0, 50))}
    decisions = selection.route(fitted, "a", margin=0.05, allowed=allowed)
    for d in decisions:
        if d.task_id is not None and d.task_id >= 50:
            assert d.chosen != "old"
    # Rollouts on reported instances are refused by the CLI.
    _write(tmp_path / "x" / "json", _doc("turning_on_radio", 305, 1.0))
    rc = selection.main(["--candidate", f"a={tmp_path / 'x'}", "--default", "a",
                         "--out", str(tmp_path / "r.yaml"), "--log", str(tmp_path / "l.md")])
    assert rc == 2


def test_selection_cli_writes_routing(tmp_path):
    for cand, q in (("a", 0.4), ("b", 0.2)):
        for t in constants.tasks()[:5]:
            for inst in (311, 312, 313):
                _write(tmp_path / cand / "json", _doc(t.name, inst, q))
    rc = selection.main(["--candidate", f"a={tmp_path / 'a'}", "--candidate", f"b={tmp_path / 'b'}",
                         "--default", "a", "--out", str(tmp_path / "routing.yaml"), "--log", str(tmp_path / "log.md")])
    assert rc == 0
    text = (tmp_path / "routing.yaml").read_text()
    assert "default: a" in text
    assert "per_task: {}" in text
    assert "Route selection log" in (tmp_path / "log.md").read_text()


# ------------------------------------------------------------------------------------------ orchestrate
def test_plan_covers_every_job_once_and_balances(tmp_path):
    rc = orchestrate.main(["plan", "--workers", "20", "--instances", "0-9", "--out", str(tmp_path)])
    assert rc == 0
    jobs = [j for p in sorted(tmp_path.glob("worker_*.jsonl")) for j in orchestrate.load_jobs(p)]
    assert len(jobs) == 1000
    assert len({(j.task, j.index) for j in jobs}) == 1000
    assert {j.instance_id for j in jobs} == set(constants.REPORTED_INSTANCE_IDS)
    loads = [sum(j.max_steps for j in orchestrate.load_jobs(p)) for p in sorted(tmp_path.glob("worker_*.jsonl"))]
    assert max(loads) / (sum(loads) / len(loads)) < 1.1
    summary = json.loads((tmp_path / "plan_summary.json").read_text())
    assert summary["worst_case_env_steps"] == 15_818_280


def test_instance_mapping_and_task_spec():
    assert orchestrate.instance_id_for("public_test", 0) == 301
    assert orchestrate.instance_id_for("public_test", 19) == 320
    assert orchestrate.instance_id_for("train", 42) == 42
    assert orchestrate.parse_task_spec("0-2,turning_on_radio,99") == [0, 1, 2, 0, 99]
    probe = Path(__file__).resolve().parents[1] / "configs" / "probe_tasks.txt"
    ids = orchestrate.parse_task_spec(f"@{probe}")
    assert len(ids) == 24 and len(set(ids)) == 24 and sum(i >= 50 for i in ids) == 12


def test_runner_resumes_retries_and_logs(tmp_path):
    jobs_file = tmp_path / "jobs.jsonl"
    jobs = orchestrate.make_jobs([0, 1], [0])
    jobs_file.write_text("\n".join(json.dumps(j.__dict__) for j in jobs) + "\n")
    out = tmp_path / "run"
    # Fake evaluator: writes the metrics JSON for task 0, writes nothing for task 1 (simulated crash, rc 0).
    fake = tmp_path / "fake_eval.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "task, inst, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])\n"
        "if task == 'turning_on_radio':\n"
        "    (out / 'json').mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'json' / f'{task}_{301 + inst}_0.json').write_text(json.dumps({'task': task, 'steps': 10, "
        "'q_score': {'final': 1.0}, 'success': True}))\n"
    )
    template = f"python {fake} {{task}} {{index}} {{output_dir}}"
    rc = orchestrate.main(["run", "--jobs", str(jobs_file), "--output-dir", str(out), "--eval-template", template,
                           "--max-attempts", "2"])
    assert rc == 1  # one job never produced metrics
    status = [json.loads(line) for line in (out / "status.jsonl").read_text().splitlines()]
    assert [s["task"] for s in status] == ["turning_on_radio", "picking_up_trash", "picking_up_trash"]
    assert status[0]["json_present"] and not status[1]["json_present"]
    # Resume: the finished job is skipped, only the failing one is attempted again.
    orchestrate.main(["run", "--jobs", str(jobs_file), "--output-dir", str(out), "--eval-template", template,
                      "--max-attempts", "1"])
    status = [json.loads(line) for line in (out / "status.jsonl").read_text().splitlines()]
    assert [s["task"] for s in status][-1] == "picking_up_trash" and len(status) == 4


def test_collect_refuses_mixed_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "json", _doc("turning_on_radio", 301, 1.0))
    _write(b / "json", _doc("turning_on_radio", 301, 0.0))
    _write(b / "json", _doc("turning_on_radio", 302, 0.0))
    rc = orchestrate.main(["collect", str(a), str(b), "--into", str(tmp_path / "merged")])
    assert rc == 1
    assert len(list((tmp_path / "merged" / "json").glob("*.json"))) == 2


# ------------------------------------------------------------------------------------------ package
def test_package_builds_clean_zips(tmp_path):
    metrics = tmp_path / "run" / "json"
    videos = tmp_path / "run" / "videos"
    videos.mkdir(parents=True)
    for t in constants.tasks():
        for inst in constants.REPORTED_INSTANCE_IDS:
            p = _write(metrics, _doc(t.name, inst, 0.25))
            (videos / (p.stem + ".mp4")).write_bytes(b"\x00\x01")
    wrapper = tmp_path / "rgbd_full_res_wrapper.py"
    wrapper.write_text("# wrapper\n")
    out = tmp_path / "submission"
    rc = package.main(["--metrics", str(metrics), "--videos", str(videos), "--out", str(out), "--team", "T",
                       "--method", "M", "--wrapper-file", str(wrapper)])
    assert rc == 0
    with zipfile.ZipFile(out / "metrics.zip") as zf:
        names = zf.namelist()
        assert len(names) == 1000 and all(n.endswith("_0.json") and "/" not in n for n in names)
    with zipfile.ZipFile(out / "package.zip") as zf:
        names = set(zf.namelist())
        assert {"README.md", "MANIFEST.sha256", "rgbd_full_res_wrapper.py"} <= names
        assert sum(n.startswith("metrics/") for n in names) == 1000
    readme = (out / "README.md").read_text()
    assert "0.2500" in readme and "RGBDFullResWrapper" in readme
    assert len((out / "videos_manifest.txt").read_text().splitlines()) == 1000


def test_package_refuses_invalid(tmp_path):
    metrics = tmp_path / "json"
    _write(metrics, _doc("turning_on_radio", 301, 0.5, rollout_id=2))
    rc = package.main(["--metrics", str(metrics), "--out", str(tmp_path / "o"), "--team", "T", "--method", "M"])
    assert rc == 1
