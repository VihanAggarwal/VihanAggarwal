"""Tests for scoring, selection, orchestration and packaging (pure CPU, no simulator)."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
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


def _train_halves(monkeypatch, ids) -> list[tuple[int, ...]]:
    """Run split_half_gain on synthetic data over ``ids`` and record the instance ids of every training half."""
    data = _synthetic({"a": 0.0, "b": 0.0}, {}, ids=ids, seed=3)
    seen: list[tuple[int, ...]] = []
    real_fit = selection.fit

    def spy(by_cand, *args, **kwargs):
        seen.append(tuple(sorted({r.instance_id for rs in by_cand.values() for r in rs})))
        return real_fit(by_cand, *args, **kwargs)

    monkeypatch.setattr(selection, "fit", spy)
    cv = selection.split_half_gain(data, "a", 0.05, 2, None, repeats=20)
    assert cv["available"] and cv["splits"] == len(seen)
    return seen


def test_split_half_cv_uses_distinct_splits(monkeypatch):
    # 10 held-out ids: 20 different training halves (the old rotate+parity scheme trained on one half 20 times).
    halves = _train_halves(monkeypatch, range(311, 321))
    assert len(halves) == 20 and len(set(halves)) == 20
    assert all(len(h) == 5 for h in halves)
    # 2 ids (PLAN's probe): both directions, each exactly once.
    halves = _train_halves(monkeypatch, range(311, 313))
    assert sorted(halves) == [(311,), (312,)]
    # 4 ids: at most C(4, 2) = 6 distinct halves exist; all are used.
    halves = _train_halves(monkeypatch, range(311, 315))
    assert len(halves) == 6 and len(set(halves)) == 6


def _r(task: str, inst: int, q: float) -> scoring.Rollout:
    return scoring.Rollout(task, inst, 0, q >= 1.0, q, 100, 1.0, {}, {}, None, "synthetic")


def test_route_ignores_unmeasured_candidates_in_the_argmax():
    """A candidate with no rollouts on a task (only a strong global effect) must not mask a measured candidate
    that clears the margin there."""
    probe = [constants.task(t).name for t in range(0, 20)]
    target = constants.task(30).name
    by = {"a": [], "b": [], "c": []}
    for t in probe:
        for i in range(311, 315):
            by["a"].append(_r(t, i, 0.30))
            by["b"].append(_r(t, i, 0.60))
            by["c"].append(_r(t, i, 0.30))
    for i in range(311, 321):
        by["a"].append(_r(target, i, 0.10 if i % 2 else 0.20))
        by["c"].append(_r(target, i, 0.70 if i % 2 else 0.80))
    fitted = selection.fit(by)
    assert fitted.posterior("b", target)[0] > fitted.posterior("c", target)[0]  # b has the best prior there
    d = selection.route(fitted, "a", margin=0.05, min_n=2, all_tasks=[target])[0]
    assert d.chosen == "c", d.reason
    assert "b has the best prior" in d.reason and d.estimates["b"][2] == 0
    # The default not being allowed: a measured candidate is still preferred over an unmeasured one.
    d = selection.route(fitted, "a", margin=0.05, min_n=2, allowed={"a": set()}, all_tasks=[target])[0]
    assert d.chosen == "c"


# ------------------------------------------------------------------------------------------ orchestrate
def test_plan_covers_every_job_once_and_balances(tmp_path):
    rc = orchestrate.main(["plan", "--workers", "20", "--instances", "0-9", "--final", "--out", str(tmp_path)])
    assert rc == 0
    jobs = [j for p in sorted(tmp_path.glob("worker_*.jsonl")) for j in orchestrate.load_jobs(p)]
    assert len(jobs) == 1000
    assert len({(j.task, j.index) for j in jobs}) == 1000
    assert {j.instance_id for j in jobs} == set(constants.REPORTED_INSTANCE_IDS)
    loads = [sum(j.max_steps for j in orchestrate.load_jobs(p)) for p in sorted(tmp_path.glob("worker_*.jsonl"))]
    assert max(loads) / (sum(loads) / len(loads)) < 1.1
    summary = json.loads((tmp_path / "plan_summary.json").read_text())
    assert summary["worst_case_env_steps"] == 15_818_280


def test_plan_runs_each_bucket_shortest_first(tmp_path):
    """rules+docs-5: inside each worker's bucket the jobs run shortest-first, so a run stopped early leaves only the
    longest jobs undone (never every instance of the shortest tasks)."""
    assert orchestrate.main(["plan", "--workers", "20", "--instances", "0-9", "--final", "--out", str(tmp_path)]) == 0
    buckets = [orchestrate.load_jobs(p) for p in sorted(tmp_path.glob("worker_*.jsonl"))]
    for b in buckets:
        assert [j.max_steps for j in b] == sorted(j.max_steps for j in b)
    # The last job of every worker (what an overrun would lose first) never includes a short task.
    shortest = sorted(constants.tasks(), key=lambda t: t.max_steps)[:10]
    assert not {b[-1].task for b in buckets} & {t.name for t in shortest}
    # The first jobs cover the short tasks.
    first = {j.task for b in buckets for j in b[:5]}
    assert {t.name for t in shortest} <= first


def test_plan_refuses_reported_instances_without_final(tmp_path, capsys):
    """rules+docs-11: --instances is required and the reported instances 301-310 need --final."""
    with pytest.raises(SystemExit):
        orchestrate.main(["plan", "--workers", "2", "--out", str(tmp_path / "a")])
    assert orchestrate.main(["plan", "--workers", "2", "--instances", "0-9", "--out", str(tmp_path / "b")]) == 2
    assert "--final" in capsys.readouterr().err and not (tmp_path / "b").exists()
    assert orchestrate.main(["plan", "--workers", "2", "--instances", "9-10", "--tasks", "0",
                             "--out", str(tmp_path / "c")]) == 2
    assert orchestrate.main(["plan", "--workers", "2", "--instances", "10-11", "--tasks", "@" + str(
        Path(__file__).resolve().parents[1] / "configs" / "probe_tasks.txt"), "--out", str(tmp_path / "d")]) == 0
    assert "REPORTED" not in capsys.readouterr().err
    assert orchestrate.main(["plan", "--workers", "2", "--instances", "0", "--tasks", "0", "--final",
                             "--out", str(tmp_path / "e")]) == 0
    assert "REPORTED instances 301-310" in capsys.readouterr().err


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
    # Attempt numbers continue across invocations, so no attempt log is overwritten.
    assert [s["attempt"] for s in status] == [1, 1, 2, 3] and len({s["log"] for s in status}) == 4
    # Every attempt records its exact command line and wrapper (the package cross-checks them).
    assert all(s["cmd"].startswith("python ") and s["wrapper"] == orchestrate.OFFICIAL_WRAPPER for s in status)
    assert status[0]["over_time_budget"] is False and "policy_failure" not in status[1]


def _jobs_file(tmp_path: Path, tasks=(0,), indices=(10,), max_steps: int | None = None) -> Path:
    jobs = orchestrate.make_jobs(list(tasks), list(indices))
    if max_steps is not None:
        jobs = [orchestrate.Job(j.task, j.task_id, j.index, j.instance_id, j.mode, max_steps) for j in jobs]
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(j.__dict__) for j in jobs) + "\n")
    return path


def _status(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "status.jsonl").read_text().splitlines()]


def test_runner_does_not_rerun_policy_failures(tmp_path):
    """rules+docs-3: an attempt that failed because of the policy (the post2 client's "Websocket connection error")
    is a failure under the rules, not an infrastructure crash: it is not re-rolled (unless --retry-policy-failures,
    for screening)."""
    jobs = _jobs_file(tmp_path)
    fake = tmp_path / "fake_eval.py"
    fake.write_text("import sys\nprint('RuntimeError: Websocket connection error: received 1011')\nsys.exit(1)\n")
    out = tmp_path / "run"
    template = f"python {fake} {{task}}"
    rc = orchestrate.main(["run", "--jobs", str(jobs), "--output-dir", str(out), "--eval-template", template])
    assert rc == 0  # the job is finished: failed because of the policy
    st = _status(out)
    assert len(st) == 1 and st[0]["policy_failure"] == "Websocket connection error"
    # A resume does not re-run it either.
    assert orchestrate.main(["run", "--jobs", str(jobs), "--output-dir", str(out), "--eval-template", template]) == 0
    assert len(_status(out)) == 1
    rc = orchestrate.main(["run", "--jobs", str(jobs), "--output-dir", str(tmp_path / "run2"), "--eval-template",
                           template, "--retry-policy-failures"])
    assert rc == 1 and len(_status(tmp_path / "run2")) == 2
    status = subprocess.run([sys.executable, "-m", "b1k26.orchestrate", "status", str(out)], capture_output=True,
                            text=True, timeout=60)
    assert "Websocket connection error" in json.loads(status.stdout)["policy_failures"][0]


def test_runner_flags_rollouts_over_the_time_budget(tmp_path):
    """rules+docs-3: a rollout slower than max_steps seconds after scene load is flagged (2026/eval would have cut
    it off), instead of the old wall/steps ratio that flagged early successes."""
    jobs = _jobs_file(tmp_path, max_steps=1)
    fake = tmp_path / "fake_eval.py"
    fake.write_text(
        "import json, sys, time, pathlib\n"
        "time.sleep(1.5)\n"
        "out = pathlib.Path(sys.argv[2]) / 'json'\nout.mkdir(parents=True, exist_ok=True)\n"
        "(out / f'{sys.argv[1]}_311_0.json').write_text(json.dumps({'steps': 1, 'q_score': {'final': 1.0}}))\n")
    out = tmp_path / "run"
    rc = orchestrate.main(["run", "--jobs", str(jobs), "--output-dir", str(out), "--eval-template",
                           f"python {fake} {{task}} {{output_dir}}", "--load-allowance-s", "0"])
    assert rc == 0 and _status(out)[0]["over_time_budget"] is True
    status = subprocess.run([sys.executable, "-m", "b1k26.orchestrate", "status", str(out)], capture_output=True,
                            text=True, timeout=60)
    assert json.loads(status.stdout)["rollouts_over_time_budget"] == ["turning_on_radio:public_test:10"]


def test_score_validate_reports_run_status(tmp_path, capsys):
    """rules+docs-3: `b1k26-score --validate` lists policy failures and over-budget rollouts from the run's
    status.jsonl (next to the metrics dir, as `b1k26-plan collect` writes it)."""
    run = tmp_path / "final_all"
    _write(run / "json", _doc("turning_on_radio", 301, 1.0))
    (run / "status.jsonl").write_text(
        json.dumps({"job": "turning_on_radio:public_test:0", "json_present": True, "over_time_budget": True}) + "\n"
        + json.dumps({"job": "picking_up_trash:public_test:0", "json_present": False,
                      "policy_failure": "Websocket connection error"}) + "\n")
    scoring.main([str(run / "json"), "--validate"])
    err = capsys.readouterr().err
    assert "POLICY FAILURE" in err and "picking_up_trash:public_test:0" in err
    assert "OVER TIME BUDGET" in err and "turning_on_radio:public_test:0" in err


def test_runner_stops_when_the_health_command_fails(tmp_path):
    """robustness-5: a failing or hanging --health-cmd stops the run (rc 1) instead of running every later job
    against a dead server."""
    jobs = _jobs_file(tmp_path, tasks=(0, 1))
    fake = tmp_path / "fake_eval.py"
    fake.write_text("print('simulator crashed')\n")  # no JSON, no policy marker: an infrastructure failure
    for name, cmd in (("fails", "exit 3"), ("hangs", "sleep 30")):
        out = tmp_path / name
        t0 = time.monotonic()
        rc = orchestrate.main(["run", "--jobs", str(jobs), "--output-dir", str(out), "--eval-template",
                               f"python {fake} {{task}}", "--health-cmd", cmd, "--health-timeout-s", "2"])
        assert rc == 1 and time.monotonic() - t0 < 20
        st = _status(out)
        assert [r.get("task") for r in st if "job" in r] == ["turning_on_radio"]  # the second job never ran
        assert st[-1]["event"] == "health_cmd_failed"


def test_stopping_the_runner_stops_its_evaluator(tmp_path):
    """robustness-7: SIGTERM to `b1k26-plan run` also stops the evaluator it started (its own process group), so
    no orphan can finish or re-run a rollout after the runner is gone."""
    jobs = _jobs_file(tmp_path)
    fake = tmp_path / "fake_eval.py"
    pidfile = tmp_path / "eval.pid"
    fake.write_text(
        "import json, os, sys, time, pathlib\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(6)\n"
        "out = pathlib.Path(sys.argv[2]) / 'json'\nout.mkdir(parents=True, exist_ok=True)\n"
        "(out / f'{sys.argv[1]}_311_0.json').write_text(json.dumps({'steps': 1, 'q_score': {'final': 1.0}}))\n")
    out = tmp_path / "run"
    runner = subprocess.Popen([sys.executable, "-m", "b1k26.orchestrate", "run", "--jobs", str(jobs), "--output-dir",
                               str(out), "--eval-template", f"{sys.executable} {fake} {{task}} {{output_dir}}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    while not pidfile.exists():
        assert time.monotonic() - t0 < 30 and runner.poll() is None
        time.sleep(0.05)
    eval_pid = int(pidfile.read_text())
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=40) == 128 + signal.SIGTERM
    try:
        os.kill(eval_pid, 0)
        alive = Path(f"/proc/{eval_pid}").exists() and "Z" not in Path(f"/proc/{eval_pid}/stat").read_text().split()[2]
    except ProcessLookupError:
        alive = False
    assert not alive, "the evaluator outlived the runner"
    time.sleep(7)
    assert not (out / "json").exists() or not any((out / "json").iterdir())
    st = _status(out)  # the stopped attempt is logged as interrupted (a resume runs the job again)
    assert len(st) == 1 and st[0]["interrupted"] is True and not st[0]["json_present"]


def test_collect_refuses_mixed_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "json", _doc("turning_on_radio", 301, 1.0))
    _write(b / "json", _doc("turning_on_radio", 301, 0.0))
    _write(b / "json", _doc("turning_on_radio", 302, 0.0))
    (a / "status.jsonl").write_text('{"job": "x"}\n')
    (b / "status.jsonl").write_text('{"job": "y"}\n{"job": "z"}\n')
    rc = orchestrate.main(["collect", str(a), str(b), "--into", str(tmp_path / "merged")])
    assert rc == 1
    assert len(list((tmp_path / "merged" / "json").glob("*.json"))) == 2
    assert len((tmp_path / "merged" / "status.jsonl").read_text().splitlines()) == 3  # every node's attempts


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
                       "--method", "M", "--wrapper-file", str(wrapper), *PKG_REFS])
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


PKG_REFS = ["--docker-image", "ghcr.io/t/img@sha256:" + "a" * 64, "--video-url", "https://huggingface.co/x"]


def _pkg_run(tmp_path: Path) -> tuple[Path, Path]:
    metrics = tmp_path / "run" / "json"
    for t in constants.tasks()[:2]:
        _write(metrics, _doc(t.name, 301, 0.5))
    wrapper = tmp_path / "rgbd_full_res_wrapper.py"
    wrapper.write_text("# wrapper\n")
    return metrics, wrapper


def _status_log(tmp_path: Path, extra: str, wrapper: str = orchestrate.OFFICIAL_WRAPPER) -> Path:
    cmd = (f"python -m omnigibson.eval.eval --task-name turning_on_radio --env-wrapper {wrapper} --output-dir o "
           f"--write-video {extra}")
    path = tmp_path / "status.jsonl"
    path.write_text(json.dumps({"job": "turning_on_radio:public_test:0", "json_present": True, "cmd": cmd,
                                "wrapper": wrapper}) + "\n"
                    + json.dumps({"job": "picking_up_trash:public_test:0", "json_present": False, "cmd": cmd,
                                  "wrapper": wrapper, "policy_failure": "Websocket connection error"}) + "\n")
    return path


def test_package_readme_states_the_chunk_size_used(tmp_path):
    """protocol-2 / rules+docs-2: the README never claims results are independent of --replay-action-chunk-size;
    it states the K the metrics used (in the command too), and a K that does not divide every routed profile's
    execute_steps is refused (with --config)."""
    metrics, wrapper = _pkg_run(tmp_path)
    fake_cfg = str(Path(__file__).resolve().parents[1] / "configs" / "fake.yaml")  # execute_steps 20
    base = ["--metrics", str(metrics), "--team", "T", "--method", "M", "--wrapper-file", str(wrapper), *PKG_REFS]
    assert package.main([*base, "--out", str(tmp_path / "a")]) == 0
    readme = (tmp_path / "a" / "README.md").read_text()
    assert "do not depend" not in readme and "replay-action-chunk-size" in readme
    assert "without `--replay-action-chunk-size`" in readme
    assert package.main([*base, "--out", str(tmp_path / "b"), "--config", fake_cfg, "--replay-chunk-size", "16"]) == 1
    assert package.main([*base, "--out", str(tmp_path / "c"), "--config", fake_cfg, "--replay-chunk-size", "10"]) == 0
    readme = (tmp_path / "c" / "README.md").read_text()
    assert "--write-video --replay-action-chunk-size 10" in readme
    assert "for this policy: 2, 4, 5, 10, 20" in readme


def test_package_refuses_placeholders_and_a_missing_wrapper(tmp_path):
    """rules+docs-4: no placeholder image/video links, and the wrapper .py must be included."""
    metrics, wrapper = _pkg_run(tmp_path)
    base = ["--metrics", str(metrics), "--team", "T", "--method", "M"]
    assert package.main([*base, "--out", str(tmp_path / "a"), "--wrapper-file", str(wrapper)]) == 1  # placeholders
    assert package.main([*base, "--out", str(tmp_path / "b"), "--wrapper-file", str(wrapper), "--docker-image",
                         "ghcr.io/t/img:final", "--video-url", "https://x"]) == 1  # no digest
    if package.find_official_wrapper(orchestrate.OFFICIAL_WRAPPER) is None:  # omnigibson is not installed here
        assert package.main([*base, "--out", str(tmp_path / "c"), *PKG_REFS]) == 1
    assert package.main([*base, "--out", str(tmp_path / "d"), *PKG_REFS, "--wrapper-file", str(wrapper)]) == 0
    readme = (tmp_path / "d" / "README.md").read_text()
    assert "status.jsonl" not in readme  # not claimed without --status-log
    # Bridge networking by default (several containers per host), and how to move the ports.
    assert "--network host" not in readme and "-p 8000-8049:8000-8049" in readme and "B1K26_PORTS" in readme
    with zipfile.ZipFile(tmp_path / "d" / "package.zip") as zf:
        assert "rgbd_full_res_wrapper.py" in zf.namelist()


def test_package_cross_checks_the_run_settings(tmp_path):
    """rules+docs-6 / protocol-2: the wrapper and chunk size stated in the README must be the ones the run's attempts
    used (status.jsonl command lines); policy failures are listed."""
    metrics, wrapper = _pkg_run(tmp_path)
    base = ["--metrics", str(metrics), "--team", "T", "--method", "M", "--wrapper-file", str(wrapper), *PKG_REFS]
    status = _status_log(tmp_path, "--replay-action-chunk-size 20")
    assert package.main([*base, "--out", str(tmp_path / "a"), "--status-log", str(status)]) == 1  # K 20 vs 0
    assert package.main([*base, "--out", str(tmp_path / "b"), "--status-log", str(status),
                         "--replay-chunk-size", "20"]) == 0
    readme = (tmp_path / "b" / "README.md").read_text()
    assert "status.jsonl" in readme and "picking_up_trash:public_test:0" in readme
    status = _status_log(tmp_path, "", wrapper="omnigibson.eval.wrappers.DefaultWrapper")
    assert package.main([*base, "--out", str(tmp_path / "c"), "--status-log", str(status)]) == 1  # wrapper differs
    assert package.main([*base, "--out", str(tmp_path / "d"), "--status-log", str(status), "--wrapper",
                         "omnigibson.eval.wrappers.DefaultWrapper"]) == 0
    assert "--env-wrapper omnigibson.eval.wrappers.DefaultWrapper" in (tmp_path / "d" / "README.md").read_text()


def test_package_refuses_invalid(tmp_path):
    metrics = tmp_path / "json"
    _write(metrics, _doc("turning_on_radio", 301, 0.5, rollout_id=2))
    rc = package.main(["--metrics", str(metrics), "--out", str(tmp_path / "o"), "--team", "T", "--method", "M"])
    assert rc == 1
