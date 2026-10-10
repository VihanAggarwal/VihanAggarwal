"""Build the submission package exactly as the 2026 rules ask for it.

Produces, under --out:
  metrics.zip        only the rollout JSONs (flat). The leaderboard extractor counts EVERY .json inside a zip that
                     has "success" and "q_score", so nothing else may go in this zip. Name contains "metric" so an
                     HF dataset upload is picked up by the extractor.
  package.zip        metrics JSONs + wrapper .py + robot config (or a statement that the default r1pro.yaml was
                     used) + README.md + MANIFEST.sha256 (the "final package" of submission.md)
  README.md          rendered evaluation README (full evaluator command, wrapper path, robot config, Docker
                     image, capacity, method, run provenance, score summary)
  videos_manifest.txt  every video file with size and sha256, for the separate video upload
  summary.json       scoring summary (official and leaderboard formulas) and validation problems
The JSON and video files are copied byte-for-byte; the rules forbid modifying them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import string
import sys
import zipfile
from pathlib import Path

from b1k26 import constants
from b1k26.scoring import load_rollouts, summarize, validate_submission

README_TEMPLATE = string.Template(r"""# $team: BEHAVIOR-1K 2026 Challenge submission

Method: **$method**. Affiliation: $affiliation. Contact: $contact.

## Policy
$policy_description

Policy inputs: RGB (head + both wrists) and proprioception from `robot_r1::proprio`, plus the evaluator-provided
`task_id`. No depth, no privileged simulator state. All inference-time logic (action chunking, compression,
gripper corrections, routing) runs inside the policy server and only sees those observations.

## How to run (final evaluation)

**Policy server (Docker, recommended path).** Image: `$docker_image`
```
$docker_run
```
- Fits one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX). Turing (sm_75) is supported: $turing_note
- Serves the BEHAVIOR websocket policy protocol on ports `$ports` (every port is an independent endpoint;
  `/healthz` returns 200 once the model is loaded and warmed up, usually within $warmup_note).
- Capacity: $capacity_note

**Evaluator command (BEHAVIOR-1K $b1k_version, one process per task instance):**
```
$eval_command
```
- Wrapper: `$wrapper` ($wrapper_note)
- Robot config: $robot_config_note
- Chunk replay: $chunk_note
- Works with the v3.9.3-post1/post2 evaluator and with the 2026/eval `--policy-endpoints` multi-port evaluator.

## Self-evaluation run (what these metrics are)
$provenance

| metric | value |
|---|---|
| rollouts | $episodes |
| tasks covered | $tasks |
| Q (official: missing instances count as zero) | $official_q |
| Q (leaderboard extractor formula) | $leaderboard_q |
| full-task success rate | $official_sr |

$routing_section

## Files
- `metrics/`: one JSON per rollout, unmodified evaluator output, names `<task>_<instance_id>_0.json`.
- `$wrapper_file`: the evaluation wrapper used.
- `$robot_config_file`
- `MANIFEST.sha256`: checksums of every file in this package.
- Videos: uploaded separately ($video_url), one MP4 per rollout with the same base name as its JSON.

## Additional data
$extra_data
""")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics)
    videos_dir = Path(args.videos) if args.videos else None
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"{out} is not empty (use --overwrite)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    problems = validate_submission(metrics_dir, videos_dir)
    blocking = [p for p in problems if "allowed for partial submissions" not in p]
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if blocking and not args.force:
        print(f"{len(blocking)} blocking problem(s); fix them or pass --force", file=sys.stderr)
        return 1

    json_files = sorted(metrics_dir.rglob("*.json"))
    rollouts = load_rollouts([metrics_dir])
    summary = summarize(rollouts)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "problems": problems}, indent=1))

    with zipfile.ZipFile(out / "metrics.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in json_files:
            zf.write(p, arcname=p.name)

    staging = out / "package"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "metrics").mkdir(parents=True)
    for p in json_files:
        shutil.copy2(p, staging / "metrics" / p.name)

    wrapper_file = "wrapper: official omnigibson.eval.wrappers.RGBDFullResWrapper (copy included)"
    if args.wrapper_file:
        src = Path(args.wrapper_file)
        shutil.copy2(src, staging / src.name)
        wrapper_file = src.name
    robot_config_file = "Robot config: the unchanged default `OmniGibson/omnigibson/eval/r1pro.yaml` (no --robot-config)."
    if args.robot_config:
        src = Path(args.robot_config)
        shutil.copy2(src, staging / src.name)
        robot_config_file = f"`{src.name}`: the exact robot config passed with --robot-config."

    routing_section = ""
    if args.routing_log and Path(args.routing_log).exists():
        shutil.copy2(args.routing_log, staging / Path(args.routing_log).name)
        routing_section = (f"## Checkpoint routing\nThe policy routes each task (by the evaluator's `task_id`) to one "
                           f"checkpoint. Routes were chosen on held-out instances only (311-320 / train-mode), never on "
                           f"the reported instances 301-310; see `{Path(args.routing_log).name}`.")

    provenance = args.provenance or (
        "Single evaluation run of the frozen final policy: every reported (task, instance) was rolled out once "
        "(rollout_id 0). Jobs that produced no metrics JSON because of an infrastructure failure (simulator "
        "crash, node loss) were re-run; every attempt is listed in `status.jsonl`.")
    if args.status_log and Path(args.status_log).exists():
        shutil.copy2(args.status_log, staging / "status.jsonl")

    fields = dict(
        team=args.team, method=args.method, affiliation=args.affiliation, contact=args.contact,
        policy_description=args.policy_description, docker_image=args.docker_image,
        docker_run=args.docker_run or f"docker run --gpus '\"device=0\"' --network host {args.docker_image}",
        turing_note=args.turing_note, ports=args.ports, warmup_note=args.warmup_note, capacity_note=args.capacity_note,
        b1k_version=args.b1k_version, eval_command=args.eval_command, wrapper=args.wrapper,
        wrapper_note=args.wrapper_note, robot_config_note=robot_config_file, chunk_note=args.chunk_note,
        provenance=provenance, episodes=summary["episodes"], tasks=summary["tasks"],
        official_q=f"{summary['official_q']:.4f}", leaderboard_q=f"{summary['leaderboard_q']:.4f}",
        official_sr=f"{summary['official_sr']:.3f}", routing_section=routing_section,
        wrapper_file=wrapper_file, robot_config_file=robot_config_file, video_url=args.video_url,
        extra_data=args.extra_data,
    )
    readme = README_TEMPLATE.substitute(fields)
    (staging / "README.md").write_text(readme)
    (out / "README.md").write_text(readme)

    manifest_lines = [f"{sha256(p)}  {p.relative_to(staging)}" for p in sorted(staging.rglob("*")) if p.is_file()]
    (staging / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
    with zipfile.ZipFile(out / "package.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(staging)))

    if videos_dir is not None:
        lines = [f"{sha256(p)}  {p.stat().st_size}  {p.name}" for p in sorted(videos_dir.rglob("*.mp4"))]
        (out / "videos_manifest.txt").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=1))
    print(f"wrote {out/'metrics.zip'}, {out/'package.zip'}, {out/'README.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the BEHAVIOR-1K 2026 submission package.")
    ap.add_argument("--metrics", required=True, help="dir with the final run's rollout JSONs")
    ap.add_argument("--videos", default=None, help="dir with the final run's MP4s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force", action="store_true", help="package despite blocking validation problems")
    ap.add_argument("--team", required=True)
    ap.add_argument("--method", required=True, help="<= 25 characters on the portal")
    ap.add_argument("--affiliation", default="Independent")
    ap.add_argument("--contact", default="")
    ap.add_argument("--policy-description", default="(describe the model, checkpoints and training data)")
    ap.add_argument("--docker-image", default="<registry>/<image>@sha256:<digest>")
    ap.add_argument("--docker-run", default="")
    ap.add_argument("--ports", default="8000-8049")
    ap.add_argument("--turing-note", default="JAX upcasts bf16 matmuls to fp32 on sm_75; no flash-attention is used.")
    ap.add_argument("--warmup-note", default="5 minutes")
    ap.add_argument("--capacity-note", default="one model copy shared by all ports; requests are micro-batched.")
    ap.add_argument("--b1k-version", default="v3.9.3-post2")
    ap.add_argument("--eval-command", default=(
        "python -m omnigibson.eval.eval --task-name <task> --mode public_test --instance-indices <i> --num-envs 1 "
        "--num-rollouts 1 --host <host> --port <port> --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper "
        "--output-dir <out> --write-video"))
    ap.add_argument("--wrapper", default="omnigibson.eval.wrappers.RGBDFullResWrapper")
    ap.add_argument("--wrapper-file", default=None, help="path of the wrapper .py to include")
    ap.add_argument("--wrapper-note", default="official wrapper, unmodified; the policy ignores depth")
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--chunk-note", default="not required; the server keeps its own action queue, so results do not "
                                            "depend on --replay-action-chunk-size")
    ap.add_argument("--provenance", default="")
    ap.add_argument("--status-log", default=None, help="status.jsonl from the run (all attempts)")
    ap.add_argument("--routing-log", default=None, help="route_selection.md from b1k26-select")
    ap.add_argument("--video-url", default="<link>")
    ap.add_argument("--extra-data", default="None. Only the official 2026 human demonstrations (via the public "
                                            "checkpoints we start from).")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
