"""Choose which candidate policy serves each task, from held-out rollouts, without fooling ourselves.

Per-rollout Q on BEHAVIOR is very noisy (within-task standard deviation around 0.28-0.35), so picking the best of
several checkpoints per task on a handful of rollouts mostly selects noise: Mirua's 2026 "best of 10 checkpoints
per task" submission scored +0.10 above its best single checkpoint on the same instances, and a simulation of
pure noise explains most of that. This module therefore:

1. fits an additive model q(c, t) = a_t + b_c + e_ct (task difficulty + candidate strength + interaction) with
   per-cell weights n_ct / sigma^2,
2. shrinks every observed interaction e_ct toward zero by tau^2 / (tau^2 + sigma^2 / n_ct) (empirical Bayes,
   method-of-moments tau^2),
3. routes a task away from the default candidate only when the shrunk gain exceeds a margin,
4. estimates the real gain of the routing with split-half cross-validation (choose on half of the instances,
   score on the other half), which is the number to trust.

Inputs must be held-out rollouts: public instances 311-320 (allowed as a pre-submission test set) or train-mode
instances. Rollouts on the reported instances 301-310 are refused unless explicitly allowed, because selecting on
them and then reporting them is the cherry-picking the rules forbid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from b1k26 import constants
from b1k26.scoring import Rollout, load_rollouts, parse_ids

DEFAULT_SIGMA = 0.30  # per-rollout Q std when it cannot be estimated (observed 0.28-0.35 on 2026 runs)


@dataclasses.dataclass
class CellStats:
    n: int
    mean: float
    instances: tuple[int, ...]


@dataclasses.dataclass
class Fit:
    candidates: list[str]
    tasks: list[str]
    task_effect: dict[str, float]
    cand_effect: dict[str, float]
    sigma2: float
    tau2: float
    cells: dict[tuple[str, str], CellStats]

    def posterior(self, cand: str, task: str) -> tuple[float, float]:
        """Posterior mean and standard deviation of q(cand, task)."""
        base = self.task_effect.get(task, 0.0) + self.cand_effect.get(cand, 0.0)
        cell = self.cells.get((cand, task))
        if cell is None or cell.n == 0:
            return _clip01(base), math.sqrt(self.tau2 + 1e-12)
        noise = self.sigma2 / cell.n
        w = self.tau2 / (self.tau2 + noise) if self.tau2 > 0 else 0.0
        resid = cell.mean - base
        mean = base + w * resid
        var = 1.0 / (1.0 / max(self.tau2, 1e-12) + 1.0 / noise)
        return _clip01(mean), math.sqrt(var)


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def cell_stats(rollouts_by_cand: dict[str, list[Rollout]]) -> dict[tuple[str, str], CellStats]:
    cells: dict[tuple[str, str], list[Rollout]] = defaultdict(list)
    for cand, rollouts in rollouts_by_cand.items():
        for r in rollouts:
            cells[(cand, r.task)].append(r)
    return {
        k: CellStats(n=len(v), mean=sum(r.q for r in v) / len(v), instances=tuple(sorted(r.instance_id for r in v)))
        for k, v in cells.items()
    }


def pooled_sigma2(rollouts_by_cand: dict[str, list[Rollout]]) -> float:
    """Pooled within-cell variance of per-rollout Q (falls back to DEFAULT_SIGMA**2)."""
    ss, dof = 0.0, 0
    for cand, rollouts in rollouts_by_cand.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rollouts:
            groups[r.task].append(r.q)
        for qs in groups.values():
            if len(qs) >= 2:
                m = sum(qs) / len(qs)
                ss += sum((q - m) ** 2 for q in qs)
                dof += len(qs) - 1
    if dof < 10:
        return DEFAULT_SIGMA**2
    return max(ss / dof, 0.01)


def fit(rollouts_by_cand: dict[str, list[Rollout]], iters: int = 50) -> Fit:
    cells = cell_stats(rollouts_by_cand)
    cands = sorted(rollouts_by_cand)
    tasks = sorted({t for (_, t) in cells})
    sigma2 = pooled_sigma2(rollouts_by_cand)
    a = {t: 0.0 for t in tasks}
    b = {c: 0.0 for c in cands}
    # Weighted alternating least squares for q_ct ~ a_t + b_c (weights n_ct), with sum_c b_c = 0.
    for _ in range(iters):
        for t in tasks:
            num = den = 0.0
            for c in cands:
                cell = cells.get((c, t))
                if cell:
                    num += cell.n * (cell.mean - b[c])
                    den += cell.n
            a[t] = num / den if den else 0.0
        for c in cands:
            num = den = 0.0
            for t in tasks:
                cell = cells.get((c, t))
                if cell:
                    num += cell.n * (cell.mean - a[t])
                    den += cell.n
            b[c] = num / den if den else 0.0
        shift = sum(b.values()) / len(b) if b else 0.0
        for c in cands:
            b[c] -= shift
        for t in tasks:
            a[t] += shift
    # Method-of-moments interaction variance: E[resid^2] = tau^2 + E[sigma^2 / n].
    resid2, noise, k = 0.0, 0.0, 0
    for (c, t), cell in cells.items():
        r = cell.mean - a[t] - b[c]
        resid2 += r * r
        noise += sigma2 / cell.n
        k += 1
    n_params = len(tasks) + len(cands) - 1
    dof = max(k - n_params, 1)
    tau2 = max((resid2 - noise * (k - n_params) / max(k, 1)) / dof, 0.0) if k > n_params else 0.0
    return Fit(cands, tasks, a, b, sigma2, tau2, cells)


@dataclasses.dataclass
class Decision:
    task: str
    task_id: int | None
    chosen: str
    default: str
    gain: float
    estimates: dict[str, tuple[float, float, int]]  # cand -> (posterior mean, posterior sd, n)
    reason: str


def route(
    fitted: Fit,
    default: str,
    margin: float = 0.05,
    min_n: int = 2,
    allowed: dict[str, set[int]] | None = None,
    all_tasks: Iterable[str] | None = None,
) -> list[Decision]:
    """Per-task decisions. allowed[cand] = set of task ids that candidate may serve (None = all)."""
    decisions = []
    task_names = list(all_tasks) if all_tasks is not None else [t.name for t in constants.tasks()]
    for task in task_names:
        info = constants.task_by_name().get(task)
        tid = info.task_id if info else None
        ok = [c for c in fitted.candidates if allowed is None or c not in allowed or (tid is not None and tid in allowed[c])]
        est = {}
        for c in ok:
            m, s = fitted.posterior(c, task)
            cell = fitted.cells.get((c, task))
            est[c] = (m, s, cell.n if cell else 0)
        if default not in ok:
            # The default cannot serve this task: take the best allowed candidate outright.
            if not est:
                decisions.append(Decision(task, tid, default, default, 0.0, {}, "no allowed candidate; default kept"))
                continue
            best = max(est, key=lambda c: est[c][0])
            decisions.append(Decision(task, tid, best, default, 0.0, est, "default not allowed for this task"))
            continue
        best = max(est, key=lambda c: est[c][0])
        gain = est[best][0] - est[default][0]
        if best != default and gain > margin and est[best][2] >= min_n:
            reason = f"shrunk gain {gain:+.3f} > margin {margin} with n={est[best][2]}"
            decisions.append(Decision(task, tid, best, default, gain, est, reason))
        else:
            why = "default is best" if best == default else (
                f"gain {gain:+.3f} <= margin {margin}" if gain <= margin else f"n={est[best][2]} < min_n {min_n}")
            decisions.append(Decision(task, tid, default, default, 0.0, est, why))
    return decisions


def split_half_gain(
    rollouts_by_cand: dict[str, list[Rollout]],
    default: str,
    margin: float,
    min_n: int,
    allowed: dict[str, set[int]] | None,
    repeats: int = 20,
) -> dict:
    """Cross-validated estimate of what the routing really gains over always using the default.

    Instances are split into two halves deterministically (several different splits); routes are chosen on one
    half and the resulting policy (default vs routed) is scored on the other half, using raw means of the
    candidates' rollouts there.
    """
    all_ids = sorted({r.instance_id for rs in rollouts_by_cand.values() for r in rs})
    if len(all_ids) < 2:
        return {"available": False, "reason": "need rollouts on at least 2 distinct instance ids"}
    gains, routed_scores, default_scores = [], [], []
    for rep in range(repeats):
        # Deterministic pseudo-random split: rotate and interleave ids by repeat index.
        order = all_ids[rep % len(all_ids):] + all_ids[: rep % len(all_ids)]
        half_a = set(order[0::2]) if rep % 2 == 0 else set(order[1::2])
        train = {c: [r for r in rs if r.instance_id in half_a] for c, rs in rollouts_by_cand.items()}
        test = {c: [r for r in rs if r.instance_id not in half_a] for c, rs in rollouts_by_cand.items()}
        if not any(train.values()) or not any(test.values()):
            continue
        decisions = route(fit(train), default, margin, max(1, min_n // 2), allowed,
                          all_tasks=sorted({r.task for rs in test.values() for r in rs}))
        g, rs_sum, ds_sum, k = 0.0, 0.0, 0.0, 0
        for d in decisions:
            test_default = [r.q for r in test.get(default, []) if r.task == d.task]
            test_chosen = [r.q for r in test.get(d.chosen, []) if r.task == d.task]
            if not test_default or not test_chosen:
                continue
            md = sum(test_default) / len(test_default)
            mc = sum(test_chosen) / len(test_chosen)
            g += mc - md
            rs_sum += mc
            ds_sum += md
            k += 1
        if k:
            gains.append(g / k)
            routed_scores.append(rs_sum / k)
            default_scores.append(ds_sum / k)
    if not gains:
        return {"available": False, "reason": "no task had held-out rollouts for both default and chosen candidate"}
    mean = sum(gains) / len(gains)
    return {
        "available": True,
        "repeats": len(gains),
        "mean_gain_per_task": mean,
        "mean_routed_q": sum(routed_scores) / len(routed_scores),
        "mean_default_q": sum(default_scores) / len(default_scores),
        "note": "Gain per task on held-out halves; negative or ~0 means per-task routing is not worth it.",
    }


def write_routing_yaml(decisions: list[Decision], default: str, path: Path) -> None:
    per_task = {d.task: d.chosen for d in decisions if d.chosen != default}
    lines = ["# Generated by b1k26.selection. Paste under `routing:` in the serving config.", f"default: {default}"]
    if per_task:
        lines.append("per_task:")
        for task, cand in sorted(per_task.items(), key=lambda kv: constants.task(kv[0]).task_id):
            lines.append(f"  {task}: {cand}")
    else:
        lines.append("per_task: {}")
    path.write_text("\n".join(lines) + "\n")


def decision_log(fitted: Fit, decisions: list[Decision], cv: dict, default: str) -> str:
    out = ["# Route selection log", "",
           f"- candidates: {', '.join(fitted.candidates)}; default: **{default}**",
           f"- pooled per-rollout sigma: {math.sqrt(fitted.sigma2):.3f}; interaction tau: {math.sqrt(fitted.tau2):.3f}",
           "- candidate effects (relative, shared task difficulty removed): "
           + ", ".join(f"{c} {fitted.cand_effect[c]:+.3f}" for c in fitted.candidates),
           f"- split-half cross-validated routing gain: {json.dumps(cv)}", "",
           "| id | task | chosen | gain | " + " | ".join(f"{c} (mean±sd, n)" for c in fitted.candidates) + " | reason |",
           "|---|---|---|---|" + "---|" * len(fitted.candidates) + "---|"]
    for d in decisions:
        cols = []
        for c in fitted.candidates:
            if c in d.estimates:
                m, s, n = d.estimates[c]
                cols.append(f"{m:.2f}±{s:.2f}, {n}")
            else:
                cols.append("n/a")
        out.append(f"| {d.task_id} | {d.task} | {d.chosen} | {d.gain:+.3f} | " + " | ".join(cols) + f" | {d.reason} |")
    return "\n".join(out) + "\n"


def parse_allowed(specs: list[str]) -> dict[str, set[int]]:
    allowed = {}
    for spec in specs:
        name, ids = spec.split("=", 1)
        allowed[name] = set(parse_ids(ids))
    return allowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Choose per-task routes from held-out rollouts with shrinkage.")
    ap.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH[,PATH...]",
                    help="candidate name and its held-out rollout dirs/zips")
    ap.add_argument("--default", required=True, help="candidate used unless a task clearly prefers another")
    ap.add_argument("--allowed", action="append", default=[], metavar="NAME=IDS",
                    help="restrict a candidate to task ids, e.g. rlc2025=0-49")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--allow-reported", action="store_true",
                    help="accept rollouts on reported instances 301-310 (do NOT use for the final submission)")
    ap.add_argument("--out", default="routing.yaml")
    ap.add_argument("--log", default="route_selection.md")
    args = ap.parse_args(argv)

    by_cand: dict[str, list[Rollout]] = {}
    for spec in args.candidate:
        name, paths = spec.split("=", 1)
        rollouts = load_rollouts(paths.split(","))
        reported = [r for r in rollouts if r.instance_id in constants.REPORTED_INSTANCE_IDS]
        if reported and not args.allow_reported:
            print(f"refusing: candidate {name} has {len(reported)} rollouts on reported instances 301-310; "
                  "select on held-out instances 311-320 or train-mode instances instead", file=sys.stderr)
            return 2
        by_cand[name] = [r for r in rollouts if r.rollout_id == 0]
    if args.default not in by_cand:
        print(f"default {args.default!r} is not a candidate", file=sys.stderr)
        return 2
    allowed = parse_allowed(args.allowed) or None
    fitted = fit(by_cand)
    decisions = route(fitted, args.default, args.margin, args.min_n, allowed)
    cv = split_half_gain(by_cand, args.default, args.margin, args.min_n, allowed)
    write_routing_yaml(decisions, args.default, Path(args.out))
    Path(args.log).write_text(decision_log(fitted, decisions, cv, args.default))
    switched = [d for d in decisions if d.chosen != d.default]
    print(f"{len(switched)} task(s) routed away from {args.default}; cross-validated gain: {cv}")
    print(f"wrote {args.out} and {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
