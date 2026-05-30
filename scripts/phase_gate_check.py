#!/usr/bin/env python3
"""改良候補が提出ゲートを通過するか一括判定する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.public_score import PUBLIC_BEST_EXPERIMENT, summarize_anchor_submission_diff  # noqa: E402

PHASE1 = [
    "candidate_linear_1f_msc",
    "candidate_linear_1f_area",
    "candidate_linear_1f_smooth",
    "candidate_linear_1f_detrend",
]
PHASE2 = [
    "candidate_linear_1f_range_cal",
    "candidate_linear_1f_lowtail_mild",
    "group_curve_blend_spectral_meta",
]
PHASE3 = ["candidate_linear_1f_moisture_bins"]
PHASE4 = [
    "candidate_linear_1f_nn_bias",
    "candidate_linear_1f_moisture_bins_v2",
    "candidate_linear_1f_msc_select_raw",
]
PREP_TUNING = [
    "candidate_linear_1f_center",
    "candidate_linear_1f_l2norm",
    "candidate_linear_1f_blend_snv25",
    "candidate_linear_1f_blend_snv50",
    "candidate_linear_1f_smooth3",
    "candidate_linear_1f_smooth5",
    "candidate_linear_idx616_raw",
    "candidate_linear_idx616_center",
    "candidate_linear_idx616_blend25",
    "candidate_linear_idx616_smooth5",
]
ALL = PHASE1 + PHASE2 + PHASE3 + PHASE4 + PREP_TUNING

PUBLIC_DIFF_MAX = 10.0
PUBLIC_DIFF_MIN = 1e-9
MOISTURE_TOLERANCE = 2.0


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def _moisture_rmse(experiment: str) -> float | None:
    log_path = ROOT / "outputs" / "logs" / f"{experiment}.json"
    if not log_path.exists():
        return None
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    cv = payload.get("cv") or {}
    if cv.get("strategy") != "moisture_quantile":
        return None
    return float(cv["mean_score"])


def _ensure_cv(experiment: str) -> None:
    log_path = ROOT / "outputs" / "logs" / f"{experiment}.json"
    sub_path = ROOT / "data" / "submissions" / f"{experiment}.csv"
    if log_path.exists():
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        if (payload.get("cv") or {}).get("strategy") == "moisture_quantile" and sub_path.exists():
            return
    _run([sys.executable, str(ROOT / "run.py"), experiment, "--cv", "moisture_quantile"])


def evaluate(experiment: str, baseline_mq: float) -> dict[str, object]:
    _ensure_cv(experiment)
    diff = summarize_anchor_submission_diff(
        load_config(),
        experiment,
        anchor_experiment=PUBLIC_BEST_EXPERIMENT,
    )
    public_diff = diff.get("diff_rmse")
    mq = _moisture_rmse(experiment)
    pd_ok = (
        public_diff is not None
        and PUBLIC_DIFF_MIN < float(public_diff) < PUBLIC_DIFF_MAX
    )
    mq_ok = mq is not None and mq <= baseline_mq + MOISTURE_TOLERANCE
    return {
        "experiment": experiment,
        "public_diff": public_diff,
        "moisture_quantile_rmse": mq,
        "pass_public_diff": pd_ok,
        "pass_moisture": mq_ok,
        "pass": pd_ok and mq_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "prep", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.phase == "1":
        experiments = PHASE1
    elif args.phase == "2":
        experiments = PHASE2
    elif args.phase == "3":
        experiments = PHASE3
    elif args.phase == "4":
        experiments = PHASE4
    elif args.phase == "prep":
        experiments = PREP_TUNING
    else:
        experiments = ALL

    baseline_mq = _moisture_rmse(PUBLIC_BEST_EXPERIMENT)
    if baseline_mq is None:
        _run([sys.executable, str(ROOT / "run.py"), PUBLIC_BEST_EXPERIMENT, "--cv", "moisture_quantile"])
        baseline_mq = _moisture_rmse(PUBLIC_BEST_EXPERIMENT)
    assert baseline_mq is not None

    print(f"baseline {PUBLIC_BEST_EXPERIMENT} moisture_quantile RMSE = {baseline_mq:.4f}")
    passed: list[str] = []
    results: list[dict[str, object]] = []
    for name in experiments:
        row = evaluate(name, baseline_mq)
        results.append(row)
        flag = "PASS" if row["pass"] else "fail"
        print(
            f"[{flag}] {name}: public_diff={row['public_diff']}, "
            f"mq={row['moisture_quantile_rmse']}"
        )
        if row["pass"]:
            passed.append(name)

    out = ROOT / "outputs" / "logs" / "phase_gate_results.json"
    out.write_text(
        json.dumps({"baseline_mq": baseline_mq, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    if passed:
        print("submit candidates:", ", ".join(passed))
    else:
        print("no candidates passed gate")


if __name__ == "__main__":
    main()
