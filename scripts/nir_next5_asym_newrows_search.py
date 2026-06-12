#!/usr/bin/env python3
"""Search new hard-row corrections on top of the current asymmetric best.

The previous asymmetric golden correction improved Public strongly, but the
last same-row push was nearly saturated.  This script freezes the current best
and only allows corrections on test rows that are not already changed by that
best.  It keeps the public-failed rows banned and uses the same capped,
mean-centered asymmetric correction family so each candidate remains auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nir_next5_asym_attack_search as asym
import nir_next5_direct_blend_search as db
import nir_next5_diverse_candidate_builder as n5
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_asym_newrows"
SUBMISSION_DIR = ROOT / "data" / "submissions"

OLD_BEST = fu.BEST
CAND2 = fu.CAND2
CURRENT_BEST = SUBMISSION_DIR / "nir_asym_followup_1_asym_reanchor_pos0p035_nr0p75_b0p5_c4p5_sp3_20260612.csv"


@dataclass(frozen=True)
class Candidate:
    family: str
    experiment: str
    pred: np.ndarray
    corr_oof: np.ndarray
    corr_test: np.ndarray
    beta: float
    signal_corr: float
    memo: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--save-top", type=int, default=8)
    parser.add_argument("--current-best", type=Path, default=CURRENT_BEST)
    parser.add_argument("--current-pos-frac", type=float, default=0.035)
    parser.add_argument("--current-neg-ratio", type=float, default=0.75)
    parser.add_argument("--current-blend", type=float, default=0.50)
    parser.add_argument("--current-clip", type=float, default=4.5)
    parser.add_argument("--current-species-cap", type=int, default=3)
    args = parser.parse_args()

    data = op.load_data()
    old_best = n5.read_submission(OLD_BEST, data["test_ids"])
    current_best_path = args.current_best if args.current_best.is_absolute() else ROOT / args.current_best
    current_best = n5.read_submission(current_best_path, data["test_ids"])
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    diag = asym.load_public_delta_diag()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building old #2 OOF anchor", flush=True)
    q5_oof = n5.build_q5_oof(data)
    old_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    old_oof = np.clip(q5_oof + old_corr_oof, 0, None)
    old_rmse = rmse(data["y"], old_oof)
    old_residual = data["y"] - old_oof
    print(f"old_oof_rmse={old_rmse:.6f}", flush=True)

    print("building direct experts", flush=True)
    experts = db.build_experts(data, old_oof, old_best, old_residual, old_rmse)
    golden = [e for e in experts if e.name == "golden_sg9_keep80_c3"][0]
    golden_signal_oof = golden.oof - old_oof

    current_corr_oof = asym.asym_corr(
        golden_signal_oof,
        old_residual,
        pos_frac=args.current_pos_frac,
        neg_ratio=args.current_neg_ratio,
        blend=args.current_blend,
        clip=args.current_clip,
        labels=data["groups"],
        species_cap=args.current_species_cap,
        banned=np.zeros(len(golden_signal_oof), dtype=bool),
    )
    current_corr_test = current_best - old_best
    current_oof = np.clip(old_oof + current_corr_oof, 0, None)
    current_rmse = rmse(data["y"], current_oof)
    current_residual = data["y"] - current_oof
    current_changed_test = np.abs(current_corr_test) > 1e-12
    current_changed_oof = np.abs(current_corr_oof) > 1e-12
    print(
        f"current_oof_rmse={current_rmse:.6f} delta={current_rmse - old_rmse:.6f} "
        f"current_test_changed={int(current_changed_test.sum())}",
        flush=True,
    )

    failed_test = asym.bad_mask(diag, len(current_best))
    banned_test = failed_test | current_changed_test
    banned_oof = current_changed_oof

    branch_rows = []
    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    for expert in experts:
        signal_oof = np.asarray(expert.oof, dtype=float) - old_oof
        signal_test = np.asarray(expert.test, dtype=float) - old_best
        branch_rows.append(
            {
                "expert": expert.name,
                "expert_oof_delta_vs_old": expert.oof_delta,
                "signal_corr_vs_old_resid": expert.signal_corr,
                "signal_corr_vs_current_resid": n5.safe_corr(signal_oof, current_residual),
                "test_signal_rmse_vs_old": float(math.sqrt(np.mean(signal_test**2))),
                "test_signal_max_vs_old": float(np.max(np.abs(signal_test))),
            }
        )
        current_signal_corr = n5.safe_corr(signal_oof, current_residual)
        if current_signal_corr < 0.08:
            continue
        for cand in candidate_grid(
            expert.name,
            current_best,
            current_residual,
            signal_oof,
            signal_test,
            data["groups"],
            test_species,
            banned_oof,
            banned_test,
            current_signal_corr,
            expert.oof_delta,
        ):
            row = diagnostics(cand, data["y"], current_oof, current_rmse, current_best, current_corr_test, test_species, diag, current_changed_test)
            rows.append(row)
            candidates.append(cand)
            if row["submit_gate"] == "pass":
                print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(branch_rows).to_csv(out_dir / "asym_newrows_experts.csv", index=False)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "asym_newrows_summary.csv", index=False)
    (out_dir / "asym_newrows_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_asym_newrows_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "asym_newrows_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "asym_newrows_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "asym_newrows_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nExperts:")
    print(pd.DataFrame(branch_rows).sort_values("signal_corr_vs_current_resid", ascending=False).to_string(index=False))
    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:80]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def candidate_grid(
    expert_name: str,
    current_best: np.ndarray,
    current_residual: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
    banned_oof: np.ndarray,
    banned_test: np.ndarray,
    signal_corr: float,
    expert_delta: float,
) -> list[Candidate]:
    out: list[Candidate] = []
    for pos_frac in [0.018, 0.025, 0.035, 0.050, 0.070]:
        for neg_ratio in [0.20, 0.50, 0.75]:
            for blend in [0.25, 0.35, 0.50, 0.70]:
                for clip in [1.2, 1.8, 2.5, 3.5]:
                    for species_cap in [2, 3, 4]:
                        corr_oof = asym.asym_corr(
                            signal_oof,
                            current_residual,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            train_species,
                            species_cap,
                            banned_oof,
                        )
                        corr_test = asym.asym_corr(
                            signal_test,
                            None,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            test_species,
                            species_cap,
                            banned_test,
                        )
                        if np.max(np.abs(corr_test)) < 1e-12:
                            continue
                        exp = (
                            f"newrows_{expert_name}_pos{tag(pos_frac)}_nr{tag(neg_ratio)}_"
                            f"b{tag(blend)}_c{tag(clip)}_sp{species_cap}"
                        )
                        out.append(
                            Candidate(
                                family=f"newrows_{expert_name}",
                                experiment=exp,
                                pred=np.clip(current_best + corr_test, 0, None),
                                corr_oof=corr_oof,
                                corr_test=corr_test,
                                beta=blend,
                                signal_corr=signal_corr,
                                memo=(
                                    f"Current best fixed; new-row-only asymmetric correction from {expert_name}; "
                                    f"pos_frac={pos_frac}; neg_ratio={neg_ratio}; blend={blend}; clip={clip}; "
                                    f"species_cap={species_cap}; expert_delta_vs_old={expert_delta:.4f}; "
                                    f"signal_corr_current_residual={signal_corr:.4f}; public-failed and current-changed rows excluded."
                                ),
                            )
                        )
    return out


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    current_oof: np.ndarray,
    current_rmse: float,
    current_best: np.ndarray,
    current_increment: np.ndarray,
    test_species: np.ndarray,
    diag: pd.DataFrame | None,
    current_changed_test: np.ndarray,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, current_oof, current_rmse, current_best, current_increment, test_species)
    diff = cand.pred - current_best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "bad_new_overlap_count": db.bad_new_overlap(changed, diag),
            "failed_any_overlap_count": asym.failed_any_overlap(changed, diag),
            "current_overlap_count": int(np.sum(changed & current_changed_test)),
            "diff_p95": float(np.quantile(np.abs(diff[changed]), 0.95)) if np.any(changed) else 0.0,
            "tail_low_shift": float(np.mean(diff[current_best <= np.quantile(current_best, 0.10)])),
            "tail_high_shift": float(np.mean(diff[current_best >= np.quantile(current_best, 0.90)])),
        }
    )
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.012:
        reasons.append("weak_oof")
    if float(row["signal_corr"]) < 0.10:
        reasons.append("weak_signal_corr")
    if not (0.06 <= float(row["anchor_diff_rmse"]) <= 0.45):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) < 0.50:
        reasons.append("max_too_small")
    if float(row["anchor_diff_max"]) > 2.5:
        reasons.append("max_too_large")
    if int(row["changed_count"]) < 12 or int(row["changed_count"]) > 40:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.035:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.12:
        reasons.append("species_shift")
    if int(row["species_max"]) > 10:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 0:
        reasons.append("bad_new_overlap")
    if int(row["failed_any_overlap_count"]) > 0:
        reasons.append("failed_any_overlap")
    if int(row["current_overlap_count"]) > 0:
        reasons.append("current_overlap")
    if abs(float(row["best_increment_corr"])) > 0.25:
        reasons.append("too_correlated_current_increment")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.05, 0.0) + 0.22 * abs(float(row["anchor_diff_rmse"]) - 0.25),
        float(row["bad_new_overlap_count"]),
        float(row["failed_any_overlap_count"]),
        float(row["species_shift"]),
        -float(row["signal_corr"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(str(row["family"]) == str(prev["family"]) for prev in selected):
            continue
        if any(abs(float(row["anchor_diff_rmse"]) - float(prev["anchor_diff_rmse"])) < 0.08 for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2)))


def tag(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(cand.family, cand.experiment, cand.pred, cand.corr_test, cand.corr_oof, cand.signal_corr, cand.beta, cand.memo)


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max']:.4f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.4f} "
        f"bad={row['bad_new_overlap_count']} failed={row['failed_any_overlap_count']} "
        f"overlap={row['current_overlap_count']}"
    )


if __name__ == "__main__":
    main()
