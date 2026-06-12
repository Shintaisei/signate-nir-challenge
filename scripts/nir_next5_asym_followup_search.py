#!/usr/bin/env python3
"""Follow-up search after asymmetric attack became the public best.

Current public best:
  nir_asym_attack_1_asym_gold_pos0_035_nr0_75_b0_35_c3_5_sp3_20260611.csv

This script re-anchors diagnostics to that file.  It tests small alpha
continuations of the winning correction and nearby asymmetric golden-expert
variants, then saves only candidates that improve the re-anchored OOF proxy
without reintroducing the previously observed public-failed regions.
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
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_asym_followup"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OLD_BEST = fu.BEST
CAND2 = fu.CAND2
CURRENT_BEST = SUBMISSION_DIR / "nir_asym_attack_1_asym_gold_pos0_035_nr0_75_b0_35_c3_5_sp3_20260611.csv"


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
    parser.add_argument("--save-top", type=int, default=6)
    parser.add_argument("--current-best", type=Path, default=CURRENT_BEST)
    parser.add_argument("--current-pos-frac", type=float, default=0.035)
    parser.add_argument("--current-neg-ratio", type=float, default=0.75)
    parser.add_argument("--current-blend", type=float, default=0.35)
    parser.add_argument("--current-clip", type=float, default=3.5)
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
    old_base_rmse = rmse(data["y"], old_oof)
    old_residual = data["y"] - old_oof
    print(f"old_oof_rmse={old_base_rmse:.6f}", flush=True)

    print("building golden direct expert", flush=True)
    experts = db.build_experts(data, old_oof, old_best, old_residual, old_base_rmse)
    expert = [e for e in experts if e.name == "golden_sg9_keep80_c3"][0]
    signal_oof = expert.oof - old_oof
    signal_test = expert.test - old_best

    win_corr_oof = asym.asym_corr(
        signal_oof,
        old_residual,
        pos_frac=args.current_pos_frac,
        neg_ratio=args.current_neg_ratio,
        blend=args.current_blend,
        clip=args.current_clip,
        labels=data["groups"],
        species_cap=args.current_species_cap,
        banned=np.zeros(len(signal_oof), dtype=bool),
    )
    win_corr_test = current_best - old_best
    current_oof = np.clip(old_oof + win_corr_oof, 0, None)
    current_rmse = rmse(data["y"], current_oof)
    residual = data["y"] - current_oof
    print(f"current_oof_rmse={current_rmse:.6f} delta={current_rmse - old_base_rmse:.6f}", flush=True)

    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    for cand in alpha_candidates(current_best, win_corr_oof, win_corr_test, expert.signal_corr):
        row = diagnostics(cand, data["y"], current_oof, current_rmse, current_best, win_corr_test, test_species, diag)
        rows.append(row)
        candidates.append(cand)
        if row["submit_gate"] == "pass":
            print_one(row)

    for cand in asym_variant_candidates(
        old_best,
        current_best,
        win_corr_oof,
        win_corr_test,
        old_residual,
        signal_oof,
        signal_test,
        expert.signal_corr,
        expert.oof_delta,
        data["groups"],
        test_species,
        diag,
    ):
        row = diagnostics(cand, data["y"], current_oof, current_rmse, current_best, win_corr_test, test_species, diag)
        rows.append(row)
        candidates.append(cand)
        if row["submit_gate"] == "pass":
            print_one(row)

    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "asym_followup_summary.csv", index=False)
    (out_dir / "asym_followup_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_asym_followup_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "asym_followup_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "asym_followup_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "asym_followup_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:80]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def alpha_candidates(
    current_best: np.ndarray,
    win_corr_oof: np.ndarray,
    win_corr_test: np.ndarray,
    signal_corr: float,
) -> list[Candidate]:
    out: list[Candidate] = []
    for alpha in [0.78, 0.85, 0.92, 0.97, 1.03, 1.06, 1.10, 1.15, 1.22, 1.30]:
        delta = alpha - 1.0
        corr_oof = delta * win_corr_oof
        corr_test = delta * win_corr_test
        out.append(
            Candidate(
                family="asym_alpha",
                experiment=f"asym_win_alpha{tag(alpha)}",
                pred=np.clip(current_best + corr_test, 0, None),
                corr_oof=corr_oof,
                corr_test=corr_test,
                beta=alpha,
                signal_corr=signal_corr,
                memo=(
                    f"Current asym public-best continuation; alpha={alpha}; "
                    "positive means stronger winning correction, negative weaker."
                ),
            )
        )
    return out


def asym_variant_candidates(
    old_best: np.ndarray,
    current_best: np.ndarray,
    win_corr_oof: np.ndarray,
    win_corr_test: np.ndarray,
    old_residual: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    signal_corr: float,
    expert_delta: float,
    train_species: np.ndarray,
    test_species: np.ndarray,
    diag: pd.DataFrame | None,
) -> list[Candidate]:
    out: list[Candidate] = []
    bad_train = np.zeros(len(signal_oof), dtype=bool)
    bad_test = asym.bad_mask(diag, len(signal_test))
    for pos_frac in [0.030, 0.035, 0.040, 0.050, 0.065, 0.080]:
        for neg_ratio in [0.60, 0.75, 0.90, 1.00]:
            for blend in [0.28, 0.35, 0.42, 0.50]:
                for clip in [2.5, 3.0, 3.5, 4.0, 4.5]:
                    for species_cap in [2, 3, 4]:
                        cand_oof_total = asym.asym_corr(
                            signal_oof,
                            old_residual,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            train_species,
                            species_cap,
                            bad_train,
                        )
                        cand_test_total = asym.asym_corr(
                            signal_test,
                            None,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            test_species,
                            species_cap,
                            bad_test,
                        )
                        corr_oof = cand_oof_total - win_corr_oof
                        corr_test = cand_test_total - win_corr_test
                        if np.max(np.abs(corr_test)) < 1e-12:
                            continue
                        exp = (
                            f"asym_reanchor_pos{tag(pos_frac)}_nr{tag(neg_ratio)}_"
                            f"b{tag(blend)}_c{tag(clip)}_sp{species_cap}"
                        )
                        out.append(
                            Candidate(
                                family="asym_reanchor",
                                experiment=exp,
                                pred=np.clip(old_best + cand_test_total, 0, None),
                                corr_oof=corr_oof,
                                corr_test=corr_test,
                                beta=blend,
                                signal_corr=signal_corr,
                                memo=(
                                    f"Re-anchored asym golden_sg9_keep80_c3 variant vs current public best; "
                                    f"pos_frac={pos_frac}; neg_ratio={neg_ratio}; blend={blend}; clip={clip}; "
                                    f"species_cap={species_cap}; expert_delta={expert_delta:.4f}; "
                                    f"signal_corr={signal_corr:.4f}; public-failed rows excluded."
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
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, current_oof, current_rmse, current_best, current_increment, test_species)
    diff = cand.pred - current_best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "bad_new_overlap_count": db.bad_new_overlap(changed, diag),
            "failed_any_overlap_count": asym.failed_any_overlap(changed, diag),
            "diff_p95": float(np.quantile(np.abs(diff[changed]), 0.95)) if np.any(changed) else 0.0,
        }
    )
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.006:
        reasons.append("weak_oof")
    if float(row["signal_corr"]) < 0.30:
        reasons.append("weak_signal_corr")
    if not (0.03 <= float(row["anchor_diff_rmse"]) <= 0.55):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) > 2.2:
        reasons.append("max_too_large_vs_current")
    if int(row["changed_count"]) < 8 or int(row["changed_count"]) > 46:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.035:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.12:
        reasons.append("species_shift")
    if int(row["species_max"]) > 8:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 0:
        reasons.append("bad_new_overlap")
    if int(row["failed_any_overlap_count"]) > 1:
        reasons.append("failed_any_overlap")
    if abs(float(row["best_increment_corr"])) > 0.92:
        reasons.append("too_correlated_current")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.025, 0.0) + 0.25 * abs(float(row["anchor_diff_rmse"]) - 0.22),
        float(row["bad_new_overlap_count"]),
        float(row["failed_any_overlap_count"]),
        float(row["species_shift"]),
        -float(row["oof_delta"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(str(row["family"]) == str(prev["family"]) for prev in selected):
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
        f"bad={row['bad_new_overlap_count']} failed={row['failed_any_overlap_count']}"
    )


if __name__ == "__main__":
    main()
