#!/usr/bin/env python3
"""Follow-up search after next5 #2 became the current best.

This script is deliberately stricter than normal candidate builders.  It uses
the new Public-best test-near golden PLS submission as the anchor and keeps
only candidates with enough local evidence to justify spending a submission.
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

import nir_next5_diverse_candidate_builder as n5
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn
import nir_testlike_golden_distill_search as tlg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_followup"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CAND2 = n5.CAND2
BEST = SUBMISSION_DIR / "nir_next5_2_next5_tn_pls_k75_q16_c5_abs_signal_cluster1_f0p03_s0p018_20260611.csv"


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
    parser.add_argument("--save-top", type=int, default=5)
    args = parser.parse_args()

    data = op.load_data()
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    best = n5.read_submission(BEST, data["test_ids"])
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reconstructing current-best OOF proxy", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    cluster_train, cluster_test = n5.cluster_labels(F_train, F_test, n_clusters=56)
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)

    candidates: list[Candidate] = []
    candidates.extend(best_mask_continuation(best, cand2, best_corr_oof))
    candidates.extend(stronger_testnear_variants(data, q5_oof, best_oof, cand2, best, residual, F_train, F_test, cluster_train, cluster_test))
    local_oof, local_test = n5.local_residual_signal(F_train, data["y"], residual, data["groups"], F_test, k=55)
    candidates.extend(detector_variants(best, best_oof, residual, local_oof, local_test, detector_oof, detector_test, cluster_train, cluster_test))
    candidates.extend(local_mbl_variants(best, best_oof, residual, local_oof, local_test, cluster_train, cluster_test))
    candidates.extend(golden_variants(data, best, best_oof, residual, cluster_train, cluster_test))

    rows = [
        diagnostics(cand, data["y"], best_oof, base_rmse, best, best - cand2, test_species)
        for cand in candidates
    ]
    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "followup_summary.csv", index=False)
    (out_dir / "followup_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_followup_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "followup_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "followup_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "followup_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop follow-up rows:")
    print(pd.DataFrame(rows_sorted[:20]).to_string(index=False))
    print("\nSubmit-worthy selected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_best2_corr_oof(data: dict[str, np.ndarray], q5_oof: np.ndarray, cand2: np.ndarray) -> np.ndarray:
    residual = data["y"] - q5_oof
    branch = stn.BranchSpec("next5_tn_pls_k75_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6)
    branch_oof = stn.make_quality_branch_oof(branch, data, residual)
    branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
    signal_oof = branch_oof - q5_oof
    signal_test = branch_test - cand2
    beta = n5.fit_beta(signal_oof, residual)
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    cluster_train, cluster_test = n5.cluster_labels(F_train, F_test, n_clusters=48)
    corr_oof, _ = stn.make_candidate(
        stn.CandidateSpec(branch, "abs_signal_cluster1", 0.03, 0.018, 0.14, False),
        signal_oof,
        signal_test,
        q5_oof,
        cand2,
        residual,
        beta,
        cluster_train,
        cluster_test,
        F_train,
        F_test,
    )
    return corr_oof


def best_mask_continuation(best: np.ndarray, cand2: np.ndarray, best_corr_oof: np.ndarray) -> list[Candidate]:
    increment = best - cand2
    out: list[Candidate] = []
    for alpha in [1.08, 1.15, 1.25]:
        corr_test = (alpha - 1.0) * increment
        corr_oof = (alpha - 1.0) * best_corr_oof
        out.append(
            Candidate(
                family="best2_mask_continuation",
                experiment=f"best2_mask_alpha{stn.tag(alpha)}",
                pred=np.clip(best + corr_test, 0, None),
                corr_oof=corr_oof,
                corr_test=corr_test,
                beta=0.0,
                signal_corr=0.0,
                memo=f"Current best #2 mask fixed continuation alpha={alpha}; no new rows.",
            )
        )
    return out


def stronger_testnear_variants(
    data: dict[str, np.ndarray],
    q5_oof: np.ndarray,
    best_oof: np.ndarray,
    cand2: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    out: list[Candidate] = []
    primary = stn.BranchSpec("fu_tn_k75_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6)
    out.extend(
        make_testnear_grid(
            primary,
            data,
            best_oof,
            best,
            residual,
            F_train,
            F_test,
            cluster_train,
            cluster_test,
            gate_modes=["abs_signal_cluster2", "abs_signal_cluster2_space02", "abs_signal_cluster2_space05", "abs_signal_cluster1_space02"],
            fracs=[0.032, 0.035, 0.038],
            shrinks=[0.020, 0.022, 0.024],
            clips=[0.12, 0.14, 0.16],
        )
    )
    out.extend(
        make_testnear_grid(
            primary,
            data,
            best_oof,
            best,
            residual,
            F_train,
            F_test,
            cluster_train,
            cluster_test,
            gate_modes=["abs_signal_cluster1_space02"],
            fracs=[0.035, 0.038, 0.040],
            shrinks=[0.028, 0.032, 0.036],
            clips=[0.14, 0.16],
        )
    )
    secondary_specs = [
        (stn.BranchSpec("fu_tn_k65_q13_c4", "sg9_snv", "pls_raw", "knn_cluster", 0.65, 4, 1.3), "abs_signal_cluster1", 0.030, 0.018, 0.14),
        (stn.BranchSpec("fu_tn_k75_q16_c5_neg", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6), "negative_signal_cluster1", 0.030, 0.020, 0.14),
    ]
    for branch, gate_mode, frac, shrink, clip in secondary_specs:
        out.extend(
            make_testnear_grid(
                branch,
                data,
                best_oof,
                best,
                residual,
                F_train,
                F_test,
                cluster_train,
                cluster_test,
                gate_modes=[gate_mode],
                fracs=[frac],
                shrinks=[shrink],
                clips=[clip],
            )
        )
    return out


def make_testnear_grid(
    branch: stn.BranchSpec,
    data: dict[str, np.ndarray],
    best_oof: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
    *,
    gate_modes: list[str],
    fracs: list[float],
    shrinks: list[float],
    clips: list[float],
) -> list[Candidate]:
    branch_oof = stn.make_quality_branch_oof(branch, data, residual)
    branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
    signal_oof = branch_oof - best_oof
    signal_test = branch_test - best
    beta = n5.fit_beta(signal_oof, residual)
    signal_corr = stn.safe_corr(signal_oof, residual)
    out: list[Candidate] = []
    for gate_mode in gate_modes:
        for frac in fracs:
            for shrink in shrinks:
                for clip in clips:
                    corr_oof, pred = stn.make_candidate(
                        stn.CandidateSpec(branch, gate_mode, frac, shrink, clip, False),
                        signal_oof,
                        signal_test,
                        best_oof,
                        best,
                        residual,
                        beta,
                        cluster_train,
                        cluster_test,
                        F_train,
                        F_test,
                    )
                    out.append(
                        Candidate(
                            family="testnear_followup",
                            experiment=f"{branch.name}_{gate_mode}_f{stn.tag(frac)}_s{stn.tag(shrink)}_c{stn.tag(clip)}",
                            pred=pred,
                            corr_oof=corr_oof,
                            corr_test=pred - best,
                            beta=beta,
                            signal_corr=signal_corr,
                            memo=f"#2 anchor; follow-up test-near PLS branch {branch.name}; {gate_mode}; frac={frac}; shrink={shrink}; clip={clip}.",
                        )
                    )
    return out


def detector_variants(
    best: np.ndarray,
    best_oof: np.ndarray,
    residual: np.ndarray,
    local_oof: np.ndarray,
    local_test: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    beta = n5.fit_beta(local_oof, residual)
    out: list[Candidate] = []
    for det_frac, frac, cap, shrink, clip in [(0.10, 0.040, 1, 0.028, 0.10), (0.16, 0.055, 2, 0.024, 0.12)]:
        score_oof = np.where(n5.top_fraction(detector_oof, det_frac), np.abs(local_oof), 0.0)
        score_test = np.where(n5.top_fraction(detector_test, det_frac), np.abs(local_test), 0.0)
        gate_oof = n5.diverse_top_fraction(score_oof, frac, cluster_train, cap)
        gate_test = n5.diverse_top_fraction(score_test, frac, cluster_test, cap)
        corr_oof = np.where(gate_oof, np.clip(shrink * beta * local_oof, -clip, clip), 0.0)
        corr_test = np.where(gate_test, np.clip(shrink * beta * local_test, -clip, clip), 0.0)
        out.append(
            Candidate(
                family="detector_followup",
                experiment=f"det_follow_df{stn.tag(det_frac)}_f{stn.tag(frac)}_cap{cap}_s{stn.tag(shrink)}",
                pred=np.clip(best + corr_test, 0, None),
                corr_oof=corr_oof,
                corr_test=corr_test,
                beta=beta,
                signal_corr=stn.safe_corr(local_oof, residual),
                memo=f"#2 anchor; detector intersected local residual; detector top={det_frac}; frac={frac}; cap={cap}; shrink={shrink}.",
            )
        )
    return out


def local_mbl_variants(
    best: np.ndarray,
    best_oof: np.ndarray,
    residual: np.ndarray,
    local_oof: np.ndarray,
    local_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    beta = n5.fit_beta(local_oof, residual)
    out: list[Candidate] = []
    for frac, cap, shrink, clip in [(0.050, 1, 0.018, 0.10), (0.065, 2, 0.016, 0.12)]:
        gate_oof = n5.diverse_top_fraction(np.abs(local_oof), frac, cluster_train, cap)
        gate_test = n5.diverse_top_fraction(np.abs(local_test), frac, cluster_test, cap)
        corr_oof = np.where(gate_oof, np.clip(shrink * beta * local_oof, -clip, clip), 0.0)
        corr_test = np.where(gate_test, np.clip(shrink * beta * local_test, -clip, clip), 0.0)
        out.append(
            Candidate(
                family="local_mbl_followup",
                experiment=f"mbl_follow_f{stn.tag(frac)}_cap{cap}_s{stn.tag(shrink)}",
                pred=np.clip(best + corr_test, 0, None),
                corr_oof=corr_oof,
                corr_test=corr_test,
                beta=beta,
                signal_corr=stn.safe_corr(local_oof, residual),
                memo=f"#2 anchor; local MBL residual follow-up; frac={frac}; cap={cap}; shrink={shrink}.",
            )
        )
    return out


def golden_variants(
    data: dict[str, np.ndarray],
    best: np.ndarray,
    best_oof: np.ndarray,
    residual: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    specs = [
        tlg.TestLikeSpec("fu_golden_msc_keep75_c4", "msc_sg9", "pls_raw", "knn_cluster", 0.75, 4),
        tlg.TestLikeSpec("fu_golden_msc_keep80_c3", "msc_sg9", "pls_raw", "knn", 0.80, 3),
    ]
    out: list[Candidate] = []
    for spec in specs:
        branch_oof, _ = tlg.make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test = tlg.fit_predict_testlike_branch(spec, data["X_train"], data["y"], data["X_test"], groups=data["groups"])
        signal_oof = branch_oof - best_oof
        signal_test = np.asarray(branch_test, dtype=float) - best
        beta = n5.fit_beta(signal_oof, residual)
        for frac, cap, shrink, clip in [(0.045, 1, 0.010, 0.07), (0.055, 1, 0.012, 0.08)]:
            raw_oof = shrink * beta * signal_oof
            raw_test = shrink * beta * signal_test
            gate_oof = n5.diverse_top_fraction(np.abs(raw_oof), frac, cluster_train, cap)
            gate_test = n5.diverse_top_fraction(np.abs(raw_test), frac, cluster_test, cap)
            corr_oof = np.where(gate_oof, np.clip(raw_oof, -clip, clip), 0.0)
            corr_test = np.where(gate_test, np.clip(raw_test, -clip, clip), 0.0)
            out.append(
                Candidate(
                    family="golden_followup",
                    experiment=f"{spec.name}_f{stn.tag(frac)}_cap{cap}_s{stn.tag(shrink)}",
                    pred=np.clip(best + corr_test, 0, None),
                    corr_oof=corr_oof,
                    corr_test=corr_test,
                    beta=beta,
                    signal_corr=stn.safe_corr(signal_oof, residual),
                    memo=f"#2 anchor; golden subset follow-up {spec.preprocess}/{spec.score_mode}; keep={spec.keep_frac}; c={spec.n_components}.",
                )
            )
    return out


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
) -> dict[str, object]:
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    corrected_oof = np.clip(best_oof + cand.corr_oof, 0, None)
    oof_rmse = rmse(y, corrected_oof)
    species_counts = pd.Series(test_species[changed]).value_counts() if changed.any() else pd.Series(dtype=int)
    top_idx = np.argsort(-np.abs(diff))[:10]
    top_species = pd.Series(test_species[top_idx[np.abs(diff[top_idx]) > 1e-12]]).value_counts()
    row = {
        "family": cand.family,
        "experiment": cand.experiment,
        "submit_gate": "pass",
        "reject_reasons": "",
        "oof_delta": oof_rmse - base_rmse,
        "oof_rmse": oof_rmse,
        "signal_corr": cand.signal_corr,
        "beta": cand.beta,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "changed_count": int(changed.sum()),
        "positive_count": int(np.sum(diff > 1e-12)),
        "negative_count": int(np.sum(diff < -1e-12)),
        "species_shift": n5.max_species_shift(diff, test_species),
        "species_max": int(species_counts.iloc[0]) if len(species_counts) else 0,
        "top10_species_max": int(top_species.iloc[0]) if len(top_species) else 0,
        "best_increment_corr": n5.safe_corr(diff, best_increment),
        "pred_min": float(np.min(cand.pred)),
        "pred_mean": float(np.mean(cand.pred)),
        "pred_median": float(np.median(cand.pred)),
        "pred_max": float(np.max(cand.pred)),
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    family = str(row["family"])
    if int(row["changed_count"]) < 18 and family not in {"best2_mask_continuation", "detector_followup"}:
        reasons.append("too_few_changes")
    if int(row["changed_count"]) > 35 and family != "best2_mask_continuation":
        reasons.append("too_many_changes")
    if float(row["anchor_diff_rmse"]) < 0.010 and family not in {"best2_mask_continuation", "detector_followup"}:
        reasons.append("too_small_for_12s_goal")
    if float(row["anchor_diff_rmse"]) > 0.020 and family not in {"detector_followup"}:
        reasons.append("large_diff_rmse")
    if float(row["anchor_diff_max"]) > 0.16:
        reasons.append("large_max_diff")
    if abs(float(row["anchor_diff_mean"])) > 0.004:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.010:
        reasons.append("species_shift")
    if int(row["species_max"]) > 12:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if family != "best2_mask_continuation" and float(row["oof_delta"]) > (-0.0035 if family == "detector_followup" else -0.0045):
        reasons.append("weak_oof_for_12s_goal")
    if family == "best2_mask_continuation" and float(row["oof_delta"]) > -0.0003:
        reasons.append("weak_alpha_oof")
    if abs(float(row["best_increment_corr"])) > 0.70 and family != "best2_mask_continuation":
        reasons.append("too_correlated_with_best2")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    oof = float(row["oof_delta"])
    diff = float(row["anchor_diff_rmse"])
    # For a 12s-goal queue, prefer non-trivial, OOF-supported movement.
    return (
        penalty + max(oof + 0.006, 0.0) + abs(diff - 0.012),
        float(row["species_shift"]),
        float(row.get("max_pairwise_corr", 0.0)),
        float(row["anchor_diff_max"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    out = [
        row
        for row in rows
        if row["submit_gate"] == "pass"
        and float(row["oof_delta"]) <= (-0.0035 if row["family"] == "detector_followup" else -0.0045)
        and float(row["anchor_diff_rmse"]) >= (0.006 if row["family"] == "detector_followup" else 0.010)
        and float(row.get("max_pairwise_corr", 0.0)) <= 0.90
        and abs(float(row.get("best_increment_corr", 0.0))) <= 0.70
    ]
    return out[:limit]


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(
        family=cand.family,
        experiment=cand.experiment,
        pred=cand.pred,
        corr_test=cand.corr_test,
        corr_oof=cand.corr_oof,
        signal_corr=cand.signal_corr,
        beta=cand.beta,
        memo=cand.memo,
    )


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(((y - pred) ** 2).mean()))


if __name__ == "__main__":
    main()
