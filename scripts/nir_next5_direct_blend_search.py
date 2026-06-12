#!/usr/bin/env python3
"""Lightweight direct expert blend search after next5 #2.

This checks whether any direct golden/local PLS expert is strong enough to
change the prediction surface, before attempting heavier MoE gates.
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
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn
import nir_testlike_golden_distill_search as tlg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_direct_blend"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BEST = fu.BEST
CAND2 = fu.CAND2


@dataclass(frozen=True)
class Expert:
    name: str
    oof: np.ndarray
    test: np.ndarray
    signal_corr: float
    oof_delta: float


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
    args = parser.parse_args()

    data = op.load_data()
    best = n5.read_submission(BEST, data["test_ids"])
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    diag = load_public_delta_diag()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building #2 anchor OOF", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    print("building direct experts", flush=True)
    experts = build_experts(data, best_oof, best, residual, base_rmse)
    branch_rows = [
        {
            "expert": expert.name,
            "oof_rmse": rmse(data["y"], expert.oof),
            "oof_delta": expert.oof_delta,
            "signal_corr": expert.signal_corr,
            "test_diff_rmse": float(math.sqrt(np.mean((expert.test - best) ** 2))),
            "test_diff_max": float(np.max(np.abs(expert.test - best))),
        }
        for expert in experts
    ]

    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)
    for expert in experts:
        for cand in candidate_variants(
            expert,
            best_oof,
            best,
            residual,
            detector_oof,
            detector_test,
            data["groups"],
            test_species,
        ):
            row = diagnostics(cand, data["y"], best_oof, base_rmse, best, best - cand2, test_species, diag, expert)
            rows.append(row)
            candidates.append(cand)
            if row["submit_gate"] == "pass":
                print_one(row)

    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(branch_rows).to_csv(out_dir / "direct_blend_experts.csv", index=False)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "direct_blend_summary.csv", index=False)
    (out_dir / "direct_blend_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_direct_blend_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "direct_blend_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "direct_blend_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "direct_blend_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nExperts:")
    print(pd.DataFrame(branch_rows).sort_values("oof_delta").to_string(index=False))
    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:60]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_experts(
    data: dict[str, np.ndarray],
    best_oof: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    base_rmse: float,
) -> list[Expert]:
    out: list[Expert] = []
    branch_specs = [
        stn.BranchSpec("sg9_k65_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.65, 5, 1.6),
        stn.BranchSpec("sg9_k75_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6),
        stn.BranchSpec("sg9_k85_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.85, 5, 1.6),
        stn.BranchSpec("msc_k65_q16_c6", "msc_sg9", "pls_raw", "knn_cluster", 0.65, 6, 1.6),
        stn.BranchSpec("msc_k75_q13_c5", "msc_sg9", "pls_raw", "knn_cluster", 0.75, 5, 1.3),
    ]
    for branch in branch_specs:
        oof = stn.make_quality_branch_oof(branch, data, residual)
        test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
        out.append(
            Expert(
                name=branch.name,
                oof=oof,
                test=np.clip(test, 0, None),
                signal_corr=n5.safe_corr(oof - best_oof, residual),
                oof_delta=rmse(data["y"], oof) - base_rmse,
            )
        )
    golden_specs = [
        tlg.TestLikeSpec("golden_msc_keep75_c4", "msc_sg9", "pls_raw", "knn_cluster", 0.75, 4),
        tlg.TestLikeSpec("golden_sg9_keep80_c3", "sg9_snv", "pls_raw", "knn", 0.80, 3),
    ]
    for spec in golden_specs:
        oof, _ = tlg.make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        test = tlg.fit_predict_testlike_branch(spec, data["X_train"], data["y"], data["X_test"], groups=data["groups"])
        out.append(
            Expert(
                name=spec.name,
                oof=np.asarray(oof, dtype=float),
                test=np.clip(np.asarray(test, dtype=float), 0, None),
                signal_corr=n5.safe_corr(np.asarray(oof, dtype=float) - best_oof, residual),
                oof_delta=rmse(data["y"], np.asarray(oof, dtype=float)) - base_rmse,
            )
        )
    return out


def candidate_variants(
    expert: Expert,
    best_oof: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> list[Candidate]:
    signal_oof = expert.oof - best_oof
    signal_test = expert.test - best
    out: list[Candidate] = []
    for blend in [0.15, 0.25, 0.35, 0.50]:
        out.append(make_candidate(expert, "global", blend, signal_oof, signal_test, best, mask_oof=None, mask_test=None))
    for frac in [0.08, 0.12, 0.18, 0.25]:
        score_oof = ppg.rank01(np.abs(signal_oof))
        score_test = ppg.rank01(np.abs(signal_test))
        for mode, so, st in [
            ("abs", score_oof, score_test),
            ("det_abs", score_oof * ppg.rank01(detector_oof), score_test * ppg.rank01(detector_test)),
            ("resid_abs", score_oof * ppg.rank01(np.abs(residual)), score_test),
        ]:
            mask_oof = top_fraction(so, frac)
            mask_test = top_fraction(st, frac)
            for blend in [0.25, 0.40, 0.60]:
                out.append(make_candidate(expert, f"{mode}_f{stn.tag(frac)}", blend, signal_oof, signal_test, best, mask_oof, mask_test))
    for frac in [0.08, 0.12, 0.18]:
        for clip in [2.0, 4.0, 6.0]:
            for blend in [0.25, 0.40, 0.60]:
                for species_cap in [0, 2, 3, 4]:
                    out.append(
                        make_balanced_candidate(
                            expert,
                            f"balanced_f{stn.tag(frac)}_clip{stn.tag(clip)}_sp{species_cap}",
                            blend,
                            clip,
                            signal_oof,
                            signal_test,
                            best,
                            frac,
                            train_species,
                            test_species,
                            species_cap,
                        )
                    )
    for frac in [0.08, 0.12, 0.18, 0.25]:
        for clip in [0.50, 0.80, 1.20, 1.60]:
            for blend in [0.12, 0.18, 0.25, 0.32]:
                for species_cap in [2, 3, 4]:
                    out.append(
                        make_balanced_candidate(
                            expert,
                            f"balanced_safe_f{stn.tag(frac)}_clip{stn.tag(clip)}_sp{species_cap}",
                            blend,
                            clip,
                            signal_oof,
                            signal_test,
                            best,
                            frac,
                            train_species,
                            test_species,
                            species_cap,
                        )
                    )
    return out


def make_candidate(
    expert: Expert,
    mode: str,
    blend: float,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    best: np.ndarray,
    mask_oof: np.ndarray | None,
    mask_test: np.ndarray | None,
) -> Candidate:
    if mask_oof is None:
        corr_oof = blend * signal_oof
        corr_test = blend * signal_test
    else:
        corr_oof = np.where(mask_oof, blend * signal_oof, 0.0)
        corr_test = np.where(mask_test, blend * signal_test, 0.0)
    return Candidate(
        family="direct_blend",
        experiment=f"db_{expert.name}_{mode}_b{stn.tag(blend)}",
        pred=np.clip(best + corr_test, 0, None),
        corr_oof=corr_oof,
        corr_test=corr_test,
        beta=blend,
        signal_corr=expert.signal_corr,
        memo=f"#2 anchor; direct blend expert={expert.name}; mode={mode}; blend={blend}; expert_oof_delta={expert.oof_delta:.4f}; signal_corr={expert.signal_corr:.4f}.",
    )


def make_balanced_candidate(
    expert: Expert,
    mode: str,
    blend: float,
    clip: float,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    best: np.ndarray,
    frac: float,
    train_species: np.ndarray | None = None,
    test_species: np.ndarray | None = None,
    species_cap: int = 0,
) -> Candidate:
    corr_oof = balanced_correction(signal_oof, blend, clip, frac, train_species, species_cap)
    corr_test = balanced_correction(signal_test, blend, clip, frac, test_species, species_cap)
    return Candidate(
        family="direct_blend",
        experiment=f"db_{expert.name}_{mode}_b{stn.tag(blend)}",
        pred=np.clip(best + corr_test, 0, None),
        corr_oof=corr_oof,
        corr_test=corr_test,
        beta=blend,
        signal_corr=expert.signal_corr,
        memo=(
            f"#2 anchor; balanced direct blend expert={expert.name}; frac={frac}; "
            f"blend={blend}; clip={clip}; species_cap={species_cap}; expert_oof_delta={expert.oof_delta:.4f}; "
            f"signal_corr={expert.signal_corr:.4f}."
        ),
    )


def balanced_correction(
    signal: np.ndarray,
    blend: float,
    clip: float,
    frac: float,
    labels: np.ndarray | None = None,
    species_cap: int = 0,
) -> np.ndarray:
    n_each = max(1, int(round(len(signal) * frac / 2.0)))
    pos_order = np.argsort(signal, kind="mergesort")[::-1]
    neg_order = np.argsort(signal, kind="mergesort")
    selected: list[int] = []
    pos_counts: dict[int, int] = {}
    neg_counts: dict[int, int] = {}
    for idx in pos_order:
        if signal[idx] <= 0 or len([i for i in selected if signal[i] > 0]) >= n_each:
            break
        if labels is not None and species_cap > 0:
            label = int(labels[idx])
            if pos_counts.get(label, 0) >= species_cap:
                continue
            pos_counts[label] = pos_counts.get(label, 0) + 1
        selected.append(int(idx))
    for idx in neg_order:
        if signal[idx] >= 0 or len([i for i in selected if signal[i] < 0]) >= n_each:
            break
        if labels is not None and species_cap > 0:
            label = int(labels[idx])
            if neg_counts.get(label, 0) >= species_cap:
                continue
            neg_counts[label] = neg_counts.get(label, 0) + 1
        selected.append(int(idx))
    corr = np.zeros(len(signal), dtype=float)
    if selected:
        idx = np.asarray(sorted(set(selected)), dtype=int)
        raw = np.clip(blend * signal[idx], -clip, clip)
        raw = raw - float(np.mean(raw))
        corr[idx] = raw
    return corr


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
    diag: pd.DataFrame | None,
    expert: Expert,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, best_oof, base_rmse, best, best_increment, test_species)
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "expert": expert.name,
            "expert_oof_delta": expert.oof_delta,
            "bad_new_overlap_count": bad_new_overlap(changed, diag),
            "diff_p95": float(np.quantile(np.abs(diff[changed]), 0.95)) if np.any(changed) else 0.0,
            "tail_low_shift": float(np.mean(diff[best <= np.quantile(best, 0.10)])),
            "tail_high_shift": float(np.mean(diff[best >= np.quantile(best, 0.90)])),
        }
    )
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.02:
        reasons.append("weak_oof")
    if float(row["expert_oof_delta"]) >= 0:
        reasons.append("bad_expert_oof")
    if float(row["signal_corr"]) < 0.25:
        reasons.append("weak_signal_corr")
    if not (0.04 <= float(row["anchor_diff_rmse"]) <= 0.65):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) < 0.20:
        reasons.append("max_too_small")
    if float(row["anchor_diff_max"]) > 2.0:
        reasons.append("max_too_large")
    if int(row["changed_count"]) < 18 or int(row["changed_count"]) > 90:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.05:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.22:
        reasons.append("species_shift")
    if int(row["species_max"]) > 12:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 3:
        reasons.append("bad_new_overlap")
    if abs(float(row["best_increment_corr"])) > 0.75:
        reasons.append("too_correlated_best")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.08, 0.0) + abs(float(row["anchor_diff_rmse"]) - 0.35),
        float(row["bad_new_overlap_count"]),
        float(row["species_shift"]),
        -float(row["signal_corr"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(str(row["expert"]) == str(prev["expert"]) for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    order = np.argsort(score, kind="mergesort")[::-1]
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:target]] = True
    return mask


def load_public_delta_diag() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "stage5_public_delta_diagnostics" / "row_public_delta_diagnostics.csv"
    return pd.read_csv(path) if path.exists() else None


def bad_new_overlap(changed: np.ndarray, diag: pd.DataFrame | None) -> int:
    if diag is None:
        return 0
    mask = np.zeros(len(changed), dtype=bool)
    if "failed_new_vs_cand2" in diag.columns:
        mask |= diag["failed_new_vs_cand2"].astype(bool).to_numpy()
    if "broad_changed" in diag.columns and "cand2_changed" in diag.columns:
        mask |= diag["broad_changed"].astype(bool).to_numpy() & ~diag["cand2_changed"].astype(bool).to_numpy()
    return int(np.sum(changed & mask))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2)))


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(cand.family, cand.experiment, cand.pred, cand.corr_test, cand.corr_oof, cand.signal_corr, cand.beta, cand.memo)


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max']:.4f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.4f} bad={row['bad_new_overlap_count']}"
    )


if __name__ == "__main__":
    main()
