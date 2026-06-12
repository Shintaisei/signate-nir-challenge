#!/usr/bin/env python3
"""Search regions similar to the rows fixed by next5 #2."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import nir_next5_diverse_candidate_builder as n5
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_winner_region"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BEST = fu.BEST
CAND2 = fu.CAND2


@dataclass(frozen=True)
class RegionSpec:
    name: str
    signal_name: str
    mode: str
    prox_power: float
    signal_power: float
    frac: float
    cap: int
    shrink: float
    clip: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--save-top", type=int, default=8)
    args = parser.parse_args()

    data = op.load_data()
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    best = n5.read_submission(BEST, data["test_ids"])
    best_increment = best - cand2
    winner_mask = np.abs(best_increment) > 1e-12
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building #2 anchor OOF", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = fu.rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}; winner_rows={int(winner_mask.sum())}", flush=True)

    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    train_clusters, test_clusters = n5.cluster_labels(F_train, F_test, n_clusters=64)
    prox_train, prox_test = winner_proximity(F_train, F_test, winner_mask)
    failed_train, failed_test = failed_new_proximity(F_train, F_test)
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)

    signal_cache = build_signals(data, best_oof, best, residual, F_train, F_test)
    rows: list[dict[str, object]] = []
    candidates: list[fu.Candidate] = []
    for spec in build_specs(signal_cache):
        signal_oof, signal_test, beta, signal_corr = signal_cache[spec.signal_name]
        score_oof, score_test = region_scores(
            spec,
            signal_oof,
            signal_test,
            prox_train,
            prox_test,
            failed_train,
            failed_test,
            detector_oof,
            detector_test,
            winner_mask,
        )
        gate_oof = n5.diverse_top_fraction(score_oof, spec.frac, train_clusters, spec.cap)
        gate_test = n5.diverse_top_fraction(score_test, spec.frac, test_clusters, spec.cap)
        corr_oof = np.where(gate_oof, np.clip(spec.shrink * beta * signal_oof, -spec.clip, spec.clip), 0.0)
        corr_test = np.where(gate_test, np.clip(spec.shrink * beta * signal_test, -spec.clip, spec.clip), 0.0)
        pred = np.clip(best + corr_test, 0, None)
        cand = fu.Candidate(
            family="winner_region",
            experiment=spec.name,
            pred=pred,
            corr_oof=corr_oof,
            corr_test=corr_test,
            beta=beta,
            signal_corr=signal_corr,
            memo=(
                f"#2 anchor; winner-region search; signal={spec.signal_name}; mode={spec.mode}; "
                f"prox_power={spec.prox_power}; signal_power={spec.signal_power}; frac={spec.frac}; "
                f"cap={spec.cap}; shrink={spec.shrink}; clip={spec.clip}."
            ),
        )
        row = diagnostics(cand, data["y"], best_oof, base_rmse, best, best_increment, test_species, winner_mask, prox_test)
        candidates.append(cand)
        rows.append(row)
        if row["submit_gate"] == "pass":
            print_one(row)

    n5.add_diversity_columns(rows, [fu.to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "winner_region_summary.csv", index=False)
    (out_dir / "winner_region_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_winner_region_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "winner_region_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "winner_region_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "winner_region_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:50]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_signals(
    data: dict[str, np.ndarray],
    best_oof: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, float, float]]:
    signals: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    local_oof, local_test = n5.local_residual_signal(F_train, data["y"], residual, data["groups"], F_test, k=55)
    signals["local_k55"] = pack_signal(local_oof, local_test, residual)
    for branch in [
        stn.BranchSpec("wr_sg9_k65_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.65, 5, 1.6),
        stn.BranchSpec("wr_sg9_k85_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.85, 5, 1.6),
        stn.BranchSpec("wr_msc_k65_q16_c6", "msc_sg9", "pls_raw", "knn_cluster", 0.65, 6, 1.6),
    ]:
        branch_oof = stn.make_quality_branch_oof(branch, data, residual)
        branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
        signals[branch.name] = pack_signal(branch_oof - best_oof, branch_test - best, residual)
    return signals


def pack_signal(signal_oof: np.ndarray, signal_test: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    beta = n5.fit_beta(signal_oof, residual)
    return signal_oof, signal_test, beta, n5.safe_corr(signal_oof, residual)


def build_specs(signal_cache: dict[str, tuple[np.ndarray, np.ndarray, float, float]]) -> list[RegionSpec]:
    specs: list[RegionSpec] = []
    for signal_name in signal_cache:
        for mode in ["contrast_near_abs", "contrast_det_abs", "new_near_abs", "det_near_abs"]:
            for frac in [0.025, 0.032, 0.04, 0.055]:
                for cap in [1, 2]:
                    for shrink in [0.022, 0.030, 0.040]:
                        specs.append(
                            RegionSpec(
                                name=(
                                    f"wr_{signal_name}_{mode}_f{stn.tag(frac)}_"
                                    f"cap{cap}_s{stn.tag(shrink)}"
                                ),
                                signal_name=signal_name,
                                mode=mode,
                                prox_power=1.5,
                                signal_power=1.0,
                                frac=frac,
                                cap=cap,
                                shrink=shrink,
                                clip=0.16,
                            )
                        )
    return specs


def winner_proximity(F_train: np.ndarray, F_test: np.ndarray, winner_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(winner_mask):
        raise ValueError("winner mask is empty")
    winner = F_test[winner_mask]
    nn = NearestNeighbors(n_neighbors=min(5, len(winner)), metric="euclidean").fit(winner)
    train_dist = nn.kneighbors(F_train, return_distance=True)[0].mean(axis=1)
    test_dist = nn.kneighbors(F_test, return_distance=True)[0].mean(axis=1)
    return ppg.rank01(-train_dist), ppg.rank01(-test_dist)


def failed_new_proximity(F_train: np.ndarray, F_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "outputs" / "stage5_public_delta_diagnostics" / "row_public_delta_diagnostics.csv"
    if not path.exists():
        return np.zeros(len(F_train), dtype=float), np.zeros(len(F_test), dtype=float)
    df = pd.read_csv(path)
    if "failed_new_vs_cand2" not in df.columns:
        return np.zeros(len(F_train), dtype=float), np.zeros(len(F_test), dtype=float)
    mask = df["failed_new_vs_cand2"].astype(bool).to_numpy()
    if not np.any(mask):
        return np.zeros(len(F_train), dtype=float), np.zeros(len(F_test), dtype=float)
    failed = F_test[mask]
    nn = NearestNeighbors(n_neighbors=min(5, len(failed)), metric="euclidean").fit(failed)
    train_dist = nn.kneighbors(F_train, return_distance=True)[0].mean(axis=1)
    test_dist = nn.kneighbors(F_test, return_distance=True)[0].mean(axis=1)
    return ppg.rank01(-train_dist), ppg.rank01(-test_dist)


def region_scores(
    spec: RegionSpec,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    prox_train: np.ndarray,
    prox_test: np.ndarray,
    failed_train: np.ndarray,
    failed_test: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    winner_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signal_score_oof = np.power(ppg.rank01(np.abs(signal_oof)), spec.signal_power)
    signal_score_test = np.power(ppg.rank01(np.abs(signal_test)), spec.signal_power)
    score_oof = np.power(prox_train, spec.prox_power) * signal_score_oof
    score_test = np.power(prox_test, spec.prox_power) * signal_score_test
    if spec.mode == "contrast_near_abs":
        score_oof *= np.power(1.0 - failed_train, 1.5)
        score_test *= np.power(1.0 - failed_test, 1.5)
        score_test = np.where(winner_mask, 0.0, score_test)
    elif spec.mode == "contrast_det_abs":
        score_oof *= np.power(1.0 - failed_train, 1.5) * ppg.rank01(detector_oof)
        score_test *= np.power(1.0 - failed_test, 1.5) * ppg.rank01(detector_test)
        score_test = np.where(winner_mask, 0.0, score_test)
    elif spec.mode == "new_near_abs":
        score_test = np.where(winner_mask, 0.0, score_test)
    elif spec.mode == "mixed_near_abs":
        pass
    elif spec.mode == "det_near_abs":
        score_oof *= ppg.rank01(detector_oof)
        score_test *= ppg.rank01(detector_test)
        score_test = np.where(winner_mask, 0.0, score_test)
    else:
        raise ValueError(spec.mode)
    return score_oof, score_test


def diagnostics(
    cand: fu.Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
    winner_mask: np.ndarray,
    prox_test: np.ndarray,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, best_oof, base_rmse, best, best_increment, test_species)
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    row["new_changed_count"] = int(np.sum(changed & ~winner_mask))
    row["winner_overlap_count"] = int(np.sum(changed & winner_mask))
    row["changed_prox_mean"] = float(np.mean(prox_test[changed])) if np.any(changed) else 0.0
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.0045:
        reasons.append("weak_oof")
    if not (0.012 <= float(row["anchor_diff_rmse"]) <= 0.028):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) > 0.18:
        reasons.append("max_diff")
    if int(row["changed_count"]) < 12 or int(row["changed_count"]) > 28:
        reasons.append("changed_count")
    if int(row["new_changed_count"]) < 12:
        reasons.append("too_few_new_rows")
    if float(row["species_shift"]) > 0.010:
        reasons.append("species_shift")
    if int(row["species_max"]) > 14:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if abs(float(row["anchor_diff_mean"])) > 0.005:
        reasons.append("mean_shift")
    if abs(float(row["best_increment_corr"])) > 0.65:
        reasons.append("too_correlated_best")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.006, 0.0) + abs(float(row["anchor_diff_rmse"]) - 0.018),
        -float(row["new_changed_count"]),
        float(row["species_shift"]),
        abs(float(row["best_increment_corr"])),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(str(row["experiment"]).split("_f")[0] == str(prev["experiment"]).split("_f")[0] for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.6f} diff={row['anchor_diff_rmse']:.6f} "
        f"changed={row['changed_count']} new={row['new_changed_count']} overlap={row['winner_overlap_count']} "
        f"sp={row['species_shift']:.6f} top10={row['top10_species_max']} corr={row['best_increment_corr']:.3f}"
    )


if __name__ == "__main__":
    main()
