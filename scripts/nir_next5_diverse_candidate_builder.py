#!/usr/bin/env python3
"""Build five diverse next-day submission candidates from the current best.

The goal of this builder is intentionally different from the Stage5 near-grid
scripts: keep the current Public-best cand2 as the protected anchor, then
prepare a small queue whose candidates answer different modeling questions.
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
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn
import nir_testlike_golden_distill_search as tlg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_diverse"
SUBMISSION_DIR = ROOT / "data" / "submissions"
Q5 = SUBMISSION_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv"
CAND2 = SUBMISSION_DIR / "nir_stage5_q5cand2_knn_k6_q1_c4_cl1_f0p045_s0p015_c0p18_b0p48411_20260609.csv"


@dataclass(frozen=True)
class Candidate:
    family: str
    experiment: str
    pred: np.ndarray
    corr_test: np.ndarray
    corr_oof: np.ndarray
    signal_corr: float
    beta: float
    memo: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    data = op.load_data()
    q5 = read_submission(Q5, data["test_ids"])
    cand2 = read_submission(CAND2, data["test_ids"])
    cand2_increment = cand2 - q5
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reconstructing q5 OOF proxy", flush=True)
    q5_oof = build_q5_oof(data)
    residual = data["y"] - q5_oof
    q5_oof_rmse = rmse(data["y"], q5_oof)
    print(f"q5_oof_rmse={q5_oof_rmse:.6f}", flush=True)

    print("building shared feature spaces", flush=True)
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    cluster_train, cluster_test = cluster_labels(F_train, F_test, n_clusters=48)
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)

    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    candidates.extend(alpha_candidates(q5, cand2, q5_oof))
    candidates.extend(testnear_candidates(data, q5_oof, cand2, residual, F_train, F_test, cluster_train, cluster_test))
    candidates.extend(golden_subset_candidates(data, q5_oof, cand2, residual, cluster_train, cluster_test))
    local_oof, local_test = local_residual_signal(F_train, data["y"], residual, data["groups"], F_test, k=55)
    candidates.extend(local_mbl_candidates(q5_oof, cand2, residual, local_oof, local_test, cluster_train, cluster_test))
    candidates.extend(
        detector_candidates(
            q5_oof,
            cand2,
            residual,
            local_oof,
            local_test,
            detector_oof,
            detector_test,
            cluster_train,
            cluster_test,
        )
    )
    candidates.extend(cluster_bias_candidates(q5_oof, cand2, residual, F_train, F_test, data["groups"], cluster_train, cluster_test))

    for cand in candidates:
        row = diagnostics(
            cand,
            data["y"],
            q5_oof,
            q5_oof_rmse,
            q5,
            cand2,
            cand2_increment,
            test_species,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    add_diversity_columns(rows_sorted, candidates)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "next5_diverse_summary.csv", index=False)
    (out_dir / "next5_diverse_summary.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = select_one_per_family(rows_sorted, args.limit)
    saved: list[dict[str, object]] = []
    by_name = {cand.experiment: cand for cand in candidates}
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        filename = f"nir_next5_{rank}_{safe_name(cand.experiment)}_{args.date_tag}.csv"
        path = SUBMISSION_DIR / filename
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "next5_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "next5_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "next5_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSelected next5 candidates:")
    print(manifest[["rank", "family", "experiment", "submit_gate", "oof_delta", "cand2_diff_rmse", "changed_count", "file"]].to_string(index=False))
    print(f"saved diagnostics: {out_dir}")


def build_q5_oof(data: dict[str, np.ndarray]) -> np.ndarray:
    base = read_submission(stn.BASE_ANCHOR, data["test_ids"])
    first = read_submission(stn.FIRST_STAGE_BEST, data["test_ids"])
    second = read_submission(stn.SECOND_STAGE_BEST, data["test_ids"])
    current = read_submission(stn.CURRENT_BEST, data["test_ids"])
    _, slot1_oof = stn.make_slot1_oof(data, base)
    current_oof = stn.make_current_best_oof(data, slot1_oof, read_submission(stn.SLOT1, data["test_ids"]), first, second)
    return stn.make_q5_oof(data, current_oof, current)


def alpha_candidates(q5: np.ndarray, cand2: np.ndarray, q5_oof: np.ndarray) -> list[Candidate]:
    out: list[Candidate] = []
    base = cand2 - q5
    for alpha in [1.05, 1.08]:
        corr = (alpha - 1.0) * base
        out.append(
            Candidate(
                family="safety_alpha",
                experiment=f"cand2_mask_alpha{stn.tag(alpha)}",
                pred=np.clip(cand2 + corr, 0, None),
                corr_test=corr,
                corr_oof=np.zeros_like(q5_oof),
                signal_corr=0.0,
                beta=0.0,
                memo=f"Current-best cand2 mask fixed; continuation alpha={alpha}; no new rows.",
            )
        )
    return out


def testnear_candidates(
    data: dict[str, np.ndarray],
    q5_oof: np.ndarray,
    cand2: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    branches = [
        stn.BranchSpec("next5_tn_pls_k55_q1_c4", "sg9_snv", "pls_raw", "knn_cluster", 0.55, 4, 1.0),
        stn.BranchSpec("next5_tn_pls_k65_q13_c4", "sg9_snv", "pls_raw", "knn_cluster", 0.65, 4, 1.3),
        stn.BranchSpec("next5_tn_pls_k75_q16_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6),
    ]
    out: list[Candidate] = []
    for branch in branches:
        branch_oof = stn.make_quality_branch_oof(branch, data, residual)
        branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
        signal_oof = branch_oof - q5_oof
        signal_test = branch_test - cand2
        beta = fit_beta(signal_oof, residual)
        for gate_mode, frac, shrink, clip in [
            ("abs_signal_cluster1", 0.030, 0.018, 0.14),
            ("abs_signal_cluster2", 0.035, 0.022, 0.16),
            ("negative_signal_cluster1", 0.030, 0.020, 0.14),
        ]:
            corr_oof, pred = stn.make_candidate(
                stn.CandidateSpec(branch, gate_mode, frac, shrink, clip, False),
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
            out.append(
                Candidate(
                    family="testnear_golden",
                    experiment=f"{branch.name}_{gate_mode}_f{stn.tag(frac)}_s{stn.tag(shrink)}",
                    pred=pred,
                    corr_test=pred - cand2,
                    corr_oof=corr_oof,
                    signal_corr=stn.safe_corr(signal_oof, residual),
                    beta=beta,
                    memo=f"cand2 anchor; test-near golden PLS branch {branch.name}; {gate_mode}; frac={frac}; shrink={shrink}; clip={clip}.",
                )
            )
    return out


def golden_subset_candidates(
    data: dict[str, np.ndarray],
    q5_oof: np.ndarray,
    cand2: np.ndarray,
    residual: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    specs = [
        tlg.TestLikeSpec("next5_golden_msc_keep80_c3", "msc_sg9", "pls_raw", "knn", 0.80, 3),
        tlg.TestLikeSpec("next5_golden_msc_keep75_c4", "msc_sg9", "pls_raw", "knn_cluster", 0.75, 4),
        tlg.TestLikeSpec("next5_golden_snv_keep80_c3", "sg9_snv", "pls_raw", "knn", 0.80, 3),
    ]
    out: list[Candidate] = []
    for spec in specs:
        branch_oof, _ = tlg.make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test = tlg.fit_predict_testlike_branch(spec, data["X_train"], data["y"], data["X_test"], groups=data["groups"])
        signal_oof = branch_oof - q5_oof
        signal_test = np.asarray(branch_test, dtype=float) - cand2
        beta = fit_beta(signal_oof, residual)
        for gate_frac, cap, shrink, clip in [(0.055, 1, 0.012, 0.08), (0.065, 2, 0.014, 0.09)]:
            raw_oof = shrink * beta * signal_oof
            raw_test = shrink * beta * signal_test
            gate_oof = diverse_top_fraction(np.abs(raw_oof), gate_frac, cluster_train, cap)
            gate_test = diverse_top_fraction(np.abs(raw_test), gate_frac, cluster_test, cap)
            corr_oof = np.where(gate_oof, np.clip(raw_oof, -clip, clip), 0.0)
            corr_test = np.where(gate_test, np.clip(raw_test, -clip, clip), 0.0)
            out.append(
                Candidate(
                    family="global_golden_subset",
                    experiment=f"{spec.name}_f{stn.tag(gate_frac)}_cap{cap}_s{stn.tag(shrink)}_c{stn.tag(clip)}",
                    pred=np.clip(cand2 + corr_test, 0, None),
                    corr_test=corr_test,
                    corr_oof=corr_oof,
                    signal_corr=stn.safe_corr(signal_oof, residual),
                    beta=beta,
                    memo=f"cand2 anchor; global test-like golden subset PLS; {spec.preprocess}/{spec.score_mode}; keep={spec.keep_frac}; c={spec.n_components}; gate frac={gate_frac}; cap={cap}.",
                )
            )
    return out


def local_mbl_candidates(
    q5_oof: np.ndarray,
    cand2: np.ndarray,
    residual: np.ndarray,
    local_oof: np.ndarray,
    local_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    beta = fit_beta(local_oof, residual)
    out: list[Candidate] = []
    for frac, shrink, clip, cap in [(0.055, 0.020, 0.12, 1), (0.075, 0.016, 0.14, 2)]:
        gate_oof = diverse_top_fraction(np.abs(local_oof), frac, cluster_train, cap)
        gate_test = diverse_top_fraction(np.abs(local_test), frac, cluster_test, cap)
        corr_oof = np.where(gate_oof, np.clip(shrink * beta * local_oof, -clip, clip), 0.0)
        corr_test = np.where(gate_test, np.clip(shrink * beta * local_test, -clip, clip), 0.0)
        out.append(
            Candidate(
                family="local_mbl_residual",
                experiment=f"mbl_resid_k55_f{stn.tag(frac)}_cap{cap}_s{stn.tag(shrink)}",
                pred=np.clip(cand2 + corr_test, 0, None),
                corr_test=corr_test,
                corr_oof=corr_oof,
                signal_corr=stn.safe_corr(local_oof, residual),
                beta=beta,
                memo=f"cand2 anchor; memory-based local residual correction; k=55; top frac={frac}; cluster cap={cap}.",
            )
        )
    return out


def detector_candidates(
    q5_oof: np.ndarray,
    cand2: np.ndarray,
    residual: np.ndarray,
    local_oof: np.ndarray,
    local_test: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    beta = fit_beta(local_oof, residual)
    out: list[Candidate] = []
    for det_frac, frac, shrink, clip, cap in [(0.12, 0.045, 0.026, 0.12, 1), (0.20, 0.060, 0.022, 0.14, 2)]:
        score_oof = np.where(top_fraction(detector_oof, det_frac), np.abs(local_oof), 0.0)
        score_test = np.where(top_fraction(detector_test, det_frac), np.abs(local_test), 0.0)
        gate_oof = diverse_top_fraction(score_oof, frac, cluster_train, cap)
        gate_test = diverse_top_fraction(score_test, frac, cluster_test, cap)
        corr_oof = np.where(gate_oof, np.clip(shrink * beta * local_oof, -clip, clip), 0.0)
        corr_test = np.where(gate_test, np.clip(shrink * beta * local_test, -clip, clip), 0.0)
        out.append(
            Candidate(
                family="detector_hardcase",
                experiment=f"det_mbl_df{stn.tag(det_frac)}_f{stn.tag(frac)}_cap{cap}_s{stn.tag(shrink)}",
                pred=np.clip(cand2 + corr_test, 0, None),
                corr_test=corr_test,
                corr_oof=corr_oof,
                signal_corr=stn.safe_corr(local_oof, residual),
                beta=beta,
                memo=f"cand2 anchor; spectral detector intersected with local residual; detector top={det_frac}; frac={frac}; cluster cap={cap}.",
            )
        )
    return out


def cluster_bias_candidates(
    q5_oof: np.ndarray,
    cand2: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    groups: np.ndarray,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
) -> list[Candidate]:
    out: list[Candidate] = []
    for n_clusters, gate_frac, cap, shrink, clip in [(24, 0.055, 1, 0.10, 0.10), (36, 0.075, 2, 0.08, 0.10)]:
        raw_oof = cluster_residual_oof(F_train, residual, groups, n_clusters, shrink, clip)
        raw_test = cluster_residual_test(F_train, residual, F_test, n_clusters, shrink, clip)
        gate_oof = diverse_top_fraction(np.abs(raw_oof), gate_frac, cluster_train, cap)
        gate_test = diverse_top_fraction(np.abs(raw_test), gate_frac, cluster_test, cap)
        corr_oof = np.where(gate_oof, raw_oof, 0.0)
        corr_test = np.where(gate_test, raw_test, 0.0)
        out.append(
            Candidate(
                family="spectral_cluster_bias",
                experiment=f"cluster_bias_k{n_clusters}_f{stn.tag(gate_frac)}_cap{cap}_s{stn.tag(shrink)}",
                pred=np.clip(cand2 + corr_test, 0, None),
                corr_test=corr_test,
                corr_oof=corr_oof,
                signal_corr=stn.safe_corr(corr_oof, residual),
                beta=1.0,
                memo=f"cand2 anchor; unsupervised spectral-cluster residual mean correction; clusters={n_clusters}; gate frac={gate_frac}; cap={cap}; shrink={shrink}.",
            )
        )
    return out


def local_residual_signal(
    F_train: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    F_test: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(F_train, y, groups):
        pred[valid_idx] = knn_residual(F_train[train_idx], residual[train_idx], F_train[valid_idx], k)
    test_pred = knn_residual(F_train, residual, F_test, k)
    return pred, test_pred


def knn_residual(F_fit: np.ndarray, residual: np.ndarray, F_pred: np.ndarray, k: int) -> np.ndarray:
    n = min(k, len(F_fit))
    dist, idx = NearestNeighbors(n_neighbors=n, metric="euclidean").fit(F_fit).kneighbors(F_pred)
    weight = 1.0 / np.maximum(dist, 1e-6)
    weight = weight / np.sum(weight, axis=1, keepdims=True)
    return np.sum(weight * residual[idx], axis=1)


def cluster_residual_oof(
    F_train: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    n_clusters: int,
    shrink: float,
    clip: float,
) -> np.ndarray:
    out = np.zeros(len(residual), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(F_train, residual, groups):
        kmeans = KMeans(n_clusters=min(n_clusters, len(train_idx) - 1), random_state=42, n_init=20)
        labels_fit = kmeans.fit_predict(F_train[train_idx])
        labels_valid = kmeans.predict(F_train[valid_idx])
        means = cluster_means(labels_fit, residual[train_idx])
        out[valid_idx] = [means.get(int(label), 0.0) for label in labels_valid]
    return np.clip(shrink * out, -clip, clip)


def cluster_residual_test(
    F_train: np.ndarray,
    residual: np.ndarray,
    F_test: np.ndarray,
    n_clusters: int,
    shrink: float,
    clip: float,
) -> np.ndarray:
    kmeans = KMeans(n_clusters=min(n_clusters, len(F_train) - 1), random_state=42, n_init=20)
    labels_fit = kmeans.fit_predict(F_train)
    labels_test = kmeans.predict(F_test)
    means = cluster_means(labels_fit, residual)
    raw = np.asarray([means.get(int(label), 0.0) for label in labels_test], dtype=float)
    return np.clip(shrink * raw, -clip, clip)


def cluster_means(labels: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(label): float(np.mean(values[labels == label])) for label in np.unique(labels)}


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    q5_oof: np.ndarray,
    q5_oof_rmse: float,
    q5: np.ndarray,
    anchor: np.ndarray,
    cand2_increment: np.ndarray,
    test_species: np.ndarray,
) -> dict[str, object]:
    diff = cand.pred - anchor
    changed = np.abs(diff) > 1e-12
    corrected_oof = np.clip(q5_oof + cand.corr_oof, 0, None)
    oof_rmse = rmse(y, corrected_oof)
    species_counts = pd.Series(test_species[changed]).value_counts() if changed.any() else pd.Series(dtype=int)
    top_idx = np.argsort(-np.abs(diff))[:10]
    top_species = pd.Series(test_species[top_idx[np.abs(diff[top_idx]) > 1e-12]]).value_counts()
    row = {
        "family": cand.family,
        "experiment": cand.experiment,
        "submit_gate": "pass",
        "reject_reasons": "",
        "oof_delta": oof_rmse - q5_oof_rmse,
        "oof_rmse": oof_rmse,
        "signal_corr": cand.signal_corr,
        "beta": cand.beta,
        "cand2_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "cand2_diff_max": float(np.max(np.abs(diff))),
        "cand2_diff_mean": float(np.mean(diff)),
        "changed_count": int(changed.sum()),
        "positive_count": int(np.sum(diff > 1e-12)),
        "negative_count": int(np.sum(diff < -1e-12)),
        "species_shift": max_species_shift(diff, test_species),
        "species_max": int(species_counts.iloc[0]) if len(species_counts) else 0,
        "top10_species_max": int(top_species.iloc[0]) if len(top_species) else 0,
        "cand2_increment_corr": safe_corr(diff, cand2_increment),
        "q5_diff_rmse": float(math.sqrt(np.mean((cand.pred - q5) ** 2))),
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
    if int(row["changed_count"]) == 0:
        reasons.append("no_change")
    if float(row["cand2_diff_rmse"]) > (0.035 if family == "spectral_cluster_bias" else 0.025):
        reasons.append("large_diff_rmse")
    if float(row["cand2_diff_max"]) > 0.22:
        reasons.append("large_max_diff")
    if abs(float(row["cand2_diff_mean"])) > 0.006:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.018:
        reasons.append("species_shift")
    if int(row["species_max"]) > 16:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 5:
        reasons.append("top10_species")
    if float(row["pred_min"]) < 0:
        reasons.append("negative_prediction")
    if family not in {"safety_alpha", "spectral_cluster_bias"} and float(row["oof_delta"]) > 0.001:
        reasons.append("oof_worse")
    if family == "safety_alpha" and float(row["cand2_diff_rmse"]) > 0.002:
        reasons.append("alpha_too_large")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    diff = float(row["cand2_diff_rmse"])
    oof = float(row["oof_delta"])
    target_diff = 0.011
    if row["family"] == "safety_alpha":
        target_diff = 0.001
    if row["family"] == "spectral_cluster_bias":
        target_diff = 0.020
    return (
        penalty + max(oof, 0.0) + abs(diff - target_diff) + 0.2 * abs(float(row["cand2_diff_mean"])),
        float(row["species_shift"]),
        -float(row["signal_corr"]),
        float(row["cand2_diff_max"]),
    )


def select_one_per_family(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    preferred = [
        "safety_alpha",
        "testnear_golden",
        "global_golden_subset",
        "local_mbl_residual",
        "detector_hardcase",
        "spectral_cluster_bias",
    ]
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for family in preferred:
        family_rows = [row for row in rows if row["family"] == family and row["submit_gate"] == "pass"]
        if not family_rows:
            family_rows = [row for row in rows if row["family"] == family]
        if family_rows:
            row = pick_diverse_row(family_rows, selected)
            selected.append(row)
            used.add(str(row["experiment"]))
        if len(selected) >= limit:
            return selected
    for row in rows:
        if str(row["experiment"]) in used:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def add_diversity_columns(rows: list[dict[str, object]], candidates: list[Candidate]) -> None:
    by_name = {cand.experiment: cand for cand in candidates}
    changed = {
        cand.experiment: set(np.flatnonzero(np.abs(cand.corr_test) > 1e-12).tolist())
        for cand in candidates
    }
    for row in rows:
        name = str(row["experiment"])
        corr_values = []
        overlap_values = []
        top_overlap_values = []
        for other in candidates:
            if other.experiment == name:
                continue
            cand = by_name[name]
            corr_values.append(abs(safe_corr(cand.corr_test, other.corr_test)))
            own = changed[name]
            other_changed = changed[other.experiment]
            union = own | other_changed
            overlap_values.append(len(own & other_changed) / max(len(union), 1))
            top_overlap_values.append(top_abs_overlap(cand.corr_test, other.corr_test, top_n=20))
        row["max_pairwise_corr"] = float(max(corr_values)) if corr_values else 0.0
        row["max_changed_jaccard"] = float(max(overlap_values)) if overlap_values else 0.0
        row["max_top20_overlap"] = float(max(top_overlap_values)) if top_overlap_values else 0.0


def pick_diverse_row(candidates: list[dict[str, object]], selected: list[dict[str, object]]) -> dict[str, object]:
    if not selected:
        return candidates[0]
    selected_names = {str(row["experiment"]) for row in selected}
    # Diversity columns are global maxima. Penalize candidates that are globally
    # highly redundant when we already have a candidate from another family.
    return min(
        candidates,
        key=lambda row: (
            str(row["experiment"]) in selected_names,
            float(row.get("max_pairwise_corr", 0.0)) > 0.80,
            float(row.get("max_changed_jaccard", 0.0)) > 0.50,
            rank_key(row),
        ),
    )


def top_abs_overlap(a: np.ndarray, b: np.ndarray, top_n: int) -> float:
    n = min(top_n, len(a), len(b))
    if n <= 0:
        return 0.0
    ia = set(np.argsort(np.abs(a), kind="mergesort")[::-1][:n].tolist())
    ib = set(np.argsort(np.abs(b), kind="mergesort")[::-1][:n].tolist())
    return len(ia & ib) / n


def read_submission(path: Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    if not np.array_equal(df.iloc[:, 0].to_numpy(), ids):
        raise ValueError(f"sample order mismatch: {path}")
    return df.iloc[:, 1].to_numpy(float)


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.dot(signal, residual) / denom, -1.0, 1.0))


def cluster_labels(F_train: np.ndarray, F_test: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    return labels[: len(F_train)], labels[len(F_train) :]


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    order = np.argsort(score, kind="mergesort")[::-1]
    gate = np.zeros(len(score), dtype=bool)
    gate[order[:target]] = True
    return gate


def diverse_top_fraction(score: np.ndarray, frac: float, labels: np.ndarray, cap: int) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    order = np.argsort(score, kind="mergesort")[::-1]
    selected: list[int] = []
    counts: dict[int, int] = {}
    for idx in order:
        if score[idx] <= 0:
            continue
        label = int(labels[idx])
        if counts.get(label, 0) >= cap:
            continue
        selected.append(int(idx))
        counts[label] = counts.get(label, 0) + 1
        if len(selected) >= target:
            break
    if len(selected) < target:
        for idx in order:
            if int(idx) in selected or score[idx] <= 0:
                continue
            selected.append(int(idx))
            if len(selected) >= target:
                break
    gate = np.zeros(len(score), dtype=bool)
    gate[np.asarray(selected, dtype=int)] = True
    return gate


def max_species_shift(diff: np.ndarray, test_species: np.ndarray) -> float:
    return max(abs(float(np.mean(diff[test_species == group]))) for group in np.unique(test_species))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)[:140]


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['family']} {row['experiment']}: gate={row['submit_gate']} "
        f"reasons={row['reject_reasons']} oof={row['oof_delta']:.6f} "
        f"diff={row['cand2_diff_rmse']:.6f} max={row['cand2_diff_max']:.6f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.6f} "
        f"corr={row['cand2_increment_corr']:.3f}"
    )


if __name__ == "__main__":
    main()
