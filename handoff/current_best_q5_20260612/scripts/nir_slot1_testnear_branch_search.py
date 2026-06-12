#!/usr/bin/env python3
"""Slot1-anchored test-near golden branch search.

This branch tests the calibration-subset idea from NIR/soil spectroscopy:
select train samples that are both test-like and internally reliable, fit a
simple local/PLS-style branch, then distill only a small gated correction into
the current public-best Slot1 anchor.
"""

from __future__ import annotations

import csv
import json
import math
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

import nir_post_public_gate_search as ppg
import nir_testlike_golden_distill_search as tlg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_slot1_testnear_branch"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BASE_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"
SLOT1 = SUBMISSION_DIR / "nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv"
SLOT2 = SUBMISSION_DIR / "nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv"
FIRST_STAGE_BEST = SUBMISSION_DIR / "nir_s1tn_pls_knncluster_k6_q13_c5_clcap3_f4_s004_c018_20260608.csv"
SECOND_STAGE_BEST = SUBMISSION_DIR / "nir_s2tn_curbest_pls_k75_q13_c5_clcap2_f045_s002_c018_20260608.csv"
CURRENT_BEST = SUBMISSION_DIR / "nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv"
Q5_BEST = SUBMISSION_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv"


@dataclass(frozen=True)
class BranchSpec:
    name: str
    preprocess: str
    model: str
    score_mode: str
    keep_frac: float
    n_components: int
    quality_power: float
    min_keep: int = 40


@dataclass(frozen=True)
class CandidateSpec:
    branch: BranchSpec
    gate_mode: str
    frac: float
    shrink: float
    clip: float
    mean_center: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branches", type=int, default=None)
    parser.add_argument("--max-candidates-per-branch", type=int, default=None)
    parser.add_argument("--cluster-count", type=int, default=24)
    parser.add_argument("--focus-best", action="store_true")
    parser.add_argument("--branch-contains", action="append", default=None)
    parser.add_argument("--anchor-current-best", action="store_true")
    parser.add_argument("--anchor-q5", action="store_true")
    parser.add_argument("--spacing-gates", action="store_true")
    parser.add_argument("--detector-gates", action="store_true")
    args = parser.parse_args()

    data = ppg.op.load_data()
    base_df = pd.read_csv(BASE_ANCHOR, header=None, names=["id", "pred"])
    slot1_df = pd.read_csv(SLOT1, header=None, names=["id", "pred"])
    slot2_df = pd.read_csv(SLOT2, header=None, names=["id", "pred"])
    first_stage_best_df = pd.read_csv(FIRST_STAGE_BEST, header=None, names=["id", "pred"])
    second_stage_best_df = pd.read_csv(SECOND_STAGE_BEST, header=None, names=["id", "pred"])
    current_best_df = pd.read_csv(CURRENT_BEST, header=None, names=["id", "pred"])
    q5_best_df = pd.read_csv(Q5_BEST, header=None, names=["id", "pred"])
    for label, df in [
        ("base", base_df),
        ("slot1", slot1_df),
        ("slot2", slot2_df),
        ("first_stage_best", first_stage_best_df),
        ("second_stage_best", second_stage_best_df),
        ("current_best", current_best_df),
        ("q5_best", q5_best_df),
    ]:
        if not np.array_equal(data["test_ids"], df["id"].to_numpy()):
            raise ValueError(f"{label} sample order mismatch")

    base_test = base_df["pred"].to_numpy(float)
    slot1_test = slot1_df["pred"].to_numpy(float)
    slot2_test = slot2_df["pred"].to_numpy(float)
    first_stage_best_test = first_stage_best_df["pred"].to_numpy(float)
    second_stage_best_test = second_stage_best_df["pred"].to_numpy(float)
    current_best_test = current_best_df["pred"].to_numpy(float)
    q5_best_test = q5_best_df["pred"].to_numpy(float)
    slot1_changed = np.abs(slot1_test - base_test) > 1e-12
    slot2_only = (np.abs(slot2_test - base_test) > 1e-12) & ~slot1_changed
    test_species = pd.read_csv(ppg.op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    cluster_labels = KMeans(n_clusters=args.cluster_count, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    train_clusters = cluster_labels[: len(F_train)]
    test_clusters = cluster_labels[len(F_train) :]

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building Slot1 OOF anchor", flush=True)
    base_oof, slot1_oof = make_slot1_oof(data, base_test)
    detector_oof = None
    detector_test = None
    if args.detector_gates:
        print("building fold-local spectral detector scores", flush=True)
        detector_oof, detector_test = build_spectral_detector_scores(data)

    q5_increment = None
    if args.anchor_current_best or args.anchor_q5:
        print("reconstructing current-best OOF anchor", flush=True)
        slot1_oof = make_current_best_oof(data, slot1_oof, slot1_test, first_stage_best_test, second_stage_best_test)
        slot1_test = current_best_test
    if args.anchor_q5:
        print("reconstructing q5 OOF anchor", flush=True)
        q5_increment = q5_best_test - current_best_test
        slot1_oof = make_q5_oof(data, slot1_oof, current_best_test)
        slot1_test = q5_best_test
    slot1_oof_rmse = rmse(data["y"], slot1_oof)
    residual = data["y"] - slot1_oof
    prior_correction = slot1_test - base_test
    prior_changed = np.abs(prior_correction) > 1e-12
    print(f"slot1_oof_rmse={slot1_oof_rmse:.6f}", flush=True)

    branch_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    branches = build_focus_branches() if args.focus_best else build_branches()
    if args.branch_contains:
        branches = [
            branch
            for branch in branches
            if any(token in branch.name for token in args.branch_contains)
        ]
    if args.max_branches is not None:
        branches = branches[: args.max_branches]
    for branch in branches:
        print(f"branch {branch.name}", flush=True)
        branch_oof = make_quality_branch_oof(branch, data, residual)
        branch_test = fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
        signal_oof = branch_oof - slot1_oof
        signal_test = branch_test - slot1_test
        beta = fit_beta(signal_oof, residual)
        signal_corr = safe_corr(signal_oof, residual)
        branch_rmse = rmse(data["y"], branch_oof)
        branch_rows.append(
            {
                "branch": branch.name,
                "preprocess": branch.preprocess,
                "model": branch.model,
                "score_mode": branch.score_mode,
                "keep_frac": branch.keep_frac,
                "n_components": branch.n_components,
                "quality_power": branch.quality_power,
                "branch_oof_rmse": branch_rmse,
                "branch_delta_vs_slot1": branch_rmse - slot1_oof_rmse,
                "signal_residual_corr": signal_corr,
                "beta": beta,
            }
        )
        candidate_specs = (
            build_focus_candidates(branch, include_spacing=args.spacing_gates, include_detector=args.detector_gates)
            if args.focus_best
            else build_candidates(branch, include_detector=args.detector_gates)
        )
        if args.max_candidates_per_branch is not None:
            candidate_specs = candidate_specs[: args.max_candidates_per_branch]
        for spec in candidate_specs:
            corr_oof, pred = make_candidate(
                spec,
                signal_oof,
                signal_test,
                slot1_oof,
                slot1_test,
                residual,
                beta,
                train_clusters,
                test_clusters,
                F_train,
                F_test,
                detector_oof,
                detector_test,
            )
            name = candidate_name(spec, beta)
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = diagnostics(
                name=name,
                spec=spec,
                path=path,
                pred=pred,
                corr_oof=corr_oof,
                slot1_test=slot1_test,
                base_test=base_test,
                slot2_only=slot2_only,
                prior_changed=prior_changed,
                prior_correction=prior_correction,
                y=data["y"],
                slot1_oof=slot1_oof,
                slot1_oof_rmse=slot1_oof_rmse,
                residual=residual,
                groups=data["groups"],
                test_species=test_species,
                beta=beta,
                branch_rmse=branch_rmse,
                signal_corr=signal_corr,
                q5_increment=q5_increment,
            )
            rows.append(row)
            try:
                print_one(row)
            except (OSError, UnicodeEncodeError):
                pass

    rows_sorted = sorted(rows, key=rank_key)
    write_csv(out_dir / "slot1_testnear_summary.csv", rows_sorted)
    write_csv(out_dir / "slot1_testnear_branch_summary.csv", branch_rows)
    with (out_dir / "slot1_testnear_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop slot1 test-near candidates:")
    for row in rows_sorted[:60]:
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['risk']:.4f} diff={row['slot1_diff_rmse']:.4f} "
            f"max={row['slot1_diff_max_abs']:.4f} changed={row['changed_count']} "
            f"new={row['new_changed_count']} pcorr={row['prior_correction_corr']:.3f} "
            f"s2only={row['slot2_only_changed_count']} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof={row['oof_delta_vs_slot1']:.4f} groups={row['improved_group_count']}/{row['group_count']}"
        )
    print(f"saved diagnostics: {out_dir}")


def make_slot1_oof(data: dict[str, np.ndarray], base_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_spec = ppg.bsa.AugSpec(
        name="current_shape_anchor",
        pca_components=18,
        alpha=3500.0,
        shape_set="shape14",
        shape_scale=0.50,
        affine_shrink=0.12,
        affine_clip=0.40,
        mean_center=True,
    )
    base_oof = ppg.bsa.make_nested_affine_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    base_residual = data["y"] - base_oof
    detector_train, detector_test, _ = ppg.s2.build_detector_features(data, base_oof, base_test)
    hard_label = np.abs(base_residual) >= np.quantile(np.abs(base_residual), 0.80)
    detector_oof = ppg.hcd.detector_oof_score("logreg", detector_train, hard_label, data["groups"])
    detector_test = ppg.hcd.detector_full_score("logreg", detector_train, hard_label, detector_test)
    signal_oof, signal_test = ppg.build_signals(data, base_oof, base_test, base_residual)["wmean_k40"]
    score_oof = ppg.rank01(detector_oof) * ppg.rank01(np.abs(signal_oof))
    score_test = ppg.rank01(detector_test) * ppg.rank01(np.abs(signal_test))
    corr_oof, _ = ppg.make_prediction(
        score_oof,
        score_test,
        signal_oof,
        signal_test,
        base_oof,
        base_test,
        base_residual,
        data["groups"],
        frac=0.05,
        shrink=0.003,
        clip=0.2,
    )
    return base_oof, np.clip(base_oof + corr_oof, 0, None)


def make_current_best_oof(
    data: dict[str, np.ndarray],
    slot1_oof: np.ndarray,
    slot1_test: np.ndarray,
    first_stage_best_test: np.ndarray,
    second_stage_best_test: np.ndarray,
) -> np.ndarray:
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])

    residual = data["y"] - slot1_oof
    branch = BranchSpec("pls_sg9_snv_knn_cluster_k6_q1p3_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.60, 5, 1.3)
    branch_oof = make_quality_branch_oof(branch, data, residual)
    branch_test = fit_quality_branch(
        branch,
        data["X_train"],
        data["y"],
        data["X_test"],
        residual,
        data["groups"],
    )
    signal_oof = branch_oof - slot1_oof
    signal_test = branch_test - slot1_test
    beta = fit_beta(signal_oof, residual)
    stage1_clusters = KMeans(n_clusters=24, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    spec = CandidateSpec(
        branch=branch,
        gate_mode="abs_signal_cluster3",
        frac=0.04,
        shrink=0.04,
        clip=0.18,
        mean_center=False,
    )
    corr_oof, _ = make_candidate(
        spec,
        signal_oof,
        signal_test,
        slot1_oof,
        slot1_test,
        residual,
        beta,
        stage1_clusters[: len(F_train)],
        stage1_clusters[len(F_train) :],
    )
    first_stage_oof = np.clip(slot1_oof + corr_oof, 0, None)

    residual2 = data["y"] - first_stage_oof
    branch2 = BranchSpec("pls_sg9_snv_knn_cluster_k75_q1p3_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.3)
    branch2_oof = make_quality_branch_oof(branch2, data, residual2)
    branch2_test = fit_quality_branch(
        branch2,
        data["X_train"],
        data["y"],
        data["X_test"],
        residual2,
        data["groups"],
    )
    signal2_oof = branch2_oof - first_stage_oof
    signal2_test = branch2_test - first_stage_best_test
    beta2 = fit_beta(signal2_oof, residual2)
    stage2_clusters = KMeans(n_clusters=40, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    spec2 = CandidateSpec(
        branch=branch2,
        gate_mode="abs_signal_cluster2",
        frac=0.045,
        shrink=0.02,
        clip=0.18,
        mean_center=False,
    )
    corr2_oof, _ = make_candidate(
        spec2,
        signal2_oof,
        signal2_test,
        first_stage_oof,
        first_stage_best_test,
        residual2,
        beta2,
        stage2_clusters[: len(F_train)],
        stage2_clusters[len(F_train) :],
    )
    second_stage_oof = np.clip(first_stage_oof + corr2_oof, 0, None)

    residual3 = data["y"] - second_stage_oof
    branch3 = BranchSpec("pls_sg9_snv_knn_cluster_k6_q1p6_c4", "sg9_snv", "pls_raw", "knn_cluster", 0.60, 4, 1.6)
    branch3_oof = make_quality_branch_oof(branch3, data, residual3)
    branch3_test = fit_quality_branch(
        branch3,
        data["X_train"],
        data["y"],
        data["X_test"],
        residual3,
        data["groups"],
    )
    signal3_oof = branch3_oof - second_stage_oof
    signal3_test = branch3_test - second_stage_best_test
    beta3 = fit_beta(signal3_oof, residual3)
    stage3_clusters = KMeans(n_clusters=32, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    spec3 = CandidateSpec(
        branch=branch3,
        gate_mode="abs_signal_cluster2",
        frac=0.04,
        shrink=0.012,
        clip=0.18,
        mean_center=False,
    )
    corr3_oof, _ = make_candidate(
        spec3,
        signal3_oof,
        signal3_test,
        second_stage_oof,
        second_stage_best_test,
        residual3,
        beta3,
        stage3_clusters[: len(F_train)],
        stage3_clusters[len(F_train) :],
    )
    return np.clip(second_stage_oof + corr3_oof, 0, None)


def make_q5_oof(data: dict[str, np.ndarray], current_oof: np.ndarray, current_best_test: np.ndarray) -> np.ndarray:
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    residual = data["y"] - current_oof
    branch = BranchSpec("pls_sg9_snv_knn_cluster_k6_q1p6_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.60, 5, 1.6)
    branch_oof = make_quality_branch_oof(branch, data, residual)
    branch_test = fit_quality_branch(
        branch,
        data["X_train"],
        data["y"],
        data["X_test"],
        residual,
        data["groups"],
    )
    signal_oof = branch_oof - current_oof
    signal_test = branch_test - current_best_test
    beta = fit_beta(signal_oof, residual)
    q5_clusters = KMeans(n_clusters=80, random_state=42, n_init=20).fit_predict(np.vstack([F_train, F_test]))
    spec = CandidateSpec(
        branch=branch,
        gate_mode="abs_signal_cluster2",
        frac=0.04,
        shrink=0.04,
        clip=0.18,
        mean_center=False,
    )
    corr_oof, _ = make_candidate(
        spec,
        signal_oof,
        signal_test,
        current_oof,
        current_best_test,
        residual,
        beta,
        q5_clusters[: len(F_train)],
        q5_clusters[len(F_train) :],
    )
    return np.clip(current_oof + corr_oof, 0, None)


def build_spectral_detector_scores(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = data["X_train"]
    X_test = data["X_test"]
    groups = data["groups"]
    scores = np.empty(len(X), dtype=float)
    splitter = tlg.GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, data["y"], groups):
        _, valid_score = fit_score_spectral_detector(X[train_idx], X[valid_idx])
        scores[valid_idx] = valid_score
    _, test_score = fit_score_spectral_detector(X, X_test)
    return scores, test_score


def fit_score_spectral_detector(X_fit_raw: np.ndarray, X_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_fit, X_pred = tlg.preprocess_pair(X_fit_raw, X_pred_raw, "sg9_snv")
    scaler = StandardScaler().fit(X_fit)
    Z_fit = scaler.transform(X_fit)
    Z_pred = scaler.transform(X_pred)
    n_comp = min(20, Z_fit.shape[0] - 2, Z_fit.shape[1])
    pca = PCA(n_components=max(2, n_comp), random_state=42).fit(Z_fit)
    T_fit = pca.transform(Z_fit)
    T_pred = pca.transform(Z_pred)
    recon = pca.inverse_transform(T_pred)
    q_resid = np.sum((Z_pred - recon) ** 2, axis=1)
    n_neighbors = min(20, max(2, len(T_fit) - 1))
    dist, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(T_fit).kneighbors(T_pred)
    nn_dist = dist.mean(axis=1)
    lof_neighbors = min(35, max(2, len(T_fit) - 1))
    lof_score = -LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True).fit(T_fit).score_samples(T_pred)
    iso_score = -IsolationForest(n_estimators=250, contamination="auto", random_state=42).fit(T_fit).score_samples(T_pred)
    score = (
        ppg.rank01(q_resid)
        + ppg.rank01(nn_dist)
        + ppg.rank01(lof_score)
        + ppg.rank01(iso_score)
    ) / 4.0
    fit_score = np.zeros(len(X_fit_raw), dtype=float)
    return fit_score, score


def build_branches() -> list[BranchSpec]:
    branches: list[BranchSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for score_mode in ["knn", "knn_cluster", "cluster"]:
            for keep_frac in [0.45, 0.60, 0.75]:
                for quality_power in [0.0, 0.7, 1.3]:
                    for comp in [3, 5, 7]:
                        branches.append(
                            BranchSpec(
                                name=(
                                    f"pls_{preprocess}_{score_mode}_"
                                    f"k{pct_tag(keep_frac)}_q{tag(quality_power)}_c{comp}"
                                ),
                                preprocess=preprocess,
                                model="pls_raw",
                                score_mode=score_mode,
                                keep_frac=keep_frac,
                                n_components=comp,
                                quality_power=quality_power,
                            )
                        )
                for quality_power in [0.7, 1.3]:
                    branches.append(
                        BranchSpec(
                            name=(
                                f"ridgeyj_{preprocess}_{score_mode}_"
                                f"k{pct_tag(keep_frac)}_q{tag(quality_power)}_p20"
                            ),
                            preprocess=preprocess,
                            model="ridge_yj",
                            score_mode=score_mode,
                            keep_frac=keep_frac,
                            n_components=20,
                            quality_power=quality_power,
                        )
                    )
    return branches


def build_focus_branches() -> list[BranchSpec]:
    branches: list[BranchSpec] = []
    for score_mode in ["knn_cluster", "knn"]:
        for keep_frac in [0.55, 0.60, 0.65, 0.75]:
            for quality_power in [1.0, 1.3, 1.6]:
                for comp in [3, 4, 5, 6]:
                    branches.append(
                        BranchSpec(
                            name=(
                                f"pls_sg9_snv_{score_mode}_"
                                f"k{pct_tag(keep_frac)}_q{tag(quality_power)}_c{comp}"
                            ),
                            preprocess="sg9_snv",
                            model="pls_raw",
                            score_mode=score_mode,
                            keep_frac=keep_frac,
                            n_components=comp,
                            quality_power=quality_power,
                        )
                    )
    return branches


def build_candidates(branch: BranchSpec, *, include_detector: bool = False) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    gate_modes = [
        "abs_signal",
        "positive_signal",
        "negative_signal",
        "abs_signal_cluster2",
        "abs_signal_cluster3",
        "positive_signal_cluster2",
        "positive_signal_cluster3",
    ]
    if include_detector:
        gate_modes.extend(
            [
                "abs_signal_det10_cluster2",
                "abs_signal_det15_cluster2",
                "abs_signal_det20_cluster2",
                "negative_signal_det10_cluster2",
                "negative_signal_det15_cluster2",
                "abs_signal_det20_cluster3",
            ]
        )
    for gate_mode in gate_modes:
        for frac in [0.04, 0.05, 0.065, 0.08]:
            for shrink in [0.015, 0.025, 0.04, 0.06]:
                for clip in [0.10, 0.14, 0.18]:
                    specs.append(CandidateSpec(branch, gate_mode, frac, shrink, clip, mean_center=False))
    return specs


def build_focus_candidates(
    branch: BranchSpec,
    *,
    include_spacing: bool = False,
    include_detector: bool = False,
) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    gate_modes = [
        "abs_signal_cluster1",
        "abs_signal_cluster2",
        "abs_signal_cluster3",
        "abs_signal_cluster4",
        "positive_signal_cluster1",
        "positive_signal_cluster2",
        "positive_signal_cluster3",
    ]
    if include_spacing:
        gate_modes.extend(
            [
                "abs_signal_space02",
                "abs_signal_space05",
                "abs_signal_cluster1_space02",
                "abs_signal_cluster1_space05",
                "positive_signal_space02",
                "positive_signal_cluster1_space02",
            ]
        )
    if include_detector:
        detector_modes = [
            "abs_signal_det08_cluster2",
            "abs_signal_det10_cluster2",
            "abs_signal_det15_cluster2",
            "abs_signal_det20_cluster2",
            "abs_signal_det30_cluster2",
            "negative_signal_det08_cluster2",
            "negative_signal_det10_cluster2",
            "negative_signal_det15_cluster2",
            "abs_signal_det10_cluster3",
            "abs_signal_det20_cluster3",
        ]
        gate_modes = detector_modes + gate_modes
    for gate_mode in gate_modes:
        for frac in [0.03, 0.035, 0.04, 0.045, 0.05]:
            for shrink in [0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]:
                for clip in [0.14, 0.18]:
                    specs.append(CandidateSpec(branch, gate_mode, frac, shrink, clip, mean_center=False))
    return specs


def make_quality_branch_oof(branch: BranchSpec, data: dict[str, np.ndarray], residual: np.ndarray) -> np.ndarray:
    pred = np.empty_like(data["y"], dtype=float)
    splitter = tlg.GroupKFold(n_splits=min(5, len(np.unique(data["groups"]))))
    for train_idx, valid_idx in splitter.split(data["X_train"], data["y"], data["groups"]):
        pred[valid_idx] = fit_quality_branch(
            branch,
            data["X_train"][train_idx],
            data["y"][train_idx],
            data["X_train"][valid_idx],
            residual[train_idx],
            data["groups"][train_idx],
        )
    return np.clip(pred, 0, None)


def fit_quality_branch(
    branch: BranchSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    inner = tlg.TestLikeSpec(
        name=branch.name,
        preprocess=branch.preprocess,
        model=branch.model,
        score_mode=branch.score_mode,
        keep_frac=branch.keep_frac,
        n_components=branch.n_components,
        min_keep=branch.min_keep,
    )
    X_train, X_pred = tlg.preprocess_pair(X_train_raw, X_pred_raw, branch.preprocess)
    test_score = tlg.test_likeness_scores(inner, X_train, X_pred)
    quality = 1.0 - ppg.rank01(np.abs(residual))
    score = ppg.rank01(test_score) * np.power(np.clip(quality, 1e-6, 1.0), branch.quality_power)
    keep_mask = tlg.select_mask(score, branch.keep_frac, branch.min_keep)
    if np.max(tlg.species_selection_share(groups, keep_mask)) > 0.65 and np.sum(keep_mask) > branch.min_keep:
        keep_mask = tlg.diversify_by_species(score, groups, keep_mask, min_per_species=3)
    return np.clip(tlg.fit_predict_selected(inner, X_train, y, X_pred, keep_mask, score), 0, None)


def make_candidate(
    spec: CandidateSpec,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    slot1_oof: np.ndarray,
    slot1_test: np.ndarray,
    residual: np.ndarray,
    beta: float,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    F_train: np.ndarray | None = None,
    F_test: np.ndarray | None = None,
    detector_oof: np.ndarray | None = None,
    detector_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gate_oof = gate_from_signal(signal_oof, spec.gate_mode, spec.frac, train_clusters, F_train, detector_oof)
    gate_test = gate_from_signal(signal_test, spec.gate_mode, spec.frac, test_clusters, F_test, detector_test)
    corr_oof = np.zeros_like(slot1_oof, dtype=float)
    corr_test = np.zeros_like(slot1_test, dtype=float)
    raw_oof = np.clip(spec.shrink * beta * signal_oof, -spec.clip, spec.clip)
    raw_test = np.clip(spec.shrink * beta * signal_test, -spec.clip, spec.clip)
    corr_oof[gate_oof] = raw_oof[gate_oof]
    corr_test[gate_test] = raw_test[gate_test]
    if spec.mean_center and np.any(gate_test):
        corr_oof[gate_oof] -= float(np.mean(corr_oof[gate_oof]))
        corr_test[gate_test] -= float(np.mean(corr_test[gate_test]))
    # Avoid a branch whose learned sign is opposite to its OOF evidence.
    if safe_corr(corr_oof[gate_oof], residual[gate_oof]) < 0:
        corr_oof *= -1.0
        corr_test *= -1.0
    return corr_oof, np.clip(slot1_test + corr_test, 0, None)


def gate_from_signal(
    signal: np.ndarray,
    mode: str,
    frac: float,
    labels: np.ndarray | None = None,
    features: np.ndarray | None = None,
    detector: np.ndarray | None = None,
) -> np.ndarray:
    cap: int | None = None
    spacing_power: float | None = None
    detector_frac: float | None = None
    base_mode = mode
    if "_space05" in base_mode:
        spacing_power = 0.5
        base_mode = base_mode.replace("_space05", "")
    elif "_space02" in base_mode:
        spacing_power = 0.2
        base_mode = base_mode.replace("_space02", "")
    for token, value in [
        ("_det08", 0.08),
        ("_det10", 0.10),
        ("_det15", 0.15),
        ("_det20", 0.20),
        ("_det30", 0.30),
        ("_det40", 0.40),
    ]:
        if token in base_mode:
            detector_frac = value
            base_mode = base_mode.replace(token, "")
            break
    if base_mode.endswith("_cluster2"):
        cap = 2
        base_mode = base_mode.removesuffix("_cluster2")
    elif base_mode.endswith("_cluster1"):
        cap = 1
        base_mode = base_mode.removesuffix("_cluster1")
    elif base_mode.endswith("_cluster3"):
        cap = 3
        base_mode = base_mode.removesuffix("_cluster3")
    elif base_mode.endswith("_cluster4"):
        cap = 4
        base_mode = base_mode.removesuffix("_cluster4")
    if base_mode == "abs_signal":
        score = np.abs(signal)
    elif base_mode == "positive_signal":
        score = np.maximum(signal, 0)
    elif base_mode == "negative_signal":
        score = np.maximum(-signal, 0)
    else:
        raise ValueError(mode)
    if detector_frac is not None:
        if detector is None:
            raise ValueError("detector scores are required for detector gate")
        det_gate = ppg.top_fraction(detector, detector_frac)
        score = np.where(det_gate, score, 0.0)
    if cap is None:
        if spacing_power is None:
            return ppg.top_fraction(score, frac)
        if features is None:
            raise ValueError("features are required for spacing gate")
        return spacing_top_fraction(score, frac, features, spacing_power)
    if labels is None:
        raise ValueError("cluster labels are required for cluster-capped gate")
    gate = diverse_top_fraction(score, frac, labels, cap)
    if spacing_power is None:
        return gate
    if features is None:
        raise ValueError("features are required for spacing gate")
    return spacing_top_fraction(score, frac, features, spacing_power, base_gate=gate)


def diverse_top_fraction(score: np.ndarray, frac: float, labels: np.ndarray, cap: int) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    selected: list[int] = []
    counts: dict[int, int] = {}
    order = np.argsort(score, kind="mergesort")[::-1]
    for idx in order:
        label = int(labels[idx])
        if counts.get(label, 0) >= cap:
            continue
        selected.append(int(idx))
        counts[label] = counts.get(label, 0) + 1
        if len(selected) >= target:
            break
    if len(selected) < target:
        used = set(selected)
        for idx in order:
            if int(idx) in used:
                continue
            selected.append(int(idx))
            if len(selected) >= target:
                break
    gate = np.zeros(len(score), dtype=bool)
    gate[np.asarray(selected, dtype=int)] = True
    return gate


def spacing_top_fraction(
    score: np.ndarray,
    frac: float,
    features: np.ndarray,
    spacing_power: float,
    *,
    base_gate: np.ndarray | None = None,
) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    base_order = np.argsort(score, kind="mergesort")[::-1]
    if base_gate is None:
        pool = base_order[: min(len(base_order), max(target * 8, target + 12))]
    else:
        gated = np.flatnonzero(base_gate)
        if len(gated) < target:
            pool = base_order[: min(len(base_order), max(target * 8, target + 12))]
        else:
            gated_set = set(int(i) for i in gated)
            pool = np.asarray([int(i) for i in base_order if int(i) in gated_set], dtype=int)
    if len(pool) <= target:
        gate = np.zeros(len(score), dtype=bool)
        gate[pool] = True
        return gate

    X = features[pool].astype(float, copy=False)
    scale = np.std(X, axis=0)
    scale[scale < 1e-12] = 1.0
    X = (X - np.mean(X, axis=0)) / scale
    score_pool = score[pool].astype(float, copy=False)
    score_range = float(score_pool.max() - score_pool.min())
    score_norm = np.ones_like(score_pool) if score_range < 1e-12 else (score_pool - score_pool.min()) / score_range

    selected_local: list[int] = [0]
    remaining = set(range(1, len(pool)))
    while len(selected_local) < target and remaining:
        sel_X = X[np.asarray(selected_local)]
        rem = np.asarray(sorted(remaining), dtype=int)
        d = np.sqrt(((X[rem, None, :] - sel_X[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        d_range = float(d.max() - d.min())
        d_norm = np.ones_like(d) if d_range < 1e-12 else (d - d.min()) / d_range
        value = score_norm[rem] + spacing_power * d_norm
        winner = int(rem[int(np.argmax(value))])
        selected_local.append(winner)
        remaining.remove(winner)

    gate = np.zeros(len(score), dtype=bool)
    gate[pool[np.asarray(selected_local, dtype=int)]] = True
    return gate


def diagnostics(
    *,
    name: str,
    spec: CandidateSpec,
    path: Path,
    pred: np.ndarray,
    corr_oof: np.ndarray,
    slot1_test: np.ndarray,
    base_test: np.ndarray,
    slot2_only: np.ndarray,
    prior_changed: np.ndarray,
    prior_correction: np.ndarray,
    y: np.ndarray,
    slot1_oof: np.ndarray,
    slot1_oof_rmse: float,
    residual: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    beta: float,
    branch_rmse: float,
    signal_corr: float,
    q5_increment: np.ndarray | None,
) -> dict[str, object]:
    diff = pred - slot1_test
    base_diff = pred - base_test
    changed = np.abs(diff) > 1e-12
    new_changed = changed & ~prior_changed
    repeated_changed = changed & prior_changed
    q5_changed = np.abs(q5_increment) > 1e-12 if q5_increment is not None else np.zeros_like(changed, dtype=bool)
    top10 = np.argsort(np.abs(diff))[-10:]
    corrected_species = pd.Series(test_species[changed]).value_counts()
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_oof = np.clip(slot1_oof + corr_oof, 0, None)
    fold = ppg.fold_delta_stats(y, corrected_oof, slot1_oof, groups)
    oof_rmse = rmse(y, corrected_oof)
    gate_oof = np.abs(corr_oof) > 1e-12
    row: dict[str, object] = {
        "experiment": name,
        "branch": spec.branch.name,
        "preprocess": spec.branch.preprocess,
        "model": spec.branch.model,
        "score_mode": spec.branch.score_mode,
        "keep_frac": spec.branch.keep_frac,
        "quality_power": spec.branch.quality_power,
        "n_components": spec.branch.n_components,
        "gate_mode": spec.gate_mode,
        "frac": spec.frac,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "anchor": "q5" if q5_increment is not None else "stage3",
        "beta": beta,
        "branch_oof_rmse": branch_rmse,
        "branch_delta_vs_slot1": branch_rmse - slot1_oof_rmse,
        "signal_residual_corr": signal_corr,
        "submission_path": str(path),
        "changed_count": int(changed.sum()),
        "negative_count": int((pred < 0).sum()),
        "slot1_diff_rmse": rmse(pred, slot1_test),
        "slot1_diff_max_abs": float(np.max(np.abs(diff))),
        "base_diff_rmse": rmse(pred, base_test),
        "base_diff_max_abs": float(np.max(np.abs(base_diff))),
        "slot2_only_changed_count": int(np.sum(changed & slot2_only)),
        "new_changed_count": int(new_changed.sum()),
        "repeated_changed_count": int(repeated_changed.sum()),
        "prior_overlap_frac": float(repeated_changed.sum() / max(1, changed.sum())),
        "prior_correction_corr": safe_corr(diff, prior_correction),
        "q5_increment_corr": safe_corr(diff, q5_increment) if q5_increment is not None else 0.0,
        "q5_overlap_frac": float(np.sum(changed & q5_changed) / max(1, changed.sum())) if q5_increment is not None else 0.0,
        "q5_new_changed_count": int(np.sum(changed & ~q5_changed)) if q5_increment is not None else int(new_changed.sum()),
        "prior_rows_mean_abs_correction": float(np.mean(np.abs(diff[repeated_changed]))) if repeated_changed.any() else 0.0,
        "new_rows_mean_abs_correction": float(np.mean(np.abs(diff[new_changed]))) if new_changed.any() else 0.0,
        "new_positive_correction_frac": float(np.mean(diff[new_changed] > 0)) if new_changed.any() else 0.0,
        "max_abs_species_mean_shift": ppg.max_species_shift(diff, test_species),
        "corrected_species_max_count": int(corrected_species.iloc[0]) if len(corrected_species) else 0,
        "top10_abs_max_species_count": int(top_species.iloc[0]) if len(top_species) else 0,
        "clip_saturation_frac": float(np.mean(np.abs(diff[changed]) >= spec.clip - 1e-12)) if changed.any() else 0.0,
        "positive_correction_frac": float(np.mean(diff[changed] > 0)) if changed.any() else 0.0,
        "oof_rmse": oof_rmse,
        "oof_delta_vs_slot1": oof_rmse - slot1_oof_rmse,
        "signal_residual_corr_on_gate": safe_corr(corr_oof[gate_oof], residual[gate_oof]) if gate_oof.any() else 0.0,
        "improved_group_count": fold["improved_count"],
        "group_count": fold["group_count"],
        "worst_group_delta": fold["worst_delta"],
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else ""
    row["submit_gate"] = "pass" if not reasons else "reject"
    row["risk"] = risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if row.get("anchor") == "q5":
        if int(row["negative_count"]) > 0:
            reasons.append("negative")
        if not (16 <= int(row["changed_count"]) <= 28):
            reasons.append("changed_count")
        if not (0.006 <= float(row["slot1_diff_rmse"]) <= 0.020):
            reasons.append("slot1_diff_range")
        if float(row["slot1_diff_max_abs"]) > 0.16:
            reasons.append("max_diff")
        if float(row["max_abs_species_mean_shift"]) > 0.008:
            reasons.append("species_shift")
        if int(row["corrected_species_max_count"]) > 6:
            reasons.append("species_concentration")
        if int(row["top10_abs_max_species_count"]) > 3:
            reasons.append("top10_species")
        if float(row["clip_saturation_frac"]) > 0.0:
            reasons.append("clip_saturation")
        if float(row["oof_delta_vs_slot1"]) > -0.0035:
            reasons.append("weak_oof")
        if int(row["improved_group_count"]) < 9:
            reasons.append("weak_groups")
        if float(row["signal_residual_corr_on_gate"]) < 0.30:
            reasons.append("bad_signal_corr")
        if int(row["q5_new_changed_count"]) < 6:
            reasons.append("too_few_q5_new_rows")
        if abs(float(row["q5_increment_corr"])) > 0.80:
            reasons.append("q5_corr")
        if float(row["q5_overlap_frac"]) > 0.65:
            reasons.append("q5_overlap")
        if abs(float(row["prior_correction_corr"])) > 0.75:
            reasons.append("prior_corr")
        return reasons
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if not (12 <= int(row["changed_count"]) <= 24):
        reasons.append("changed_count")
    if not (0.006 <= float(row["slot1_diff_rmse"]) <= 0.016):
        reasons.append("slot1_diff_range")
    if float(row["slot1_diff_max_abs"]) > 0.10:
        reasons.append("max_diff")
    if float(row["max_abs_species_mean_shift"]) > 0.008:
        reasons.append("species_shift")
    if int(row["corrected_species_max_count"]) > 5:
        reasons.append("species_concentration")
    if int(row["top10_abs_max_species_count"]) > 3:
        reasons.append("top10_species")
    if float(row["clip_saturation_frac"]) > 0.0:
        reasons.append("clip_saturation")
    if float(row["oof_delta_vs_slot1"]) > -0.003:
        reasons.append("weak_oof")
    if int(row["improved_group_count"]) < 9:
        reasons.append("weak_groups")
    if float(row["signal_residual_corr_on_gate"]) < 0.30:
        reasons.append("bad_signal_corr")
    if int(row["slot2_only_changed_count"]) > 1:
        reasons.append("slot2_only_rows")
    if int(row["new_changed_count"]) < 6:
        reasons.append("too_few_new_rows")
    if abs(float(row["prior_correction_corr"])) > 0.75:
        reasons.append("prior_corr")
    if (
        int(row["new_changed_count"]) > 0
        and float(row["prior_rows_mean_abs_correction"]) > float(row["new_rows_mean_abs_correction"]) * 2.0
    ):
        reasons.append("prior_rows_dominate")
    return reasons


def risk(row: dict[str, object]) -> float:
    out = 0.0
    out += max(0.0, float(row["slot1_diff_rmse"]) - 0.022) * 4.0
    out += max(0.0, float(row["slot1_diff_max_abs"]) - 0.16) * 1.5
    out += max(0.0, float(row["max_abs_species_mean_shift"]) - 0.010) * 5.0
    out += 0.04 * max(0, int(row["corrected_species_max_count"]) - 8)
    out += 0.05 * max(0, int(row["top10_abs_max_species_count"]) - 3)
    out += 0.03 * int(row["slot2_only_changed_count"])
    out += 0.04 * max(0, 6 - int(row["new_changed_count"]))
    out += max(0.0, abs(float(row["prior_correction_corr"])) - 0.65) * 0.5
    out += 0.20 if float(row["positive_correction_frac"]) in {0.0, 1.0} else 0.0
    return float(out)


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(signal, residual) / denom), -1.0, 1.0))


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    pass_penalty = 0 if row["submit_gate"] == "pass" else 1
    return (
        pass_penalty,
        float(row["risk"]) - min(0.03, -float(row["oof_delta_vs_slot1"])),
        float(row["slot1_diff_rmse"]),
    )


def candidate_name(spec: CandidateSpec, beta: float) -> str:
    return (
        f"nir_s1tn_{spec.branch.name}_{spec.gate_mode}_"
        f"f{tag(spec.frac)}_s{tag(spec.shrink)}_c{tag(spec.clip)}_b{tag(beta)}"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def pct_tag(value: float) -> str:
    return tag(value).replace("0p", "")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
        f"risk={row['risk']:.4f} diff={row['slot1_diff_rmse']:.4f} "
        f"oof={row['oof_delta_vs_slot1']:.4f} groups={row['improved_group_count']}/{row['group_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
