#!/usr/bin/env python3
"""Proper stage-2 local residual models for validated hard cases.

The current public-best anchor is kept fixed.  This script only learns a small
correction for rows selected by the validated hard-case detector:

    corrected = anchor + clipped(shrink * local_residual_estimate)

The local residual estimate is fit from fold-local anchor OOF residuals, not
from direct target replacement.  This follows the safer Locally-Biased
Regression / local error correction pattern for spectroscopy calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import nir_base_shape_aug_guard_search as bsa
import nir_gated_local_correction_search as glc
import nir_hard_case_detector_diagnostics as hcd
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op
import nir_slot1_testnear_branch_search as stn
import nir_stage2_hardcase_correction_search as s2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_stage2_local_residual_model"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv"


@dataclass(frozen=True)
class ResidualBranchSpec:
    name: str
    feature_space: str
    method: str
    k: int
    alpha: float = 1.0
    anchor_weight: float = 0.35
    distance_power: float = 1.0
    quality_mode: str = "none"
    quality_keep: float = 1.0
    quality_power: float = 0.0


@dataclass(frozen=True)
class ResidualSubmitSpec:
    name: str
    branch_name: str
    detector_frac: float
    shrink: float
    clip: float
    gate_mode: str = "detector"
    soft_power: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branches", type=int, default=None)
    parser.add_argument("--max-submits", type=int, default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated local methods to include.")
    parser.add_argument("--anchor-weights", default=None, help="Comma-separated anchor weights to include.")
    parser.add_argument("--ks", default=None, help="Comma-separated k values to include.")
    parser.add_argument("--gate-modes", default=None, help="Comma-separated gate modes to include.")
    parser.add_argument("--quality-modes", default=None, help="Comma-separated train quality modes to include.")
    parser.add_argument("--quality-keeps", default=None, help="Comma-separated quality keep fractions.")
    parser.add_argument("--quality-powers", default=None, help="Comma-separated quality weight powers.")
    parser.add_argument("--detector-fracs", default=None, help="Comma-separated gate fractions.")
    parser.add_argument("--shrinks", default=None, help="Comma-separated correction shrink values.")
    parser.add_argument("--clips", default=None, help="Comma-separated correction clip values.")
    parser.add_argument("--soft-powers", default=None, help="Comma-separated soft correction powers.")
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    refs = nl.load_reference_diffs(anchor_test)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current Stage3 anchor OOF", flush=True)
    base_df = pd.read_csv(stn.BASE_ANCHOR, header=None, names=["id", "pred"])
    slot1_df = pd.read_csv(stn.SLOT1, header=None, names=["id", "pred"])
    first_stage_best_df = pd.read_csv(stn.FIRST_STAGE_BEST, header=None, names=["id", "pred"])
    second_stage_best_df = pd.read_csv(stn.SECOND_STAGE_BEST, header=None, names=["id", "pred"])
    for label, df in [
        ("base", base_df),
        ("slot1", slot1_df),
        ("first_stage_best", first_stage_best_df),
        ("second_stage_best", second_stage_best_df),
    ]:
        if not np.array_equal(data["test_ids"], df["id"].to_numpy()):
            raise ValueError(f"{label} sample order mismatch")
    _, slot1_oof = stn.make_slot1_oof(data, base_df["pred"].to_numpy(float))
    anchor_oof = stn.make_current_best_oof(
        data,
        slot1_oof,
        slot1_df["pred"].to_numpy(float),
        first_stage_best_df["pred"].to_numpy(float),
        second_stage_best_df["pred"].to_numpy(float),
    )
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    residual = data["y"] - anchor_oof
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    print("building validated hard-case detector", flush=True)
    detector_features_train, detector_features_test, _ = s2.build_detector_features(data, anchor_oof, anchor_test)
    hard_label = np.abs(residual) >= np.quantile(np.abs(residual), 0.80)
    detector_oof = hcd.detector_oof_score("logreg", detector_features_train, hard_label, data["groups"])
    detector_test = hcd.detector_full_score("logreg", detector_features_train, hard_label, detector_features_test)
    positive_label = residual > 0
    signed_oof = hcd.detector_oof_score("logreg", detector_features_train, positive_label, data["groups"])
    signed_test = hcd.detector_full_score("logreg", detector_features_train, positive_label, detector_features_test)
    detector_rows = s2.detector_summary(detector_oof, detector_test, hard_label, np.abs(residual), data["groups"], test_species)

    branches = build_branches(
        methods=parse_str_list(args.methods),
        anchor_weights=parse_float_list(args.anchor_weights),
        ks=parse_int_list(args.ks),
        quality_modes=parse_str_list(args.quality_modes),
        quality_keeps=parse_float_list(args.quality_keeps),
        quality_powers=parse_float_list(args.quality_powers),
    )
    if args.max_branches is not None:
        branches = branches[: args.max_branches]
    submits = build_submit_specs(
        branches,
        gate_modes=parse_str_list(args.gate_modes),
        detector_fracs=parse_float_list(args.detector_fracs),
        shrinks=parse_float_list(args.shrinks),
        clips=parse_float_list(args.clips),
        soft_powers=parse_float_list(args.soft_powers),
    )
    if args.max_submits is not None:
        submits = submits[: args.max_submits]

    branch_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, branch in enumerate(branches, start=1):
        print(f"[branch {i}/{len(branches)}] {branch.name}", flush=True)
        signal_oof, iso_oof = make_residual_signal_oof(branch, data, anchor_oof, anchor_test, residual)
        signal_test, iso_test = fit_predict_residual_signal(
            branch,
            data["X_train"],
            residual,
            data["X_test"],
            anchor_oof,
            anchor_test,
            data["X_test"],
            anchor_test,
        )
        branch_cache[branch.name] = (signal_oof, signal_test, iso_oof, iso_test)

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(submits, start=1):
        print(f"[submit {i}/{len(submits)}] {spec.name}", flush=True)
        branch = next(b for b in branches if b.name == spec.branch_name)
        signal_oof, signal_test, _, _ = branch_cache[branch.name]
        gate_oof = make_gate(detector_oof, signed_oof, signal_oof, spec.detector_frac, spec.gate_mode)
        gate_test = make_gate(detector_test, signed_test, signal_test, spec.detector_frac, spec.gate_mode)
        weight_oof = correction_weight(detector_oof, signed_oof, signal_oof, gate_oof, spec.soft_power)
        weight_test = correction_weight(detector_test, signed_test, signal_test, gate_test, spec.soft_power)
        corr_oof = np.where(gate_oof, np.clip(spec.shrink * signal_oof * weight_oof, -spec.clip, spec.clip), 0.0)
        corr_test = np.where(gate_test, np.clip(spec.shrink * signal_test * weight_test, -spec.clip, spec.clip), 0.0)
        corrected_oof = np.clip(anchor_oof + corr_oof, 0, None)
        pred = np.clip(anchor_test + corr_test, 0, None)

        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            branch=branch,
            path=path,
            pred=pred,
            corrected_oof=corrected_oof,
            corr_test=corr_test,
            corr_oof=corr_oof,
            signal_oof=signal_oof,
            signal_test=signal_test,
            gate_oof=gate_oof,
            gate_test=gate_test,
            hard_label=hard_label,
            positive_label=positive_label,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            refs=refs,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "stage2_local_residual_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "stage2_local_residual_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"detector_summary": detector_rows, "candidate_summary": rows_sorted}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(detector_rows).to_csv(out_dir / "detector_gate_summary.csv", index=False)

    print("\nDetector gate summary:")
    print(pd.DataFrame(detector_rows).to_string(index=False))
    print("\nTop local residual candidates:")
    for row in rows_sorted[:50]:
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} corrmax={row['max_abs_correction']:.4f} "
            f"clipfrac={row['clip_saturation_frac_on_gate']:.3f} var={row['correction_std_on_gate']:.4f} "
            f"gfrac={row['gate_test_frac']:.3f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"sigcorr={row['signal_residual_corr']:.4f} oof={row['oof_delta_vs_anchor']:.4f} "
            f"fold={row['improved_fold_count']}/5"
        )
    print(f"saved Stage2 local residual diagnostics: {out_dir}")


def build_branches(
    *,
    methods: list[str] | None = None,
    anchor_weights: list[float] | None = None,
    ks: list[int] | None = None,
    quality_modes: list[str] | None = None,
    quality_keeps: list[float] | None = None,
    quality_powers: list[float] | None = None,
) -> list[ResidualBranchSpec]:
    branches: list[ResidualBranchSpec] = []
    for feature_space in ["current"]:
        for anchor_weight in (anchor_weights or [0.0, 0.35, 0.70]):
            for method in (methods or ["wmean", "median", "ridge", "pls1", "pls2", "pls3"]):
                for k in (ks or [40, 80, 120]):
                    alpha = 5.0 if method == "ridge" else 0.0
                    for quality_mode in (quality_modes or ["none"]):
                        keep_values = quality_keeps or ([1.0] if quality_mode == "none" else [0.65, 0.80])
                        power_values = quality_powers or ([0.0] if quality_mode == "none" else [0.5, 1.0])
                        for quality_keep in keep_values:
                            for quality_power in power_values:
                                if quality_mode == "none" and (quality_keep != 1.0 or quality_power != 0.0):
                                    continue
                                branches.append(
                                    ResidualBranchSpec(
                                        name=(
                                            f"lres_{feature_space}_{method}_k{k}_aw{tag(anchor_weight)}_"
                                            f"a{tag(alpha)}_q{quality_mode}_keep{tag(quality_keep)}_"
                                            f"qp{tag(quality_power)}"
                                        ),
                                        feature_space=feature_space,
                                        method=method,
                                        k=k,
                                        alpha=alpha,
                                        anchor_weight=anchor_weight,
                                        quality_mode=quality_mode,
                                        quality_keep=quality_keep,
                                        quality_power=quality_power,
                                    )
                        )
    return branches


def build_submit_specs(
    branches: list[ResidualBranchSpec],
    *,
    gate_modes: list[str] | None = None,
    detector_fracs: list[float] | None = None,
    shrinks: list[float] | None = None,
    clips: list[float] | None = None,
    soft_powers: list[float] | None = None,
) -> list[ResidualSubmitSpec]:
    specs: list[ResidualSubmitSpec] = []
    for branch in branches:
        for detector_frac in (detector_fracs or [0.05, 0.075]):
            for gate_mode in (
                gate_modes
                or ["detector", "impact_abs", "impact_signed_pos", "signed_pos", "signed_impact_abs", "signed_impact_pos"]
            ):
                for soft_power in (soft_powers or [0.0, 0.5]):
                    for shrink in (shrinks or [0.003, 0.005, 0.01, 0.02, 0.05, 0.10]):
                        for clip in (clips or [0.05, 0.10, 0.15, 0.20, 0.30]):
                            specs.append(
                                ResidualSubmitSpec(
                                    name=(
                                        f"nir_s2lbr_{gate_mode}_top{tag(detector_frac)}_"
                                        f"{branch.name}_s{tag(shrink)}_c{tag(clip)}_"
                                        f"soft{tag(soft_power)}"
                                    ),
                                    branch_name=branch.name,
                                    detector_frac=detector_frac,
                                    shrink=shrink,
                                    clip=clip,
                                    gate_mode=gate_mode,
                                    soft_power=soft_power,
                                )
                            )
    return specs


def parse_str_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_list(value: str | None) -> list[float] | None:
    parts = parse_str_list(value)
    return None if parts is None else [float(part) for part in parts]


def parse_int_list(value: str | None) -> list[int] | None:
    parts = parse_str_list(value)
    return None if parts is None else [int(part) for part in parts]


def make_residual_signal_oof(
    spec: ResidualBranchSpec,
    data: dict[str, np.ndarray],
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = data["X_train"]
    groups = data["groups"]
    pred = np.empty(len(residual), dtype=float)
    iso = np.empty(len(residual), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, residual, groups):
        pred[valid_idx], iso[valid_idx] = fit_predict_residual_signal(
            spec,
            X[train_idx],
            residual[train_idx],
            X[valid_idx],
            anchor_oof[train_idx],
            anchor_oof[valid_idx],
            data["X_test"],
            anchor_test,
        )
    return pred, iso


def fit_predict_residual_signal(
    spec: ResidualBranchSpec,
    X_train_raw: np.ndarray,
    residual_train: np.ndarray,
    X_pred_raw: np.ndarray,
    anchor_train: np.ndarray,
    anchor_pred: np.ndarray,
    X_quality_raw: np.ndarray | None = None,
    anchor_quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    F_train, F_pred, F_quality = make_residual_features(
        spec,
        X_train_raw,
        X_pred_raw,
        anchor_train,
        anchor_pred,
        X_quality_raw,
        anchor_quality,
    )
    quality = train_quality_score(spec, F_train, F_quality, residual_train)
    train_mask = quality >= np.quantile(quality, 1.0 - spec.quality_keep)
    if np.sum(train_mask) < min(spec.k, len(residual_train)):
        train_mask = np.ones(len(residual_train), dtype=bool)
    F_pool = F_train[train_mask]
    residual_pool = residual_train[train_mask]
    quality_pool = quality[train_mask]
    k = min(spec.k, len(residual_pool))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(F_pool)
    distances, indices = nn.kneighbors(F_pred, return_distance=True)
    pred = np.empty(len(X_pred_raw), dtype=float)
    iso = distances.mean(axis=1)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        pred[i] = fit_one_local_residual(spec, F_pool[idx], residual_pool[idx], F_pred[i], dist, quality_pool[idx])
    return pred, iso


def make_residual_features(
    spec: ResidualBranchSpec,
    X_train_raw: np.ndarray,
    X_pred_raw: np.ndarray,
    anchor_train: np.ndarray,
    anchor_pred: np.ndarray,
    X_quality_raw: np.ndarray | None = None,
    anchor_quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    local_spec = glc.LocalSpec(name="feature_builder", feature_space=spec.feature_space)
    if X_quality_raw is not None and anchor_quality is not None:
        X_all_pred = np.vstack([X_pred_raw, X_quality_raw])
        F_train, F_all_pred = glc.make_features(local_spec, X_train_raw, X_all_pred)
        F_pred = F_all_pred[: len(X_pred_raw)]
        F_quality = F_all_pred[len(X_pred_raw) :]
    else:
        F_train, F_pred = glc.make_features(local_spec, X_train_raw, X_pred_raw)
        F_quality = None
    if spec.anchor_weight <= 0:
        return F_train, F_pred, F_quality
    scaler = StandardScaler()
    a_train = scaler.fit_transform(anchor_train.reshape(-1, 1)) * spec.anchor_weight
    a_pred = scaler.transform(anchor_pred.reshape(-1, 1)) * spec.anchor_weight
    F_train_out = np.hstack([F_train, a_train])
    F_pred_out = np.hstack([F_pred, a_pred])
    if F_quality is None or anchor_quality is None:
        return F_train_out, F_pred_out, None
    a_quality = scaler.transform(anchor_quality.reshape(-1, 1)) * spec.anchor_weight
    return F_train_out, F_pred_out, np.hstack([F_quality, a_quality])


def train_quality_score(
    spec: ResidualBranchSpec,
    F_train: np.ndarray,
    F_quality: np.ndarray | None,
    residual_train: np.ndarray,
) -> np.ndarray:
    if spec.quality_mode == "none" or F_quality is None or len(F_quality) == 0:
        return np.ones(len(F_train), dtype=float)
    nn = NearestNeighbors(n_neighbors=min(10, len(F_quality)))
    nn.fit(F_quality)
    dist, _ = nn.kneighbors(F_train, return_distance=True)
    testlike = rank01(-dist.mean(axis=1))
    stable = rank01(-np.abs(residual_train))
    if spec.quality_mode == "testlike":
        quality = testlike
    elif spec.quality_mode == "stable":
        quality = stable
    elif spec.quality_mode == "testlike_stable":
        quality = np.sqrt(np.clip(testlike, 1e-6, None) * np.clip(stable, 1e-6, None))
    else:
        raise ValueError(spec.quality_mode)
    return np.asarray(quality, dtype=float)


def fit_one_local_residual(
    spec: ResidualBranchSpec,
    F_local: np.ndarray,
    residual_local: np.ndarray,
    f_query: np.ndarray,
    dist: np.ndarray,
    quality_local: np.ndarray,
) -> float:
    weight = 1.0 / np.power(dist + 1e-6, spec.distance_power)
    if spec.quality_power > 0.0:
        weight = weight * np.power(np.clip(quality_local, 1e-6, None), spec.quality_power)
    weight = weight / np.mean(weight)
    if spec.method == "wmean":
        return float(np.average(residual_local, weights=weight))
    if spec.method == "median":
        return float(np.median(residual_local))
    if spec.method == "ridge":
        model = Ridge(alpha=spec.alpha)
        model.fit(F_local, residual_local, sample_weight=weight)
        return float(model.predict(f_query.reshape(1, -1))[0])
    if spec.method.startswith("pls"):
        n_components = int(spec.method.replace("pls", ""))
        n_components = max(1, min(n_components, F_local.shape[0] - 1, F_local.shape[1]))
        Xw, yw = weighted_center(F_local, residual_local, weight)
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(Xw, yw)
        xq = f_query.reshape(1, -1) - np.average(F_local, axis=0, weights=weight)
        return float(model.predict(xq)[0, 0] + np.average(residual_local, weights=weight))
    raise ValueError(spec.method)


def weighted_center(X: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_center = np.average(X, axis=0, weights=weight)
    y_center = float(np.average(y, weights=weight))
    sqrt_w = np.sqrt(weight / np.mean(weight))
    return (X - x_center) * sqrt_w[:, None], (y - y_center).reshape(-1, 1) * sqrt_w[:, None]


def make_gate(detector_score: np.ndarray, signed_score: np.ndarray, signal: np.ndarray, frac: float, mode: str) -> np.ndarray:
    if mode == "detector":
        score = detector_score
    elif mode == "impact_abs":
        score = rank01(detector_score) * rank01(np.abs(signal))
    elif mode == "impact_signed_pos":
        score = rank01(detector_score) * rank01(np.maximum(signal, 0.0))
    elif mode == "signed_pos":
        score = rank01(detector_score) * rank01(signed_score)
    elif mode == "signed_impact_abs":
        score = rank01(detector_score) * rank01(signed_score) * rank01(np.abs(signal))
    elif mode == "signed_impact_pos":
        score = rank01(detector_score) * rank01(signed_score) * rank01(np.maximum(signal, 0.0))
    else:
        raise ValueError(mode)
    return s2.top_fraction(score, frac)


def correction_weight(
    detector_score: np.ndarray,
    signed_score: np.ndarray,
    signal: np.ndarray,
    gate: np.ndarray,
    soft_power: float,
) -> np.ndarray:
    if soft_power <= 0:
        return np.ones_like(detector_score, dtype=float)
    score = rank01(detector_score) * rank01(signed_score) * rank01(np.abs(signal))
    out = np.ones_like(score, dtype=float)
    if np.any(gate):
        local = score[gate]
        denom = float(np.max(local) - np.min(local))
        scaled = np.ones_like(local) if denom < 1e-12 else 0.5 + 0.5 * (local - np.min(local)) / denom
        out[gate] = np.power(scaled, soft_power)
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def diagnostics(
    *,
    spec: ResidualSubmitSpec,
    branch: ResidualBranchSpec,
    path: Path,
    pred: np.ndarray,
    corrected_oof: np.ndarray,
    corr_test: np.ndarray,
    corr_oof: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    gate_oof: np.ndarray,
    gate_test: np.ndarray,
    hard_label: np.ndarray,
    positive_label: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
) -> dict[str, object]:
    diff = pred - anchor_test
    oof_rmse = rmse(y, corrected_oof)
    fold = nl.fold_delta_stats(y, corrected_oof, anchor_oof, groups)
    residual = y - anchor_oof
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_species = pd.Series(test_species[gate_test]).value_counts() if np.any(gate_test) else pd.Series(dtype=int)
    on_gate = corr_test[gate_test]
    raw_on_gate = spec.shrink * signal_test[gate_test]
    clipped = np.abs(raw_on_gate) > spec.clip + 1e-12 if len(raw_on_gate) else np.array([], dtype=bool)
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "branch": branch.name,
        "feature_space": branch.feature_space,
        "method": branch.method,
        "k": branch.k,
        "alpha": branch.alpha,
        "anchor_weight": branch.anchor_weight,
        "distance_power": branch.distance_power,
        "quality_mode": branch.quality_mode,
        "quality_keep": branch.quality_keep,
        "quality_power": branch.quality_power,
        "detector_frac": spec.detector_frac,
        "gate_mode": spec.gate_mode,
        "soft_power": spec.soft_power,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "anchor_oof_rmse": anchor_oof_rmse,
        "oof_rmse": oof_rmse,
        "oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "improved_fold_count": fold["improved_count"],
        "worst_fold_delta": fold["worst_delta"],
        "fold_delta_std": fold["delta_std"],
        "signal_residual_corr": nl.safe_corr(signal_oof[gate_oof], residual[gate_oof]) if np.any(gate_oof) else 0.0,
        "signal_residual_corr_all": nl.safe_corr(signal_oof, residual),
        "hard_precision_on_gate": float(np.mean(hard_label[gate_oof])) if np.any(gate_oof) else 0.0,
        "positive_precision_on_gate": float(np.mean(positive_label[gate_oof])) if np.any(gate_oof) else 0.0,
        "residual_lift_on_gate": float(np.mean(np.abs(residual[gate_oof])) / np.mean(np.abs(residual))) if np.any(gate_oof) else 0.0,
        "signal_test_mean_on_gate": float(np.mean(signal_test[gate_test])) if np.any(gate_test) else 0.0,
        "signal_test_std_on_gate": float(np.std(signal_test[gate_test])) if np.any(gate_test) else 0.0,
        "gate_test_count": int(np.sum(gate_test)),
        "gate_test_frac": float(np.mean(gate_test)),
        "gate_oof_frac": float(np.mean(gate_oof)),
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": nl.rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": nl.safe_corr(pred, anchor_test),
        "max_abs_correction": float(np.max(np.abs(corr_test))),
        "mean_abs_correction_on_gate": float(np.mean(np.abs(on_gate))) if len(on_gate) else 0.0,
        "correction_std_on_gate": float(np.std(on_gate)) if len(on_gate) else 0.0,
        "clip_saturation_frac_on_gate": float(np.mean(clipped)) if len(clipped) else 0.0,
        "positive_correction_frac_on_gate": float(np.mean(on_gate > 0)) if len(on_gate) else 0.0,
        "max_abs_species_mean_shift": nl.max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "pred_std_ratio_vs_anchor": float(np.std(pred) / np.std(anchor_test)),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": nl.safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]) if len(top_species) else 0,
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
        "corrected_species_max_count": int(corrected_species.iloc[0]) if len(corrected_species) else 0,
        "corrected_species_set": ",".join(map(str, sorted(corrected_species.index.tolist()))),
    }
    for ref_name, ref_diff in refs.items():
        row[f"corr_diff_{ref_name}"] = nl.safe_corr(diff, ref_diff)
        row[f"rmse_diff_{ref_name}"] = nl.rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    if diff < 0.015:
        reasons.append("near_anchor")
    if diff > 0.10:
        reasons.append("anchor_diff_gt0p10")
    if float(row["max_abs_correction"]) > 0.30:
        reasons.append("correction_gt0p30")
    if float(row["anchor_diff_max_abs"]) > 0.35:
        reasons.append("max_diff_gt0p35")
    if float(row["max_abs_species_mean_shift"]) > 0.04:
        reasons.append("species_shift_gt0p04")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if int(row["corrected_species_max_count"]) > 12:
        reasons.append("corrected_species_concentration")
    if float(row["oof_delta_vs_anchor"]) > 0.0:
        reasons.append("oof_not_improved")
    if int(row["improved_fold_count"]) < 3:
        reasons.append("folds_lt3")
    if float(row["worst_fold_delta"]) > 0.08:
        reasons.append("worst_fold_gt0p08")
    if float(row["signal_residual_corr"]) <= 0.0:
        reasons.append("signal_corr_nonpositive")
    if float(row["signal_residual_corr_all"]) <= 0.0:
        reasons.append("signal_corr_all_nonpositive")
    if float(row["clip_saturation_frac_on_gate"]) > 0.50:
        reasons.append("clip_saturated")
    if float(row["correction_std_on_gate"]) < 0.005 and float(row["mean_abs_correction_on_gate"]) > 0:
        reasons.append("correction_not_variable")
    if float(row["positive_correction_frac_on_gate"]) > 0.98 or float(row["positive_correction_frac_on_gate"]) < 0.02:
        reasons.append("one_sided_correction")
    if float(row.get("corr_diff_bad_alpha3000", 0.0)) > 0.35:
        reasons.append("bad_alpha3000_corr_gt0p35")
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    return float(
        2.0 * max(float(row.get("corr_diff_bad_alpha3000", 0.0)), 0.0)
        + 1.0 * abs(float(row["decile_gap_top_minus_bottom"]))
        + 0.7 * abs(float(row["sample_order_corr"]))
        + 1.0 * float(row["max_abs_species_mean_shift"])
        + 0.5 * max(float(row["oof_delta_vs_anchor"]), 0.0)
        + 0.08 * float(row["anchor_diff_max_abs"])
        + 0.5 * float(row["clip_saturation_frac_on_gate"])
        + (0.5 if int(row["top10_abs_max_species_count"]) > 4 else 0.0)
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(row["submit_gate"]) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.5 * max(float(row["oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.04),
        float(row["oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"corrmax={row['max_abs_correction']:.4f} clipfrac={row['clip_saturation_frac_on_gate']:.3f} "
        f"var={row['correction_std_on_gate']:.4f} gfrac={row['gate_test_frac']:.3f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} sigcorr={row['signal_residual_corr']:.4f} "
        f"oof={row['oof_delta_vs_anchor']:.4f} fold={row['improved_fold_count']}/5 "
        f"reasons={row['reject_reasons']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
