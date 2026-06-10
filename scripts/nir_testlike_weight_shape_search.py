#!/usr/bin/env python3
"""Test-like golden weighting on the current shape-augmented YJ Ridge anchor.

This is a re-do of the golden-subset idea in the feature space that currently
works best:

    SG9/SNV -> PCA18 + shape14(scale=0.5) -> Yeo-Johnson Ridge

The experiment keeps the estimator simple and changes only sample_weight.  Two
confidence signals are combined:

* residual confidence: downweight training samples with high group-OOF YJ
  residuals.
* test-likeness: upweight training samples whose latent/shape descriptor is
  close to the public test spectra.

The intent is not to hard-delete samples, but to create a soft "golden"
calibration set while preserving the current anchor's prediction scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_base_shape_aug_guard_search as bsa
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_testlike_weight_shape"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"


@dataclass(frozen=True)
class WeightSpec:
    name: str
    pca_components: int = 18
    alpha: float = 3500.0
    shape_set: str = "shape14"
    shape_scale: float = 0.50
    residual_top: float = 0.00
    residual_min_weight: float = 1.00
    residual_gamma: float = 1.0
    test_mode: str = "knn"
    test_k: int = 20
    test_top: float = 0.00
    test_max_weight: float = 1.00
    test_gamma: float = 1.0
    species_balance: bool = False
    species_mean_cap: float = 0.0
    affine_shrink: float = 0.12
    affine_clip: float = 0.40
    mean_center: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--nested-top", type=int, default=24)
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

    print("building current shape-anchor OOF", flush=True)
    anchor_spec = as_aug_spec(WeightSpec(name="current_anchor"))
    anchor_direct_oof = bsa.make_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    anchor_oof = bsa.make_nested_affine_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    anchor_direct_rmse = rmse(data["y"], anchor_direct_oof)
    print(f"anchor_direct_oof_rmse={anchor_direct_rmse:.6f}", flush=True)
    print(f"anchor_nested_affine_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    specs = build_specs()
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"weight_specs={len(specs)}", flush=True)

    rows: list[dict[str, object]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            full_weights, full_weight_meta = make_weights(
                spec,
                data["X_train"],
                data["y"],
                data["groups"],
                data["X_test"],
            )
            direct_oof = np.clip(make_weighted_oof(spec, data), 0, None)
            direct_test = np.clip(
                fit_predict_weighted(
                    spec,
                    data["X_train"],
                    data["y"],
                    data["X_test"],
                    full_weights,
                ),
                0,
                None,
            )
            cache[spec.name] = (direct_oof, direct_test, full_weights)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        pred, affine_oof = apply_affine_from_oof(spec, direct_test, direct_oof, data["y"])
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            direct_test=direct_test,
            direct_oof=direct_oof,
            affine_oof=affine_oof,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_direct_oof=anchor_direct_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            weights=full_weights,
            weight_meta=full_weight_meta,
            refs=refs,
            nested=False,
        )
        rows.append(row)
        print_one(row)

    nested_names = pick_nested(rows, args.nested_top)
    print(f"nested variants={len(nested_names)}", flush=True)
    spec_by_name = {spec.name: spec for spec in specs}
    row_by_name = {str(row["experiment"]): row for row in rows}
    for name in nested_names:
        spec = spec_by_name[name]
        direct_oof, direct_test, full_weights = cache[name]
        nested_oof = make_nested_affine_oof(spec, data)
        pred, _ = apply_affine_from_oof(spec, direct_test, direct_oof, data["y"])
        path = candidate_dir / f"{spec.name}_nested_ranked.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        full_weight_meta = weight_summary(full_weights, np.zeros_like(full_weights), np.zeros_like(full_weights))
        row_by_name[name].update(
            diagnostics(
                spec=spec,
                path=path,
                pred=pred,
                direct_test=direct_test,
                direct_oof=direct_oof,
                affine_oof=nested_oof,
                anchor_test=anchor_test,
                anchor_oof=anchor_oof,
                anchor_direct_oof=anchor_direct_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                y=data["y"],
                groups=data["groups"],
                test_species=test_species,
                weights=full_weights,
                weight_meta=full_weight_meta,
                refs=refs,
                nested=True,
            )
        )
        print("nested", end=" ")
        print_one(row_by_name[name])

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "testlike_weight_shape_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "testlike_weight_shape_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop test-like weighted candidates:")
    for row in rows_sorted[:80]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"direct={row['direct_oof_delta_vs_anchor_direct']:.4f} "
            f"affine={row['affine_oof_delta_vs_anchor']:.4f} "
            f"fold={row['affine_improved_fold_count']}/5 nested={row['nested_evaluated']}"
        )
    print(f"saved test-like weight diagnostics: {out_dir}")


def build_specs() -> list[WeightSpec]:
    specs: list[WeightSpec] = [
        WeightSpec(name="nir_tlw_anchor_control"),
    ]

    for test_top in [0.15, 0.20]:
        for test_max in [1.10, 1.12, 1.15]:
            for species_mean_cap in [0.0, 1.12]:
                cap_suffix = "" if species_mean_cap == 0.0 else f"_spcap{tag(species_mean_cap)}"
                specs.append(
                    WeightSpec(
                        name=(
                            f"nir_tlw_acap_k10_top{tag(test_top)}_"
                            f"up{tag(test_max)}_g1{cap_suffix}"
                        ),
                        test_mode="knn",
                        test_k=10,
                        test_top=test_top,
                        test_max_weight=test_max,
                        test_gamma=1.0,
                        species_mean_cap=species_mean_cap,
                    )
                )

    for residual_top in [0.10, 0.15]:
        for residual_min in [0.90, 0.95]:
            for species_mean_cap in [0.0, 1.08, 1.10]:
                for test_max in [1.10, 1.15, 1.22]:
                    cap_suffix = "" if species_mean_cap == 0.0 else f"_spcap{tag(species_mean_cap)}"
                    specs.append(
                        WeightSpec(
                            name=(
                                f"nir_tlw_focus_k10_top0p2_up{tag(test_max)}_g1_"
                                f"yjres{tag(residual_top)}_dn{tag(residual_min)}{cap_suffix}"
                            ),
                            residual_top=residual_top,
                            residual_min_weight=residual_min,
                            residual_gamma=1.0,
                            test_mode="knn",
                            test_k=10,
                            test_top=0.20,
                            test_max_weight=test_max,
                            test_gamma=1.0,
                            species_mean_cap=species_mean_cap,
                        )
                    )

    for test_mode in ["knn", "ratio"]:
        for test_k in [10, 20, 40]:
            for test_top in [0.20, 0.35, 0.50, 0.70]:
                for test_max in [1.05, 1.10, 1.15, 1.22]:
                    for test_gamma in [0.7, 1.0, 1.6]:
                        for species_balance in [True, False]:
                            suffix = "_spbal" if species_balance else ""
                            specs.append(
                                WeightSpec(
                                    name=(
                                        f"nir_tlw_{test_mode}k{test_k}_top{tag(test_top)}_"
                                        f"up{tag(test_max)}_g{tag(test_gamma)}{suffix}"
                                    ),
                                    test_mode=test_mode,
                                    test_k=test_k,
                                    test_top=test_top,
                                    test_max_weight=test_max,
                                    test_gamma=test_gamma,
                                    species_balance=species_balance,
                                )
                            )

    for residual_top in [0.02, 0.03, 0.05, 0.08]:
        for residual_min in [0.78, 0.85, 0.90]:
            for test_mode in ["knn", "ratio"]:
                for test_k in [10, 20, 40]:
                    for test_top in [0.35, 0.50, 0.70]:
                        for test_max in [1.08, 1.15, 1.22]:
                            for species_balance in [True, False]:
                                suffix = "_spbal" if species_balance else ""
                                specs.append(
                                    WeightSpec(
                                        name=(
                                            f"nir_tlw_yjres{tag(residual_top)}_dn{tag(residual_min)}_"
                                            f"{test_mode}k{test_k}_top{tag(test_top)}_up{tag(test_max)}{suffix}"
                                        ),
                                        residual_top=residual_top,
                                        residual_min_weight=residual_min,
                                        residual_gamma=1.0,
                                        test_mode=test_mode,
                                        test_k=test_k,
                                        test_top=test_top,
                                        test_max_weight=test_max,
                                        test_gamma=1.0,
                                        species_balance=species_balance,
                                    )
                                )

    focused: list[WeightSpec] = []
    for base in specs:
        if base.name == "nir_tlw_anchor_control":
            focused.append(base)
            continue
        for alpha in [3500.0, 4000.0]:
            for pca_components in [18, 20]:
                if alpha == 3500.0 and pca_components == 18:
                    focused.append(base)
                else:
                    focused.append(
                        replace(
                            base,
                            name=f"{base.name}_p{pca_components}_a{int(alpha)}",
                            pca_components=pca_components,
                            alpha=alpha,
                        )
                    )
    return focused


def make_weighted_oof(spec: WeightSpec, data: dict[str, np.ndarray]) -> np.ndarray:
    X = data["X_train"]
    y = data["y"]
    groups = data["groups"]
    X_test = data["X_test"]
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        weights, _ = make_weights(spec, X[train_idx], y[train_idx], groups[train_idx], X_test)
        pred[valid_idx] = fit_predict_weighted(
            spec,
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            weights,
        )
    return pred


def make_nested_affine_oof(spec: WeightSpec, data: dict[str, np.ndarray]) -> np.ndarray:
    if spec.affine_shrink == 0.0:
        return make_weighted_oof(spec, data)
    X = data["X_train"]
    y = data["y"]
    groups = data["groups"]
    X_test = data["X_test"]
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        weights, _ = make_weights(spec, X[train_idx], y[train_idx], groups[train_idx], X_test)
        valid_direct = fit_predict_weighted(spec, X[train_idx], y[train_idx], X[valid_idx], weights)
        inner_data = {
            "X_train": X[train_idx],
            "y": y[train_idx],
            "groups": groups[train_idx],
            "X_test": X_test,
        }
        inner_oof = make_weighted_oof(spec, inner_data)
        params = bsa.fit_affine_params(inner_oof, y[train_idx])
        pred[valid_idx] = bsa.apply_affine(valid_direct, params, as_aug_spec(spec))
    return np.clip(pred, 0, None)


def fit_predict_weighted(
    spec: WeightSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    F_train, F_pred = make_model_features(spec, X_train_raw, X_pred_raw)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=spec.alpha)
    model.fit(F_train, yt, sample_weight=sample_weight)
    pred_t = model.predict(F_pred)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def make_model_features(
    spec: WeightSpec,
    X_train_raw: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, "sg9_snv")
    pca = PCA(n_components=min(spec.pca_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    S_train, S_pred = bsa.shape_pair(X_train, X_pred, spec.shape_set)
    scaler = StandardScaler()
    S_train = spec.shape_scale * scaler.fit_transform(S_train)
    S_pred = spec.shape_scale * scaler.transform(S_pred)
    return np.hstack([Z_train, S_train]), np.hstack([Z_pred, S_pred])


def make_weights(
    spec: WeightSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.ones(len(y), dtype=float)
    residual_score = np.zeros(len(y), dtype=float)
    test_score = np.zeros(len(y), dtype=float)

    if spec.residual_top > 0.0 and spec.residual_min_weight < 1.0:
        residual_score = yj_oof_abs_residual(spec, X_train_raw, y, groups)
        weights *= tail_down_weight(
            residual_score,
            top_frac=spec.residual_top,
            min_weight=spec.residual_min_weight,
            gamma=spec.residual_gamma,
        )

    if spec.test_top > 0.0 and spec.test_max_weight > 1.0:
        test_score = test_likeness_score(spec, X_train_raw, X_test_raw)
        weights *= tail_up_weight(
            test_score,
            top_frac=spec.test_top,
            max_weight=spec.test_max_weight,
            gamma=spec.test_gamma,
        )

    if spec.species_balance:
        weights = balance_species_mean(weights, groups)
    if spec.species_mean_cap > 0.0:
        weights = cap_species_mean(weights, groups, spec.species_mean_cap)

    weights = np.clip(weights, 0.50, 1.35)
    weights = weights / np.mean(weights)
    return weights, weight_summary(weights, residual_score, test_score)


def balance_species_mean(weights: np.ndarray, groups: np.ndarray) -> np.ndarray:
    balanced = weights.copy()
    for species in np.unique(groups):
        mask = groups == species
        mean = float(np.mean(balanced[mask]))
        if mean > 0:
            balanced[mask] /= mean
    return balanced


def cap_species_mean(weights: np.ndarray, groups: np.ndarray, cap: float) -> np.ndarray:
    capped = weights.copy()
    low = max(0.0, 2.0 - cap)
    for species in np.unique(groups):
        mask = groups == species
        mean = float(np.mean(capped[mask]))
        if mean > cap:
            capped[mask] *= cap / mean
        elif low > 0.0 and mean < low:
            capped[mask] *= low / max(mean, 1e-12)
    return capped


def yj_oof_abs_residual(spec: WeightSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred_t = np.empty(len(y), dtype=float)
    y_t_all = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    base_spec = replace(spec, residual_top=0.0, test_top=0.0)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        F_train, F_valid = make_model_features(base_spec, X[train_idx], X[valid_idx])
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_train_t = transformer.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
        y_valid_t = transformer.transform(y[valid_idx].reshape(-1, 1)).ravel()
        model = Ridge(alpha=base_spec.alpha)
        model.fit(F_train, y_train_t)
        pred_t[valid_idx] = model.predict(F_valid)
        y_t_all[valid_idx] = y_valid_t
    return np.abs(y_t_all - pred_t)


def test_likeness_score(spec: WeightSpec, X_train_raw: np.ndarray, X_test_raw: np.ndarray) -> np.ndarray:
    F_train, F_test = make_model_features(spec, X_train_raw, X_test_raw)
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)
    k_test = min(spec.test_k, len(F_test))
    nn_test = NearestNeighbors(n_neighbors=k_test)
    nn_test.fit(F_test)
    test_dist = nn_test.kneighbors(F_train, return_distance=True)[0].mean(axis=1)
    if spec.test_mode == "knn":
        return -test_dist
    if spec.test_mode == "ratio":
        k_train = min(max(2, spec.test_k), len(F_train))
        nn_train = NearestNeighbors(n_neighbors=k_train)
        nn_train.fit(F_train)
        train_dist = nn_train.kneighbors(F_train, return_distance=True)[0][:, 1:].mean(axis=1)
        return -(test_dist / np.maximum(train_dist, 1e-12))
    raise ValueError(f"unknown test_mode={spec.test_mode}")


def tail_down_weight(score: np.ndarray, *, top_frac: float, min_weight: float, gamma: float) -> np.ndarray:
    if top_frac <= 0.0:
        return np.ones_like(score, dtype=float)
    q = np.quantile(score, 1.0 - top_frac)
    tail = np.clip((score - q) / max(np.max(score) - q, 1e-12), 0.0, 1.0)
    return 1.0 - (1.0 - min_weight) * np.power(tail, gamma)


def tail_up_weight(score: np.ndarray, *, top_frac: float, max_weight: float, gamma: float) -> np.ndarray:
    if top_frac <= 0.0:
        return np.ones_like(score, dtype=float)
    q = np.quantile(score, 1.0 - top_frac)
    tail = np.clip((score - q) / max(np.max(score) - q, 1e-12), 0.0, 1.0)
    return 1.0 + (max_weight - 1.0) * np.power(tail, gamma)


def apply_affine_from_oof(
    spec: WeightSpec,
    direct_test: np.ndarray,
    direct_oof: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.affine_shrink == 0.0:
        return direct_test, direct_oof
    params = bsa.fit_affine_params(direct_oof, y)
    aug = as_aug_spec(spec)
    pred = bsa.apply_affine(direct_test, params, aug)
    affine_oof = bsa.apply_affine(direct_oof, params, aug)
    return np.clip(pred, 0, None), np.clip(affine_oof, 0, None)


def diagnostics(
    *,
    spec: WeightSpec,
    path: Path,
    pred: np.ndarray,
    direct_test: np.ndarray,
    direct_oof: np.ndarray,
    affine_oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_direct_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    weights: np.ndarray,
    weight_meta: dict[str, float],
    refs: dict[str, np.ndarray],
    nested: bool,
) -> dict[str, object]:
    diff = pred - anchor_test
    direct_diff = direct_test - anchor_test
    affine_rmse = rmse(y, affine_oof)
    direct_rmse = rmse(y, direct_oof)
    anchor_direct_rmse = rmse(y, anchor_direct_oof)
    affine_fold = nl.fold_delta_stats(y, affine_oof, anchor_oof, groups)
    direct_fold = nl.fold_delta_stats(y, direct_oof, anchor_direct_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    if float(np.std(weights)) < 1e-12:
        low_weight = np.zeros_like(weights, dtype=bool)
        high_weight = np.zeros_like(weights, dtype=bool)
    else:
        low_weight = weights < np.quantile(weights, 0.10)
        high_weight = weights > np.quantile(weights, 0.90)

    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "pca_components": spec.pca_components,
        "alpha": spec.alpha,
        "shape_set": spec.shape_set,
        "shape_scale": spec.shape_scale,
        "residual_top": spec.residual_top,
        "residual_min_weight": spec.residual_min_weight,
        "residual_gamma": spec.residual_gamma,
        "test_mode": spec.test_mode,
        "test_k": spec.test_k,
        "test_top": spec.test_top,
        "test_max_weight": spec.test_max_weight,
        "test_gamma": spec.test_gamma,
        "species_balance": spec.species_balance,
        "species_mean_cap": spec.species_mean_cap,
        "affine_shrink": spec.affine_shrink,
        "affine_clip": spec.affine_clip,
        "mean_center": spec.mean_center,
        "nested_evaluated": nested,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": direct_rmse,
        "affine_oof_rmse": affine_rmse,
        "direct_oof_delta_vs_anchor": direct_rmse - anchor_oof_rmse,
        "direct_oof_delta_vs_anchor_direct": direct_rmse - anchor_direct_rmse,
        "affine_oof_delta_vs_anchor": affine_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": direct_fold["improved_count"],
        "direct_worst_fold_delta": direct_fold["worst_delta"],
        "direct_fold_delta_std": direct_fold["delta_std"],
        "affine_improved_fold_count": affine_fold["improved_count"],
        "affine_worst_fold_delta": affine_fold["worst_delta"],
        "affine_fold_delta_std": affine_fold["delta_std"],
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "pred_std": float(np.std(pred)),
        "pred_std_ratio_vs_anchor": float(np.std(pred) / np.std(anchor_test)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": nl.rmse(diff),
        "anchor_direct_diff_rmse": nl.rmse(direct_diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": nl.safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": nl.max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": nl.safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
        "min_sample_weight": float(np.min(weights)),
        "p10_sample_weight": float(np.quantile(weights, 0.10)),
        "mean_sample_weight": float(np.mean(weights)),
        "p90_sample_weight": float(np.quantile(weights, 0.90)),
        "max_sample_weight": float(np.max(weights)),
        "effective_n": float((np.sum(weights) ** 2) / np.sum(weights**2)),
        "weight_y_corr": nl.safe_corr(weights, y),
        "weighted_y_mean_delta": float(np.average(y, weights=weights) - np.mean(y)),
        "low_weight_species_max_rate": species_flag_max_rate(low_weight, groups),
        "high_weight_species_max_rate": species_flag_max_rate(high_weight, groups),
        **weight_meta,
    }
    for ref_name, ref_diff in refs.items():
        row[f"corr_diff_{ref_name}"] = nl.safe_corr(diff, ref_diff)
        row[f"rmse_diff_{ref_name}"] = nl.rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = nl.public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    if diff < 0.03:
        reasons.append("near_anchor")
    if diff < 0.05:
        reasons.append("diff_below_target")
    if diff > 0.35:
        reasons.append("anchor_diff_gt0p35")
    if float(row["anchor_diff_max_abs"]) > 1.25:
        reasons.append("max_diff_gt1p25")
    if float(row["max_abs_species_mean_shift"]) > 0.14:
        reasons.append("species_shift_gt0p14")
    if float(row["max_abs_species_mean_shift"]) > 0.11:
        reasons.append("species_shift_caution")
    if abs(float(row["decile_gap_top_minus_bottom"])) > 0.55:
        reasons.append("decile_gap_gt0p55")
    if int(row["top10_abs_max_species_count"]) > 5:
        reasons.append("top10_species_concentration")
    if not (0.95 <= float(row["range_ratio_vs_anchor"]) <= 1.05):
        reasons.append("range_shift")
    if not (0.95 <= float(row["pred_std_ratio_vs_anchor"]) <= 1.05):
        reasons.append("pred_std_ratio_shift")
    if float(row["direct_oof_delta_vs_anchor_direct"]) > 0.04:
        reasons.append("direct_oof_bad")
    if float(row["direct_worst_fold_delta"]) > 0.20:
        reasons.append("direct_worst_fold_bad")
    if float(row["affine_oof_delta_vs_anchor"]) > 0.04:
        reasons.append("affine_oof_bad")
    if float(row["affine_worst_fold_delta"]) > 0.20:
        reasons.append("worst_fold_bad")
    if int(row["affine_improved_fold_count"]) < 2:
        reasons.append("affine_folds_lt2")
    if float(row["effective_n"]) / 1322.0 < 0.96:
        reasons.append("effective_n_low")
    if abs(float(row["weight_y_corr"])) > 0.25:
        reasons.append("weight_y_corr_gt0p25")
    if abs(float(row["weighted_y_mean_delta"])) > 1.0:
        reasons.append("weighted_y_shift_gt1")
    if float(row["low_weight_species_max_rate"]) > 0.45:
        reasons.append("low_weight_species_concentration")
    if float(row["high_weight_species_max_rate"]) > 0.45:
        reasons.append("high_weight_species_concentration")
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    return reasons


def pick_nested(rows: list[dict[str, object]], limit: int) -> list[str]:
    ok = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(ok, key=rough_rank_key)
    return [str(row["experiment"]) for row in ranked[:limit]]


def rough_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(row.get("submit_gate")) else 10.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.5 * max(float(row["affine_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmSE".lower()]) - 0.12),
        float(row["affine_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    if not bool(row.get("nested_evaluated")):
        penalty += 3.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.8 * max(float(row["affine_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.12),
        float(row["affine_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def failed_row(spec: WeightSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
        "affine_oof_delta_vs_anchor": float("inf"),
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
    }


def as_aug_spec(spec: WeightSpec) -> bsa.AugSpec:
    return bsa.AugSpec(
        name=spec.name,
        pca_components=spec.pca_components,
        alpha=spec.alpha,
        shape_set=spec.shape_set,
        shape_scale=spec.shape_scale,
        affine_shrink=spec.affine_shrink,
        affine_clip=spec.affine_clip,
        mean_center=spec.mean_center,
    )


def weight_summary(weights: np.ndarray, residual_score: np.ndarray, test_score: np.ndarray) -> dict[str, float]:
    return {
        "weight_residual_corr": nl.safe_corr(weights, residual_score),
        "weight_testlike_corr": nl.safe_corr(weights, test_score),
        "residual_score_p90": float(np.quantile(residual_score, 0.90)),
        "testlike_score_p90": float(np.quantile(test_score, 0.90)),
    }


def species_flag_max_rate(flag: np.ndarray, groups: np.ndarray) -> float:
    rates = []
    for species in np.unique(groups):
        mask = groups == species
        rates.append(float(np.mean(flag[mask])))
    return max(rates) if rates else 0.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor_direct']:.4f} "
        f"affine={row['affine_oof_delta_vs_anchor']:.4f} "
        f"fold={row['affine_improved_fold_count']}/5 reasons={row['reject_reasons']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
