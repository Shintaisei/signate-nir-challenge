#!/usr/bin/env python3
"""Focused adversarial-validation golden-subset residual search.

This script narrows the broad search around the best local family:
train/test-likeness weighting plus a small residual correction to the protected
current anchor. The domain classifier is target-free and fold-local for OOF.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nir_broad_signal_search as broad
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_adv_golden_focused_search"


@dataclass(frozen=True)
class FocusSpec:
    name: str
    preprocess: str
    model: str
    weight_mode: str
    model_components: int
    domain_components: int
    weight_low: float
    weight_high: float
    alpha: float = 1200.0
    seed: int = 42


def main() -> None:
    data = op.load_data()
    anchor = pd.read_csv(broad.CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    pls_primary = pd.read_csv(broad.PLS_PRIMARY, header=None)
    if not np.array_equal(data["test_ids"], pls_primary[0].to_numpy()):
        raise ValueError("PLS primary order mismatch")
    pls_primary_delta = pls_primary[1].to_numpy(float) - anchor_test

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    anchor_oof_rmse = broad.rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    rows: list[dict[str, object]] = []
    for spec in build_specs():
        print(f"branch {spec.name}", flush=True)
        branch_oof, oof_meta = make_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test, test_meta = fit_adv_weighted(spec, data["X_train"], data["y"], data["X_test"])
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = op.fit_beta(signal_oof, residual)
        signal_corr = op.safe_corr(signal_oof, residual)
        branch_oof_rmse = broad.rmse(data["y"], branch_oof)

        broad_spec = broad.BroadSpec(
            name=spec.name,
            family="adv_golden_focused",
            feature_set="domain_pca",
            model=spec.model,
            n_components=spec.model_components,
            alpha=spec.alpha,
            preprocess=spec.preprocess,
            weight_mode=spec.weight_mode,
            seed=spec.seed,
        )
        for distill in build_distills():
            correction_oof = op.make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = op.make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_advgold_{spec.name}_b{op.tag(beta)}_"
                f"s{op.tag(distill.shrink)}_c{op.tag(distill.clip)}_mc1"
            )
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = broad.diagnostics(
                name=name,
                spec=broad_spec,
                distill=distill,
                pred=pred,
                anchor=anchor_test,
                y=data["y"],
                corrected_oof=corrected_oof,
                branch_oof=branch_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                branch_oof_rmse=branch_oof_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
                oof_meta=oof_meta,
                test_meta=test_meta,
                pls_primary_delta=pls_primary_delta,
            )
            row["domain_components"] = spec.domain_components
            row["weight_low"] = spec.weight_low
            row["weight_high"] = spec.weight_high
            rows.append(row)
            broad.print_one(row)

    rows_sorted = sorted(rows, key=broad.rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "adv_golden_focused_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "adv_golden_focused_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:40]:
        print(
            f"{row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f} pls_corr={row['corr_with_pls_primary']:.4f}"
        )
    print(f"saved focused diagnostics: {out_dir}")


def build_specs() -> list[FocusSpec]:
    specs: list[FocusSpec] = []
    weight_ranges = [(0.80, 1.25), (0.65, 1.65), (0.50, 2.00)]
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for weight_mode in ["rf", "logistic"]:
            for domain_components in [6, 10, 14]:
                for weight_low, weight_high in weight_ranges:
                    for comps in [4, 5, 6, 7]:
                        specs.append(
                            FocusSpec(
                                name=(
                                    f"pls_{preprocess}_{weight_mode}_dc{domain_components}_"
                                    f"w{op.tag(weight_low)}to{op.tag(weight_high)}_c{comps}"
                                ),
                                preprocess=preprocess,
                                model="adv_pls_raw",
                                weight_mode=weight_mode,
                                model_components=comps,
                                domain_components=domain_components,
                                weight_low=weight_low,
                                weight_high=weight_high,
                            )
                        )
                    for comps in [5, 6]:
                        specs.append(
                            FocusSpec(
                                name=(
                                    f"huber_{preprocess}_{weight_mode}_dc{domain_components}_"
                                    f"w{op.tag(weight_low)}to{op.tag(weight_high)}_c{comps}"
                                ),
                                preprocess=preprocess,
                                model="adv_huber_raw",
                                weight_mode=weight_mode,
                                model_components=comps,
                                domain_components=domain_components,
                                weight_low=weight_low,
                                weight_high=weight_high,
                            )
                        )
    return specs


def build_distills() -> list[op.DistillSpec]:
    return [
        op.DistillSpec(0.012, 0.06),
        op.DistillSpec(0.015, 0.08),
        op.DistillSpec(0.018, 0.08),
        op.DistillSpec(0.020, 0.08),
        op.DistillSpec(0.020, 0.10),
    ]


def make_oof(spec: FocusSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.empty(len(y), dtype=float)
    meta_rows: list[dict[str, float]] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_pred, meta = fit_adv_weighted(spec, X[train_idx], y[train_idx], X[valid_idx])
        pred[valid_idx] = fold_pred
        meta_rows.append(meta)
    return np.clip(pred, 0, None), broad.summarize_meta(meta_rows)


def fit_adv_weighted(
    spec: FocusSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(np.vstack([X_train, X_pred]))
    comps = min(spec.domain_components, X_all.shape[0] - 1, X_all.shape[1])
    pca = PCA(n_components=max(1, comps), random_state=spec.seed)
    Z_all = pca.fit_transform(X_all)
    Z_train = Z_all[: len(X_train)]
    labels = np.r_[np.zeros(len(X_train)), np.ones(len(X_pred))]
    if spec.weight_mode == "logistic":
        clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=spec.seed)
    elif spec.weight_mode == "rf":
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=3,
            min_samples_leaf=18,
            class_weight="balanced",
            random_state=spec.seed,
        )
    else:
        raise ValueError(spec.weight_mode)
    clf.fit(Z_all, labels)
    domain_score_train = clf.predict_proba(Z_train)[:, 1]
    domain_score_all = clf.predict_proba(Z_all)[:, 1]
    weights = broad.bounded_rank_weights(domain_score_train, low=spec.weight_low, high=spec.weight_high)
    X_model_train, X_model_pred = broad.make_adv_model_features(X_train, X_pred, spec.model_components)
    pred = broad.fit_predict_model(
        spec.model,
        X_model_train,
        y,
        X_model_pred,
        spec.model_components,
        spec.alpha,
        weights,
    )
    return pred, {
        "domain_auc": broad.safe_auc(labels, domain_score_all),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_std": float(np.std(weights)),
    }


if __name__ == "__main__":
    main()
