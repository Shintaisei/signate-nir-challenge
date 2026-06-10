#!/usr/bin/env python3
"""Base YJ Ridge with small spectral-shape feature augmentation."""

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
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_anchor_rebuild_search as ar
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_base_shape_aug_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class AugSpec:
    name: str
    pca_components: int
    alpha: float
    shape_set: str
    shape_scale: float
    affine_shrink: float
    affine_clip: float = 0.40
    mean_center: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--nested-top", type=int, default=40)
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    refs = nl.load_reference_diffs(anchor_test)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF", flush=True)
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    base_direct_oof = op.make_base_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)
    print(f"base_direct_oof_rmse={rmse(data['y'], base_direct_oof):.6f}", flush=True)
    ar.assert_anchor_reproduction(data, anchor_test)

    base_specs = build_base_specs()
    if args.max_specs is not None:
        base_specs = base_specs[: args.max_specs]
    print(f"base_specs={len(base_specs)}", flush=True)

    rows: list[dict[str, object]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, base in enumerate(base_specs, start=1):
        print(f"[{i}/{len(base_specs)}] {base.name}", flush=True)
        try:
            direct_oof = np.clip(make_oof(base, data["X_train"], data["y"], data["groups"]), 0, None)
            direct_test = np.clip(fit_predict_spec(base, data["X_train"], data["y"], data["X_test"]), 0, None)
        except Exception as exc:
            row = failed_row(base, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue
        cache[base.name] = (direct_oof, direct_test)

        for shrink in [0.0, 0.04, 0.06, 0.08, 0.10, 0.12]:
            spec = replace(base, name=f"{base.name}_affs{nl.tag(shrink)}", affine_shrink=shrink)
            pred, affine_oof = apply_variant(
                spec,
                direct_oof,
                direct_test,
                data["X_train"],
                data["y"],
                data["groups"],
                nested=False,
            )
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
                base_direct_oof=base_direct_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                y=data["y"],
                groups=data["groups"],
                test_species=test_species,
                refs=refs,
                nested=False,
            )
            rows.append(row)
            print_one(row)

    nested_names = pick_nested(rows, args.nested_top)
    print(f"nested variants={len(nested_names)}", flush=True)
    base_by_name = {base.name: base for base in base_specs}
    row_by_name = {str(row["experiment"]): row for row in rows}
    nested_cache: dict[tuple[str, float, float, bool], np.ndarray] = {}
    for name in nested_names:
        row = row_by_name[name]
        base_name = str(row["base_experiment"])
        base = base_by_name[base_name]
        spec = replace(
            base,
            name=name,
            affine_shrink=float(row["affine_shrink"]),
            affine_clip=float(row["affine_clip"]),
            mean_center=bool(row["mean_center"]),
        )
        direct_oof, direct_test = cache[base_name]
        cache_key = (base_name, spec.affine_shrink, spec.affine_clip, spec.mean_center)
        if cache_key not in nested_cache:
            nested_cache[cache_key] = make_nested_affine_oof(
                spec, data["X_train"], data["y"], data["groups"]
            )
        pred, _ = apply_variant(
            spec,
            direct_oof,
            direct_test,
            data["X_train"],
            data["y"],
            data["groups"],
            nested=False,
        )
        path = candidate_dir / f"{spec.name}_nested_ranked.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row.update(
            diagnostics(
                spec=spec,
                path=path,
                pred=pred,
                direct_test=direct_test,
                direct_oof=direct_oof,
                affine_oof=nested_cache[cache_key],
                anchor_test=anchor_test,
                anchor_oof=anchor_oof,
                base_direct_oof=base_direct_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                y=data["y"],
                groups=data["groups"],
                test_species=test_species,
                refs=refs,
                nested=True,
            )
        )
        print("nested", end=" ")
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "base_shape_aug_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "base_shape_aug_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop base shape-aug candidates:")
    for row in rows_sorted[:80]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} direct={row['direct_oof_delta_vs_anchor']:.4f} "
            f"affine={row['affine_oof_delta_vs_anchor']:.4f} fold={row['affine_improved_fold_count']}/5 "
            f"nested={row['nested_evaluated']}"
        )
    print(f"saved base shape-aug diagnostics: {out_dir}")


def build_base_specs() -> list[AugSpec]:
    specs: list[AugSpec] = []
    for shape_set in ["shape4", "shape6", "shape10", "shape14"]:
        for shape_scale in [0.25, 0.50, 1.00]:
            for pca_components in [18, 20, 24]:
                for alpha in [3500.0, 4000.0, 5000.0, 6500.0]:
                    specs.append(
                        AugSpec(
                            name=(
                                f"nir_baseaug_{shape_set}_sc{nl.tag(shape_scale)}_"
                                f"p{pca_components}_a{int(alpha)}"
                            ),
                            pca_components=pca_components,
                            alpha=alpha,
                            shape_set=shape_set,
                            shape_scale=shape_scale,
                            affine_shrink=0.0,
                        )
                    )
    return specs


def make_oof(spec: AugSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
    return pred


def make_nested_affine_oof(spec: AugSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    if spec.affine_shrink == 0.0:
        return make_oof(spec, X, y, groups)
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        valid_direct = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
        inner_oof = make_oof(spec, X[train_idx], y[train_idx], groups[train_idx])
        params = fit_affine_params(inner_oof, y[train_idx])
        pred[valid_idx] = apply_affine(valid_direct, params, spec)
    return np.clip(pred, 0, None)


def fit_predict_spec(spec: AugSpec, X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, "sg9_snv")
    pca = PCA(n_components=min(spec.pca_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    S_train, S_pred = shape_pair(X_train, X_pred, spec.shape_set)
    scaler = StandardScaler()
    S_train = spec.shape_scale * scaler.fit_transform(S_train)
    S_pred = spec.shape_scale * scaler.transform(S_pred)
    F_train = np.hstack([Z_train, S_train])
    F_pred = np.hstack([Z_pred, S_pred])
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=spec.alpha)
    model.fit(F_train, yt)
    pred_t = model.predict(F_pred)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def shape_pair(X_train: np.ndarray, X_pred: np.ndarray, shape_set: str) -> tuple[np.ndarray, np.ndarray]:
    return shape_features(X_train, shape_set), shape_features(X_pred, shape_set)


def shape_features(X: np.ndarray, shape_set: str) -> np.ndarray:
    d1 = np.diff(X, axis=1)
    d2 = np.diff(X, n=2, axis=1)
    feats: list[np.ndarray] = [
        np.mean(d1, axis=1),
        np.std(d1, axis=1),
        np.max(np.abs(d1), axis=1),
        np.mean(np.abs(d2), axis=1),
    ]
    if shape_set in {"shape6", "shape10", "shape14"}:
        feats.extend(
            [
                np.std(d2, axis=1),
                np.max(np.abs(d2), axis=1),
            ]
        )
    if shape_set in {"shape10", "shape14"}:
        for idx in np.array_split(np.arange(X.shape[1]), 4):
            feats.append(np.mean(X[:, idx], axis=1))
    if shape_set == "shape14":
        coeff = dct(X, type=2, norm="ortho", axis=1)
        for j in range(4):
            feats.append(coeff[:, j])
    return np.vstack(feats).T


def apply_variant(
    spec: AugSpec,
    direct_oof: np.ndarray,
    direct_test: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    nested: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.affine_shrink == 0.0:
        return direct_test, direct_oof
    params = fit_affine_params(direct_oof, y)
    pred = apply_affine(direct_test, params, spec)
    if nested:
        affine_oof = make_nested_affine_oof(spec, X, y, groups)
    else:
        affine_oof = apply_affine(direct_oof, params, spec)
    return np.clip(pred, 0, None), np.clip(affine_oof, 0, None)


def fit_affine_params(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, y, deg=1)
    return float(intercept), float(slope)


def apply_affine(pred: np.ndarray, params: tuple[float, float], spec: AugSpec) -> np.ndarray:
    intercept, slope = params
    delta = spec.affine_shrink * (intercept + slope * pred - pred)
    delta = np.clip(delta, -spec.affine_clip, spec.affine_clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return pred + delta


def diagnostics(
    *,
    spec: AugSpec,
    path: Path,
    pred: np.ndarray,
    direct_test: np.ndarray,
    direct_oof: np.ndarray,
    affine_oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    base_direct_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
    nested: bool,
) -> dict[str, object]:
    diff = pred - anchor_test
    direct_diff = direct_test - anchor_test
    direct_rmse = rmse(y, direct_oof)
    base_direct_rmse = rmse(y, base_direct_oof)
    affine_rmse = rmse(y, affine_oof)
    direct_fold = nl.fold_delta_stats(y, direct_oof, anchor_oof, groups)
    direct_base_fold = nl.fold_delta_stats(y, direct_oof, base_direct_oof, groups)
    affine_fold = nl.fold_delta_stats(y, affine_oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "base_experiment": base_name(spec),
        "submission_path": str(path),
        "pca_components": spec.pca_components,
        "alpha": spec.alpha,
        "shape_set": spec.shape_set,
        "shape_scale": spec.shape_scale,
        "affine_shrink": spec.affine_shrink,
        "affine_clip": spec.affine_clip,
        "mean_center": spec.mean_center,
        "nested_evaluated": nested,
        "anchor_oof_rmse": anchor_oof_rmse,
        "base_direct_oof_rmse": base_direct_rmse,
        "direct_oof_rmse": direct_rmse,
        "affine_oof_rmse": affine_rmse,
        "direct_oof_delta_vs_anchor": direct_rmse - anchor_oof_rmse,
        "direct_oof_delta_vs_base_direct": direct_rmse - base_direct_rmse,
        "affine_oof_delta_vs_anchor": affine_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": direct_fold["improved_count"],
        "direct_worst_fold_delta": direct_fold["worst_delta"],
        "direct_fold_delta_std": direct_fold["delta_std"],
        "direct_vs_base_improved_fold_count": direct_base_fold["improved_count"],
        "direct_vs_base_worst_fold_delta": direct_base_fold["worst_delta"],
        "direct_vs_base_fold_delta_std": direct_base_fold["delta_std"],
        "affine_improved_fold_count": affine_fold["improved_count"],
        "affine_worst_fold_delta": affine_fold["worst_delta"],
        "affine_fold_delta_std": affine_fold["delta_std"],
        "seed_pred_std_mean": 0.0,
        "seed_pred_std_max": 0.0,
        "seed_pred_rmse_mean": 0.0,
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
    if diff > 0.30:
        reasons.append("anchor_diff_gt0p30")
    if float(row["anchor_diff_max_abs"]) > 1.00:
        reasons.append("max_diff_gt1")
    if float(row["max_abs_species_mean_shift"]) > 0.12:
        reasons.append("species_shift_gt0p12")
    if float(row["max_abs_species_mean_shift"]) > 0.10:
        reasons.append("species_shift_caution")
    if float(row["corr_diff_bad_alpha3000"]) > 0.40:
        reasons.append("bad_alpha3000_corr_gt0p40")
    if abs(float(row["decile_gap_top_minus_bottom"])) > 0.45:
        reasons.append("decile_gap_gt0p45")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if not (0.95 <= float(row["range_ratio_vs_anchor"]) <= 1.05):
        reasons.append("range_shift")
    if not (0.95 <= float(row["pred_std_ratio_vs_anchor"]) <= 1.05):
        reasons.append("pred_std_ratio_shift")
    if float(row["direct_oof_delta_vs_base_direct"]) > 0.03:
        reasons.append("direct_oof_bad")
    if float(row["direct_vs_base_worst_fold_delta"]) > 0.20:
        reasons.append("direct_worst_fold_bad")
    if int(row["direct_vs_base_improved_fold_count"]) < 2:
        reasons.append("direct_folds_lt2")
    if float(row["affine_oof_delta_vs_anchor"]) > 0.03:
        reasons.append("affine_oof_bad")
    if float(row["affine_worst_fold_delta"]) > 0.15:
        reasons.append("worst_fold_bad")
    if int(row["affine_improved_fold_count"]) < 3:
        reasons.append("affine_folds_lt3")
    if float(row["affine_oof_delta_vs_anchor"]) < -0.01 and float(row["direct_oof_delta_vs_base_direct"]) > 0.03:
        reasons.append("affine_only_improvement")
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if abs(float(row["corr_diff_weighted_golden"])) > 0.85:
        reasons.append("weighted_golden_like")
    if abs(float(row["corr_diff_operator_residual"])) > 0.85:
        reasons.append("operator_residual_like")
    return reasons


def pick_nested(rows: list[dict[str, object]], limit: int) -> list[str]:
    ok = [row for row in rows if row.get("status") == "ok" and float(row.get("affine_shrink", 0.0)) > 0.0]
    ranked = sorted(ok, key=rough_rank_key)
    return [str(row["experiment"]) for row in ranked[:limit]]


def rough_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(row.get("submit_gate")) else 10.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.6 * max(float(row["affine_oof_delta_vs_anchor"]), 0.0)
        + abs(float(row["anchor_diff_rmse"]) - 0.12) * 0.5,
        float(row["affine_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    if not bool(row.get("nested_evaluated")) and float(row.get("affine_shrink", 0.0)) > 0.0:
        penalty += 3.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.8 * max(float(row["affine_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.12),
        float(row["affine_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def failed_row(spec: AugSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "base_experiment": base_name(spec),
        "shape_set": spec.shape_set,
        "shape_scale": spec.shape_scale,
        "pca_components": spec.pca_components,
        "alpha": spec.alpha,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
        "affine_oof_delta_vs_anchor": float("inf"),
        "direct_oof_delta_vs_anchor": float("inf"),
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
    }


def base_name(spec: AugSpec) -> str:
    return (
        f"nir_baseaug_{spec.shape_set}_sc{nl.tag(spec.shape_scale)}_"
        f"p{spec.pca_components}_a{int(spec.alpha)}"
    )


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    if b is None:
        values = np.asarray(a, dtype=float)
        return float(math.sqrt(np.mean(values**2)))
    return float(math.sqrt(mean_squared_error(a, b)))


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} badcorr={row['corr_diff_bad_alpha3000']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor']:.4f} affine={row['affine_oof_delta_vs_anchor']:.4f} "
        f"fold={row['affine_improved_fold_count']}/5 reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
