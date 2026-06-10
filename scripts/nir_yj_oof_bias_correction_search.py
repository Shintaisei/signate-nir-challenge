#!/usr/bin/env python3
"""Small OOF residual corrections around the true YJ Ridge3500 anchor."""

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
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PowerTransformer


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_yj_oof_bias"
CURRENT_BEST = SUBMISSION_DIR / "nir_ms_target2_yeojohnson_pca20_ridge3500.csv"


@dataclass(frozen=True)
class CorrectionSpec:
    name: str
    family: str
    shrink: float
    mean_center: bool
    clip: float | None = None
    bins: int | None = None
    statistic: str = "mean"


@dataclass(frozen=True)
class CorrectionParams:
    spec: CorrectionSpec
    mean_residual: float = 0.0
    affine_intercept: float = 0.0
    affine_slope: float = 1.0
    bin_centers: np.ndarray | None = None
    bin_values: np.ndarray | None = None


@dataclass(frozen=True)
class CvContext:
    fold_id: int
    train_size: int
    valid_size: int
    train_oof_pred: np.ndarray
    y_train: np.ndarray
    valid_base_pred: np.ndarray
    y_valid: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-top", type=int, default=24)
    args = parser.parse_args()

    data = load_data()
    current = pd.read_csv(CURRENT_BEST, header=None)
    if not np.array_equal(data["test_ids"], current[0].to_numpy()):
        raise ValueError("current-best sample order mismatch")

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    anchor = current[1].to_numpy(float)
    base_pred = fit_base_full(data)
    base_diff = base_pred - anchor
    base_repro_rmse = math.sqrt(float(np.mean(base_diff**2)))
    base_repro_max_abs = float(np.max(np.abs(base_diff)))
    print(f"base reproduction: rmse={base_repro_rmse:.12f} max_abs={base_repro_max_abs:.12f}")
    if base_repro_max_abs > 1e-8:
        raise ValueError("base prediction does not reproduce current best")

    full_oof = make_oof_base_predictions(data["X_train"], data["y"], data["groups"])
    specs = build_specs()
    rows: list[dict[str, object]] = []
    row_by_name: dict[str, dict[str, object]] = {}
    for spec in specs:
        params = fit_correction(spec, full_oof, data["y"])
        pred = np.clip(apply_correction(base_pred, params), 0, None)
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(spec, pred, anchor, path)
        row.update(empty_cv())
        row["base_repro_rmse"] = base_repro_rmse
        row["base_repro_max_abs"] = base_repro_max_abs
        row["cv_evaluated"] = False
        row["prediction_set_mean_centering"] = spec.mean_center
        rows.append(row)
        row_by_name[spec.name] = row
        print_diag(row)

    group_contexts = build_cv_contexts(data, exclude_species=None)
    ex15_contexts = build_cv_contexts(data, exclude_species=15)
    base_cv = {
        **score_contexts(None, group_contexts, prefix="cv_group_species"),
        **score_contexts(None, ex15_contexts, prefix="cv_exclude15"),
    }
    picked_names = pick_cv_candidates(rows, specs, args.cv_top)
    print(f"\nCV candidates: {len(picked_names)}")
    for name in picked_names:
        spec = next(item for item in specs if item.name == name)
        row = row_by_name[name]
        row.update(score_contexts(spec, group_contexts, prefix="cv_group_species"))
        row.update(score_contexts(spec, ex15_contexts, prefix="cv_exclude15"))
        row["cv_group_delta_vs_base3500"] = float(row["cv_group_species_rmse"]) - base_cv["cv_group_species_rmse"]
        row["cv_exclude15_delta_vs_base3500"] = float(row["cv_exclude15_rmse"]) - base_cv["cv_exclude15_rmse"]
        row["cv_evaluated"] = True
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "oof_bias_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "oof_bias_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nBase CV:")
    print(
        f"group={base_cv['cv_group_species_rmse']:.6f}+/-{base_cv['cv_group_species_std']:.6f} "
        f"ex15={base_cv['cv_exclude15_rmse']:.6f}+/-{base_cv['cv_exclude15_std']:.6f}"
    )
    print("\nTop gated candidates:")
    for row in rows_sorted[:20]:
        print(
            f"{row['experiment']}: {row['family']} rmse={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} mean={row['anchor_diff_mean']:.4f} "
            f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
            f"spmax={row['max_abs_species_mean_shift']:.4f} "
            f"gcv={row['cv_group_delta_vs_base3500']:.4f} "
            f"ex15={row['cv_exclude15_delta_vs_base3500']:.4f}"
        )
    print(f"saved OOF-bias diagnostics: {out_dir}")
    write_cv_context_log(out_dir, "group_species_contexts.csv", group_contexts)
    write_cv_context_log(out_dir, "exclude15_contexts.csv", ex15_contexts)


def build_specs() -> list[CorrectionSpec]:
    specs: list[CorrectionSpec] = []
    for mean_center in [False, True]:
        for shrink in [0.05, 0.10, 0.15, 0.20]:
            specs.append(
                CorrectionSpec(
                    name=f"nir_yj_oof_global_s{tag(shrink)}_mc{int(mean_center)}",
                    family="global_mean",
                    shrink=shrink,
                    mean_center=mean_center,
                    clip=0.3,
                )
            )
        for shrink in [0.03, 0.05, 0.08, 0.10, 0.15]:
            specs.append(
                CorrectionSpec(
                    name=f"nir_yj_oof_affine_s{tag(shrink)}_mc{int(mean_center)}",
                    family="affine",
                    shrink=shrink,
                    mean_center=mean_center,
                    clip=0.4,
                )
            )
        for bins in [5, 8, 10]:
            for statistic in ["mean", "median"]:
                for shrink in [0.05, 0.08, 0.10, 0.15]:
                    for clip in [0.2, 0.3, 0.4]:
                        specs.append(
                            CorrectionSpec(
                                name=(
                                    f"nir_yj_oof_bin{bins}_{statistic}_s{tag(shrink)}_"
                                    f"c{tag(clip)}_mc{int(mean_center)}"
                                ),
                                family="bin_residual",
                                shrink=shrink,
                                mean_center=mean_center,
                                clip=clip,
                                bins=bins,
                                statistic=statistic,
                            )
                        )
    return specs


def empty_cv() -> dict[str, object]:
    return {
        "cv_group_species_rmse": None,
        "cv_group_species_std": None,
        "cv_exclude15_rmse": None,
        "cv_exclude15_std": None,
        "cv_group_delta_vs_base3500": None,
        "cv_exclude15_delta_vs_base3500": None,
    }


def pick_cv_candidates(
    rows: list[dict[str, object]],
    specs: list[CorrectionSpec],
    cv_top: int,
) -> list[str]:
    spec_by_name = {spec.name: spec for spec in specs}
    sorted_rows = sorted(rows, key=pre_cv_rank_key)
    picked: list[str] = []

    def add(name: str) -> None:
        if name not in picked:
            picked.append(name)

    for row in sorted_rows:
        if len(picked) >= cv_top:
            break
        if float(row["anchor_diff_rmse"]) >= 0.05:
            add(str(row["experiment"]))

    for family in ["global_mean", "affine", "bin_residual"]:
        for mean_center in [False, True]:
            added = 0
            for row in sorted_rows:
                spec = spec_by_name[str(row["experiment"])]
                if spec.family != family or spec.mean_center != mean_center:
                    continue
                if float(row["anchor_diff_rmse"]) < 0.05:
                    continue
                add(spec.name)
                added += 1
                if added >= 2:
                    break

    # If a mean-centered spec is evaluated, compare its non-centered pair where available.
    for name in list(picked):
        spec = spec_by_name[name]
        if not spec.mean_center:
            continue
        pair_name = name[:-1] + "0"
        if pair_name in spec_by_name:
            add(pair_name)

    return picked


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def load_data() -> dict[str, np.ndarray]:
    train = pd.read_csv(TRAIN_PATH, encoding="cp932")
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample = pd.read_csv(SAMPLE_PATH, header=None)
    feature_cols = [c for c in train.columns if is_float_like(c)]
    target_cols = [
        c
        for c in train.columns
        if c not in feature_cols
        and c not in {"sample number", "species number"}
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    if len(target_cols) != 1:
        raise ValueError(f"could not infer target column, candidates={target_cols}")
    ids = test["sample number"].to_numpy()
    if not np.array_equal(ids, sample[0].to_numpy()):
        raise ValueError("sample_submit order mismatch")
    return {
        "X_train": train[feature_cols].to_numpy(float),
        "X_test": test[feature_cols].to_numpy(float),
        "y": train[target_cols[0]].to_numpy(float),
        "groups": train["species number"].to_numpy(),
        "test_ids": ids,
        "test_species": test["species number"].to_numpy(),
    }


def is_float_like(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def make_base_features(X: np.ndarray) -> np.ndarray:
    return snv(savgol_filter(X, 9, 2, axis=1, mode="interp"))


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / np.where(std == 0, 1.0, std)


def fit_base_full(data: dict[str, np.ndarray]) -> np.ndarray:
    X_train = make_base_features(data["X_train"])
    X_test = make_base_features(data["X_test"])
    return fit_predict_yj_ridge(X_train, data["y"], X_test)


def make_oof_base_predictions(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X_raw, y, groups):
        X_train = make_base_features(X_raw[train_idx])
        X_valid = make_base_features(X_raw[valid_idx])
        pred[valid_idx] = fit_predict_yj_ridge(X_train, y[train_idx], X_valid)
    return np.clip(pred, 0, None)


def build_cv_contexts(data: dict[str, np.ndarray], *, exclude_species: int | None) -> list[CvContext]:
    X_raw = data["X_train"]
    y = data["y"]
    groups = data["groups"]
    if exclude_species is None:
        mask = np.ones(len(y), dtype=bool)
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        splits = splitter.split(X_raw[mask], y[mask], groups[mask])
    else:
        mask = groups != exclude_species
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(X_raw[mask], y[mask])

    X_m = X_raw[mask]
    y_m = y[mask]
    groups_m = groups[mask]
    contexts: list[CvContext] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        X_train_raw = X_m[train_idx]
        y_train = y_m[train_idx]
        groups_train = groups_m[train_idx]
        X_train = make_base_features(X_train_raw)
        X_valid = make_base_features(X_m[valid_idx])
        pred_valid = np.clip(fit_predict_yj_ridge(X_train, y_train, X_valid), 0, None)
        inner_oof = make_oof_base_predictions(X_train_raw, y_train, groups_train)
        contexts.append(
            CvContext(
                fold_id=fold_id,
                train_size=len(train_idx),
                valid_size=len(valid_idx),
                train_oof_pred=inner_oof,
                y_train=y_train,
                valid_base_pred=pred_valid,
                y_valid=y_m[valid_idx],
            )
        )
    return contexts


def score_contexts(
    spec: CorrectionSpec | None,
    contexts: list[CvContext],
    *,
    prefix: str,
) -> dict[str, float]:
    rmses: list[float] = []
    for context in contexts:
        pred_valid = context.valid_base_pred
        if spec is not None:
            params = fit_correction(spec, context.train_oof_pred, context.y_train)
            pred_valid = np.clip(apply_correction(pred_valid, params), 0, None)
        rmses.append(math.sqrt(mean_squared_error(context.y_valid, pred_valid)))
    return {f"{prefix}_rmse": float(np.mean(rmses)), f"{prefix}_std": float(np.std(rmses))}


def write_cv_context_log(out_dir: Path, filename: str, contexts: list[CvContext]) -> None:
    path = out_dir / filename
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold_id", "train_size", "valid_size"])
        writer.writeheader()
        for context in contexts:
            writer.writerow(
                {
                    "fold_id": context.fold_id,
                    "train_size": context.train_size,
                    "valid_size": context.valid_size,
                }
            )


def fit_predict_yj_ridge(X_train: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=20, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=3500.0)
    model.fit(Z_train, yt)
    pred_t = model.predict(Z_test)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def fit_correction(spec: CorrectionSpec, pred: np.ndarray, y: np.ndarray) -> CorrectionParams:
    residual = y - pred
    if spec.family == "global_mean":
        return CorrectionParams(spec=spec, mean_residual=float(np.mean(residual)))
    if spec.family == "affine":
        slope, intercept = np.polyfit(pred, y, deg=1)
        return CorrectionParams(spec=spec, affine_intercept=float(intercept), affine_slope=float(slope))
    if spec.family == "bin_residual":
        if spec.bins is None:
            raise ValueError("bin_residual requires bins")
        quantiles = np.linspace(0, 1, spec.bins + 1)
        edges = np.unique(np.quantile(pred, quantiles))
        centers: list[float] = []
        values: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi == edges[-1]:
                mask = (pred >= lo) & (pred <= hi)
            else:
                mask = (pred >= lo) & (pred < hi)
            if not np.any(mask):
                continue
            centers.append(float(np.mean(pred[mask])))
            if spec.statistic == "median":
                values.append(float(np.median(residual[mask])))
            else:
                values.append(float(np.mean(residual[mask])))
        if len(centers) < 2:
            return CorrectionParams(spec=spec, mean_residual=float(np.mean(residual)))
        return CorrectionParams(spec=spec, bin_centers=np.asarray(centers), bin_values=np.asarray(values))
    raise ValueError(f"unknown correction family: {spec.family}")


def apply_correction(pred: np.ndarray, params: CorrectionParams) -> np.ndarray:
    spec = params.spec
    if spec.family == "global_mean":
        delta = np.full_like(pred, params.mean_residual)
    elif spec.family == "affine":
        delta = params.affine_intercept + params.affine_slope * pred - pred
    elif spec.family == "bin_residual":
        if params.bin_centers is None or params.bin_values is None:
            delta = np.full_like(pred, params.mean_residual)
        else:
            delta = np.interp(pred, params.bin_centers, params.bin_values)
    else:
        raise ValueError(f"unknown correction family: {spec.family}")
    delta = spec.shrink * delta
    if spec.clip is not None:
        delta = np.clip(delta, -spec.clip, spec.clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return pred + delta


def diagnostics(spec: CorrectionSpec, pred: np.ndarray, anchor: np.ndarray, path: Path) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    return {
        "experiment": spec.name,
        "family": spec.family,
        "submission_path": str(path),
        "shrink": spec.shrink,
        "mean_center": spec.mean_center,
        "clip": spec.clip,
        "bins": spec.bins,
        "statistic": spec.statistic,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int((pred < 0).sum()),
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": float(np.corrcoef(anchor, pred)[0, 1]),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor) - np.min(anchor))),
        "max_abs_species_mean_shift": float(max_abs_species_shift(diff)),
    }


def max_abs_species_shift(diff: np.ndarray) -> float:
    # Test set species are ordered in blocks in this competition's sample file.
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    species = test["species number"].to_numpy()
    shifts = [abs(float(np.mean(diff[species == value]))) for value in np.unique(species)]
    return max(shifts) if shifts else 0.0


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    rmse = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    mean = abs(float(row["anchor_diff_mean"]))
    species = float(row["max_abs_species_mean_shift"])
    corr = float(row["anchor_corr"])
    if not row.get("cv_evaluated"):
        return (100.0, max_abs, species, -corr)
    group_delta = float(row["cv_group_delta_vs_base3500"])
    ex15_delta = float(row["cv_exclude15_delta_vs_base3500"])
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    hard_penalty = 0.0
    if rmse < 0.05 or rmse > 0.15:
        hard_penalty += 5.0
    if max_abs >= 0.5 or mean >= 0.02 or species > 0.10 or corr < 0.99999:
        hard_penalty += 5.0
    if group_delta > 0.02 or ex15_delta > 0.02:
        hard_penalty += 3.0
    if low > 0.15 or top > 0.15:
        hard_penalty += 2.0
    cv_penalty = max(group_delta, 0.0) + max(ex15_delta, 0.0)
    movement = abs(rmse - 0.09)
    return (hard_penalty + cv_penalty + movement + 0.25 * species + mean, max_abs, species, -corr)


def pre_cv_rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    rmse = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    mean = abs(float(row["anchor_diff_mean"]))
    species = float(row["max_abs_species_mean_shift"])
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    corr = float(row["anchor_corr"])
    hard_penalty = 0.0
    if rmse < 0.05 or rmse > 0.15:
        hard_penalty += 5.0
    if max_abs >= 0.5 or mean >= 0.02 or species > 0.10 or corr < 0.99999:
        hard_penalty += 5.0
    if low > 0.15 or top > 0.15:
        hard_penalty += 2.0
    return (hard_penalty + abs(rmse - 0.09) + 0.25 * species + mean, max_abs, species, -corr)


def print_diag(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: rmse={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} mean={row['anchor_diff_mean']:.4f} "
        f"spmax={row['max_abs_species_mean_shift']:.4f}"
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: rmse={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} mean={row['anchor_diff_mean']:.4f} "
        f"spmax={row['max_abs_species_mean_shift']:.4f} "
        f"gcv={row['cv_group_delta_vs_base3500']:.4f} "
        f"ex15={row['cv_exclude15_delta_vs_base3500']:.4f}"
    )


if __name__ == "__main__":
    main()
