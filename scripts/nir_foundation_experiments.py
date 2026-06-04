#!/usr/bin/env python3
"""Build foundation NIR submission candidates and diagnostics.

This script is intentionally standalone because the current repository has
historical analysis scripts but no populated src pipeline. It implements the
2026-06-04 plan around scatter correction, Savitzky-Golay transforms, compact
PCA features, log-target Ridge models, and anchor-drift checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ARDRegression, BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_SUBMIT_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_foundation"
PUBLIC_LOG_PATH = OUTPUT_ROOT / "public_log.csv"
ANCHOR_SUBMISSION = SUBMISSION_DIR / "submission_minato_predicting_zip_20260604.csv"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    transform: str
    pca_components: int
    ridge_alpha: float
    sg_window: int | None = None
    sg_polyorder: int = 2
    deriv: int = 0


@dataclass
class CandidateResult:
    experiment: str
    family: str
    submission_path: str
    pca_components: int
    ridge_alpha: float
    pred_min: float
    pred_mean: float
    pred_median: float
    pred_max: float
    negative_count_before_clip: int
    sample_order_match: bool
    anchor_diff_rmse: float | None
    anchor_diff_max_abs: float | None
    anchor_corr: float | None
    cv_group_species_rmse: float | None
    cv_group_species_std: float | None
    cv_group_species_exclude15_rmse: float | None
    cv_group_species_exclude15_std: float | None
    cv_random_rmse: float | None
    cv_random_std: float | None


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "nir_sg9_snv_pca20_ridge2500": ExperimentSpec(
        name="nir_sg9_snv_pca20_ridge2500",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=2500.0,
    ),
    "nir_sg7_snv_pca20_ridge2500": ExperimentSpec(
        name="nir_sg7_snv_pca20_ridge2500",
        family="sg_snv",
        transform="sg_snv",
        sg_window=7,
        pca_components=20,
        ridge_alpha=2500.0,
    ),
    "nir_sg11_snv_pca20_ridge2500": ExperimentSpec(
        name="nir_sg11_snv_pca20_ridge2500",
        family="sg_snv",
        transform="sg_snv",
        sg_window=11,
        pca_components=20,
        ridge_alpha=2500.0,
    ),
    "nir_sg9_snv_pca15_ridge2500": ExperimentSpec(
        name="nir_sg9_snv_pca15_ridge2500",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=15,
        ridge_alpha=2500.0,
    ),
    "nir_sg9_snv_pca20_ridge5000": ExperimentSpec(
        name="nir_sg9_snv_pca20_ridge5000",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=5000.0,
    ),
    "nir_sg9_snv_pca20_ridge3000": ExperimentSpec(
        name="nir_sg9_snv_pca20_ridge3000",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3000.0,
    ),
    "nir_sg9_snv_pca20_ridge3500": ExperimentSpec(
        name="nir_sg9_snv_pca20_ridge3500",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3500.0,
    ),
    "nir_sg9_snv_pca20_ridge4000": ExperimentSpec(
        name="nir_sg9_snv_pca20_ridge4000",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=4000.0,
    ),
    "nir_sg9_snv_pca20_shape16_ridge2500": ExperimentSpec(
        name="nir_sg9_snv_pca20_shape16_ridge2500",
        family="sg_snv_shape",
        transform="sg_snv_shape16",
        sg_window=9,
        pca_components=20,
        ridge_alpha=2500.0,
    ),
    "nir_sg9_snv_pca20_shape16_ridge3500": ExperimentSpec(
        name="nir_sg9_snv_pca20_shape16_ridge3500",
        family="sg_snv_shape",
        transform="sg_snv_shape16",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3500.0,
    ),
    "nir_ms_target_yeojohnson_ridge3200": ExperimentSpec(
        name="nir_ms_target_yeojohnson_ridge3200",
        family="target_transform",
        transform="target_yeojohnson",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3200.0,
    ),
    "nir_ms_target_boxcox_ridge3500": ExperimentSpec(
        name="nir_ms_target_boxcox_ridge3500",
        family="target_transform",
        transform="target_boxcox",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3500.0,
    ),
    "nir_ms_target_yeojohnson_ridge3500": ExperimentSpec(
        name="nir_ms_target_yeojohnson_ridge3500",
        family="target_transform",
        transform="target_yeojohnson",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3500.0,
    ),
    "nir_msc_pca20_ridge2500": ExperimentSpec(
        name="nir_msc_pca20_ridge2500",
        family="msc",
        transform="msc",
        pca_components=20,
        ridge_alpha=2500.0,
    ),
    "nir_msc_pca15_ridge2500": ExperimentSpec(
        name="nir_msc_pca15_ridge2500",
        family="msc",
        transform="msc",
        pca_components=15,
        ridge_alpha=2500.0,
    ),
    "nir_msc_pca20_ridge5000": ExperimentSpec(
        name="nir_msc_pca20_ridge5000",
        family="msc",
        transform="msc",
        pca_components=20,
        ridge_alpha=5000.0,
    ),
    "nir_snv_deriv1_sg15_pca20_ridge2500": ExperimentSpec(
        name="nir_snv_deriv1_sg15_pca20_ridge2500",
        family="snv_deriv",
        transform="snv_deriv",
        sg_window=15,
        pca_components=20,
        ridge_alpha=2500.0,
        deriv=1,
    ),
    "nir_minato_stack_snv_sg1w15_pca20_ridge2500": ExperimentSpec(
        name="nir_minato_stack_snv_sg1w15_pca20_ridge2500",
        family="minato_stack",
        transform="minato_stack",
        sg_window=15,
        pca_components=20,
        ridge_alpha=2500.0,
        deriv=1,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available experiments.")
    parser.add_argument("--all", action="store_true", help="Run all registered experiments.")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS), help="Experiment to run.")
    parser.add_argument("--cv", action="store_true", help="Run CV diagnostics.")
    parser.add_argument("--random-cv", action="store_true", help="Also run random KFold CV.")
    parser.add_argument("--no-write-submission", action="store_true", help="Do not write submission CSV files.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Diagnostics output directory.")
    parser.add_argument(
        "--attack-blends",
        action="store_true",
        help="Build post-public attack candidates from validated SG9 direction and weak regularization blends.",
    )
    parser.add_argument(
        "--model-search",
        action="store_true",
        help="Build non-blend model candidates from target transforms, structured PCR, robust PCR, and compact PLS.",
    )
    parser.add_argument(
        "--record-public",
        nargs=2,
        metavar=("EXPERIMENT", "SCORE"),
        help="Append a submitted Public score to outputs/nir_foundation/public_log.csv.",
    )
    parser.add_argument("--memo", default="", help="Memo used with --record-public.")
    args = parser.parse_args()

    if args.record_public:
        experiment_name, score_text = args.record_public
        record_public_score(experiment_name, float(score_text), args.memo)
        return

    if args.list:
        for name, spec in EXPERIMENTS.items():
            print(f"{name:45s} {spec.family:12s} PCA={spec.pca_components:2d} alpha={spec.ridge_alpha:g}")
        print("\nattack blend candidates are built with --attack-blends")
        return

    if args.attack_blends:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_dir or (OUTPUT_ROOT / f"attack_{timestamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        build_attack_blends(out_dir)
        return

    if args.model_search:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_dir or (OUTPUT_ROOT / f"model_search_{timestamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        data = load_data()
        build_model_search_candidates(data, out_dir)
        return

    selected = list(EXPERIMENTS) if args.all else args.experiment
    if not selected:
        parser.error("Use --experiment NAME, --all, or --list.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (OUTPUT_ROOT / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    anchor = load_anchor_submission()
    results: list[CandidateResult] = []

    for name in selected:
        spec = EXPERIMENTS[name]
        print(f"running {name}")
        result = run_candidate(
            spec=spec,
            data=data,
            anchor=anchor,
            write_submission=not args.no_write_submission,
            run_cv=args.cv,
            run_random_cv=args.random_cv,
        )
        results.append(result)
        print_result(result)

    write_summary(out_dir, results)
    print(f"saved diagnostics: {out_dir}")


def load_data() -> dict[str, object]:
    train = pd.read_csv(TRAIN_PATH, encoding="cp932")
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample_submit = pd.read_csv(SAMPLE_SUBMIT_PATH, header=None)

    train_feature_cols = [c for c in train.columns if _is_float_like(c)]
    test_feature_cols = [c for c in test.columns if _is_float_like(c)]
    if train_feature_cols != test_feature_cols:
        raise ValueError("train/test spectral feature columns do not match")

    return {
        "train": train,
        "test": test,
        "sample_submit": sample_submit,
        "feature_cols": train_feature_cols,
        "X_train": train[train_feature_cols].to_numpy(dtype=float),
        "X_test": test[test_feature_cols].to_numpy(dtype=float),
        "y": train["含水率"].to_numpy(dtype=float),
        "groups": train["species number"].to_numpy(),
        "test_ids": test["sample number"].to_numpy(),
    }


def _is_float_like(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def load_anchor_submission() -> pd.DataFrame | None:
    if not ANCHOR_SUBMISSION.exists():
        return None
    return pd.read_csv(ANCHOR_SUBMISSION, header=None)


def run_candidate(
    *,
    spec: ExperimentSpec,
    data: dict[str, object],
    anchor: pd.DataFrame | None,
    write_submission: bool,
    run_cv: bool,
    run_random_cv: bool,
) -> CandidateResult:
    X_train = data["X_train"]  # type: ignore[assignment]
    X_test = data["X_test"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    test_ids = data["test_ids"]  # type: ignore[assignment]
    sample_submit = data["sample_submit"]  # type: ignore[assignment]

    Xtr, Xte = transform_pair(spec, X_train, X_test)
    pred, negative_count = fit_predict_pca_ridge(Xtr, y, Xte, spec)

    submission = pd.DataFrame({0: test_ids, 1: pred})
    sample_order_match = bool((sample_submit[0].to_numpy() == submission[0].to_numpy()).all())
    if not sample_order_match:
        raise ValueError(f"{spec.name}: sample order does not match sample_submit.csv")

    submission_path = SUBMISSION_DIR / f"{spec.name}.csv"
    if write_submission:
        submission.to_csv(submission_path, index=False, header=False)

    anchor_diff_rmse, anchor_diff_max_abs, anchor_corr = compare_anchor(submission, anchor)

    cv_group = (None, None)
    cv_exclude15 = (None, None)
    cv_random = (None, None)
    if run_cv:
        cv_group = cross_validate(data, spec, exclude_species=None, random=False)
        cv_exclude15 = cross_validate(data, spec, exclude_species=15, random=False)
        if run_random_cv:
            cv_random = cross_validate(data, spec, exclude_species=None, random=True)

    return CandidateResult(
        experiment=spec.name,
        family=spec.family,
        submission_path=str(submission_path),
        pca_components=spec.pca_components,
        ridge_alpha=spec.ridge_alpha,
        pred_min=float(np.min(pred)),
        pred_mean=float(np.mean(pred)),
        pred_median=float(np.median(pred)),
        pred_max=float(np.max(pred)),
        negative_count_before_clip=negative_count,
        sample_order_match=sample_order_match,
        anchor_diff_rmse=anchor_diff_rmse,
        anchor_diff_max_abs=anchor_diff_max_abs,
        anchor_corr=anchor_corr,
        cv_group_species_rmse=cv_group[0],
        cv_group_species_std=cv_group[1],
        cv_group_species_exclude15_rmse=cv_exclude15[0],
        cv_group_species_exclude15_std=cv_exclude15[1],
        cv_random_rmse=cv_random[0],
        cv_random_std=cv_random[1],
    )


def transform_pair(spec: ExperimentSpec, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec.transform == "sg_snv":
        return snv(savgol(X_train, spec)), snv(savgol(X_test, spec))
    if spec.transform == "sg_snv_shape16":
        return snv(savgol(X_train, spec)), snv(savgol(X_test, spec))
    if spec.transform in {"target_yeojohnson", "target_boxcox"}:
        return snv(savgol(X_train, spec)), snv(savgol(X_test, spec))
    if spec.transform == "msc":
        return msc_pair(X_train, X_test)
    if spec.transform == "snv_deriv":
        return savgol(snv(X_train), spec), savgol(snv(X_test), spec)
    if spec.transform == "minato_stack":
        Xtr_snv = snv(X_train)
        Xte_snv = snv(X_test)
        return (
            np.hstack([Xtr_snv, savgol(Xtr_snv, spec)]),
            np.hstack([Xte_snv, savgol(Xte_snv, spec)]),
        )
    raise ValueError(f"unknown transform: {spec.transform}")


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def savgol(X: np.ndarray, spec: ExperimentSpec) -> np.ndarray:
    if spec.sg_window is None:
        raise ValueError(f"{spec.name}: sg_window is required")
    return savgol_filter(
        X,
        window_length=spec.sg_window,
        polyorder=spec.sg_polyorder,
        deriv=spec.deriv,
        axis=1,
        mode="interp",
    )


def msc_pair(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = X_train.mean(axis=0)
    return msc_transform(X_train, reference), msc_transform(X_test, reference)


def msc_transform(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    centered_ref = reference - reference.mean()
    denom = float(np.dot(centered_ref, centered_ref))
    if denom == 0:
        raise ValueError("MSC reference spectrum has zero variance")
    out = np.empty_like(X, dtype=float)
    for i, row in enumerate(X):
        slope = float(np.dot(row - row.mean(), centered_ref) / denom)
        intercept = float(row.mean() - slope * reference.mean())
        if abs(slope) < 1e-12:
            slope = 1.0
        out[i] = (row - intercept) / slope
    return out


def fit_predict_pca_ridge(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, int]:
    if spec.transform == "target_yeojohnson":
        pred_raw = fit_predict_target_transform(
            X_train,
            y,
            X_test,
            alpha=spec.ridge_alpha,
            target="yeojohnson",
        )
        return np.clip(pred_raw, 0, None), int((pred_raw < 0).sum())
    if spec.transform == "target_boxcox":
        pred_raw = fit_predict_target_transform(
            X_train,
            y,
            X_test,
            alpha=spec.ridge_alpha,
            target="boxcox",
        )
        return np.clip(pred_raw, 0, None), int((pred_raw < 0).sum())

    Z_train, Z_test = make_model_features(X_train, X_test, spec)
    model = Ridge(alpha=spec.ridge_alpha)
    model.fit(Z_train, np.log1p(y))
    pred_raw = np.expm1(model.predict(Z_test))
    negative_count = int((pred_raw < 0).sum())
    return np.clip(pred_raw, 0, None), negative_count


def make_model_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.transform == "sg_snv_shape16":
        pca = PCA(n_components=spec.pca_components, random_state=42)
        Z_train = pca.fit_transform(X_train)
        Z_test = pca.transform(X_test)
        F_train = local_shape_features(X_train)
        F_test = local_shape_features(X_test)
        scaler = StandardScaler()
        F_train = scaler.fit_transform(F_train)
        F_test = scaler.transform(F_test)
        return np.hstack([Z_train, F_train]), np.hstack([Z_test, F_test])

    pca = PCA(n_components=spec.pca_components, random_state=42)
    return pca.fit_transform(X_train), pca.transform(X_test)


def local_shape_features(X: np.ndarray) -> np.ndarray:
    windows = [
        (1332, 1336),
        (1352, 1356),
        (680, 720),
        (1192, 1196),
        (908, 912),
        (708, 712),
        (1344, 1348),
        (1330, 1340),
        (576, 580),
        (828, 832),
        (1320, 1324),
        (1200, 1440),
        (1320, 1360),
        (1480, 1500),
        (640, 720),
        (816, 832),
    ]
    features = []
    row_mean = X.mean(axis=1)
    for start, end in windows:
        lo = max(0, start)
        hi = min(X.shape[1], end)
        if lo >= hi:
            values = np.zeros(X.shape[0])
        else:
            values = X[:, lo:hi].mean(axis=1) - row_mean
        features.append(values)
    return np.vstack(features).T


def compare_anchor(
    submission: pd.DataFrame,
    anchor: pd.DataFrame | None,
) -> tuple[float | None, float | None, float | None]:
    if anchor is None:
        return None, None, None
    if len(anchor) != len(submission) or not np.array_equal(anchor[0].to_numpy(), submission[0].to_numpy()):
        return None, None, None
    diff = submission[1].to_numpy(dtype=float) - anchor[1].to_numpy(dtype=float)
    rmse = float(math.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    corr = float(np.corrcoef(anchor[1].to_numpy(dtype=float), submission[1].to_numpy(dtype=float))[0, 1])
    return rmse, max_abs, corr


def cross_validate(
    data: dict[str, object],
    spec: ExperimentSpec,
    *,
    exclude_species: int | None,
    random: bool,
) -> tuple[float, float]:
    X_all = data["X_train"]  # type: ignore[assignment]
    y_all = data["y"]  # type: ignore[assignment]
    groups_all = data["groups"]  # type: ignore[assignment]

    mask = np.ones(len(y_all), dtype=bool)
    if exclude_species is not None:
        mask &= groups_all != exclude_species
    X = X_all[mask]
    y = y_all[mask]
    groups = groups_all[mask]

    if random:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(X, y, groups)

    rmses: list[float] = []
    for train_idx, valid_idx in split_iter:
        Xtr_raw, Xva_raw = X[train_idx], X[valid_idx]
        ytr, yva = y[train_idx], y[valid_idx]
        Xtr, Xva = transform_pair(spec, Xtr_raw, Xva_raw)
        pred, _ = fit_predict_pca_ridge(Xtr, ytr, Xva, spec)
        rmses.append(float(math.sqrt(mean_squared_error(yva, pred))))
    return float(np.mean(rmses)), float(np.std(rmses))


def print_result(result: CandidateResult) -> None:
    print(
        "  pred "
        f"min={result.pred_min:.4f} mean={result.pred_mean:.4f} "
        f"median={result.pred_median:.4f} max={result.pred_max:.4f} "
        f"neg_before_clip={result.negative_count_before_clip}"
    )
    if result.anchor_diff_rmse is not None:
        print(
            "  anchor "
            f"diff_rmse={result.anchor_diff_rmse:.6f} "
            f"max_abs={result.anchor_diff_max_abs:.6f} corr={result.anchor_corr:.6f}"
        )
    if result.cv_group_species_rmse is not None:
        print(
            "  cv "
            f"group={result.cv_group_species_rmse:.4f}+/-{result.cv_group_species_std:.4f} "
            f"exclude15={result.cv_group_species_exclude15_rmse:.4f}+/-{result.cv_group_species_exclude15_std:.4f}"
        )
    print(f"  submission {result.submission_path}")


def write_summary(out_dir: Path, results: list[CandidateResult]) -> None:
    rows = [asdict(r) for r in results]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def record_public_score(experiment_name: str, score: float, memo: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exists = PUBLIC_LOG_PATH.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "public_score": score,
        "anchor_public": 14.39987078261004,
        "delta_vs_anchor": score - 14.39987078261004,
        "submission_path": str(SUBMISSION_DIR / f"{experiment_name}.csv"),
        "memo": memo,
    }
    with PUBLIC_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"saved public log: {PUBLIC_LOG_PATH}")
    print(f"delta_vs_anchor: {row['delta_vs_anchor']:.12f}")


def build_attack_blends(out_dir: Path) -> None:
    sample_submit = pd.read_csv(SAMPLE_SUBMIT_PATH, header=None)
    anchor = _read_submission(ANCHOR_SUBMISSION)
    sg9 = _read_submission(SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge2500.csv")
    ridge5000 = _read_submission(SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge5000.csv")
    sg7 = _read_submission(SUBMISSION_DIR / "nir_sg7_snv_pca20_ridge2500.csv")
    sg11 = _read_submission(SUBMISSION_DIR / "nir_sg11_snv_pca20_ridge2500.csv")
    pca15 = _read_submission(SUBMISSION_DIR / "nir_sg9_snv_pca15_ridge2500.csv")

    sources = [anchor, sg9, ridge5000, sg7, sg11, pca15]
    ids = anchor[0].to_numpy()
    for source in sources:
        if not np.array_equal(ids, source[0].to_numpy()):
            raise ValueError("attack source sample order mismatch")
    if not np.array_equal(ids, sample_submit[0].to_numpy()):
        raise ValueError("attack source sample order does not match sample_submit.csv")

    y_anchor = anchor[1].to_numpy(dtype=float)
    y_sg9 = sg9[1].to_numpy(dtype=float)
    y_ridge5000 = ridge5000[1].to_numpy(dtype=float)
    y_sg7 = sg7[1].to_numpy(dtype=float)
    y_sg11 = sg11[1].to_numpy(dtype=float)
    y_pca15 = pca15[1].to_numpy(dtype=float)

    candidates: dict[str, tuple[np.ndarray, str]] = {
        "nir_attack_sg9_direction_x2": (
            y_anchor + 2.0 * (y_sg9 - y_anchor),
            "Validated SG9 smoothing direction extrapolated 2x from anchor.",
        ),
        "nir_attack_sg9_direction_x3": (
            y_anchor + 3.0 * (y_sg9 - y_anchor),
            "Validated SG9 smoothing direction extrapolated 3x from anchor.",
        ),
        "nir_attack_sg_window_vote": (
            np.mean([y_sg7, y_sg9, y_sg11], axis=0),
            "Average of SG window 7/9/11 variants.",
        ),
        "nir_attack_sg_pca15_vote": (
            np.mean([y_sg9, y_pca15], axis=0),
            "Average of SG9 PCA20 and SG9 PCA15 variants.",
        ),
        "nir_attack_sg9_ridge5000_w005": (
            0.95 * y_sg9 + 0.05 * y_ridge5000,
            "Weak 5% blend toward stronger Ridge5000 regularization.",
        ),
        "nir_attack_sg9_ridge5000_w010": (
            0.90 * y_sg9 + 0.10 * y_ridge5000,
            "Weak 10% blend toward stronger Ridge5000 regularization.",
        ),
    }

    rows = []
    for name, (pred, memo) in candidates.items():
        pred = np.clip(pred, 0, None)
        submission = pd.DataFrame({0: ids, 1: pred})
        path = SUBMISSION_DIR / f"{name}.csv"
        submission.to_csv(path, index=False, header=False)
        diff_anchor = pred - y_anchor
        diff_sg9 = pred - y_sg9
        row = {
            "experiment": name,
            "submission_path": str(path),
            "memo": memo,
            "pred_min": float(np.min(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_median": float(np.median(pred)),
            "pred_max": float(np.max(pred)),
            "negative_count": int((pred < 0).sum()),
            "anchor_diff_rmse": float(math.sqrt(np.mean(diff_anchor**2))),
            "anchor_diff_max_abs": float(np.max(np.abs(diff_anchor))),
            "anchor_corr": float(np.corrcoef(y_anchor, pred)[0, 1]),
            "sg9_diff_rmse": float(math.sqrt(np.mean(diff_sg9**2))),
            "sg9_diff_max_abs": float(np.max(np.abs(diff_sg9))),
            "sg9_corr": float(np.corrcoef(y_sg9, pred)[0, 1]),
        }
        rows.append(row)
        print(
            f"{name}: anchor_rmse={row['anchor_diff_rmse']:.6f} "
            f"sg9_rmse={row['sg9_diff_rmse']:.6f} min/median/max="
            f"{row['pred_min']:.4f}/{row['pred_median']:.4f}/{row['pred_max']:.4f}"
        )

    with (out_dir / "attack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (out_dir / "attack_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved attack diagnostics: {out_dir}")


def _read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing submission source: {path}")
    return pd.read_csv(path, header=None)


def build_model_search_candidates(data: dict[str, object], out_dir: Path) -> None:
    best_path = SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge3000.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"current best submission missing: {best_path}")
    current_best = _read_submission(best_path)
    sample_submit = pd.read_csv(SAMPLE_SUBMIT_PATH, header=None)

    X_train = data["X_train"]  # type: ignore[assignment]
    X_test = data["X_test"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    ids = data["test_ids"]  # type: ignore[assignment]

    if not np.array_equal(ids, current_best[0].to_numpy()):
        raise ValueError("current best sample order mismatch")
    if not np.array_equal(ids, sample_submit[0].to_numpy()):
        raise ValueError("sample_submit sample order mismatch")

    sg_spec = ExperimentSpec(
        name="model_search_base",
        family="sg_snv",
        transform="sg_snv",
        sg_window=9,
        pca_components=20,
        ridge_alpha=3000.0,
    )
    Xtr, Xte = transform_pair(sg_spec, X_train, X_test)
    y_best = current_best[1].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []

    def add_candidate(name: str, pred: np.ndarray, family: str, memo: str) -> None:
        pred = np.clip(np.asarray(pred, dtype=float), 0, None)
        submission = pd.DataFrame({0: ids, 1: pred})
        path = SUBMISSION_DIR / f"{name}.csv"
        submission.to_csv(path, index=False, header=False)
        diff = pred - y_best
        bottom, top = decile_delta(y_best, pred)
        row = {
            "experiment": name,
            "family": family,
            "submission_path": str(path),
            "memo": memo,
            "pred_min": float(np.min(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_median": float(np.median(pred)),
            "pred_max": float(np.max(pred)),
            "pred_std": float(np.std(pred)),
            "negative_count": int((pred < 0).sum()),
            "best_diff_rmse": float(math.sqrt(np.mean(diff**2))),
            "best_diff_max_abs": float(np.max(np.abs(diff))),
            "best_corr": float(np.corrcoef(y_best, pred)[0, 1]),
            "bottom_decile_delta": bottom,
            "top_decile_delta": top,
            "range_ratio_vs_best": float((np.max(pred) - np.min(pred)) / (np.max(y_best) - np.min(y_best))),
        }
        rows.append(row)
        print(
            f"{name}: family={family} rmse_vs_best={row['best_diff_rmse']:.4f} "
            f"corr={row['best_corr']:.6f} bottom={bottom:.4f} top={top:.4f} "
            f"range_ratio={row['range_ratio_vs_best']:.4f}"
        )

    for alpha in [2800.0, 3000.0, 3200.0, 3500.0, 4000.0]:
        for target in ["identity", "log1p", "sqrt", "boxcox", "yeojohnson", "quantile_normal"]:
            name = f"nir_ms_target_{target}_ridge{int(alpha)}"
            pred = fit_predict_target_transform(Xtr, y, Xte, alpha=alpha, target=target)
            add_candidate(name, pred, "target_transform", f"SG9+SNV PCA20 Ridge alpha={alpha:g} target={target}")

    for alpha in [2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0]:
        for target in ["log1p", "boxcox", "yeojohnson"]:
            for n_components in [15, 18, 20, 22]:
                name = f"nir_ms_target2_{target}_pca{n_components}_ridge{int(alpha)}"
                pred = fit_predict_target_transform(
                    Xtr,
                    y,
                    Xte,
                    alpha=alpha,
                    target=target,
                    n_components=n_components,
                )
                add_candidate(
                    name,
                    pred,
                    "target_transform_fine",
                    f"SG9+SNV PCA{n_components} Ridge alpha={alpha:g} target={target}",
                )

    for n_components in [15, 20, 25]:
        for whiten in [False, True]:
            for alpha in [2800.0, 3000.0, 3200.0, 3500.0]:
                name = f"nir_ms_pca{n_components}_{'white' if whiten else 'plain'}_ridge{int(alpha)}"
                pred = fit_predict_structured_pcr(Xtr, y, Xte, n_components, whiten, "ridge", alpha)
                add_candidate(name, pred, "structured_pcr", f"PCA{n_components} whiten={whiten} Ridge alpha={alpha:g}")
            for model in ["bayesian", "ard"]:
                name = f"nir_ms_pca{n_components}_{'white' if whiten else 'plain'}_{model}"
                pred = fit_predict_structured_pcr(Xtr, y, Xte, n_components, whiten, model, 0.0)
                add_candidate(name, pred, "structured_pcr", f"PCA{n_components} whiten={whiten} {model}")

    for n_components in [15, 20, 25]:
        for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
            name = f"nir_ms_pca{n_components}_white_ridge_small{str(alpha).replace('.', 'p')}"
            pred = fit_predict_structured_pcr(Xtr, y, Xte, n_components, True, "ridge", alpha)
            add_candidate(name, pred, "structured_pcr_whiten_small_alpha", f"PCA{n_components} whiten=True Ridge alpha={alpha:g}")

    for epsilon in [1.2, 1.35, 1.5, 2.0]:
        for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            name = f"nir_ms_huber_e{str(epsilon).replace('.', 'p')}_a{str(alpha).replace('.', 'p')}"
            pred = fit_predict_huber_pcr(Xtr, y, Xte, epsilon=epsilon, alpha=alpha)
            add_candidate(name, pred, "robust_pcr", f"PCA20 Huber epsilon={epsilon:g} alpha={alpha:g}")

    for n_components in [2, 3, 4, 5, 6, 8]:
        for model in ["ridge", "bayesian"]:
            for alpha in ([3000.0, 10000.0, 30000.0] if model == "ridge" else [0.0]):
                suffix = model if model == "bayesian" else f"ridge{int(alpha)}"
                name = f"nir_ms_pls{n_components}_{suffix}"
                pred = fit_predict_pls(Xtr, y, Xte, n_components=n_components, model=model, alpha=alpha)
                add_candidate(name, pred, "compact_pls", f"PLS{n_components} + {model} alpha={alpha:g}")

    rows_sorted = sorted(rows, key=model_search_rank_key)
    with (out_dir / "model_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "model_search_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    print("\nTop gated candidates:")
    for row in rows_sorted[:12]:
        print(
            f"{row['experiment']}: family={row['family']} rmse={row['best_diff_rmse']:.4f} "
            f"corr={row['best_corr']:.6f} bottom={row['bottom_decile_delta']:.4f} "
            f"top={row['top_decile_delta']:.4f} range={row['range_ratio_vs_best']:.4f}"
        )
    print(f"saved model-search diagnostics: {out_dir}")


def decile_delta(y_ref: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    lo = np.quantile(y_ref, 0.10)
    hi = np.quantile(y_ref, 0.90)
    bottom = y_ref <= lo
    top = y_ref >= hi
    return float(np.mean(y_pred[bottom] - y_ref[bottom])), float(np.mean(y_pred[top] - y_ref[top]))


def model_search_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    corr = float(row["best_corr"])
    range_ratio = float(row["range_ratio_vs_best"])
    bottom = float(row["bottom_decile_delta"])
    top = float(row["top_decile_delta"])
    rmse = float(row["best_diff_rmse"])
    shape_penalty = abs(range_ratio - 1.0)
    direction_bonus = 0.0
    if bottom > 0 and top < 0:
        direction_bonus = -1.0
    if corr < 0.97:
        shape_penalty += 10.0
    if range_ratio < 0.80 or range_ratio > 1.08:
        shape_penalty += 5.0
    return (shape_penalty + direction_bonus, abs(rmse - 0.8), -corr)


def fit_predict_target_transform(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
    target: str,
    n_components: int = 20,
) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    if target == "identity":
        yt = y
        inverse = lambda z: z
    elif target == "log1p":
        yt = np.log1p(y)
        inverse = np.expm1
    elif target == "sqrt":
        yt = np.sqrt(y)
        inverse = lambda z: np.square(np.maximum(z, 0))
    elif target in {"boxcox", "yeojohnson"}:
        method = "box-cox" if target == "boxcox" else "yeo-johnson"
        transformer = PowerTransformer(method=method, standardize=True)
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        inverse = lambda z, tr=transformer: tr.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    elif target == "quantile_normal":
        transformer = QuantileTransformer(
            n_quantiles=min(1000, len(y)),
            output_distribution="normal",
            random_state=42,
        )
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        inverse = lambda z, tr=transformer: tr.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"unknown target transform: {target}")

    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt)
    return np.asarray(inverse(model.predict(Z_test)), dtype=float)


def fit_predict_structured_pcr(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    whiten: bool,
    model_name: str,
    alpha: float,
) -> np.ndarray:
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    yt = np.log1p(y)
    if model_name == "ridge":
        model = Ridge(alpha=alpha)
    elif model_name == "bayesian":
        model = BayesianRidge()
    elif model_name == "ard":
        model = ARDRegression()
    else:
        raise ValueError(f"unknown PCR model: {model_name}")
    model.fit(Z_train, yt)
    return np.expm1(model.predict(Z_test))


def fit_predict_huber_pcr(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    epsilon: float,
    alpha: float,
) -> np.ndarray:
    pca = PCA(n_components=20, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
    model.fit(Z_train, np.log1p(y))
    return np.expm1(model.predict(Z_test))


def fit_predict_pls(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    n_components: int,
    model: str,
    alpha: float,
) -> np.ndarray:
    pls = PLSRegression(n_components=n_components, scale=True)
    Z_train = pls.fit_transform(X_train, np.log1p(y))[0]
    Z_test = pls.transform(X_test)
    if model == "ridge":
        reg = Ridge(alpha=alpha)
    elif model == "bayesian":
        reg = BayesianRidge()
    else:
        raise ValueError(f"unknown PLS model: {model}")
    reg.fit(Z_train, np.log1p(y))
    return np.expm1(reg.predict(Z_test).ravel())


if __name__ == "__main__":
    main()
