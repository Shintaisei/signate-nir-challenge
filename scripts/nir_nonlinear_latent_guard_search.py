#!/usr/bin/env python3
"""Direct nonlinear latent-model search under current-anchor Public-risk guards.

This intentionally does not distill a tiny correction into the current anchor.
Each candidate is a standalone prediction path, then diagnosed against the
13.986 anchor and known Public-failure directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR

import nir_anchor_rebuild_search as ar
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_nonlinear_latent_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
BAD_ALPHA3000 = SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge3000.csv"
WEIGHTED_GOLDEN = SUBMISSION_DIR / "nir_20260607_advgold_rf_shrink_s0p012_c0p06.csv"
OPERATOR_RESIDUAL = SUBMISSION_DIR / "nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv"


@dataclass(frozen=True)
class NonlinearSpec:
    name: str
    preprocess: str
    latent: str
    n_components: int
    target: str
    model: str
    alpha: float = math.nan
    gamma: str = ""
    c: float = math.nan
    epsilon: float = math.nan
    n_features: int = 0
    hidden: int = 0
    seed: int = 42
    seed_count: int = 1


@dataclass(frozen=True)
class TargetTransform:
    z: np.ndarray
    inverse: Callable[[np.ndarray], np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--families", default="svr,krr,rff,elm")
    parser.add_argument("--preprocesses", default="sg9_snv,msc_sg9,detrend_snv,sg9_snv_stack_d1,sg9_snv_d1")
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    refs = load_reference_diffs(anchor_test)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF", flush=True)
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    families = set(filter(None, (part.strip() for part in args.families.split(","))))
    preprocesses = [part.strip() for part in args.preprocesses.split(",") if part.strip()]
    specs = build_specs(preprocesses, families)
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}", flush=True)

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            oof = np.clip(make_oof(spec, data["X_train"], data["y"], data["groups"]), 0, None)
            pred_raw, test_meta = fit_predict_spec_with_meta(spec, data["X_train"], data["y"], data["X_test"])
            pred = np.clip(pred_raw, 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            oof=oof,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            refs=refs,
            test_meta=test_meta,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "nonlinear_latent_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "nonlinear_latent_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop nonlinear latent candidates:")
    for row in rows_sorted[:80]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} opcorr={row['corr_diff_operator_residual']:.4f} "
            f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5 "
            f"range={row['range_ratio_vs_anchor']:.4f}"
        )
    print(f"saved nonlinear latent diagnostics: {out_dir}")


def build_specs(preprocesses: list[str], families: set[str]) -> list[NonlinearSpec]:
    specs: list[NonlinearSpec] = []
    latent_grid = [("pls", 3), ("pls", 5), ("pls", 8), ("pca", 12), ("pca", 20)]
    targets = ["raw", "yj"]
    for preprocess in preprocesses:
        for latent, n_components in latent_grid:
            for target in targets:
                prefix = f"nir_nl_{preprocess}_{latent}{n_components}_{target}"
                if "svr" in families:
                    for c in [1.0, 3.0, 10.0]:
                        for epsilon in [0.10, 0.20]:
                            for gamma in ["scale", "0.03"]:
                                specs.append(
                                    NonlinearSpec(
                                        name=f"{prefix}_svr_C{tag(c)}_e{tag(epsilon)}_g{tag_gamma(gamma)}",
                                        preprocess=preprocess,
                                        latent=latent,
                                        n_components=n_components,
                                        target=target,
                                        model="svr",
                                        c=c,
                                        epsilon=epsilon,
                                        gamma=gamma,
                                    )
                                )
                if "krr" in families:
                    for alpha in [30.0, 100.0, 300.0]:
                        for gamma in ["0.01", "0.03", "0.10"]:
                            specs.append(
                                NonlinearSpec(
                                    name=f"{prefix}_krr_a{tag(alpha)}_g{tag_gamma(gamma)}",
                                    preprocess=preprocess,
                                    latent=latent,
                                    n_components=n_components,
                                    target=target,
                                    model="krr",
                                    alpha=alpha,
                                    gamma=gamma,
                                )
                            )
                if "rff" in families:
                    for n_features in [64, 128]:
                        for alpha in [1000.0, 3000.0]:
                            for gamma in ["0.01", "0.03"]:
                                specs.append(
                                    NonlinearSpec(
                                        name=f"{prefix}_rff_nf{n_features}_a{tag(alpha)}_g{tag_gamma(gamma)}",
                                        preprocess=preprocess,
                                        latent=latent,
                                        n_components=n_components,
                                        target=target,
                                        model="rff",
                                        alpha=alpha,
                                        gamma=gamma,
                                        n_features=n_features,
                                        seed=11,
                                        seed_count=3,
                                    )
                                )
                if "elm" in families:
                    for hidden in [64, 128]:
                        for alpha in [1000.0, 3000.0]:
                            specs.append(
                                NonlinearSpec(
                                    name=f"{prefix}_elm_h{hidden}_a{tag(alpha)}",
                                    preprocess=preprocess,
                                    latent=latent,
                                    n_components=n_components,
                                    target=target,
                                    model="elm",
                                    alpha=alpha,
                                    hidden=hidden,
                                    seed=17,
                                    seed_count=5,
                                )
                            )
    return specs


def make_oof(spec: NonlinearSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
    return pred


def fit_predict_spec(spec: NonlinearSpec, X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    pred, _ = fit_predict_spec_with_meta(spec, X_train_raw, y, X_pred_raw)
    return pred


def fit_predict_spec_with_meta(
    spec: NonlinearSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X_train, X_pred = ar.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    target = transform_target(spec.target, y)
    Z_train, Z_pred = latent_features(spec, X_train, X_pred, target.z)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_pred = scaler.transform(Z_pred)
    pred_z, meta = fit_predict_model(spec, Z_train, target.z, Z_pred)
    pred = target.inverse(pred_z)
    seed_stack = meta.pop("seed_pred_z_stack", None)
    if seed_stack is None:
        meta.update(
            {
                "seed_pred_std_mean": 0.0,
                "seed_pred_std_max": 0.0,
                "seed_pred_rmse_mean": 0.0,
            }
        )
    else:
        seed_preds = np.vstack([target.inverse(row) for row in seed_stack])
        seed_std = np.std(seed_preds, axis=0)
        meta.update(
            {
                "seed_pred_std_mean": float(np.mean(seed_std)),
                "seed_pred_std_max": float(np.max(seed_std)),
                "seed_pred_rmse_mean": float(math.sqrt(np.mean(seed_std**2))),
            }
        )
    return pred, meta


def latent_features(
    spec: NonlinearSpec,
    X_train: np.ndarray,
    X_pred: np.ndarray,
    y_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    comps = min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1])
    comps = max(1, comps)
    if spec.latent == "pca":
        pca = PCA(n_components=comps, random_state=42)
        return pca.fit_transform(X_train), pca.transform(X_pred)
    if spec.latent == "pls":
        pls = PLSRegression(n_components=comps, scale=True)
        pls.fit(X_train, y_z)
        return pls.transform(X_train), pls.transform(X_pred)
    raise ValueError(spec.latent)


def fit_predict_model(
    spec: NonlinearSpec,
    Z_train: np.ndarray,
    y_z: np.ndarray,
    Z_pred: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if spec.model == "svr":
        gamma: str | float = "scale" if spec.gamma == "scale" else float(spec.gamma)
        model = SVR(kernel="rbf", C=spec.c, epsilon=spec.epsilon, gamma=gamma)
        model.fit(Z_train, y_z)
        return np.asarray(model.predict(Z_pred), dtype=float), {}
    if spec.model == "krr":
        model = KernelRidge(alpha=spec.alpha, kernel="rbf", gamma=float(spec.gamma))
        model.fit(Z_train, y_z)
        return np.asarray(model.predict(Z_pred), dtype=float), {}
    if spec.model == "rff":
        preds = []
        for seed in seeds(spec.seed, spec.seed_count):
            sampler = RBFSampler(gamma=float(spec.gamma), n_components=spec.n_features, random_state=seed)
            H_train = sampler.fit_transform(Z_train)
            H_pred = sampler.transform(Z_pred)
            model = Ridge(alpha=spec.alpha)
            model.fit(H_train, y_z)
            preds.append(model.predict(H_pred))
        stack = np.vstack(preds)
        return np.mean(stack, axis=0), {"seed_pred_z_stack": stack}
    if spec.model == "elm":
        preds = []
        for seed in seeds(spec.seed, spec.seed_count):
            H_train, H_pred = elm_features(Z_train, Z_pred, spec.hidden, seed)
            model = Ridge(alpha=spec.alpha)
            model.fit(H_train, y_z)
            preds.append(model.predict(H_pred))
        stack = np.vstack(preds)
        return np.mean(stack, axis=0), {"seed_pred_z_stack": stack}
    raise ValueError(spec.model)


def elm_features(Z_train: np.ndarray, Z_pred: np.ndarray, hidden: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(max(Z_train.shape[1], 1))
    weights = rng.normal(0.0, scale, size=(Z_train.shape[1], hidden))
    bias = rng.normal(0.0, 0.5, size=hidden)
    return np.tanh(Z_train @ weights + bias), np.tanh(Z_pred @ weights + bias)


def seeds(start: int, count: int) -> list[int]:
    return [start + 104729 * i for i in range(count)]


def transform_target(target: str, y: np.ndarray) -> TargetTransform:
    if target == "raw":
        return TargetTransform(np.asarray(y, dtype=float), lambda pred: np.asarray(pred, dtype=float))
    if target == "yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        z = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        return TargetTransform(z, lambda pred, tr=transformer: tr.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel())
    raise ValueError(target)


def load_reference_diffs(anchor_test: np.ndarray) -> dict[str, np.ndarray]:
    refs = {
        "bad_alpha3000": BAD_ALPHA3000,
        "weighted_golden": WEIGHTED_GOLDEN,
        "operator_residual": OPERATOR_RESIDUAL,
    }
    out: dict[str, np.ndarray] = {}
    for name, path in refs.items():
        df = pd.read_csv(path, header=None)
        out[name] = df[1].to_numpy(float) - anchor_test
    return out


def diagnostics(
    *,
    spec: NonlinearSpec,
    path: Path,
    pred: np.ndarray,
    oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
    test_meta: dict[str, float],
) -> dict[str, object]:
    diff = pred - anchor_test
    oof_rmse = rmse(y, oof)
    fold = fold_delta_stats(y, oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "preprocess": spec.preprocess,
        "latent": spec.latent,
        "n_components": spec.n_components,
        "target": spec.target,
        "model": spec.model,
        "alpha": spec.alpha,
        "gamma": spec.gamma,
        "C": spec.c,
        "epsilon": spec.epsilon,
        "n_features": spec.n_features,
        "hidden": spec.hidden,
        "seed_count": spec.seed_count,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": oof_rmse,
        "direct_oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": fold["improved_count"],
        "direct_worst_fold_delta": fold["worst_delta"],
        "direct_fold_delta_std": fold["delta_std"],
        "direct_oof_corr_with_anchor": safe_corr(oof, anchor_oof),
        "seed_pred_std_mean": test_meta["seed_pred_std_mean"],
        "seed_pred_std_max": test_meta["seed_pred_std_max"],
        "seed_pred_rmse_mean": test_meta["seed_pred_rmse_mean"],
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "pred_std": float(np.std(pred)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
    }
    for ref_name, ref_diff in refs.items():
        row[f"corr_diff_{ref_name}"] = safe_corr(diff, ref_diff)
        row[f"rmse_diff_{ref_name}"] = rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    if diff < 0.08:
        reasons.append("near_anchor")
    if diff > 0.45:
        reasons.append("anchor_diff_gt0p45")
    if float(row["anchor_diff_max_abs"]) > 2.00:
        reasons.append("max_diff_gt2")
    if float(row["max_abs_species_mean_shift"]) > 0.18:
        reasons.append("species_shift_gt0p18")
    if float(row["max_abs_species_mean_shift"]) > 0.12:
        reasons.append("species_shift_caution")
    if float(row["corr_diff_bad_alpha3000"]) > 0.40:
        reasons.append("bad_alpha3000_corr_gt0p40")
    if abs(float(row["decile_gap_top_minus_bottom"])) > 0.55:
        reasons.append("decile_gap_gt0p55")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if not (0.90 <= float(row["range_ratio_vs_anchor"]) <= 1.10):
        reasons.append("range_shift")
    if float(row["direct_oof_delta_vs_anchor"]) > 0.03:
        reasons.append("direct_oof_bad")
    if float(row["direct_worst_fold_delta"]) > 0.20:
        reasons.append("worst_fold_bad")
    if int(row["direct_improved_fold_count"]) < 3:
        reasons.append("direct_folds_lt3")
    if float(row["seed_pred_std_mean"]) > 0.05:
        reasons.append("seed_std_mean_gt0p05")
    if float(row["seed_pred_std_max"]) > 0.15:
        reasons.append("seed_std_max_gt0p15")
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if abs(float(row["corr_diff_weighted_golden"])) > 0.85:
        reasons.append("weighted_golden_like")
    if abs(float(row["corr_diff_operator_residual"])) > 0.85:
        reasons.append("operator_residual_like")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    return float(
        2.0 * max(float(row["corr_diff_bad_alpha3000"]), 0.0)
        + 0.8 * abs(float(row["decile_gap_top_minus_bottom"]))
        + 0.7 * abs(float(row["sample_order_corr"]))
        + 0.8 * float(row["max_abs_species_mean_shift"])
        + 0.08 * float(row["anchor_diff_max_abs"])
        + 0.5 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
        + (0.5 if int(row["top10_abs_max_species_count"]) > 4 else 0.0)
        + (0.4 if not (0.90 <= float(row["range_ratio_vs_anchor"]) <= 1.10) else 0.0)
    )


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    deltas: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for _, valid_idx in splitter.split(np.zeros((len(y), 1)), y, groups):
        deltas.append(rmse(y[valid_idx], pred[valid_idx]) - rmse(y[valid_idx], anchor_pred[valid_idx]))
    arr = np.asarray(deltas)
    return {
        "improved_count": int(np.sum(arr < 0)),
        "worst_delta": float(np.max(arr)),
        "delta_std": float(np.std(arr)),
    }


def failed_row(spec: NonlinearSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "preprocess": spec.preprocess,
        "latent": spec.latent,
        "n_components": spec.n_components,
        "target": spec.target,
        "model": spec.model,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
        "direct_oof_delta_vs_anchor": float("inf"),
        "direct_improved_fold_count": 0,
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
    }


def max_abs_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    if b is None:
        values = np.asarray(a, dtype=float)
        return float(math.sqrt(np.mean(values**2)))
    return float(math.sqrt(mean_squared_error(a, b)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.8 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.18),
        float(row["direct_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def tag_gamma(value: str) -> str:
    return "scale" if value == "scale" else tag(float(value))


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} badcorr={row['corr_diff_bad_alpha3000']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5 "
        f"reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
