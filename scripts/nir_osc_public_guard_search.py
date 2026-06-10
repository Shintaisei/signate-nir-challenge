#!/usr/bin/env python3
"""OSC/domain-component search with Public-failure hard gates."""

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
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer

import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_osc_public_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
BAD_ALPHA3000 = SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge3000.csv"


@dataclass(frozen=True)
class OscSpec:
    name: str
    strategy: str
    search_components: int
    n_remove: int
    n_components: int
    alpha: float


@dataclass(frozen=True)
class RemovalInfo:
    removed_target_corr_max: float
    removed_energy_ratio: float
    removed_indices: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    bad_diff = pd.read_csv(BAD_ALPHA3000, header=None)[1].to_numpy(float) - anchor_test
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    specs = build_specs()
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}")

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            pred, info = fit_predict_spec(spec, data["X_train"], data["y"], data["groups"], data["X_test"], data["groups"])
            oof, oof_infos = make_oof(spec, data["X_train"], data["y"], data["groups"])
            pred = np.clip(pred, 0, None)
            oof = np.clip(oof, 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}")
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
            bad_diff=bad_diff,
            info=info,
            oof_infos=oof_infos,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "osc_public_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "osc_public_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop OSC candidates:")
    for row in rows_sorted[:50]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} direct={row['direct_oof_delta_vs_anchor']:.4f} "
            f"fold={row['direct_improved_fold_count']}/5 tcorr={row['removed_target_corr_max']:.4f} "
            f"energy={row['removed_energy_ratio']:.4f}"
        )
    print(f"saved OSC diagnostics: {out_dir}")


def build_specs() -> list[OscSpec]:
    specs: list[OscSpec] = []
    strategies = [
        "low_target_corr_pc",
        "domain_shift_pc",
        "domain_shift_low_target_pc",
        "species_centroid_pc",
        "domain_mean_vector",
    ]
    for strategy in strategies:
        for search_components in [8, 12, 16]:
            for n_remove in [1, 2]:
                if strategy == "domain_mean_vector" and (search_components != 8 or n_remove != 1):
                    continue
                for n_components in [18, 20, 22, 24]:
                    for alpha in [3400.0, 3500.0, 3600.0]:
                        specs.append(
                            OscSpec(
                                name=(
                                    f"nir_osc_{strategy}_sc{search_components}_r{n_remove}_"
                                    f"p{n_components}_a{int(alpha)}"
                                ),
                                strategy=strategy,
                                search_components=search_components,
                                n_remove=n_remove,
                                n_components=n_components,
                                alpha=alpha,
                            )
                        )
    return specs


def make_oof(spec: OscSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[RemovalInfo]]:
    pred = np.empty(len(y), dtype=float)
    infos: list[RemovalInfo] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_pred, info = fit_predict_spec(
            spec,
            X[train_idx],
            y[train_idx],
            groups[train_idx],
            X[valid_idx],
            groups[valid_idx],
        )
        pred[valid_idx] = fold_pred
        infos.append(info)
    return pred, infos


def fit_predict_spec(
    spec: OscSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_pred_raw: np.ndarray,
    pred_groups: np.ndarray,
) -> tuple[np.ndarray, RemovalInfo]:
    X_train = base_features(X_train_raw)
    X_pred = base_features(X_pred_raw)
    X_train, X_pred, info = remove_components(spec, X_train, X_pred, y, groups)
    pred = fit_yj_ridge(X_train, y, X_pred, spec.n_components, spec.alpha)
    return pred, info


def base_features(X: np.ndarray) -> np.ndarray:
    return snv(savgol_filter(X, 9, 2, axis=1, mode="interp"))


def remove_components(
    spec: OscSpec,
    X_train: np.ndarray,
    X_pred: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, RemovalInfo]:
    x_mean = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - x_mean
    Pc = X_pred - x_mean
    total_energy = float(np.sum(np.var(Xc, axis=0)))
    yj = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(y.reshape(-1, 1)).ravel()

    if spec.strategy == "domain_mean_vector":
        direction = Pc.mean(axis=0) - Xc.mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise ValueError("empty domain mean direction")
        loading = direction / norm
        score = Xc @ loading
        pred_score = Pc @ loading
        target_corr = abs(safe_corr(score, yj))
        energy = float(np.var(score) / total_energy) if total_energy > 0 else 0.0
        Xnew = Xc - np.outer(score, loading)
        Pnew = Pc - np.outer(pred_score, loading)
        return Xnew, Pnew, RemovalInfo(target_corr, energy, "domain_mean")

    pca = PCA(n_components=spec.search_components, random_state=42)
    scores = pca.fit_transform(Xc)
    pred_scores = pca.transform(Pc)
    loadings = pca.components_
    target_corr = np.array([abs(safe_corr(scores[:, i], yj)) for i in range(scores.shape[1])])
    domain_shift = np.abs(pred_scores.mean(axis=0) - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-12)
    species_score = species_centroid_score(scores, groups)

    if spec.strategy == "low_target_corr_pc":
        order = np.argsort(target_corr)
    elif spec.strategy == "domain_shift_pc":
        order = np.argsort(-domain_shift)
    elif spec.strategy == "domain_shift_low_target_pc":
        order = np.argsort(-(domain_shift / (target_corr + 0.05)))
    elif spec.strategy == "species_centroid_pc":
        order = np.argsort(-species_score)
    else:
        raise ValueError(spec.strategy)

    selected = order[: spec.n_remove]
    Xnew = Xc.copy()
    Pnew = Pc.copy()
    for idx in selected:
        loading = loadings[idx]
        Xnew -= np.outer(Xnew @ loading, loading)
        Pnew -= np.outer(Pnew @ loading, loading)
    energy = float(np.sum(pca.explained_variance_ratio_[selected]))
    return (
        Xnew,
        Pnew,
        RemovalInfo(
            float(np.max(target_corr[selected])),
            energy,
            ",".join(str(int(i)) for i in selected),
        ),
    )


def fit_yj_ridge(X_train: np.ndarray, y: np.ndarray, X_pred: np.ndarray, n_components: int, alpha: float) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt)
    pred_t = model.predict(Z_pred)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def diagnostics(
    *,
    spec: OscSpec,
    path: Path,
    pred: np.ndarray,
    oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    bad_diff: np.ndarray,
    info: RemovalInfo,
    oof_infos: list[RemovalInfo],
) -> dict[str, object]:
    diff = pred - anchor_test
    oof_rmse = rmse(y, oof)
    fold = fold_delta_stats(y, oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    range_ratio = float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min()))
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "strategy": spec.strategy,
        "search_components": spec.search_components,
        "n_remove": spec.n_remove,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": oof_rmse,
        "direct_oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": fold["improved_count"],
        "direct_worst_fold_delta": fold["worst_delta"],
        "direct_fold_delta_std": fold["delta_std"],
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": range_ratio,
        "corr_diff_bad_alpha3000": safe_corr(diff, bad_diff),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
        "removed_target_corr_max": info.removed_target_corr_max,
        "removed_energy_ratio": info.removed_energy_ratio,
        "removed_indices": info.removed_indices,
        "oof_removed_target_corr_max": float(max(item.removed_target_corr_max for item in oof_infos)),
        "oof_removed_energy_ratio_mean": float(np.mean([item.removed_energy_ratio for item in oof_infos])),
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    if diff < 0.05:
        reasons.append("near_anchor")
    if diff > 0.25:
        reasons.append("anchor_diff_gt0p25")
    if float(row["anchor_diff_max_abs"]) > 1.50:
        reasons.append("max_diff_gt1p5")
    if float(row["max_abs_species_mean_shift"]) > 0.15:
        reasons.append("species_shift_gt0p15")
    if float(row["corr_diff_bad_alpha3000"]) > 0.50:
        reasons.append("bad_alpha3000_corr_gt0p50")
    if abs(float(row["decile_gap_top_minus_bottom"])) > 0.50:
        reasons.append("decile_gap_gt0p50")
    if int(row["top10_abs_max_species_count"]) >= 5:
        reasons.append("top10_species_concentration")
    if not (0.90 <= float(row["range_ratio_vs_anchor"]) <= 1.10):
        reasons.append("range_shift")
    if float(row["removed_target_corr_max"]) > 0.30:
        reasons.append("removed_target_corr_gt0p30")
    if float(row["removed_energy_ratio"]) > 0.15:
        reasons.append("removed_energy_gt0p15")
    if float(row["direct_oof_delta_vs_anchor"]) > 0.10:
        reasons.append("direct_oof_bad")
    if int(row["direct_improved_fold_count"]) < 2:
        reasons.append("direct_folds_lt2")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    return float(
        2.0 * max(float(row["corr_diff_bad_alpha3000"]), 0.0)
        + 0.8 * abs(float(row["decile_gap_top_minus_bottom"]))
        + 0.7 * abs(float(row["sample_order_corr"]))
        + 0.8 * float(row["max_abs_species_mean_shift"])
        + 0.08 * float(row["anchor_diff_max_abs"])
        + 0.7 * max(float(row["removed_target_corr_max"]) - 0.20, 0.0)
        + 0.7 * max(float(row["removed_energy_ratio"]) - 0.10, 0.0)
        + (0.5 if int(row["top10_abs_max_species_count"]) >= 5 else 0.0)
    )


def failed_row(spec: OscSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "strategy": spec.strategy,
        "search_components": spec.search_components,
        "n_remove": spec.n_remove,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
    }


def species_centroid_score(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = []
    for j in range(scores.shape[1]):
        values = scores[:, j]
        means = np.array([values[groups == group].mean() for group in np.unique(groups)])
        out.append(float(np.var(means) / (np.var(values) + 1e-12)))
    return np.asarray(out)


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    deltas: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for _, valid_idx in splitter.split(np.zeros((len(y), 1)), y, groups):
        deltas.append(rmse(y[valid_idx] - pred[valid_idx]) - rmse(y[valid_idx] - anchor_pred[valid_idx]))
    arr = np.asarray(deltas)
    return {
        "improved_count": int(np.sum(arr < 0)),
        "worst_delta": float(np.max(arr)),
        "delta_std": float(np.std(arr)),
    }


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def max_abs_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    values = np.asarray(a, dtype=float) if b is None else np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(math.sqrt(mean_squared_error(np.zeros_like(values), values)))


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
    direct_delta = float(row["direct_oof_delta_vs_anchor"])
    diff = float(row["anchor_diff_rmse"])
    return (
        penalty + float(row["public_failure_risk"]) + 0.6 * max(direct_delta, 0.0) + abs(diff - 0.10) * 0.5,
        direct_delta,
        float(row["anchor_diff_max_abs"]),
    )


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
