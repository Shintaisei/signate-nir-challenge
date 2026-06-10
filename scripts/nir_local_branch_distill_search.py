#!/usr/bin/env python3
"""Local-calibration branch diagnostics distilled into the current anchor."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = ROOT / "scripts" / "nir_operator_branch_distill_search.py"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
OUTPUT_ROOT = ROOT / "outputs" / "nir_local_branch_distill"
OP_CACHE = None


@dataclass(frozen=True)
class LocalSpec:
    name: str
    preprocess: str
    model: str
    k: int
    n_components: int
    distance_components: int = 10
    alpha: float = 1500.0


@dataclass(frozen=True)
class DistillSpec:
    shrink: float
    clip: float
    mean_center: bool = True


def main() -> None:
    op = load_operator_module()
    data = op.load_data()
    anchor = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    distills = [
        DistillSpec(0.01, 0.06),
        DistillSpec(0.02, 0.06),
        DistillSpec(0.03, 0.06),
        DistillSpec(0.02, 0.08),
        DistillSpec(0.03, 0.08),
        DistillSpec(0.04, 0.08),
    ]
    for spec in build_specs():
        print(f"branch {spec.name}", flush=True)
        branch_oof = make_local_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test, neighbor_species = fit_predict_local_branch(
            spec,
            data["X_train"],
            data["y"],
            data["X_test"],
            train_species=data["groups"],
            return_neighbor_species=True,
        )
        neighbor_rows.extend(neighbor_diagnostics(spec, data["test_species"], neighbor_species))
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = fit_beta(signal_oof, residual)
        signal_corr = safe_corr(signal_oof, residual)
        branch_rmse = rmse(data["y"], branch_oof)
        for distill in distills:
            correction_oof = make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_lbd_{spec.name}_b{tag(beta)}_"
                f"s{tag(distill.shrink)}_c{tag(distill.clip)}_mc1"
            )
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = diagnostics(
                name=name,
                spec=spec,
                distill=distill,
                pred=pred,
                anchor=anchor_test,
                y=data["y"],
                corrected_oof=corrected_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                branch_oof_rmse=branch_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "local_branch_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "local_branch_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "local_neighbor_species.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(neighbor_rows[0]))
        writer.writeheader()
        writer.writerows(neighbor_rows)

    print("\nTop candidates:")
    for row in rows_sorted[:25]:
        print(
            f"{row['experiment']}: branch={row['branch']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f}"
        )
    print(f"saved local-branch diagnostics: {out_dir}")


def load_operator_module():
    global OP_CACHE
    if OP_CACHE is not None:
        return OP_CACHE
    spec = importlib.util.spec_from_file_location("nir_operator_branch_distill_search", OPERATOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load operator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    OP_CACHE = module
    return module


def build_specs() -> list[LocalSpec]:
    specs: list[LocalSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for k in [40, 80, 120, 180]:
            for n_components in [2, 4]:
                specs.append(
                    LocalSpec(
                        name=f"local_pls_{preprocess}_k{k}_c{n_components}",
                        preprocess=preprocess,
                        model="pls_raw",
                        k=k,
                        n_components=n_components,
                    )
                )
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for k in [80, 120, 180]:
            specs.append(
                LocalSpec(
                    name=f"local_ridge_{preprocess}_k{k}_p10",
                    preprocess=preprocess,
                    model="ridge_yj",
                    k=k,
                    n_components=10,
                    alpha=1500.0,
                )
            )
    return specs


def make_local_oof(spec: LocalSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_local_branch(spec, X[train_idx], y[train_idx], X[valid_idx])
    return np.clip(pred, 0, None)


def fit_predict_local_branch(
    spec: LocalSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    *,
    train_species: np.ndarray | None = None,
    return_neighbor_species: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    X_train, X_pred = preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pred_s = scaler.transform(X_pred)
    n_dist = min(spec.distance_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
    pca = PCA(n_components=n_dist, random_state=42)
    Z_train = pca.fit_transform(X_train_s)
    Z_pred = pca.transform(X_pred_s)
    k = min(spec.k, len(y))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Z_train)
    neighbor_idx = nn.kneighbors(Z_pred, return_distance=False)

    preds = np.empty(X_pred.shape[0], dtype=float)
    neighbor_species: list[np.ndarray] = []
    for i, idx in enumerate(neighbor_idx):
        preds[i] = fit_one_local(spec, X_train[idx], y[idx], X_pred[i : i + 1])
        if train_species is not None:
            neighbor_species.append(train_species[idx])
    preds = np.clip(preds, 0, None)
    if return_neighbor_species:
        return preds, neighbor_species
    return preds


def fit_one_local(spec: LocalSpec, X_train: np.ndarray, y: np.ndarray, X_pred: np.ndarray) -> float:
    if spec.model == "pls_raw":
        n_comp = min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1])
        model = PLSRegression(n_components=n_comp, scale=True)
        model.fit(X_train, y)
        return float(model.predict(X_pred).ravel()[0])
    if spec.model == "ridge_yj":
        n_comp = min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        Z_train = pca.fit_transform(X_train)
        Z_pred = pca.transform(X_pred)
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        model = Ridge(alpha=spec.alpha)
        model.fit(Z_train, yt)
        pred_t = model.predict(Z_pred)
        return float(transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()[0])
    raise ValueError(spec.model)


def preprocess_pair(X_train: np.ndarray, X_pred: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    op = load_operator_module()
    return op.preprocess_pair(X_train, X_pred, name)


def neighbor_diagnostics(
    spec: LocalSpec,
    test_species: np.ndarray,
    neighbor_species: list[np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for group in sorted(np.unique(test_species)):
        mask = test_species == group
        stacked = np.concatenate([neighbor_species[i] for i in np.where(mask)[0]])
        values, counts = np.unique(stacked, return_counts=True)
        order = np.argsort(counts)[::-1]
        top_values = values[order[:3]]
        top_counts = counts[order[:3]]
        rows.append(
            {
                "branch": spec.name,
                "test_species": int(group),
                "n_test": int(np.sum(mask)),
                "top_train_species": "|".join(str(int(x)) for x in top_values),
                "top_train_share": "|".join(f"{c / len(stacked):.4f}" for c in top_counts),
            }
        )
    return rows


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(signal, residual) / denom), -1.0, 1.0))


def make_correction(beta: float, signal: np.ndarray, spec: DistillSpec) -> np.ndarray:
    delta = spec.shrink * beta * signal
    delta = np.clip(delta, -spec.clip, spec.clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return delta


def diagnostics(
    *,
    name: str,
    spec: LocalSpec,
    distill: DistillSpec,
    pred: np.ndarray,
    anchor: np.ndarray,
    y: np.ndarray,
    corrected_oof: np.ndarray,
    anchor_oof_rmse: float,
    branch_oof_rmse: float,
    beta: float,
    signal_corr: float,
    path: Path,
) -> dict[str, object]:
    diff = pred - anchor
    test_species = pd.read_csv(ROOT / "data" / "raw" / "test.csv", encoding="cp932")["species number"].to_numpy()
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    corrected_oof_rmse = rmse(y, corrected_oof)
    return {
        "experiment": name,
        "branch": spec.name,
        "preprocess": spec.preprocess,
        "model": spec.model,
        "k": spec.k,
        "n_components": spec.n_components,
        "shrink": distill.shrink,
        "clip": distill.clip,
        "beta": beta,
        "signal_residual_corr": signal_corr,
        "anchor_oof_rmse": anchor_oof_rmse,
        "branch_oof_rmse": branch_oof_rmse,
        "corrected_oof_rmse": corrected_oof_rmse,
        "oof_delta_vs_anchor": corrected_oof_rmse - anchor_oof_rmse,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "max_abs_species_mean_shift": max(abs(float(np.mean(diff[test_species == group]))) for group in np.unique(test_species)),
        "negative_count": int(np.sum(pred < 0)),
        "submission_path": str(path),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    rmse_diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    penalty = 0.0
    if rmse_diff < 0.03 or rmse_diff > 0.08:
        penalty += 5.0
    if max_abs > 0.15 or species > 0.03 or low > 0.08 or top > 0.08:
        penalty += 5.0
    if oof_delta > -0.01:
        penalty += 1.0
    return (penalty + max(oof_delta, 0.0) + abs(rmse_diff - 0.05) + species, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
