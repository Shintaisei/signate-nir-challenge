#!/usr/bin/env python3
"""Test-like golden subset branch distillation.

This tests a domain-adaptive golden subset idea: train samples close to the
unlabeled test spectra get selected/upweighted, then the resulting branch signal
is distilled into the current Public-best anchor.
"""

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
from sklearn.cluster import KMeans
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
OUTPUT_ROOT = ROOT / "outputs" / "nir_testlike_golden_distill"


@dataclass(frozen=True)
class TestLikeSpec:
    name: str
    preprocess: str
    model: str
    score_mode: str
    keep_frac: float
    n_components: int
    latent_components: int = 10
    n_clusters: int = 10
    min_keep: int = 30
    alpha: float = 3500.0


@dataclass(frozen=True)
class DistillSpec:
    shrink: float
    clip: float
    mean_center: bool = True


OP_CACHE = None


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
    selection_rows: list[dict[str, object]] = []
    for spec in build_specs():
        print(f"branch {spec.name}", flush=True)
        branch_oof, oof_selection = make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test, full_selection = fit_predict_testlike_branch(
            spec,
            data["X_train"],
            data["y"],
            data["X_test"],
            groups=data["groups"],
            return_selection=True,
        )
        selection_rows.extend(selection_diagnostics(spec, "full_test", full_selection, data["groups"]))
        selection_rows.extend(oof_selection)
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = fit_beta(signal_oof, residual)
        signal_corr = safe_corr(signal_oof, residual)
        branch_oof_rmse = rmse(data["y"], branch_oof)
        for distill in build_distills():
            correction_oof = make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_tlg_{spec.name}_b{tag(beta)}_"
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
                branch_oof_rmse=branch_oof_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
                full_selection=full_selection,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "testlike_golden_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "testlike_golden_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "testlike_selection_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)

    print("\nTop candidates:")
    for row in rows_sorted[:30]:
        print(
            f"{row['experiment']}: branch={row['branch']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f} keep={row['full_keep_count']}"
        )
    print(f"saved test-like golden diagnostics: {out_dir}")


def build_specs() -> list[TestLikeSpec]:
    specs: list[TestLikeSpec] = []
    for preprocess in ["msc_sg9", "sg9_snv"]:
        for score_mode in ["knn", "centroid", "cluster", "knn_cluster"]:
            for keep_frac in [0.35, 0.50, 0.65, 0.80]:
                for n_components in [3, 4, 5, 6]:
                    specs.append(
                        TestLikeSpec(
                            name=(
                                f"pls_raw_{preprocess}_{score_mode}_"
                                f"keep{pct_tag(keep_frac)}_c{n_components}"
                            ),
                            preprocess=preprocess,
                            model="pls_raw",
                            score_mode=score_mode,
                            keep_frac=keep_frac,
                            n_components=n_components,
                        )
                    )
        for score_mode in ["knn", "cluster", "knn_cluster"]:
            for keep_frac in [0.50, 0.65, 0.80]:
                specs.append(
                    TestLikeSpec(
                        name=f"ridge_yj_{preprocess}_{score_mode}_keep{pct_tag(keep_frac)}_p20",
                        preprocess=preprocess,
                        model="ridge_yj",
                        score_mode=score_mode,
                        keep_frac=keep_frac,
                        n_components=20,
                    )
                )
    return specs


def build_distills() -> list[DistillSpec]:
    return [
        DistillSpec(0.01, 0.06),
        DistillSpec(0.02, 0.08),
        DistillSpec(0.03, 0.08),
        DistillSpec(0.02, 0.10),
        DistillSpec(0.03, 0.10),
        DistillSpec(0.04, 0.10),
        DistillSpec(0.03, 0.12),
        DistillSpec(0.04, 0.12),
    ]


def make_branch_oof(
    spec: TestLikeSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    pred = np.empty(len(y), dtype=float)
    rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
        fold_pred, selection = fit_predict_testlike_branch(
            spec,
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            groups=groups[train_idx],
            return_selection=True,
        )
        pred[valid_idx] = fold_pred
        rows.extend(selection_diagnostics(spec, f"oof_fold{fold}", selection, groups[train_idx]))
    return np.clip(pred, 0, None), rows


def fit_predict_testlike_branch(
    spec: TestLikeSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pseudo_test_raw: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    return_selection: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    X_train, X_pseudo_test = preprocess_pair(X_train_raw, X_pseudo_test_raw, spec.preprocess)
    scores = test_likeness_scores(spec, X_train, X_pseudo_test)
    keep_mask = select_mask(scores, spec.keep_frac, spec.min_keep)
    if groups is not None and np.max(species_selection_share(groups, keep_mask)) > 0.75 and np.sum(keep_mask) > spec.min_keep:
        # Avoid extremely single-species subsets by adding the next best samples.
        keep_mask = diversify_by_species(scores, groups, keep_mask, min_per_species=3)
    pred = fit_predict_selected(spec, X_train, y, X_pseudo_test, keep_mask, scores)
    selection = {
        "scores": scores,
        "keep_mask": keep_mask,
    }
    if return_selection:
        return np.clip(pred, 0, None), selection
    return np.clip(pred, 0, None)


def test_likeness_scores(spec: TestLikeSpec, X_train: np.ndarray, X_test_like: np.ndarray) -> np.ndarray:
    Z_train, Z_test = latent_pair(X_train, X_test_like, spec.latent_components)
    parts: list[np.ndarray] = []
    if spec.score_mode in {"knn", "knn_cluster"}:
        nn = NearestNeighbors(n_neighbors=min(10, len(Z_test)), metric="euclidean")
        nn.fit(Z_test)
        distances = nn.kneighbors(Z_train, return_distance=True)[0]
        parts.append(rank01(-np.mean(distances, axis=1)))
    if spec.score_mode in {"centroid"}:
        centroid = np.mean(Z_test, axis=0)
        distance = np.linalg.norm(Z_train - centroid, axis=1)
        parts.append(rank01(-distance))
    if spec.score_mode in {"cluster", "knn_cluster"}:
        n_clusters = min(spec.n_clusters, max(2, len(Z_train) + len(Z_test) - 1))
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(np.vstack([Z_train, Z_test]))
        train_labels = labels[: len(Z_train)]
        test_labels = labels[len(Z_train) :]
        cluster_scores = np.zeros(len(Z_train), dtype=float)
        for label in np.unique(labels):
            train_count = int(np.sum(train_labels == label))
            test_count = int(np.sum(test_labels == label))
            share = test_count / max(train_count + test_count, 1)
            cluster_scores[train_labels == label] = share
        parts.append(rank01(cluster_scores))
    if not parts:
        raise ValueError(spec.score_mode)
    return np.mean(np.vstack(parts), axis=0)


def latent_pair(X_train: np.ndarray, X_test_like: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test_like)
    n = min(n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
    pca = PCA(n_components=n, random_state=42)
    Z_train = pca.fit_transform(X_train_s)
    Z_test = pca.transform(X_test_s)
    return Z_train, Z_test


def select_mask(scores: np.ndarray, keep_frac: float, min_keep: int) -> np.ndarray:
    n_keep = max(min_keep, int(round(len(scores) * keep_frac)))
    n_keep = min(n_keep, len(scores))
    order = np.argsort(scores, kind="mergesort")[::-1]
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def diversify_by_species(scores: np.ndarray, groups: np.ndarray, keep_mask: np.ndarray, min_per_species: int) -> np.ndarray:
    out = keep_mask.copy()
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        if np.sum(out[idx]) >= min_per_species:
            continue
        add_count = min(min_per_species - int(np.sum(out[idx])), len(idx))
        order = idx[np.argsort(scores[idx], kind="mergesort")[::-1]]
        out[order[:add_count]] = True
    return out


def species_selection_share(groups: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    kept = groups[keep_mask]
    if len(kept) == 0:
        return np.array([1.0])
    return np.array([np.mean(kept == group) for group in np.unique(kept)], dtype=float)


def fit_predict_selected(
    spec: TestLikeSpec,
    X_train: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    keep_mask: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    if np.sum(keep_mask) < spec.min_keep:
        keep_mask = select_mask(scores, 0.8, spec.min_keep)
    X_keep = X_train[keep_mask]
    y_keep = y[keep_mask]
    if spec.model == "pls_raw":
        n_comp = min(spec.n_components, X_keep.shape[0] - 1, X_keep.shape[1])
        model = PLSRegression(n_components=n_comp, scale=True)
        model.fit(X_keep, y_keep)
        return model.predict(X_pred).ravel()
    if spec.model == "ridge_yj":
        pca = PCA(n_components=min(spec.n_components, X_keep.shape[0] - 1, X_keep.shape[1]), random_state=42)
        Z_keep = pca.fit_transform(X_keep)
        Z_pred = pca.transform(X_pred)
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y_keep.reshape(-1, 1)).ravel()
        model = Ridge(alpha=spec.alpha)
        model.fit(Z_keep, yt)
        pred_t = model.predict(Z_pred)
        return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    raise ValueError(spec.model)


def preprocess_pair(X_train: np.ndarray, X_pred: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    op = load_operator_module()
    return op.preprocess_pair(X_train, X_pred, name)


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
    spec: TestLikeSpec,
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
    full_selection: dict[str, np.ndarray],
) -> dict[str, object]:
    diff = pred - anchor
    test_species = pd.read_csv(ROOT / "data" / "raw" / "test.csv", encoding="cp932")["species number"].to_numpy()
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    corrected_oof_rmse = rmse(y, corrected_oof)
    keep_mask = full_selection["keep_mask"]
    return {
        "experiment": name,
        "branch": spec.name,
        "preprocess": spec.preprocess,
        "model": spec.model,
        "score_mode": spec.score_mode,
        "keep_frac": spec.keep_frac,
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
        "full_keep_count": int(np.sum(keep_mask)),
        "full_max_species_keep_share": float(np.max(species_selection_share(load_train_groups(), keep_mask))),
        "submission_path": str(path),
    }


def selection_diagnostics(
    spec: TestLikeSpec,
    phase: str,
    selection: dict[str, np.ndarray],
    groups: np.ndarray,
) -> list[dict[str, object]]:
    keep_mask = selection["keep_mask"]
    scores = selection["scores"]
    rows = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append(
            {
                "branch": spec.name,
                "phase": phase,
                "species": int(group),
                "n": int(np.sum(mask)),
                "keep_count": int(np.sum(keep_mask[mask])),
                "keep_rate": float(np.mean(keep_mask[mask])),
                "score_mean": float(np.mean(scores[mask])),
                "score_median": float(np.median(scores[mask])),
            }
        )
    return rows


def load_train_groups() -> np.ndarray:
    return pd.read_csv(ROOT / "data" / "raw" / "train.csv", encoding="cp932")["species number"].to_numpy()


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    rmse_diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    keep_share = float(row["full_max_species_keep_share"])
    penalty = 0.0
    if rmse_diff < 0.03 or rmse_diff > 0.12:
        penalty += 5.0
    if max_abs > 0.18 or species > 0.04 or low > 0.10 or top > 0.10:
        penalty += 5.0
    if keep_share > 0.75:
        penalty += 2.0
    if oof_delta > -0.005:
        penalty += 1.0
    return (penalty + max(oof_delta, 0.0) + abs(rmse_diff - 0.07) + species, max_abs, species)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def pct_tag(value: float) -> str:
    return tag(value).replace("0p", "")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f} keep={row['full_keep_count']}"
    )


if __name__ == "__main__":
    main()
