#!/usr/bin/env python3
"""Gated local residual correction for high-impact minority test regions.

Current best is kept as an anchor.  This script builds local kNN Ridge branches
and applies only tiny, OOF-learned corrections to selected high-risk rows.

The submit hypothesis is:

    global model is mostly right, but a small high-prediction / high-disagreement
    subset needs local calibration.

No candidate gates directly on species number or sample number.
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
OUTPUT_ROOT = ROOT / "outputs" / "nir_gated_local_correction"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"


@dataclass(frozen=True)
class LocalSpec:
    name: str
    feature_space: str = "current"
    local_target: str = "raw"
    k: int = 80
    alpha: float = 10.0
    distance_power: float = 1.0
    gate: str = "high_disagree"
    high_q: float = 0.90
    disagree_q: float = 0.90
    iso_q: float = 0.90
    shrink: float = 0.04
    clip: float = 0.20
    max_gate_frac: float = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
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
    anchor_spec = bsa.AugSpec(
        name="current_shape_anchor",
        pca_components=18,
        alpha=3500.0,
        shape_set="shape14",
        shape_scale=0.50,
        affine_shrink=0.12,
        affine_clip=0.40,
        mean_center=True,
    )
    anchor_oof = bsa.make_nested_affine_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    specs = build_specs()
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"local_specs={len(specs)}", flush=True)

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            local_oof, iso_oof = make_local_oof(spec, data)
            local_test, iso_test = fit_predict_local(
                spec,
                data["X_train"],
                data["y"],
                data["X_test"],
            )
            pred, corrected_oof, corr_test, corr_oof, gate_test, gate_oof, beta = apply_gated_correction(
                spec=spec,
                anchor_test=anchor_test,
                anchor_oof=anchor_oof,
                local_test=local_test,
                local_oof=local_oof,
                iso_test=iso_test,
                iso_oof=iso_oof,
                y=data["y"],
                groups=data["groups"],
            )
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
            corrected_oof=corrected_oof,
            corr_test=corr_test,
            corr_oof=corr_oof,
            gate_test=gate_test,
            gate_oof=gate_oof,
            beta=beta,
            local_oof=local_oof,
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
    with (out_dir / "gated_local_correction_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "gated_local_correction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop gated local correction candidates:")
    for row in rows_sorted[:60]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} corrmax={row['max_abs_correction']:.4f} "
            f"gfrac={row['gate_test_frac']:.3f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"beta={row['beta']:.4f} sigcorr={row['signal_residual_corr']:.4f} "
            f"oof={row['oof_delta_vs_anchor']:.4f} fold={row['improved_fold_count']}/5"
        )
    print(f"saved gated local correction diagnostics: {out_dir}")


def build_specs() -> list[LocalSpec]:
    specs: list[LocalSpec] = []
    for high_q in [0.90, 0.95]:
        for disagree_q in [0.85, 0.90, 0.95]:
            for gate in ["high", "high_disagree", "high_iso", "high_disagree_iso"]:
                for k in [40, 80]:
                    for clip in [0.10, 0.15, 0.20]:
                        name = (
                            f"nir_glc_focus_raw_k{k}_a1_{gate}_hq{tag(high_q)}_"
                            f"dq{tag(disagree_q)}_s0p02_c{tag(clip)}"
                        )
                        specs.append(
                            LocalSpec(
                                name=name,
                                feature_space="current",
                                local_target="raw",
                                k=k,
                                alpha=1.0,
                                gate=gate,
                                high_q=high_q,
                                disagree_q=disagree_q,
                                shrink=0.02,
                                clip=clip,
                                max_gate_frac=0.10,
                            )
                        )
    for feature_space in ["current"]:
        for local_target in ["raw", "yj"]:
            for k in [40, 80, 120]:
                for alpha in [1.0, 10.0, 100.0]:
                    for gate in ["high", "high_disagree", "high_iso", "disagree", "high_or_disagree"]:
                        for shrink in [0.02, 0.04, 0.06]:
                            for clip in [0.10, 0.20, 0.30]:
                                name = (
                                    f"nir_glc_{feature_space}_{local_target}_k{k}_a{tag(alpha)}_"
                                    f"{gate}_s{tag(shrink)}_c{tag(clip)}"
                                )
                                specs.append(
                                    LocalSpec(
                                        name=name,
                                        feature_space=feature_space,
                                        local_target=local_target,
                                        k=k,
                                        alpha=alpha,
                                        gate=gate,
                                        shrink=shrink,
                                        clip=clip,
                                    )
                                )
    return specs


def make_local_oof(spec: LocalSpec, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = data["X_train"]
    y = data["y"]
    groups = data["groups"]
    pred = np.empty(len(y), dtype=float)
    iso = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx], iso[valid_idx] = fit_predict_local(
            spec,
            X[train_idx],
            y[train_idx],
            X[valid_idx],
        )
    return np.clip(pred, 0, None), iso


def fit_predict_local(
    spec: LocalSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    F_train, F_pred = make_features(spec, X_train_raw, X_pred_raw)
    k = min(spec.k, len(y))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(F_train)
    distances, indices = nn.kneighbors(F_pred, return_distance=True)
    pred = np.empty(len(X_pred_raw), dtype=float)
    iso = distances.mean(axis=1)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        pred[i] = fit_one_local(spec, F_train[idx], y[idx], F_pred[i], dist)
    return pred, iso


def fit_one_local(spec: LocalSpec, F_local: np.ndarray, y_local: np.ndarray, f_query: np.ndarray, dist: np.ndarray) -> float:
    weight = 1.0 / np.power(dist + 1e-6, spec.distance_power)
    weight = weight / np.mean(weight)
    model = Ridge(alpha=spec.alpha)
    if spec.local_target == "raw":
        model.fit(F_local, y_local, sample_weight=weight)
        return float(model.predict(f_query.reshape(1, -1))[0])
    if spec.local_target == "yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y_local.reshape(-1, 1)).ravel()
        model.fit(F_local, yt, sample_weight=weight)
        pred_t = model.predict(f_query.reshape(1, -1))[0]
        return float(transformer.inverse_transform([[pred_t]])[0, 0])
    raise ValueError(spec.local_target)


def make_features(
    spec: LocalSpec,
    X_train_raw: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.feature_space == "current":
        X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, "sg9_snv")
        pca = PCA(n_components=min(18, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
        Z_train = pca.fit_transform(X_train)
        Z_pred = pca.transform(X_pred)
        S_train, S_pred = bsa.shape_pair(X_train, X_pred, "shape14")
        shape_scaler = StandardScaler()
        S_train = 0.5 * shape_scaler.fit_transform(S_train)
        S_pred = 0.5 * shape_scaler.transform(S_pred)
        F_train = np.hstack([Z_train, S_train])
        F_pred = np.hstack([Z_pred, S_pred])
    elif spec.feature_space == "msc_pca":
        X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, "msc_sg9")
        pca = PCA(n_components=min(20, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
        F_train = pca.fit_transform(X_train)
        F_pred = pca.transform(X_pred)
    else:
        raise ValueError(spec.feature_space)
    scaler = StandardScaler()
    return scaler.fit_transform(F_train), scaler.transform(F_pred)


def apply_gated_correction(
    *,
    spec: LocalSpec,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    local_test: np.ndarray,
    local_oof: np.ndarray,
    iso_test: np.ndarray,
    iso_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    signal_test = local_test - anchor_test
    signal_oof = local_oof - anchor_oof
    gate_test = make_gate(spec, anchor_test, signal_test, iso_test)
    gate_oof = make_gate(spec, anchor_oof, signal_oof, iso_oof)
    gate_test = limit_gate(gate_test, anchor_test, spec.max_gate_frac)
    gate_oof = limit_gate(gate_oof, anchor_oof, spec.max_gate_frac)
    beta = fit_beta(signal_oof, y - anchor_oof, gate_oof)
    corr_test = spec.shrink * beta * signal_test
    corr_oof = nested_oof_correction(spec, signal_oof, y - anchor_oof, gate_oof, groups)
    corr_test = np.where(gate_test, np.clip(corr_test, -spec.clip, spec.clip), 0.0)
    pred = np.clip(anchor_test + corr_test, 0, None)
    corrected_oof = np.clip(anchor_oof + corr_oof, 0, None)
    return pred, corrected_oof, corr_test, corr_oof, gate_test, gate_oof, beta


def nested_oof_correction(
    spec: LocalSpec,
    signal_oof: np.ndarray,
    residual: np.ndarray,
    gate_oof: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    corr = np.zeros_like(signal_oof, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(np.zeros((len(groups), 1)), residual, groups):
        beta = fit_beta(signal_oof[train_idx], residual[train_idx], gate_oof[train_idx])
        raw = spec.shrink * beta * signal_oof[valid_idx]
        corr[valid_idx] = np.where(gate_oof[valid_idx], np.clip(raw, -spec.clip, spec.clip), 0.0)
    return corr


def make_gate(spec: LocalSpec, anchor: np.ndarray, signal: np.ndarray, iso: np.ndarray) -> np.ndarray:
    high = anchor >= np.quantile(anchor, spec.high_q)
    disagree = np.abs(signal) >= np.quantile(np.abs(signal), spec.disagree_q)
    isolated = iso >= np.quantile(iso, spec.iso_q)
    if spec.gate == "high":
        return high
    if spec.gate == "disagree":
        return disagree
    if spec.gate == "iso":
        return isolated
    if spec.gate == "high_disagree":
        return high & disagree
    if spec.gate == "high_iso":
        return high & isolated
    if spec.gate == "high_or_disagree":
        return high | disagree
    if spec.gate == "high_disagree_iso":
        return high & disagree & isolated
    raise ValueError(spec.gate)


def limit_gate(gate: np.ndarray, priority: np.ndarray, max_frac: float) -> np.ndarray:
    max_count = max(1, int(math.floor(len(gate) * max_frac)))
    idx = np.flatnonzero(gate)
    if len(idx) <= max_count:
        return gate
    keep = idx[np.argsort(priority[idx])[-max_count:]]
    limited = np.zeros_like(gate, dtype=bool)
    limited[keep] = True
    return limited


def fit_beta(signal: np.ndarray, residual: np.ndarray, gate: np.ndarray) -> float:
    mask = gate & np.isfinite(signal) & np.isfinite(residual)
    if np.sum(mask) < 10:
        return 0.0
    x = signal[mask]
    y = residual[mask]
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def diagnostics(
    *,
    spec: LocalSpec,
    path: Path,
    pred: np.ndarray,
    corrected_oof: np.ndarray,
    corr_test: np.ndarray,
    corr_oof: np.ndarray,
    gate_test: np.ndarray,
    gate_oof: np.ndarray,
    beta: float,
    local_oof: np.ndarray,
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
    signal_oof = local_oof - anchor_oof
    residual = y - anchor_oof
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_species = pd.Series(test_species[gate_test]).value_counts() if np.any(gate_test) else pd.Series(dtype=int)
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "feature_space": spec.feature_space,
        "local_target": spec.local_target,
        "k": spec.k,
        "alpha": spec.alpha,
        "distance_power": spec.distance_power,
        "gate": spec.gate,
        "high_q": spec.high_q,
        "disagree_q": spec.disagree_q,
        "iso_q": spec.iso_q,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "beta": beta,
        "anchor_oof_rmse": anchor_oof_rmse,
        "oof_rmse": oof_rmse,
        "oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "improved_fold_count": fold["improved_count"],
        "worst_fold_delta": fold["worst_delta"],
        "fold_delta_std": fold["delta_std"],
        "signal_residual_corr": nl.safe_corr(signal_oof[gate_oof], residual[gate_oof]) if np.any(gate_oof) else 0.0,
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
        "mean_abs_correction_on_gate": float(np.mean(np.abs(corr_test[gate_test]))) if np.any(gate_test) else 0.0,
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
    if diff < 0.02:
        reasons.append("near_anchor")
    if diff > 0.12:
        reasons.append("anchor_diff_gt0p12")
    if float(row["max_abs_correction"]) > 0.35:
        reasons.append("correction_gt0p35")
    if float(row["anchor_diff_max_abs"]) > 0.45:
        reasons.append("max_diff_gt0p45")
    if float(row["max_abs_species_mean_shift"]) > 0.06:
        reasons.append("species_shift_gt0p06")
    if float(row["gate_test_frac"]) > 0.15:
        reasons.append("gate_frac_gt0p15")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if int(row["corrected_species_max_count"]) > 12:
        reasons.append("corrected_species_concentration")
    if float(row["oof_delta_vs_anchor"]) > 0.01:
        reasons.append("oof_bad")
    if int(row["improved_fold_count"]) < 3:
        reasons.append("folds_lt3")
    if float(row["beta"]) <= 0.0:
        reasons.append("beta_nonpositive")
    if float(row["signal_residual_corr"]) <= 0.0:
        reasons.append("signal_corr_nonpositive")
    if float(row.get("corr_diff_bad_alpha3000", 0.0)) > 0.35:
        reasons.append("bad_alpha3000_corr_gt0p35")
    if abs(float(row.get("decile_gap_top_minus_bottom", 0.0))) > 0.35:
        reasons.append("decile_gap_gt0p35")
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
        + (0.5 if int(row["top10_abs_max_species_count"]) > 4 else 0.0)
        + (0.3 if int(row["corrected_species_max_count"]) > 12 else 0.0)
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row["submit_gate"]) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.5 * max(float(row["oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.06),
        float(row["oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def failed_row(spec: LocalSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"corrmax={row['max_abs_correction']:.4f} gfrac={row['gate_test_frac']:.3f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} beta={row['beta']:.4f} "
        f"sigcorr={row['signal_residual_corr']:.4f} oof={row['oof_delta_vs_anchor']:.4f} "
        f"fold={row['improved_fold_count']}/5 reasons={row['reject_reasons']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
