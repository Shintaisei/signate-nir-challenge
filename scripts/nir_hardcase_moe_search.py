#!/usr/bin/env python3
"""Hard/high detector and gated-expert search for the current NIR anchor.

This is intentionally separate from the conservative local-residual script.
The goal is to test broader but still guarded hypotheses:

* stronger hard/high/positive detectors,
* soft test-like density-ratio weighting for training samples,
* small calibrated expert corrections on selected regions.

The protected anchor is never replaced globally.  Candidate predictions are:

    anchor + gate * clipped(shrink * beta * expert_signal)

where beta is fit from OOF expert_signal vs anchor residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_base_shape_aug_guard_search as bsa
import nir_gated_local_correction_search as glc
import nir_hard_case_detector_diagnostics as hcd
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op
import nir_stage2_hardcase_correction_search as s2
import nir_stage2_local_residual_model_search as lres


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_hardcase_moe"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    kind: str
    density_power: float = 0.0
    hard_power: float = 0.0
    alpha: float = 3500.0


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    expert_name: str
    gate_mode: str
    frac: float
    shrink: float
    clip: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--experts", default=None, help="Comma-separated expert names to keep.")
    parser.add_argument("--gate-modes", default=None, help="Comma-separated gate modes to keep.")
    parser.add_argument("--fracs", default=None, help="Comma-separated gate fractions.")
    parser.add_argument("--shrinks", default=None, help="Comma-separated correction shrinks.")
    parser.add_argument("--clips", default=None, help="Comma-separated correction clips.")
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
    residual = data["y"] - anchor_oof
    residual_abs = np.abs(residual)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    print("building current feature space and density ratio", flush=True)
    F_train, F_test = make_current_features(data["X_train"], data["X_test"])
    domain = domain_scores(F_train, F_test, data["groups"])
    domain_rows = [domain_diagnostics(domain, data["y"], data["groups"])]

    print("building detector scores", flush=True)
    detector_train, detector_test, branch_cache = build_detector_feature_table(data, anchor_oof, anchor_test, domain)
    detector_scores = build_detector_scores(detector_train, detector_test, data, residual, anchor_oof)
    detector_rows = detector_diagnostics(detector_scores, residual, data["y"], data["groups"], test_species)

    expert_specs = build_experts()
    if args.experts:
        keep = set(parse_strs(args.experts))
        expert_specs = [spec for spec in expert_specs if spec.name in keep]

    expert_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, spec in enumerate(expert_specs, start=1):
        print(f"[expert {i}/{len(expert_specs)}] {spec.name}", flush=True)
        expert_cache[spec.name] = make_expert_signal(spec, data, anchor_oof, anchor_test, domain, residual_abs, branch_cache)

    candidate_specs = build_candidates(
        expert_specs,
        fracs=parse_floats(args.fracs),
        shrinks=parse_floats(args.shrinks),
        clips=parse_floats(args.clips),
    )
    if args.gate_modes:
        keep_gates = set(parse_strs(args.gate_modes))
        candidate_specs = [spec for spec in candidate_specs if spec.gate_mode in keep_gates]
    if args.max_candidates is not None:
        candidate_specs = candidate_specs[: args.max_candidates]

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(candidate_specs, start=1):
        print(f"[candidate {i}/{len(candidate_specs)}] {spec.name}", flush=True)
        signal_oof, signal_test = expert_cache[spec.expert_name]
        score_oof, score_test = gate_score(spec.gate_mode, detector_scores, signal_oof, signal_test, domain)
        gate_oof = top_fraction(score_oof, spec.frac)
        gate_test = top_fraction(score_test, spec.frac)
        beta = fit_beta(signal_oof, residual, gate_oof)
        corr_oof = nested_oof_correction(signal_oof, residual, gate_oof, data["groups"], spec.shrink, spec.clip)
        corr_test = np.where(gate_test, np.clip(spec.shrink * beta * signal_test, -spec.clip, spec.clip), 0.0)
        corrected_oof = np.clip(anchor_oof + corr_oof, 0, None)
        pred = np.clip(anchor_test + corr_test, 0, None)
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
            signal_oof=signal_oof,
            signal_test=signal_test,
            beta=beta,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            residual=residual,
            residual_abs=residual_abs,
            groups=data["groups"],
            test_species=test_species,
            refs=refs,
            domain=domain,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    write_csv(out_dir / "hardcase_moe_summary.csv", rows_sorted)
    pd.DataFrame(detector_rows).to_csv(out_dir / "detector_diagnostics.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(out_dir / "domain_diagnostics.csv", index=False)
    with (out_dir / "hardcase_moe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "domain_diagnostics": domain_rows,
                "detector_diagnostics": detector_rows,
                "candidate_summary": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDomain diagnostics:")
    print(pd.DataFrame(domain_rows).to_string(index=False))
    print("\nDetector diagnostics:")
    print(pd.DataFrame(detector_rows).sort_values(["label", "model", "frac"]).to_string(index=False))
    print("\nTop hardcase MoE candidates:")
    for row in rows_sorted[:50]:
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} changed={row['changed_count']} "
            f"sp={row['max_abs_species_mean_shift']:.4f} beta={row['beta']:.4f} "
            f"sigcorr={row['signal_residual_corr']:.4f} oof={row['oof_delta_vs_anchor']:.4f} "
            f"group={row['improved_group_count']}/{row['group_count']} hp={row['hard_precision_on_gate']:.3f} "
            f"pp={row['positive_precision_on_gate']:.3f}"
        )
    print(f"saved hardcase MoE diagnostics: {out_dir}")


def build_experts() -> list[ExpertSpec]:
    specs = [
        ExpertSpec("local_yj_k40", "local_yj_k40"),
        ExpertSpec("local_yj_k80", "local_yj_k80"),
        ExpertSpec("local_raw_k40", "local_raw_k40"),
        ExpertSpec("local_raw_k80", "local_raw_k80"),
        ExpertSpec("lres_wmean_k40", "lres_wmean_k40"),
        ExpertSpec("lres_wmean_k80", "lres_wmean_k80"),
        ExpertSpec("lres_pls1_k40", "lres_pls1_k40"),
    ]
    for density_power in [0.25, 0.50]:
        specs.append(ExpertSpec(f"ridge_yj_dr{tag(density_power)}", "ridge_yj", density_power=density_power))
        specs.append(ExpertSpec(f"huber_yj_dr{tag(density_power)}", "huber_yj", density_power=density_power))
        for hard_power in [0.25, 0.50]:
            specs.append(
                ExpertSpec(
                    f"ridge_yj_dr{tag(density_power)}_hard{tag(hard_power)}",
                    "ridge_yj",
                    density_power=density_power,
                    hard_power=hard_power,
                )
            )
    return specs


def build_candidates(
    experts: list[ExpertSpec],
    *,
    fracs: list[float] | None = None,
    shrinks: list[float] | None = None,
    clips: list[float] | None = None,
) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for expert in experts:
        if expert.kind.startswith("local_") or expert.kind.startswith("lres_"):
            shrink_values = shrinks or [0.0015, 0.002, 0.003, 0.005, 0.008, 0.010]
            clip_values = clips or [0.10, 0.15, 0.20]
        else:
            shrink_values = shrinks or [0.02, 0.04, 0.06, 0.08, 0.10]
            clip_values = clips or [0.15, 0.25, 0.35]
        for gate_mode in [
            "hard80",
            "hard80_rf",
            "poshard70",
            "poshard70_rf",
            "high80",
            "high80_rf",
            "under_high_rf",
            "hard_pos_impact",
            "hardrf_posrf_impact",
            "hard_domain_impact",
            "pos_domain_impact",
            "high_pos_impact",
            "underhighrf_pos_impact",
        ]:
            for frac in (fracs or [0.05, 0.075, 0.10]):
                for shrink in shrink_values:
                    for clip in clip_values:
                        specs.append(
                            CandidateSpec(
                                name=(
                                    f"nir_hmoe_{gate_mode}_top{tag(frac)}_{expert.name}_"
                                    f"s{tag(shrink)}_c{tag(clip)}"
                                ),
                                expert_name=expert.name,
                                gate_mode=gate_mode,
                                frac=frac,
                                shrink=shrink,
                                clip=clip,
                            )
                        )
    return specs


def make_current_features(X_train_raw: np.ndarray, X_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    scaler = StandardScaler()
    return scaler.fit_transform(F_train), scaler.transform(F_pred)


def domain_scores(F_train: np.ndarray, F_test: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray | float]:
    X = np.vstack([F_train, F_test])
    y = np.r_[np.zeros(len(F_train), dtype=int), np.ones(len(F_test), dtype=int)]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000, random_state=42)
    model.fit(Xs, y)
    proba = model.predict_proba(Xs)[:, 1]
    p_train = np.clip(proba[: len(F_train)], 1e-4, 1 - 1e-4)
    p_test = np.clip(proba[len(F_train) :], 1e-4, 1 - 1e-4)
    ratio = p_train / (1.0 - p_train)
    ratio = np.clip(ratio / np.mean(ratio), 0.2, 5.0)
    pseudo = np.empty(len(F_train), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(F_train, np.zeros(len(F_train)), groups):
        X_fold = np.vstack([F_train[train_idx], F_train[valid_idx]])
        y_fold = np.r_[np.zeros(len(train_idx), dtype=int), np.ones(len(valid_idx), dtype=int)]
        fs = StandardScaler()
        Xfs = fs.fit_transform(X_fold)
        fold_model = LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000, random_state=42)
        fold_model.fit(Xfs, y_fold)
        pseudo[valid_idx] = fold_model.predict_proba(fs.transform(F_train[valid_idx]))[:, 1]
    return {
        "train_score": p_train,
        "test_score": p_test,
        "train_ratio": ratio,
        "pseudo_oof_score": pseudo,
    }


def domain_diagnostics(domain: dict[str, np.ndarray | float], y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    w = np.asarray(domain["train_ratio"], dtype=float)
    species_weight = pd.Series(w).groupby(groups).sum()
    species_weight = species_weight / species_weight.sum()
    return {
        "ess_ratio": ess_ratio(w),
        "weight_y_corr": nl.safe_corr(w, y),
        "weight_abs_y_corr": nl.safe_corr(w, np.abs(y)),
        "weighted_species_share_max": float(species_weight.max()),
        "train_score_min": float(np.min(domain["train_score"])),
        "train_score_mean": float(np.mean(domain["train_score"])),
        "train_score_max": float(np.max(domain["train_score"])),
        "test_score_min": float(np.min(domain["test_score"])),
        "test_score_mean": float(np.mean(domain["test_score"])),
        "test_score_max": float(np.max(domain["test_score"])),
        "pseudo_score_mean": float(np.mean(domain["pseudo_oof_score"])),
    }


def build_detector_feature_table(
    data: dict[str, np.ndarray],
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    domain: dict[str, np.ndarray | float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    X_train_df, X_test_df, branch_cache = s2.build_detector_features(data, anchor_oof, anchor_test)
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df["domain_score"] = np.asarray(domain["train_score"], dtype=float)
    X_train_df["domain_ratio"] = np.asarray(domain["train_ratio"], dtype=float)
    X_train_df["domain_pseudo_oof"] = np.asarray(domain["pseudo_oof_score"], dtype=float)
    X_test_df["domain_score"] = np.asarray(domain["test_score"], dtype=float)
    X_test_df["domain_ratio"] = 1.0
    X_test_df["domain_pseudo_oof"] = np.asarray(domain["test_score"], dtype=float)
    return hcd.add_rank_features(X_train_df), hcd.add_rank_features(X_test_df), branch_cache


def build_detector_scores(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    data: dict[str, np.ndarray],
    residual: np.ndarray,
    anchor_oof: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    residual_abs = np.abs(residual)
    labels = {
        "hard80": residual_abs >= np.quantile(residual_abs, 0.80),
        "poshard70": (residual > 0) & (residual_abs >= np.quantile(residual_abs, 0.70)),
        "high80": data["y"] >= np.quantile(data["y"], 0.80),
        "under_high": (residual > 0) & (anchor_oof >= np.quantile(anchor_oof, 0.70)),
    }
    scores: dict[str, dict[str, np.ndarray]] = {}
    for label_name, label in labels.items():
        for model_name in ["logreg", "rf"]:
            key = f"{label_name}_{model_name}"
            scores[key] = {
                "oof": detector_oof_score(model_name, X_train_df, label, data["groups"]),
                "test": detector_full_score(model_name, X_train_df, label, X_test_df),
                "label": label,
            }
    return scores


def detector_oof_score(model_name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    score = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        model, scaler = fit_detector(model_name, X.iloc[train_idx], y[train_idx])
        score[valid_idx] = predict_detector(model, scaler, X.iloc[valid_idx])
    return score


def detector_full_score(model_name: str, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    model, scaler = fit_detector(model_name, X, y)
    return predict_detector(model, scaler, X_test)


def fit_detector(model_name: str, X: pd.DataFrame, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if model_name == "logreg":
        model = LogisticRegression(C=0.15, class_weight="balanced", max_iter=3000, random_state=42)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=18,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)
    model.fit(Xs, y)
    return model, scaler


def predict_detector(model, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def detector_diagnostics(
    scores: dict[str, dict[str, np.ndarray]],
    residual: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    residual_abs = np.abs(residual)
    hard80 = residual_abs >= np.quantile(residual_abs, 0.80)
    positive = residual > 0
    high80 = y >= np.quantile(y, 0.80)
    for key, item in scores.items():
        score = np.asarray(item["oof"], dtype=float)
        label = np.asarray(item["label"], dtype=bool)
        for frac in [0.05, 0.075, 0.10]:
            gate = top_fraction(score, frac)
            top_test = top_fraction(np.asarray(item["test"], dtype=float), frac)
            species_counts = pd.Series(test_species[top_test]).value_counts()
            rows.append(
                {
                    "detector": key,
                    "label": key.rsplit("_", 1)[0],
                    "model": key.rsplit("_", 1)[1],
                    "frac": frac,
                    "auc": safe_auc(label, score),
                    "ap": float(average_precision_score(label, score)),
                    "label_precision": float(label[gate].mean()),
                    "hard_precision": float(hard80[gate].mean()),
                    "positive_precision": float(positive[gate].mean()),
                    "high_precision": float(high80[gate].mean()),
                    "residual_lift": float(residual_abs[gate].mean() / max(residual_abs.mean(), 1e-12)),
                    "fold_precision_min": fold_precision_min(label, gate, groups),
                    "test_gate_count": int(top_test.sum()),
                    "test_species_max_count": int(species_counts.iloc[0]) if len(species_counts) else 0,
                }
            )
    return rows


def make_expert_signal(
    spec: ExpertSpec,
    data: dict[str, np.ndarray],
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    domain: dict[str, np.ndarray | float],
    residual_abs: np.ndarray,
    branch_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if spec.kind.startswith("local_"):
        local_oof, local_test = local_predictions(spec.kind, data, branch_cache)
        return local_oof - anchor_oof, local_test - anchor_test
    if spec.kind.startswith("lres_"):
        branch = residual_branch(spec.kind)
        signal_oof, _ = lres.make_residual_signal_oof(branch, data, anchor_oof, anchor_test, data["y"] - anchor_oof)
        signal_test, _ = lres.fit_predict_residual_signal(
            branch,
            data["X_train"],
            data["y"] - anchor_oof,
            data["X_test"],
            anchor_oof,
            anchor_test,
            data["X_test"],
            anchor_test,
        )
        return signal_oof, signal_test

    base_weight = np.ones(len(data["y"]), dtype=float)
    if spec.density_power > 0:
        base_weight *= np.power(np.asarray(domain["train_ratio"], dtype=float), spec.density_power)
    if spec.hard_power > 0:
        base_weight *= np.power(0.5 + rank01(residual_abs), spec.hard_power)
    base_weight = base_weight / np.mean(base_weight)

    oof = make_weighted_expert_oof(spec, data["X_train"], data["y"], data["groups"], base_weight)
    test = fit_predict_weighted_expert(spec, data["X_train"], data["y"], data["X_test"], base_weight)
    return np.clip(oof, 0, None) - anchor_oof, np.clip(test, 0, None) - anchor_test


def local_predictions(
    kind: str,
    data: dict[str, np.ndarray],
    branch_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key_map = {
        "local_raw_k40": "local_raw_k40_a1",
        "local_raw_k80": "local_raw_k80_a1",
        "local_yj_k40": "local_yj_k40_a10",
        "local_yj_k80": "local_yj_k80_a10",
    }
    if kind in key_map and key_map[kind] in branch_cache:
        return branch_cache[key_map[kind]]
    if kind == "local_raw_k40":
        local_spec = glc.LocalSpec("local_raw_k40", "raw", 40, 1.0)
    elif kind == "local_raw_k80":
        local_spec = glc.LocalSpec("local_raw_k80", "raw", 80, 1.0)
    elif kind == "local_yj_k40":
        local_spec = glc.LocalSpec("local_yj_k40", "yj", 40, 10.0)
    elif kind == "local_yj_k80":
        local_spec = glc.LocalSpec("local_yj_k80", "yj", 80, 10.0)
    else:
        raise ValueError(kind)
    oof, _ = glc.make_local_oof(local_spec, data)
    test, _ = glc.fit_predict_local(local_spec, data["X_train"], data["y"], data["X_test"])
    return oof, test


def residual_branch(kind: str) -> lres.ResidualBranchSpec:
    if kind == "lres_wmean_k40":
        return lres.ResidualBranchSpec("lres_wmean_k40", "current", "wmean", 40, anchor_weight=0.0)
    if kind == "lres_wmean_k80":
        return lres.ResidualBranchSpec("lres_wmean_k80", "current", "wmean", 80, anchor_weight=0.0)
    if kind == "lres_pls1_k40":
        return lres.ResidualBranchSpec("lres_pls1_k40", "current", "pls1", 40, anchor_weight=0.0)
    raise ValueError(kind)


def make_weighted_expert_oof(
    spec: ExpertSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_weighted_expert(
            spec,
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            weight[train_idx],
        )
    return pred


def fit_predict_weighted_expert(
    spec: ExpertSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    F_train, F_pred = make_current_features(X_train_raw, X_pred_raw)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    if spec.kind == "ridge_yj":
        model = Ridge(alpha=spec.alpha)
        model.fit(F_train, yt, sample_weight=weight)
        pred_t = model.predict(F_pred)
    elif spec.kind == "huber_yj":
        model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=5000)
        model.fit(F_train, yt, sample_weight=weight)
        pred_t = model.predict(F_pred)
    else:
        raise ValueError(spec.kind)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def gate_score(
    mode: str,
    scores: dict[str, dict[str, np.ndarray]],
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    domain: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    hard_oof = np.asarray(scores["hard80_logreg"]["oof"], dtype=float)
    hard_test = np.asarray(scores["hard80_logreg"]["test"], dtype=float)
    hard_rf_oof = np.asarray(scores["hard80_rf"]["oof"], dtype=float)
    hard_rf_test = np.asarray(scores["hard80_rf"]["test"], dtype=float)
    pos_oof = np.asarray(scores["poshard70_logreg"]["oof"], dtype=float)
    pos_test = np.asarray(scores["poshard70_logreg"]["test"], dtype=float)
    pos_rf_oof = np.asarray(scores["poshard70_rf"]["oof"], dtype=float)
    pos_rf_test = np.asarray(scores["poshard70_rf"]["test"], dtype=float)
    high_oof = np.asarray(scores["high80_logreg"]["oof"], dtype=float)
    high_test = np.asarray(scores["high80_logreg"]["test"], dtype=float)
    high_rf_oof = np.asarray(scores["high80_rf"]["oof"], dtype=float)
    high_rf_test = np.asarray(scores["high80_rf"]["test"], dtype=float)
    under_high_rf_oof = np.asarray(scores["under_high_rf"]["oof"], dtype=float)
    under_high_rf_test = np.asarray(scores["under_high_rf"]["test"], dtype=float)
    dom_oof = np.asarray(domain["train_score"], dtype=float)
    dom_test = np.asarray(domain["test_score"], dtype=float)
    if mode == "hard80":
        return hard_oof, hard_test
    if mode == "hard80_rf":
        return hard_rf_oof, hard_rf_test
    if mode == "poshard70":
        return pos_oof, pos_test
    if mode == "poshard70_rf":
        return pos_rf_oof, pos_rf_test
    if mode == "high80":
        return high_oof, high_test
    if mode == "high80_rf":
        return high_rf_oof, high_rf_test
    if mode == "under_high_rf":
        return under_high_rf_oof, under_high_rf_test
    impact_oof = rank01(np.abs(signal_oof))
    impact_test = rank01(np.abs(signal_test))
    pos_impact_oof = rank01(np.maximum(signal_oof, 0))
    pos_impact_test = rank01(np.maximum(signal_test, 0))
    if mode == "hard_pos_impact":
        return rank01(hard_oof) * rank01(pos_oof) * impact_oof, rank01(hard_test) * rank01(pos_test) * impact_test
    if mode == "hardrf_posrf_impact":
        return (
            rank01(hard_rf_oof) * rank01(pos_rf_oof) * impact_oof,
            rank01(hard_rf_test) * rank01(pos_rf_test) * impact_test,
        )
    if mode == "hard_domain_impact":
        return rank01(hard_oof) * rank01(dom_oof) * impact_oof, rank01(hard_test) * rank01(dom_test) * impact_test
    if mode == "pos_domain_impact":
        return rank01(pos_oof) * rank01(dom_oof) * pos_impact_oof, rank01(pos_test) * rank01(dom_test) * pos_impact_test
    if mode == "high_pos_impact":
        return rank01(high_oof) * rank01(pos_oof) * pos_impact_oof, rank01(high_test) * rank01(pos_test) * pos_impact_test
    if mode == "underhighrf_pos_impact":
        return (
            rank01(under_high_rf_oof) * rank01(pos_oof) * pos_impact_oof,
            rank01(under_high_rf_test) * rank01(pos_test) * pos_impact_test,
        )
    raise ValueError(mode)


def nested_oof_correction(
    signal: np.ndarray,
    residual: np.ndarray,
    gate: np.ndarray,
    groups: np.ndarray,
    shrink: float,
    clip: float,
) -> np.ndarray:
    corr = np.zeros_like(signal, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(np.zeros((len(groups), 1)), residual, groups):
        beta = fit_beta(signal[train_idx], residual[train_idx], gate[train_idx])
        raw = shrink * beta * signal[valid_idx]
        corr[valid_idx] = np.where(gate[valid_idx], np.clip(raw, -clip, clip), 0.0)
    return corr


def fit_beta(signal: np.ndarray, residual: np.ndarray, gate: np.ndarray) -> float:
    x = signal[gate]
    y = residual[gate]
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    beta = float(np.dot(x, y) / denom)
    return float(np.clip(beta, -2.0, 2.0))


def diagnostics(
    *,
    spec: CandidateSpec,
    path: Path,
    pred: np.ndarray,
    corrected_oof: np.ndarray,
    corr_test: np.ndarray,
    corr_oof: np.ndarray,
    gate_test: np.ndarray,
    gate_oof: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    beta: float,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    residual: np.ndarray,
    residual_abs: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
    domain: dict[str, np.ndarray | float],
) -> dict[str, object]:
    diff = pred - anchor_test
    rmse_oof = rmse(y, corrected_oof)
    fold = fold_delta_stats(y, corrected_oof, anchor_oof, groups)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_species = pd.Series(test_species[np.abs(diff) > 1e-12]).value_counts()
    gate_species = pd.Series(test_species[gate_test]).value_counts()
    bad_ref = refs.get("bad_alpha3000")
    row: dict[str, object] = {
        "experiment": spec.name,
        "expert": spec.expert_name,
        "gate_mode": spec.gate_mode,
        "uses_domain": ("dr" in spec.expert_name) or ("domain" in spec.gate_mode),
        "frac": spec.frac,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "beta": beta,
        "submission_path": str(path),
        "changed_count": int(np.sum(np.abs(diff) > 1e-12)),
        "gate_test_count": int(gate_test.sum()),
        "gate_oof_count": int(gate_oof.sum()),
        "negative_count": int((pred < 0).sum()),
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std_ratio_vs_anchor": float(np.std(pred) / max(np.std(anchor_test), 1e-12)),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / max(np.max(anchor_test) - np.min(anchor_test), 1e-12)),
        "anchor_diff_rmse": rmse(pred, anchor_test),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "max_abs_correction": float(np.max(np.abs(corr_test))) if len(corr_test) else 0.0,
        "mean_abs_correction_on_gate": float(np.mean(np.abs(corr_test[gate_test]))) if gate_test.any() else 0.0,
        "correction_std_on_gate": float(np.std(corr_test[gate_test])) if gate_test.any() else 0.0,
        "positive_correction_frac_on_gate": float(np.mean(corr_test[gate_test] > 0)) if gate_test.any() else 0.0,
        "clip_saturation_frac_on_gate": float(np.mean(np.abs(corr_test[gate_test]) >= spec.clip - 1e-12)) if gate_test.any() else 0.0,
        "max_abs_species_mean_shift": max_species_shift(diff, test_species),
        "corrected_species_max_count": int(corrected_species.iloc[0]) if len(corrected_species) else 0,
        "top10_abs_max_species_count": int(top_species.iloc[0]) if len(top_species) else 0,
        "gate_species_max_count": int(gate_species.iloc[0]) if len(gate_species) else 0,
        "signal_residual_corr": nl.safe_corr(signal_oof, residual),
        "signal_residual_corr_all": nl.safe_corr(np.r_[signal_oof, signal_test], np.r_[residual, np.zeros_like(signal_test)]),
        "hard_precision_on_gate": float((residual_abs >= np.quantile(residual_abs, 0.80))[gate_oof].mean()) if gate_oof.any() else 0.0,
        "positive_precision_on_gate": float((residual > 0)[gate_oof].mean()) if gate_oof.any() else 0.0,
        "high_precision_on_gate": float((y >= np.quantile(y, 0.80))[gate_oof].mean()) if gate_oof.any() else 0.0,
        "residual_lift_on_gate": float(residual_abs[gate_oof].mean() / max(residual_abs.mean(), 1e-12)) if gate_oof.any() else 0.0,
        "anchor_oof_rmse": anchor_oof_rmse,
        "oof_rmse": rmse_oof,
        "oof_delta_vs_anchor": rmse_oof - anchor_oof_rmse,
        "improved_fold_count": fold["improved_count"],
        "improved_group_count": fold["improved_count"],
        "group_count": fold["group_count"],
        "improved_group_ratio": fold["improved_ratio"],
        "worst_fold_delta": fold["worst_delta"],
        "fold_delta_std": fold["delta_std"],
        "domain_ess_ratio": ess_ratio(np.asarray(domain["train_ratio"], dtype=float)),
        "domain_weight_y_corr": nl.safe_corr(np.asarray(domain["train_ratio"], dtype=float), y),
        "corr_diff_bad_alpha3000": nl.safe_corr(diff, bad_ref) if bad_ref is not None else 0.0,
        "rmse_diff_bad_alpha3000": rmse(diff, bad_ref) if bad_ref is not None else 0.0,
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else ""
    row["submit_gate"] = "pass" if not reasons else "reject"
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if float(row["anchor_diff_rmse"]) < 0.015:
        reasons.append("near_anchor")
    if float(row["anchor_diff_rmse"]) > 0.08:
        reasons.append("too_far")
    if float(row["anchor_diff_max_abs"]) > 0.35:
        reasons.append("max_diff_high")
    if float(row["clip_saturation_frac_on_gate"]) > 0.05:
        reasons.append("clip_saturation")
    if float(row["correction_std_on_gate"]) < 0.010:
        reasons.append("low_correction_var")
    if float(row["max_abs_species_mean_shift"]) > 0.03:
        reasons.append("species_shift")
    if int(row["corrected_species_max_count"]) > 12:
        reasons.append("species_concentration")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if float(row["oof_delta_vs_anchor"]) >= -0.003:
        reasons.append("weak_oof")
    if float(row["improved_group_ratio"]) < 0.55:
        reasons.append("weak_groups")
    if float(row["worst_fold_delta"]) > 0.05:
        reasons.append("bad_worst_fold")
    if float(row["signal_residual_corr"]) <= 0.05:
        reasons.append("weak_signal_corr")
    if abs(float(row["corr_diff_bad_alpha3000"])) > 0.30:
        reasons.append("bad_alpha_corr")
    if not (0.98 <= float(row["pred_std_ratio_vs_anchor"]) <= 1.02):
        reasons.append("std_ratio_shift")
    if not (0.98 <= float(row["range_ratio_vs_anchor"]) <= 1.02):
        reasons.append("range_ratio_shift")
    if bool(row["uses_domain"]) and float(row["domain_ess_ratio"]) < 0.75:
        reasons.append("low_domain_ess")
    if bool(row["uses_domain"]) and abs(float(row["domain_weight_y_corr"])) > 0.20:
        reasons.append("domain_weight_y_corr")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    risk = 0.0
    risk += 2.0 * max(0.0, float(row["anchor_diff_rmse"]) - 0.025)
    risk += 1.0 * max(0.0, float(row["anchor_diff_max_abs"]) - 0.18)
    risk += 4.0 * max(0.0, float(row["max_abs_species_mean_shift"]) - 0.012)
    risk += 0.03 * max(0, int(row["corrected_species_max_count"]) - 8)
    risk += 0.04 * max(0, int(row["top10_abs_max_species_count"]) - 3)
    risk += 0.50 * float(row["clip_saturation_frac_on_gate"])
    risk += 0.20 if float(row["positive_correction_frac_on_gate"]) in {0.0, 1.0} else 0.0
    risk += 0.10 * max(0.0, abs(float(row["corr_diff_bad_alpha3000"])) - 0.20)
    risk += 0.20 * max(0.0, 0.75 - float(row["domain_ess_ratio"]))
    risk += 0.20 * max(0.0, abs(float(row["domain_weight_y_corr"])) - 0.20)
    return float(risk)


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    pass_penalty = 0 if row["submit_gate"] == "pass" else 1
    return (
        pass_penalty,
        float(row["public_failure_risk"]) - min(0.03, -float(row["oof_delta_vs_anchor"])),
        -float(row["residual_lift_on_gate"]),
        float(row["anchor_diff_rmse"]),
    )


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(len(score) * frac)))
    idx = np.argsort(score)[-k:]
    gate = np.zeros(len(score), dtype=bool)
    gate[idx] = True
    return gate


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def ess_ratio(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float((np.sum(w) ** 2) / max(np.sum(w * w) * len(w), 1e-12))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def fold_precision_min(label: np.ndarray, gate: np.ndarray, groups: np.ndarray) -> float:
    vals: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        if np.any(gate[mask]):
            vals.append(float(label[mask][gate[mask]].mean()))
    return float(min(vals)) if vals else 0.0


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    deltas: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        if np.sum(mask) == 0:
            continue
        deltas.append(rmse(y[mask], pred[mask]) - rmse(y[mask], anchor[mask]))
    return {
        "improved_count": int(sum(d < 0 for d in deltas)),
        "group_count": int(len(deltas)),
        "improved_ratio": float(sum(d < 0 for d in deltas) / max(len(deltas), 1)),
        "worst_delta": float(max(deltas)) if deltas else 0.0,
        "delta_std": float(np.std(deltas)) if deltas else 0.0,
    }


def max_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    values = []
    for sp in np.unique(species):
        mask = species == sp
        values.append(abs(float(np.mean(diff[mask]))))
    return float(max(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def tag(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
        f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} oof={row['oof_delta_vs_anchor']:.4f} "
        f"group={row['improved_group_count']}/{row['group_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
