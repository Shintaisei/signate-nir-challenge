#!/usr/bin/env python3
"""Stage-2 correction using a validated hard-case detector gate.

The detector gate is intentionally fixed to the most stable diagnostic found:

    LogisticRegression hard label = top20% anchor OOF residual, test gate top5%.

Then only tiny corrections are tested on that gate.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import nir_base_shape_aug_guard_search as bsa
import nir_gated_local_correction_search as glc
import nir_hard_case_detector_diagnostics as hcd
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_stage2_hardcase_correction"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"


@dataclass(frozen=True)
class Stage2Spec:
    name: str
    branch_target: str
    k: int
    alpha: float
    detector_frac: float
    shrink: float
    clip: float


def main() -> None:
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

    print("building anchor OOF", flush=True)
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
    residual_abs = np.abs(data["y"] - anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    print("building detector features", flush=True)
    detector_features_train, detector_features_test, branch_cache = build_detector_features(data, anchor_oof, anchor_test)
    hard_label = residual_abs >= np.quantile(residual_abs, 0.80)
    detector_oof = hcd.detector_oof_score("logreg", detector_features_train, hard_label, data["groups"])
    detector_test = hcd.detector_full_score("logreg", detector_features_train, hard_label, detector_features_test)

    detector_rows = detector_summary(detector_oof, detector_test, hard_label, residual_abs, data["groups"], test_species)
    specs = build_specs()
    rows: list[dict[str, object]] = []

    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        local_oof, local_test = get_branch_predictions(spec, data, branch_cache)
        signal_oof = local_oof - anchor_oof
        signal_test = local_test - anchor_test
        gate_oof = top_fraction(detector_oof, spec.detector_frac)
        gate_test = top_fraction(detector_test, spec.detector_frac)
        beta = glc.fit_beta(signal_oof, data["y"] - anchor_oof, gate_oof)
        corr_oof = nested_oof_correction(spec, signal_oof, data["y"] - anchor_oof, gate_oof, data["groups"])
        corr_test = np.where(
            gate_test,
            np.clip(spec.shrink * beta * signal_test, -spec.clip, spec.clip),
            0.0,
        )
        pred = np.clip(anchor_test + corr_test, 0, None)
        corrected_oof = np.clip(anchor_oof + corr_oof, 0, None)
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
            signal_oof=signal_oof,
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
    with (out_dir / "stage2_hardcase_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "stage2_hardcase_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "detector_summary": detector_rows,
                "candidate_summary": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    pd.DataFrame(detector_rows).to_csv(out_dir / "detector_gate_summary.csv", index=False)

    print("\nDetector gate summary:")
    print(pd.DataFrame(detector_rows).to_string(index=False))
    print("\nTop Stage2 hardcase candidates:")
    for row in rows_sorted[:40]:
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} corrmax={row['max_abs_correction']:.4f} "
            f"gfrac={row['gate_test_frac']:.3f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"beta={row['beta']:.4f} sigcorr={row['signal_residual_corr']:.4f} "
            f"oof={row['oof_delta_vs_anchor']:.4f} fold={row['improved_fold_count']}/5"
        )
    print(f"saved Stage2 hardcase diagnostics: {out_dir}")


def build_specs() -> list[Stage2Spec]:
    specs: list[Stage2Spec] = []
    for branch_target, k, alpha in [
        ("raw", 40, 1.0),
        ("raw", 80, 1.0),
        ("yj", 40, 10.0),
        ("yj", 80, 10.0),
    ]:
        for detector_frac in [0.05, 0.075, 0.10]:
            for shrink in [0.01, 0.02, 0.03]:
                for clip in [0.10, 0.15, 0.20]:
                    specs.append(
                        Stage2Spec(
                            name=(
                                f"nir_s2_logreg80_top{tag(detector_frac)}_"
                                f"{branch_target}_k{k}_a{tag(alpha)}_s{tag(shrink)}_c{tag(clip)}"
                            ),
                            branch_target=branch_target,
                            k=k,
                            alpha=alpha,
                            detector_frac=detector_frac,
                            shrink=shrink,
                            clip=clip,
                        )
                    )
    return specs


def build_detector_features(
    data: dict[str, np.ndarray],
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    branch_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    train_frames = [hcd.base_features(anchor_oof, None, prefix="anchor")]
    test_frames = [hcd.base_features(anchor_test, None, prefix="anchor")]
    for branch in [
        hcd.BranchDiag("local_raw_k40_a1", "raw", 40, 1.0),
        hcd.BranchDiag("local_raw_k80_a1", "raw", 80, 1.0),
        hcd.BranchDiag("local_yj_k40_a10", "yj", 40, 10.0),
        hcd.BranchDiag("local_yj_k80_a10", "yj", 80, 10.0),
    ]:
        spec = glc.LocalSpec(
            name=branch.name,
            local_target=branch.local_target,
            k=branch.k,
            alpha=branch.alpha,
        )
        local_oof, iso_oof = glc.make_local_oof(spec, data)
        local_test, iso_test = glc.fit_predict_local(spec, data["X_train"], data["y"], data["X_test"])
        branch_cache[branch.name] = (local_oof, local_test)
        train_frames.append(hcd.branch_features(branch.name, local_oof - anchor_oof, iso_oof, local_oof))
        test_frames.append(hcd.branch_features(branch.name, local_test - anchor_test, iso_test, local_test))
    return hcd.add_rank_features(pd.concat(train_frames, axis=1)), hcd.add_rank_features(pd.concat(test_frames, axis=1)), branch_cache


def get_branch_predictions(
    spec: Stage2Spec,
    data: dict[str, np.ndarray],
    branch_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if spec.branch_target == "raw" and spec.k == 40 and spec.alpha == 1.0:
        return branch_cache["local_raw_k40_a1"]
    if spec.branch_target == "raw" and spec.k == 80 and spec.alpha == 1.0:
        return branch_cache["local_raw_k80_a1"]
    if spec.branch_target == "yj" and spec.k == 40 and spec.alpha == 10.0:
        return branch_cache["local_yj_k40_a10"]
    if spec.branch_target == "yj" and spec.k == 80 and spec.alpha == 10.0:
        return branch_cache["local_yj_k80_a10"]
    local_spec = glc.LocalSpec(name="local", local_target=spec.branch_target, k=spec.k, alpha=spec.alpha)
    local_oof, _ = glc.make_local_oof(local_spec, data)
    local_test, _ = glc.fit_predict_local(local_spec, data["X_train"], data["y"], data["X_test"])
    return local_oof, local_test


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(len(score) * frac)))
    idx = np.argsort(score)[-k:]
    gate = np.zeros(len(score), dtype=bool)
    gate[idx] = True
    return gate


def nested_oof_correction(
    spec: Stage2Spec,
    signal_oof: np.ndarray,
    residual: np.ndarray,
    gate_oof: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    corr = np.zeros_like(signal_oof, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(np.zeros((len(groups), 1)), residual, groups):
        beta = glc.fit_beta(signal_oof[train_idx], residual[train_idx], gate_oof[train_idx])
        raw = spec.shrink * beta * signal_oof[valid_idx]
        corr[valid_idx] = np.where(gate_oof[valid_idx], np.clip(raw, -spec.clip, spec.clip), 0.0)
    return corr


def detector_summary(
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    hard_label: np.ndarray,
    residual_abs: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for frac in [0.05, 0.075, 0.10]:
        gate_oof = top_fraction(detector_oof, frac)
        gate_test = top_fraction(detector_test, frac)
        fold_hits = []
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for _, valid_idx in splitter.split(np.zeros((len(groups), 1)), hard_label, groups):
            selected = gate_oof[valid_idx]
            if np.any(selected):
                fold_hits.append(float(np.mean(hard_label[valid_idx][selected])))
        top_species = pd.Series(test_species[gate_test]).value_counts()
        rows.append(
            {
                "frac": frac,
                "oof_precision": float(np.mean(hard_label[gate_oof])),
                "oof_residual_lift": float(np.mean(residual_abs[gate_oof]) / np.mean(residual_abs)),
                "fold_precision_min": float(np.min(fold_hits)) if fold_hits else 0.0,
                "fold_precision_mean": float(np.mean(fold_hits)) if fold_hits else 0.0,
                "test_count": int(np.sum(gate_test)),
                "test_species_max_count": int(top_species.iloc[0]) if len(top_species) else 0,
                "test_species_counts": json.dumps(top_species.sort_index().to_dict(), ensure_ascii=False),
            }
        )
    return rows


def diagnostics(
    *,
    spec: Stage2Spec,
    path: Path,
    pred: np.ndarray,
    corrected_oof: np.ndarray,
    corr_test: np.ndarray,
    corr_oof: np.ndarray,
    gate_test: np.ndarray,
    gate_oof: np.ndarray,
    beta: float,
    signal_oof: np.ndarray,
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
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_species = pd.Series(test_species[gate_test]).value_counts()
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "branch_target": spec.branch_target,
        "k": spec.k,
        "alpha": spec.alpha,
        "detector_frac": spec.detector_frac,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "beta": beta,
        "anchor_oof_rmse": anchor_oof_rmse,
        "oof_rmse": oof_rmse,
        "oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "improved_fold_count": fold["improved_count"],
        "worst_fold_delta": fold["worst_delta"],
        "fold_delta_std": fold["delta_std"],
        "signal_residual_corr": nl.safe_corr(signal_oof[gate_oof], (y - anchor_oof)[gate_oof]) if np.any(gate_oof) else 0.0,
        "gate_test_count": int(np.sum(gate_test)),
        "gate_test_frac": float(np.mean(gate_test)),
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
    if diff < 0.015:
        reasons.append("near_anchor")
    if diff > 0.08:
        reasons.append("anchor_diff_gt0p08")
    if float(row["max_abs_correction"]) > 0.20:
        reasons.append("correction_gt0p20")
    if float(row["anchor_diff_max_abs"]) > 0.25:
        reasons.append("max_diff_gt0p25")
    if float(row["max_abs_species_mean_shift"]) > 0.04:
        reasons.append("species_shift_gt0p04")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if int(row["corrected_species_max_count"]) > 12:
        reasons.append("corrected_species_concentration")
    if float(row["oof_delta_vs_anchor"]) > 0.0:
        reasons.append("oof_not_improved")
    if int(row["improved_fold_count"]) < 3:
        reasons.append("folds_lt3")
    if float(row["beta"]) <= 0.0:
        reasons.append("beta_nonpositive")
    if float(row["signal_residual_corr"]) <= 0.0:
        reasons.append("signal_corr_nonpositive")
    if float(row.get("corr_diff_bad_alpha3000", 0.0)) > 0.35:
        reasons.append("bad_alpha3000_corr_gt0p35")
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
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(row["submit_gate"]) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.5 * max(float(row["oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.04),
        float(row["oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


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
