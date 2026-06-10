#!/usr/bin/env python3
"""Post-public hard-case gate search after Slot1/Slot2 results.

Public feedback:
* Slot1 improved: impact_abs top5 + local residual weighted mean k40.
* Slot2 worsened vs Slot1: high80_rf top6.5 + lres PLS1.

This script keeps the current base anchor fixed and searches only conservative
variants that either:
* intersect Slot1 with high80_rf,
* slightly tune Slot1 correction strength,
* or test high80_rf with safer wmean residual signals.
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
from sklearn.metrics import mean_squared_error

import nir_base_shape_aug_guard_search as bsa
import nir_hardcase_moe_search as hmoe
import nir_hard_case_detector_diagnostics as hcd
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op
import nir_stage2_hardcase_correction_search as s2
import nir_stage2_local_residual_model_search as lres


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_post_public_gate"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BASE_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"
SLOT1 = SUBMISSION_DIR / "nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv"
SLOT2 = SUBMISSION_DIR / "nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    gate_mode: str
    signal_name: str
    slot1_frac: float
    high_frac: float
    final_frac: float
    shrink: float
    clip: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    data = op.load_data()
    base_df = pd.read_csv(BASE_ANCHOR, header=None, names=["id", "pred"])
    slot1_df = pd.read_csv(SLOT1, header=None, names=["id", "pred"])
    slot2_df = pd.read_csv(SLOT2, header=None, names=["id", "pred"])
    if not np.array_equal(data["test_ids"], base_df["id"].to_numpy()):
        raise ValueError("base anchor sample order mismatch")
    if not np.array_equal(data["test_ids"], slot1_df["id"].to_numpy()):
        raise ValueError("slot1 sample order mismatch")
    if not np.array_equal(data["test_ids"], slot2_df["id"].to_numpy()):
        raise ValueError("slot2 sample order mismatch")
    base_test = base_df["pred"].to_numpy(float)
    slot1_test = slot1_df["pred"].to_numpy(float)
    slot2_test = slot2_df["pred"].to_numpy(float)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    refs = nl.load_reference_diffs(base_test)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building base anchor OOF", flush=True)
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
    base_oof = bsa.make_nested_affine_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    residual = data["y"] - base_oof
    residual_abs = np.abs(residual)
    base_oof_rmse = rmse(data["y"], base_oof)
    print(f"base_oof_rmse={base_oof_rmse:.6f}", flush=True)

    print("building Slot1 and high80_rf scores", flush=True)
    detector_train, detector_test, _ = s2.build_detector_features(data, base_oof, base_test)
    hard_label = residual_abs >= np.quantile(residual_abs, 0.80)
    detector_oof = hcd.detector_oof_score("logreg", detector_train, hard_label, data["groups"])
    detector_test_score = hcd.detector_full_score("logreg", detector_train, hard_label, detector_test)
    signed_oof = hcd.detector_oof_score("logreg", detector_train, residual > 0, data["groups"])
    signed_test = hcd.detector_full_score("logreg", detector_train, residual > 0, detector_test)

    F_train, F_test = hmoe.make_current_features(data["X_train"], data["X_test"])
    domain = hmoe.domain_scores(F_train, F_test, data["groups"])
    moe_train, moe_test, _ = hmoe.build_detector_feature_table(data, base_oof, base_test, domain)
    moe_scores = hmoe.build_detector_scores(moe_train, moe_test, data, residual, base_oof)
    high_oof = np.asarray(moe_scores["high80_rf"]["oof"], dtype=float)
    high_test = np.asarray(moe_scores["high80_rf"]["test"], dtype=float)

    signals = build_signals(data, base_oof, base_test, residual)
    slot1_signal_oof, slot1_signal_test = signals["wmean_k40"]
    slot1_score_oof = rank01(detector_oof) * rank01(np.abs(slot1_signal_oof))
    slot1_score_test = rank01(detector_test_score) * rank01(np.abs(slot1_signal_test))

    slot1_repro = make_prediction(
        slot1_score_oof,
        slot1_score_test,
        slot1_signal_oof,
        slot1_signal_test,
        base_oof,
        base_test,
        residual,
        data["groups"],
        frac=0.05,
        shrink=0.003,
        clip=0.2,
    )[1]
    print(f"slot1_repro_test_diff={rmse(slot1_repro, slot1_test):.8f}", flush=True)

    specs = build_specs()
    if args.max_candidates is not None:
        specs = specs[: args.max_candidates]

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        signal_oof, signal_test = signals[spec.signal_name]
        score_oof, score_test, gate_oof, gate_test = build_gate(
            spec,
            slot1_score_oof,
            slot1_score_test,
            high_oof,
            high_test,
        )
        corr_oof, pred = make_prediction(
            score_oof,
            score_test,
            signal_oof,
            signal_test,
            base_oof,
            base_test,
            residual,
            data["groups"],
            frac=spec.final_frac,
            shrink=spec.shrink,
            clip=spec.clip,
            gate_oof=gate_oof,
            gate_test=gate_test,
        )
        corrected_oof = np.clip(base_oof + corr_oof, 0, None)
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            corrected_oof=corrected_oof,
            corr_oof=corr_oof,
            base_test=base_test,
            slot1_test=slot1_test,
            slot2_test=slot2_test,
            base_oof=base_oof,
            base_oof_rmse=base_oof_rmse,
            y=data["y"],
            residual=residual,
            residual_abs=residual_abs,
            groups=data["groups"],
            gate_oof=np.abs(corr_oof) > 1e-12,
            test_species=test_species,
            refs=refs,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    write_csv(out_dir / "post_public_gate_summary.csv", rows_sorted)
    with (out_dir / "post_public_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop post-public candidates:")
    for row in rows_sorted[:60]:
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} base_diff={row['base_diff_rmse']:.4f} "
            f"slot1_diff={row['slot1_diff_rmse']:.4f} max={row['base_diff_max_abs']:.4f} "
            f"changed={row['changed_count']} s2only={row['slot2_only_changed_count']} "
            f"sp={row['max_abs_species_mean_shift']:.4f} oof={row['oof_delta_vs_base']:.4f} "
            f"group={row['improved_group_count']}/{row['group_count']}"
        )
    print(f"saved post-public diagnostics: {out_dir}")


def build_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for signal_name in ["wmean_k40", "wmean_k80", "pls1_k40"]:
        for gate_mode in ["slot1", "slot1_high_intersection", "slot1_high_product", "high_top5"]:
            if gate_mode == "slot1":
                final_fracs = [0.05]
                high_fracs = [0.05]
            elif gate_mode == "high_top5":
                final_fracs = [0.05]
                high_fracs = [0.05]
            elif gate_mode == "slot1_high_intersection":
                final_fracs = [1.0]
                high_fracs = [0.05, 0.06, 0.065, 0.075]
            else:
                final_fracs = [0.04, 0.05, 0.055, 0.06]
                high_fracs = [0.05, 0.06, 0.065]
            for high_frac in high_fracs:
                for final_frac in final_fracs:
                    for shrink in [0.0025, 0.003, 0.0033, 0.0035, 0.004, 0.005]:
                        for clip in [0.12, 0.15, 0.18]:
                            if signal_name == "pls1_k40" and shrink > 0.0035:
                                continue
                            specs.append(
                                CandidateSpec(
                                    name=(
                                        f"nir_ppg_{gate_mode}_{signal_name}_"
                                        f"hf{tag(high_frac)}_ff{tag(final_frac)}_"
                                        f"s{tag(shrink)}_c{tag(clip)}"
                                    ),
                                    gate_mode=gate_mode,
                                    signal_name=signal_name,
                                    slot1_frac=0.05,
                                    high_frac=high_frac,
                                    final_frac=final_frac,
                                    shrink=shrink,
                                    clip=clip,
                                )
                            )
    return specs


def build_signals(
    data: dict[str, np.ndarray],
    base_oof: np.ndarray,
    base_test: np.ndarray,
    residual: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    specs = {
        "wmean_k40": lres.ResidualBranchSpec("wmean_k40", "current", "wmean", 40, anchor_weight=0.0),
        "wmean_k80": lres.ResidualBranchSpec("wmean_k80", "current", "wmean", 80, anchor_weight=0.0),
        "pls1_k40": lres.ResidualBranchSpec("pls1_k40", "current", "pls1", 40, anchor_weight=0.0),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, spec in specs.items():
        oof, _ = lres.make_residual_signal_oof(spec, data, base_oof, base_test, residual)
        test, _ = lres.fit_predict_residual_signal(
            spec,
            data["X_train"],
            residual,
            data["X_test"],
            base_oof,
            base_test,
            data["X_test"],
            base_test,
        )
        out[name] = (oof, test)
    return out


def build_gate(
    spec: CandidateSpec,
    slot1_oof: np.ndarray,
    slot1_test: np.ndarray,
    high_oof: np.ndarray,
    high_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    slot_gate_oof = top_fraction(slot1_oof, spec.slot1_frac)
    slot_gate_test = top_fraction(slot1_test, spec.slot1_frac)
    high_gate_oof = top_fraction(high_oof, spec.high_frac)
    high_gate_test = top_fraction(high_test, spec.high_frac)
    if spec.gate_mode == "slot1":
        return slot1_oof, slot1_test, slot_gate_oof, slot_gate_test
    if spec.gate_mode == "high_top5":
        return high_oof, high_test, high_gate_oof, high_gate_test
    if spec.gate_mode == "slot1_high_intersection":
        return slot1_oof * high_oof, slot1_test * high_test, slot_gate_oof & high_gate_oof, slot_gate_test & high_gate_test
    if spec.gate_mode == "slot1_high_product":
        return rank01(slot1_oof) * rank01(high_oof), rank01(slot1_test) * rank01(high_test), None, None
    raise ValueError(spec.gate_mode)


def make_prediction(
    score_oof: np.ndarray,
    score_test: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    *,
    frac: float,
    shrink: float,
    clip: float,
    gate_oof: np.ndarray | None = None,
    gate_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if gate_oof is None:
        gate_oof = top_fraction(score_oof, frac)
    if gate_test is None:
        gate_test = top_fraction(score_test, frac)
    corr_oof = np.zeros_like(base_oof, dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        raw = shrink * signal_oof[mask]
        corr_oof[mask] = np.where(gate_oof[mask], np.clip(raw, -clip, clip), 0.0)
    corr_test = np.where(gate_test, np.clip(shrink * signal_test, -clip, clip), 0.0)
    return corr_oof, np.clip(base_test + corr_test, 0, None)


def diagnostics(
    *,
    spec: CandidateSpec,
    path: Path,
    pred: np.ndarray,
    corrected_oof: np.ndarray,
    corr_oof: np.ndarray,
    base_test: np.ndarray,
    slot1_test: np.ndarray,
    slot2_test: np.ndarray,
    base_oof: np.ndarray,
    base_oof_rmse: float,
    y: np.ndarray,
    residual: np.ndarray,
    residual_abs: np.ndarray,
    groups: np.ndarray,
    gate_oof: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
) -> dict[str, object]:
    base_diff = pred - base_test
    slot1_diff = pred - slot1_test
    slot1_base_diff = slot1_test - base_test
    slot2_base_diff = slot2_test - base_test
    changed = np.abs(base_diff) > 1e-12
    slot1_changed = np.abs(slot1_base_diff) > 1e-12
    slot2_changed = np.abs(slot2_base_diff) > 1e-12
    slot2_only = slot2_changed & ~slot1_changed
    top10 = np.argsort(np.abs(base_diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    corrected_species = pd.Series(test_species[changed]).value_counts()
    fold = fold_delta_stats(y, corrected_oof, base_oof, groups)
    oof_rmse = rmse(y, corrected_oof)
    row: dict[str, object] = {
        "experiment": spec.name,
        "gate_mode": spec.gate_mode,
        "signal_name": spec.signal_name,
        "high_frac": spec.high_frac,
        "final_frac": spec.final_frac,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "submission_path": str(path),
        "changed_count": int(changed.sum()),
        "negative_count": int((pred < 0).sum()),
        "base_diff_rmse": rmse(pred, base_test),
        "base_diff_max_abs": float(np.max(np.abs(base_diff))),
        "slot1_diff_rmse": rmse(pred, slot1_test),
        "slot1_diff_max_abs": float(np.max(np.abs(slot1_diff))),
        "slot1_overlap_count": int(np.sum(changed & slot1_changed)),
        "slot1_only_lost_count": int(np.sum(slot1_changed & ~changed)),
        "slot2_only_changed_count": int(np.sum(changed & slot2_only)),
        "max_abs_species_mean_shift": max_species_shift(base_diff, test_species),
        "corrected_species_max_count": int(corrected_species.iloc[0]) if len(corrected_species) else 0,
        "top10_abs_max_species_count": int(top_species.iloc[0]) if len(top_species) else 0,
        "clip_saturation_frac": float(np.mean(np.abs(base_diff[changed]) >= spec.clip - 1e-12)) if changed.any() else 0.0,
        "positive_correction_frac": float(np.mean(base_diff[changed] > 0)) if changed.any() else 0.0,
        "corr_diff_bad_alpha3000": nl.safe_corr(base_diff, refs["bad_alpha3000"]),
        "oof_rmse": oof_rmse,
        "oof_delta_vs_base": oof_rmse - base_oof_rmse,
        "signal_residual_corr_on_gate": nl.safe_corr(corr_oof[gate_oof], residual[gate_oof]) if gate_oof.any() else 0.0,
        "hard_precision_on_gate": float((residual_abs >= np.quantile(residual_abs, 0.80))[gate_oof].mean()) if gate_oof.any() else 0.0,
        "positive_precision_on_gate": float((residual > 0)[gate_oof].mean()) if gate_oof.any() else 0.0,
        "residual_lift_on_gate": float(residual_abs[gate_oof].mean() / max(residual_abs.mean(), 1e-12)) if gate_oof.any() else 0.0,
        "improved_group_count": fold["improved_count"],
        "group_count": fold["group_count"],
        "worst_group_delta": fold["worst_delta"],
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
    if not (15 <= int(row["changed_count"]) <= 30):
        reasons.append("changed_count")
    if not (0.012 <= float(row["base_diff_rmse"]) <= 0.035):
        reasons.append("base_diff_range")
    if float(row["base_diff_max_abs"]) > 0.18:
        reasons.append("max_diff")
    if float(row["slot1_diff_rmse"]) > 0.018:
        reasons.append("too_far_from_slot1")
    if float(row["max_abs_species_mean_shift"]) > 0.015:
        reasons.append("species_shift")
    if int(row["corrected_species_max_count"]) > 9:
        reasons.append("species_concentration")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species")
    if float(row["clip_saturation_frac"]) > 0.0:
        reasons.append("clip_saturation")
    if float(row["oof_delta_vs_base"]) > -0.0048:
        reasons.append("weak_oof")
    if int(row["improved_group_count"]) < 8:
        reasons.append("weak_groups")
    if abs(float(row["corr_diff_bad_alpha3000"])) > 0.25:
        reasons.append("bad_alpha_corr")
    if int(row["slot2_only_changed_count"]) > 10:
        reasons.append("slot2_only_rows")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    risk = 0.0
    risk += max(0.0, float(row["base_diff_rmse"]) - 0.022) * 4.0
    risk += max(0.0, float(row["base_diff_max_abs"]) - 0.16) * 1.5
    risk += max(0.0, float(row["slot1_diff_rmse"]) - 0.010) * 4.0
    risk += max(0.0, float(row["max_abs_species_mean_shift"]) - 0.010) * 5.0
    risk += 0.04 * max(0, int(row["corrected_species_max_count"]) - 8)
    risk += 0.05 * max(0, int(row["top10_abs_max_species_count"]) - 3)
    risk += 0.03 * int(row["slot2_only_changed_count"])
    risk += 0.20 if float(row["positive_correction_frac"]) in {0.0, 1.0} else 0.0
    return float(risk)


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    pass_penalty = 0 if row["submit_gate"] == "pass" else 1
    return (
        pass_penalty,
        float(row["public_failure_risk"]) - min(0.02, -float(row["oof_delta_vs_base"])),
        float(row["slot1_diff_rmse"]),
    )


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(len(score) * frac)))
    idx = np.argsort(score)[-k:]
    gate = np.zeros(len(score), dtype=bool)
    gate[idx] = True
    return gate


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    deltas: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        deltas.append(rmse(y[mask], pred[mask]) - rmse(y[mask], anchor[mask]))
    return {
        "improved_count": int(sum(d < 0 for d in deltas)),
        "group_count": int(len(deltas)),
        "worst_delta": float(max(deltas)) if deltas else 0.0,
    }


def max_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return float(max(abs(float(np.mean(diff[species == sp]))) for sp in np.unique(species)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
        f"risk={row['public_failure_risk']:.4f} base_diff={row['base_diff_rmse']:.4f} "
        f"slot1_diff={row['slot1_diff_rmse']:.4f} oof={row['oof_delta_vs_base']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
