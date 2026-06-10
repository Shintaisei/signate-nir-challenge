#!/usr/bin/env python3
"""Public-aware focused search after anchor rebuild saturation.

Lane A searches around the current public-best Yeo-Johnson Ridge anchor.
Lane B keeps a smaller set of direct alternatives so we do not only optimize
tiny residual corrections. Ranking is intentionally conservative and encodes
known Public failures from prior submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nir_anchor_rebuild_search as ar
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_anchor_focused_public"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-top", type=int, default=90, help="Number of affine variants to nested-evaluate.")
    parser.add_argument("--max-base-specs", type=int, default=None, help="Optional cap for smoke tests.")
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor nested OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = ar.rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")
    ar.assert_anchor_reproduction(data, anchor_test)

    base_specs = build_base_specs()
    if args.max_base_specs is not None:
        base_specs = base_specs[: args.max_base_specs]
    print(f"base_specs={len(base_specs)}")

    rows: list[dict[str, object]] = []
    direct_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, spec in enumerate(base_specs, start=1):
        print(f"[{i}/{len(base_specs)}] {spec.name}", flush=True)
        try:
            direct_oof = ar.make_oof(spec, data["X_train"], data["y"], data["groups"])
            direct_test = ar.fit_predict_spec(spec, data["X_train"], data["y"], data["X_test"])
        except Exception as exc:
            row = ar.failed_row(spec, exc)
            row["lane"] = lane_for(spec)
            rows.append(add_public_flags(row))
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        direct_cache[spec.name] = (direct_oof, direct_test)
        params = ar.fit_affine_params(direct_oof, data["y"])
        for shrink in affine_shrinks_for(spec):
            variant = replace(
                spec,
                name=f"{spec.name}_affs{tag(shrink)}",
                affine_shrink=shrink,
                affine_clip=0.40,
                mean_center=True,
            )
            affine_oof = np.clip(ar.apply_affine(direct_oof, params, variant), 0, None)
            pred = np.clip(ar.apply_affine(direct_test, params, variant), 0, None)
            path = candidate_dir / f"{variant.name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = ar.diagnostics(
                spec=variant,
                pred=pred,
                direct_test=direct_test,
                anchor_test=anchor_test,
                y=data["y"],
                groups=data["groups"],
                direct_oof=direct_oof,
                affine_oof=affine_oof,
                anchor_oof=anchor_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                path=path,
                nested=False,
            )
            row["base_experiment"] = spec.name
            row["lane"] = lane_for(spec)
            rows.append(add_public_flags(row))
            print_one(rows[-1])

    nested_names = pick_nested_public_aware(rows, args.nested_top)
    print(f"\nnested variants={len(nested_names)}")
    nested_oof_cache: dict[str, np.ndarray] = {}
    row_by_name = {str(row["experiment"]): row for row in rows}
    spec_by_name = {spec.name: spec for spec in base_specs}
    for name in nested_names:
        row = row_by_name[name]
        if row.get("status") != "ok":
            continue
        base_name = str(row["base_experiment"])
        base_spec = spec_by_name[base_name]
        variant = replace(
            base_spec,
            name=name,
            affine_shrink=float(row["affine_shrink"]),
            affine_clip=float(row["affine_clip"]),
            mean_center=bool(row["mean_center"]),
        )
        if base_name not in nested_oof_cache:
            nested_oof_cache[base_name] = ar.make_nested_affine_oof(
                base_spec, data["X_train"], data["y"], data["groups"]
            )
        direct_oof, direct_test = direct_cache[base_name]
        params = ar.fit_affine_params(direct_oof, data["y"])
        pred = np.clip(ar.apply_affine(direct_test, params, variant), 0, None)
        path = candidate_dir / f"{variant.name}_nested_ranked.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row.update(
            ar.diagnostics(
                spec=variant,
                pred=pred,
                direct_test=direct_test,
                anchor_test=anchor_test,
                y=data["y"],
                groups=data["groups"],
                direct_oof=direct_oof,
                affine_oof=nested_oof_cache[base_name],
                anchor_oof=anchor_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                path=path,
                nested=True,
            )
        )
        row["base_experiment"] = base_name
        row["lane"] = lane_for(base_spec)
        row.update(add_public_flags(row))
        print("nested", end=" ")
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "focused_public_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "focused_public_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop public-aware candidates:")
    for row in rows_sorted[:50]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: lane={row['lane']} submit={row['submit_gate']} "
            f"reasons={row['reject_reasons']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"direct={row['direct_oof_delta_vs_anchor']:.4f} affine={row['affine_oof_delta_vs_anchor']:.4f} "
            f"fold={row['direct_improved_fold_count']}/5 nested={row['nested_evaluated']}"
        )
    print(f"saved focused public diagnostics: {out_dir}")


def build_base_specs() -> list[ar.AnchorSpec]:
    specs: list[ar.AnchorSpec] = []
    lane_a_preprocesses = [
        "sg7_snv",
        "sg9_snv",
        "sg11_snv",
        "snv",
        "detrend_snv",
        "sg9_snv_stack_d1",
    ]
    for preprocess in lane_a_preprocesses:
        for n_components in [12, 15, 18, 20, 24, 28, 32]:
            for alpha in [3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 4000.0, 4500.0, 5000.0]:
                specs.append(
                    ar.AnchorSpec(
                        name=f"laneA_{preprocess}_ridge_yj_p{n_components}_a{int(alpha)}",
                        preprocess=preprocess,
                        model="ridge",
                        target="yj",
                        n_components=n_components,
                        alpha=alpha,
                    )
                )

    lane_b_ridge = [
        ("msc_sg7", "yj"),
        ("msc_sg9", "yj"),
        ("msc_sg11", "yj"),
        ("detrend_snv", "log1p"),
        ("sg9_snv_d1", "yj"),
        ("sg9_snv_d2", "yj"),
    ]
    for preprocess, target in lane_b_ridge:
        for n_components in [8, 10, 12, 15, 18, 20, 24]:
            for alpha in [3000.0, 3500.0, 4000.0, 5000.0, 6500.0, 8000.0]:
                specs.append(
                    ar.AnchorSpec(
                        name=f"laneB_{preprocess}_ridge_{target}_p{n_components}_a{int(alpha)}",
                        preprocess=preprocess,
                        model="ridge",
                        target=target,
                        n_components=n_components,
                        alpha=alpha,
                    )
                )

    for preprocess in ["sg9_snv", "msc_sg9", "detrend_snv", "sg9_snv_stack_d1"]:
        for target in ["raw", "yj"]:
            for n_components in [3, 4, 5, 6]:
                specs.append(
                    ar.AnchorSpec(
                        name=f"laneB_{preprocess}_pls_{target}_c{n_components}",
                        preprocess=preprocess,
                        model="pls",
                        target=target,
                        n_components=n_components,
                        alpha=0.0,
                    )
                )

    for preprocess in ["sg9_snv", "msc_sg9", "detrend_snv"]:
        for target in ["raw", "yj"]:
            for n_components in [10, 12, 15, 18, 20]:
                specs.append(
                    ar.AnchorSpec(
                        name=f"laneB_{preprocess}_huber_{target}_p{n_components}",
                        preprocess=preprocess,
                        model="huber",
                        target=target,
                        n_components=n_components,
                        alpha=0.0001,
                    )
                )
    return specs


def affine_shrinks_for(spec: ar.AnchorSpec) -> list[float]:
    if lane_for(spec) == "A_public_yj_ridge":
        return [0.05, 0.08, 0.10, 0.12]
    return [0.05, 0.08, 0.10]


def lane_for(spec: ar.AnchorSpec) -> str:
    if spec.name.startswith("laneA_"):
        return "A_public_yj_ridge"
    return "B_direct_alt"


def add_public_flags(row: dict[str, object]) -> dict[str, object]:
    if row.get("status") != "ok":
        row["submit_gate"] = False
        row["reject_reasons"] = "failed"
        return row

    reasons: list[str] = []
    lane = str(row.get("lane", ""))
    model = str(row.get("model", ""))
    target = str(row.get("target", ""))
    alpha = float(row.get("alpha", 0.0) or 0.0)
    diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    direct_delta = float(row["direct_oof_delta_vs_anchor"])
    affine_delta = float(row["affine_oof_delta_vs_anchor"])
    direct_folds = int(row.get("direct_improved_fold_count", 0) or 0)
    affine_folds = int(row.get("affine_improved_fold_count", 0) or 0)
    range_ratio = float(row["range_ratio_vs_anchor"])
    shrink = float(row.get("affine_shrink", 0.0) or 0.0)

    if model == "ridge" and target == "yj" and alpha <= 3000.0:
        reasons.append("public_prior_yj_alpha3000")
    if lane == "A_public_yj_ridge" and alpha == 3300.0:
        reasons.append("caution_alpha3300")
    if diff < 0.05:
        reasons.append("near_anchor")
    if diff > 0.35 and lane == "A_public_yj_ridge":
        reasons.append("laneA_too_far")
    if max_abs > 2.0:
        reasons.append("max_diff_gt2")
    if lane == "B_direct_alt" and max_abs > 1.5:
        reasons.append("laneB_max_diff_gt1p5")
    if species > 0.25:
        reasons.append("species_shift_gt0p25")
    if lane == "B_direct_alt" and species > 0.18:
        reasons.append("laneB_species_shift_gt0p18")
    if not (0.90 <= range_ratio <= 1.10):
        reasons.append("range_shift")
    if direct_folds < 3:
        reasons.append("direct_folds_lt3")
    if affine_folds < 3:
        reasons.append("affine_folds_lt3")
    if direct_delta > 0.10:
        reasons.append("direct_oof_bad")
    if direct_delta > 0.05:
        reasons.append("direct_oof_caution")
    if affine_delta < -0.02 and direct_delta > 0.10:
        reasons.append("affine_only_winner")
    if shrink > 0.12:
        reasons.append("shrink_gt0p12")

    row["submit_gate"] = len(reasons) == 0
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    return row


def pick_nested_public_aware(rows: list[dict[str, object]], limit: int) -> list[str]:
    ok = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(ok, key=rough_rank_key)
    return [str(row["experiment"]) for row in ranked[:limit]]


def rough_rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"), float("inf"))
    reject_reasons = str(row.get("reject_reasons", ""))
    penalty = 0.0
    if "public_prior_yj_alpha3000" in reject_reasons:
        penalty += 100.0
    if "near_anchor" in reject_reasons:
        penalty += 8.0
    if "max_diff_gt2" in reject_reasons or "species_shift_gt0p25" in reject_reasons:
        penalty += 20.0
    if "direct_folds_lt3" in reject_reasons or "affine_folds_lt3" in reject_reasons:
        penalty += 8.0
    if "affine_only_winner" in reject_reasons:
        penalty += 10.0
    if "caution_alpha3300" in reject_reasons:
        penalty += 3.0
    if bool(row.get("submit_gate")):
        penalty -= 3.0

    diff = float(row["anchor_diff_rmse"])
    species = float(row["max_abs_species_mean_shift"])
    max_abs = float(row["anchor_diff_max_abs"])
    direct_delta = float(row["direct_oof_delta_vs_anchor"])
    affine_delta = float(row["affine_oof_delta_vs_anchor"])
    target_diff = 0.18 if str(row.get("lane")) == "A_public_yj_ridge" else 0.30
    return (
        penalty + max(affine_delta, -0.1) + 0.4 * max(direct_delta, 0.0) + abs(diff - target_diff) * 0.25,
        max_abs,
        species,
        diff,
    )


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    base = rough_rank_key(row)
    if row.get("status") != "ok":
        return base
    penalty = 0.0 if bool(row.get("nested_evaluated")) else 30.0
    if bool(row.get("submit_gate")):
        penalty -= 5.0
    nested_delta = float(row["affine_oof_delta_vs_anchor"])
    direct_delta = float(row["direct_oof_delta_vs_anchor"])
    diff = float(row["anchor_diff_rmse"])
    species = float(row["max_abs_species_mean_shift"])
    return (
        base[0] + penalty + max(nested_delta, -0.15) + 0.25 * max(direct_delta, 0.0),
        float(row["anchor_diff_max_abs"]),
        species,
        abs(diff - 0.20),
    )


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: lane={row['lane']} gate={row['submit_gate']} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} direct={row['direct_oof_delta_vs_anchor']:.4f} "
        f"affine={row['affine_oof_delta_vs_anchor']:.4f} reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
