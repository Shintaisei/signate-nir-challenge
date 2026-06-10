#!/usr/bin/env python3
"""Stage5 soft all-row residual search on top of the q5 public-best anchor.

Unlike the hard-gated Stage5 search, this script lets every row receive a tiny
second-stage residual correction, then controls risk with continuous weights
and distribution gates. It writes only selected candidates to avoid producing
large candidate directories.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import nir_slot1_testnear_branch_search as stn
import nir_post_public_gate_search as ppg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_stage5_soft_residual"
SUBMISSION_DIR = ROOT / "data" / "submissions"


@dataclass(frozen=True)
class SoftSpec:
    name: str
    branches: tuple[stn.BranchSpec, ...]
    mode: str
    signal_power: float
    detector_power: float
    shrink: float
    clip: float
    mean_center: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=10)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ppg.op.load_data()
    q5_df = pd.read_csv(stn.Q5_BEST, header=None, names=["id", "pred"])
    stage3_df = pd.read_csv(stn.CURRENT_BEST, header=None, names=["id", "pred"])
    if not np.array_equal(data["test_ids"], q5_df["id"].to_numpy()):
        raise ValueError("q5 sample order mismatch")
    q5_test = q5_df["pred"].to_numpy(float)
    stage3_test = stage3_df["pred"].to_numpy(float)
    q5_increment = q5_test - stage3_test
    test_species = pd.read_csv(ppg.op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    print("building q5 OOF anchor", flush=True)
    _, slot1_oof = stn.make_slot1_oof(data, stage3_test)
    stage3_oof = stn.make_current_best_oof(
        data,
        slot1_oof,
        stage3_test,
        pd.read_csv(stn.FIRST_STAGE_BEST, header=None).iloc[:, 1].to_numpy(float),
        pd.read_csv(stn.SECOND_STAGE_BEST, header=None).iloc[:, 1].to_numpy(float),
    )
    q5_oof = stn.make_q5_oof(data, stage3_oof, stage3_test)
    q5_oof_rmse = rmse(data["y"], q5_oof)
    residual = data["y"] - q5_oof

    print("building detector scores", flush=True)
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)
    detector_oof_rank = ppg.rank01(detector_oof)
    detector_test_rank = ppg.rank01(detector_test)
    balance_train_labels, balance_test_labels = make_balance_labels(data, q5_oof, q5_test)

    branch_pool = [
        stn.BranchSpec("pls_sg9_snv_knn_k6_q1_c3", "sg9_snv", "pls_raw", "knn", 0.60, 3, 1.0),
        stn.BranchSpec("pls_sg9_snv_knn_k65_q1_c3", "sg9_snv", "pls_raw", "knn", 0.65, 3, 1.0),
        stn.BranchSpec("pls_sg9_snv_knn_k55_q1_c3", "sg9_snv", "pls_raw", "knn", 0.55, 3, 1.0),
        stn.BranchSpec("pls_sg9_snv_knn_k6_q1_c4", "sg9_snv", "pls_raw", "knn", 0.60, 4, 1.0),
        stn.BranchSpec("pls_sg9_snv_knn_cluster_k55_q1p3_c4", "sg9_snv", "pls_raw", "knn_cluster", 0.55, 4, 1.3),
        stn.BranchSpec("pls_sg9_snv_knn_cluster_k75_q1p6_c5", "sg9_snv", "pls_raw", "knn_cluster", 0.75, 5, 1.6),
    ]
    print("building branch signals", flush=True)
    branch_signals: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for branch in branch_pool:
        branch_oof = stn.make_quality_branch_oof(branch, data, residual)
        branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
        signal_oof = branch_oof - q5_oof
        signal_test = branch_test - q5_test
        beta = stn.fit_beta(signal_oof, residual)
        branch_signals[branch.name] = (signal_oof, signal_test, beta, stn.safe_corr(signal_oof, residual))
        print(f"  {branch.name}: beta={beta:.4f} corr={branch_signals[branch.name][3]:.4f}", flush=True)

    specs = build_specs(branch_pool)
    rows: list[dict[str, object]] = []
    saved: list[dict[str, object]] = []
    for spec in specs:
        corr_oof, pred = make_soft_candidate(
            spec,
            branch_signals,
            q5_oof,
            q5_test,
            residual,
            detector_oof_rank,
            detector_test_rank,
            balance_train_labels,
            balance_test_labels,
        )
        row = diagnostics(
            spec,
            pred,
            corr_oof,
            q5_test,
            q5_increment,
            q5_oof,
            q5_oof_rmse,
            data["y"],
            residual,
            data["groups"],
            test_species,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    summary = pd.DataFrame(rows_sorted)
    summary.to_csv(out_dir / "soft_residual_summary.csv", index=False)
    json_rows = [{k: v for k, v in row.items() if k != "_spec"} for row in rows_sorted]
    (out_dir / "soft_residual_summary.json").write_text(json.dumps(json_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    selected = select_diverse(rows_sorted, branch_signals, q5_test, args.max_candidates)
    for i, row in enumerate(selected, start=1):
        spec = row["_spec"]
        _, pred = make_soft_candidate(
            spec,
            branch_signals,
            q5_oof,
            q5_test,
            residual,
            detector_oof_rank,
            detector_test_rank,
            balance_train_labels,
            balance_test_labels,
        )
        path = SUBMISSION_DIR / f"nir_stage5_soft{i}_{spec.name}_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        out = {k: v for k, v in row.items() if k != "_spec"}
        out["rank"] = i
        out["file"] = str(path)
        out["memo"] = (
            f"Stage5 soft residual q5 anchor; {spec.name}; mode={spec.mode}; "
            f"diff_rmse={row['diff_rmse']:.6f} max={row['max_abs_diff']:.6f} "
            f"eff={row['effective_count']} oof_delta={row['oof_delta']:.6f}"
        )
        saved.append(out)

    pd.DataFrame([{k: v for k, v in r.items() if k != "_spec"} for r in rows_sorted]).to_csv(
        out_dir / "soft_residual_summary_public.csv", index=False
    )
    pd.DataFrame(saved).to_csv(OUTPUT_ROOT / "stage5_soft_candidate_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "stage5_soft_candidate_manifest_latest.json").write_text(
        json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSelected soft candidates:")
    print(pd.DataFrame(saved).to_string(index=False) if saved else "none")
    print(f"saved diagnostics: {out_dir}")


def build_specs(branches: list[stn.BranchSpec]) -> list[SoftSpec]:
    specs: list[SoftSpec] = []
    for branch in branches:
        for mode in ["soft_abs", "soft_detector", "soft_balanced", "soft_detector_balanced"]:
            for signal_power in [2.0, 3.0, 4.0, 6.0]:
                for detector_power in ([0.0] if mode in {"soft_abs", "soft_balanced"} else [0.5, 1.0, 1.5]):
                    for shrink in [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]:
                        for clip in [0.05, 0.08, 0.12]:
                            name = (
                                f"{mode}_{short_branch(branch.name)}_"
                                f"sp{tag(signal_power)}_dp{tag(detector_power)}_s{tag(shrink)}_c{tag(clip)}"
                            )
                            specs.append(SoftSpec(name, (branch,), mode, signal_power, detector_power, shrink, clip))
    # A small agreement family: average signs from two lower-correlation branches.
    pairs = [
        (branches[0], branches[1]),
        (branches[0], branches[2]),
        (branches[1], branches[4]),
    ]
    for pair in pairs:
        for signal_power in [2.0, 3.0, 4.0]:
            for shrink in [0.004, 0.006, 0.008, 0.010]:
                name = f"soft_agree_{short_branch(pair[0].name)}_{short_branch(pair[1].name)}_sp{tag(signal_power)}_s{tag(shrink)}_c0p08"
                specs.append(SoftSpec(name, pair, "soft_agree", signal_power, 0.0, shrink, 0.08))
                bname = f"soft_agree_bal_{short_branch(pair[0].name)}_{short_branch(pair[1].name)}_sp{tag(signal_power)}_s{tag(shrink)}_c0p08"
                specs.append(SoftSpec(bname, pair, "soft_agree_balanced", signal_power, 0.0, shrink, 0.08))
    return specs


def make_soft_candidate(
    spec: SoftSpec,
    branch_signals: dict[str, tuple[np.ndarray, np.ndarray, float, float]],
    q5_oof: np.ndarray,
    q5_test: np.ndarray,
    residual: np.ndarray,
    detector_oof_rank: np.ndarray,
    detector_test_rank: np.ndarray,
    balance_train_labels: np.ndarray,
    balance_test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof_parts = []
    test_parts = []
    for branch in spec.branches:
        signal_oof, signal_test, beta, _ = branch_signals[branch.name]
        oof_parts.append(beta * signal_oof)
        test_parts.append(beta * signal_test)
    raw_oof = np.mean(np.vstack(oof_parts), axis=0)
    raw_test = np.mean(np.vstack(test_parts), axis=0)

    if spec.mode == "soft_agree" and len(oof_parts) > 1:
        sign_oof = np.mean(np.sign(np.vstack(oof_parts)), axis=0)
        sign_test = np.mean(np.sign(np.vstack(test_parts)), axis=0)
        raw_oof *= np.abs(sign_oof)
        raw_test *= np.abs(sign_test)

    weight_oof = np.power(ppg.rank01(np.abs(raw_oof)), spec.signal_power)
    weight_test = np.power(ppg.rank01(np.abs(raw_test)), spec.signal_power)
    if "detector" in spec.mode:
        weight_oof *= np.power(np.clip(detector_oof_rank, 0, 1), spec.detector_power)
        weight_test *= np.power(np.clip(detector_test_rank, 0, 1), spec.detector_power)

    corr_oof = np.clip(spec.shrink * raw_oof * weight_oof, -spec.clip, spec.clip)
    corr_test = np.clip(spec.shrink * raw_test * weight_test, -spec.clip, spec.clip)
    if "balanced" in spec.mode:
        corr_oof = balance_large_corrections(corr_oof, balance_train_labels, cap=1, power=1.4)
        corr_test = balance_large_corrections(corr_test, balance_test_labels, cap=1, power=1.4)
    if spec.mean_center:
        corr_oof -= float(np.mean(corr_oof))
        corr_test -= float(np.mean(corr_test))

    active = np.abs(corr_oof) > 0.002
    if active.any() and stn.safe_corr(corr_oof[active], residual[active]) < 0:
        corr_oof *= -1.0
        corr_test *= -1.0
    return corr_oof, np.clip(q5_test + corr_test, 0, None)


def diagnostics(
    spec: SoftSpec,
    pred: np.ndarray,
    corr_oof: np.ndarray,
    q5_test: np.ndarray,
    q5_increment: np.ndarray,
    q5_oof: np.ndarray,
    q5_oof_rmse: float,
    y: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
) -> dict[str, object]:
    diff = pred - q5_test
    abs_diff = np.abs(diff)
    corrected_oof = np.clip(q5_oof + corr_oof, 0, None)
    fold = ppg.fold_delta_stats(y, corrected_oof, q5_oof, groups)
    top10 = np.argsort(abs_diff)[-10:]
    effective = abs_diff > 0.01
    corrected_species = pd.Series(test_species[effective]).value_counts()
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "_spec": spec,
        "experiment": spec.name,
        "mode": spec.mode,
        "branches": "+".join(branch.name for branch in spec.branches),
        "signal_power": spec.signal_power,
        "detector_power": spec.detector_power,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "diff_rmse": rmse(pred, q5_test),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "p95_abs_diff": float(np.quantile(abs_diff, 0.95)),
        "p99_abs_diff": float(np.quantile(abs_diff, 0.99)),
        "mean_shift": float(np.mean(diff)),
        "std_ratio": float(np.std(pred) / max(1e-12, np.std(q5_test))),
        "effective_count": int(effective.sum()),
        "over_003_count": int((abs_diff > 0.03).sum()),
        "over_005_count": int((abs_diff > 0.005).sum()),
        "negative_count": int((pred < 0).sum()),
        "species_shift": ppg.max_species_shift(diff, test_species),
        "effective_species_max": int(corrected_species.iloc[0]) if len(corrected_species) else 0,
        "top10_species_max": int(top_species.iloc[0]) if len(top_species) else 0,
        "q5_increment_corr": stn.safe_corr(diff, q5_increment),
        "positive_frac_effective": float(np.mean(diff[effective] > 0)) if effective.any() else 0.0,
        "bottom_decile_shift": decile_shift(diff, q5_test, 0.0, 0.1),
        "top_decile_shift": decile_shift(diff, q5_test, 0.9, 1.0),
        "oof_rmse": rmse(y, corrected_oof),
        "oof_delta": rmse(y, corrected_oof) - q5_oof_rmse,
        "groups_improved": fold["improved_count"],
        "group_count": fold["group_count"],
        "worst_group_delta": fold["worst_delta"],
        "signal_corr_effective": stn.safe_corr(corr_oof[np.abs(corr_oof) > 0.002], residual[np.abs(corr_oof) > 0.002])
        if np.any(np.abs(corr_oof) > 0.002)
        else 0.0,
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "reject"
    row["risk"] = risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if not (0.003 <= float(row["diff_rmse"]) <= 0.020):
        reasons.append("diff_rmse_range")
    if float(row["max_abs_diff"]) > 0.16:
        reasons.append("max_diff")
    if float(row["p95_abs_diff"]) > 0.04:
        reasons.append("p95_diff")
    if float(row["p99_abs_diff"]) > 0.08:
        reasons.append("p99_diff")
    if float(row["mean_abs_diff"]) > 0.015:
        reasons.append("mean_diff")
    if abs(float(row["mean_shift"])) > 0.003:
        reasons.append("mean_shift")
    if not (0.99 <= float(row["std_ratio"]) <= 1.01):
        reasons.append("std_ratio")
    if not (20 <= int(row["effective_count"]) <= 140):
        reasons.append("effective_count")
    if int(row["over_003_count"]) > 30:
        reasons.append("too_many_large")
    if float(row["species_shift"]) > 0.006:
        reasons.append("species_shift")
    if int(row["effective_species_max"]) > 30:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if float(row["oof_delta"]) > -0.0035:
        reasons.append("weak_oof")
    if int(row["groups_improved"]) < 9:
        reasons.append("weak_groups")
    if abs(float(row["q5_increment_corr"])) > 0.75:
        reasons.append("q5_corr")
    if float(row["positive_frac_effective"]) in {0.0, 1.0}:
        reasons.append("one_sided")
    if abs(float(row["bottom_decile_shift"])) > 0.015 or abs(float(row["top_decile_shift"])) > 0.015:
        reasons.append("tail_shift")
    if float(row["signal_corr_effective"]) < 0.25:
        reasons.append("bad_signal_corr")
    return reasons


def risk(row: dict[str, object]) -> float:
    out = 0.0
    out += max(0.0, float(row["diff_rmse"]) - 0.016) * 5.0
    out += max(0.0, float(row["max_abs_diff"]) - 0.12) * 1.0
    out += max(0.0, float(row["species_shift"]) - 0.004) * 10.0
    out += abs(float(row["mean_shift"])) * 10.0
    out += max(0.0, float(row["p95_abs_diff"]) - 0.035) * 2.0
    out += max(0.0, abs(float(row["q5_increment_corr"])) - 0.55) * 0.5
    out += 0.02 * max(0, int(row["top10_species_max"]) - 3)
    return float(out)


def select_diverse(rows: list[dict[str, object]], branch_signals: dict[str, tuple[np.ndarray, np.ndarray, float, float]], q5_test: np.ndarray, limit: int) -> list[dict[str, object]]:
    passes = [row for row in rows if row["submit_gate"] == "pass"]
    passes.sort(key=rank_key)
    selected: list[dict[str, object]] = []
    for row in passes:
        if len(selected) >= limit:
            break
        selected.append(row)
    return selected


def make_balance_labels(data: dict[str, np.ndarray], q5_oof: np.ndarray, q5_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    n_clusters = min(64, max(4, len(F_train) // 4))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit(F_train)
    train_cluster = km.predict(F_train)
    test_cluster = km.predict(F_test)
    qs = np.quantile(q5_oof, np.linspace(0, 1, 11))
    qs = np.unique(qs)
    train_bin = np.digitize(q5_oof, qs[1:-1], right=True)
    test_bin = np.digitize(q5_test, qs[1:-1], right=True)
    return train_cluster * 20 + train_bin, test_cluster * 20 + test_bin


def balance_large_corrections(corr: np.ndarray, labels: np.ndarray, *, cap: int, power: float) -> np.ndarray:
    out = corr.copy()
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) <= cap:
            continue
        order = idx[np.argsort(np.abs(corr[idx]))[::-1]]
        ranks = np.arange(1, len(order) + 1, dtype=float)
        mult = np.minimum(1.0, np.power(cap / ranks, power))
        out[order] *= mult
    return out


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    pass_penalty = 0 if row["submit_gate"] == "pass" else 1
    return (
        pass_penalty,
        float(row["risk"]) - min(0.04, -float(row["oof_delta"])),
        -float(row["oof_delta"]),
        float(row["diff_rmse"]),
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
        f"diff={row['diff_rmse']:.4f} max={row['max_abs_diff']:.4f} eff={row['effective_count']} "
        f"sp={row['species_shift']:.4f} oof={row['oof_delta']:.4f} "
        f"groups={row['groups_improved']}/{row['group_count']}",
        flush=True,
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


def decile_shift(diff: np.ndarray, pred: np.ndarray, low: float, high: float) -> float:
    lo = np.quantile(pred, low)
    hi = np.quantile(pred, high)
    mask = (pred >= lo) & (pred <= hi)
    return float(np.mean(diff[mask])) if mask.any() else 0.0


def short_branch(name: str) -> str:
    return name.replace("pls_sg9_snv_", "").replace("cluster_", "cl")


def tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
