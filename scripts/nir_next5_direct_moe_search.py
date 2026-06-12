#!/usr/bin/env python3
"""Direct model / MoE-style search after next5 #2.

This is intentionally more aggressive than the Stage5 residual searches.  Each
branch is a direct predictor.  A fold-local gate learns where the branch beats
the #2 anchor OOF, then test rows in the same region are blended toward the
branch prediction.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nir_next5_diverse_candidate_builder as n5
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_direct_moe"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BEST = fu.BEST
CAND2 = fu.CAND2


@dataclass(frozen=True)
class Candidate:
    family: str
    experiment: str
    pred: np.ndarray
    corr_oof: np.ndarray
    corr_test: np.ndarray
    beta: float
    signal_corr: float
    memo: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--save-top", type=int, default=8)
    args = parser.parse_args()

    data = op.load_data()
    best = n5.read_submission(BEST, data["test_ids"])
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building #2 anchor OOF", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    test_diag = load_public_delta_diag()

    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    branch_rows: list[dict[str, object]] = []
    for i, branch in enumerate(build_branches(), start=1):
        print(f"[{i}] {branch.name}", flush=True)
        branch_oof = stn.make_quality_branch_oof(branch, data, data["y"] - best_oof)
        branch_test = stn.fit_quality_branch(
            branch,
            data["X_train"],
            data["y"],
            data["X_test"],
            data["y"] - best_oof,
            data["groups"],
        )
        branch_rmse = rmse(data["y"], branch_oof)
        signal_oof = branch_oof - best_oof
        signal_test = branch_test - best
        signal_corr = n5.safe_corr(signal_oof, residual)
        branch_rows.append(
            {
                "branch": branch.name,
                "branch_oof_rmse": branch_rmse,
                "branch_delta_vs_anchor": branch_rmse - base_rmse,
                "signal_corr": signal_corr,
                "signal_test_rmse": float(math.sqrt(np.mean(signal_test**2))),
                "signal_test_max": float(np.max(np.abs(signal_test))),
            }
        )
        if branch_rmse - base_rmse > 5.0 and signal_corr < 0.03:
            continue
        labels = branch_win_labels(data["y"], best_oof, branch_oof)
        if int(labels.sum()) < 20:
            continue
        for gate_model in ["logreg", "rf"]:
            gate_oof, gate_test, gate_auc = gate_scores(gate_model, F_train, F_test, labels, data["groups"])
            for frac in [0.05, 0.075, 0.10, 0.15, 0.20]:
                gate_oof_mask = top_fraction(gate_oof * rank01(np.abs(signal_oof)), frac)
                gate_test_mask = top_fraction(gate_test * rank01(np.abs(signal_test)), frac)
                for blend in [0.25, 0.40, 0.60, 0.80, 1.00]:
                    corr_oof = np.where(gate_oof_mask, blend * signal_oof, 0.0)
                    corr_test = np.where(gate_test_mask, blend * signal_test, 0.0)
                    pred = np.clip(best + corr_test, 0, None)
                    cand = Candidate(
                        family="direct_moe",
                        experiment=(
                            f"dm_{branch.name}_{gate_model}_f{stn.tag(frac)}_"
                            f"b{stn.tag(blend)}"
                        ),
                        pred=pred,
                        corr_oof=corr_oof,
                        corr_test=corr_test,
                        beta=blend,
                        signal_corr=signal_corr,
                        memo=(
                            f"#2 anchor; direct MoE; branch={branch.name}; gate={gate_model}; "
                            f"frac={frac}; blend={blend}; branch_delta={branch_rmse - base_rmse:.4f}; "
                            f"gate_auc={gate_auc:.4f}; signal_corr={signal_corr:.4f}."
                        ),
                    )
                    row = diagnostics(
                        cand,
                        data["y"],
                        best_oof,
                        base_rmse,
                        best,
                        best - cand2,
                        test_species,
                        test_diag,
                        branch.name,
                        gate_model,
                        gate_auc,
                        branch_rmse - base_rmse,
                    )
                    candidates.append(cand)
                    rows.append(row)
                    if row["submit_gate"] == "pass":
                        print_one(row)

    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(branch_rows).to_csv(out_dir / "direct_moe_branch_summary.csv", index=False)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "direct_moe_summary.csv", index=False)
    (out_dir / "direct_moe_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_direct_moe_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "direct_moe_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "direct_moe_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "direct_moe_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:50]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_branches() -> list[stn.BranchSpec]:
    out: list[stn.BranchSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for keep_frac in [0.55, 0.65, 0.75, 0.85]:
            for quality_power in [0.0, 0.7, 1.3, 1.6]:
                for comp in [3, 4, 5, 6, 8]:
                    out.append(
                        stn.BranchSpec(
                            name=(
                                f"direct_{preprocess}_kc_"
                                f"k{stn.pct_tag(keep_frac)}_q{stn.tag(quality_power)}_c{comp}"
                            ),
                            preprocess=preprocess,
                            model="pls_raw",
                            score_mode="knn_cluster",
                            keep_frac=keep_frac,
                            n_components=comp,
                            quality_power=quality_power,
                        )
                    )
    return out


def branch_win_labels(y: np.ndarray, anchor_oof: np.ndarray, branch_oof: np.ndarray) -> np.ndarray:
    anchor_abs = np.abs(y - anchor_oof)
    branch_abs = np.abs(y - branch_oof)
    margin = np.maximum(1.0, 0.08 * anchor_abs)
    return (branch_abs + margin < anchor_abs).astype(int)


def gate_scores(
    model_name: str,
    F_train: np.ndarray,
    F_test: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.zeros(len(labels), dtype=float)
    for train_idx, valid_idx in splitter.split(F_train, labels, groups):
        model = make_gate_model(model_name)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(F_train[train_idx])
        Xva = scaler.transform(F_train[valid_idx])
        model.fit(Xtr, labels[train_idx])
        oof[valid_idx] = model.predict_proba(Xva)[:, 1]
    model = make_gate_model(model_name)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(F_train)
    Xte = scaler.transform(F_test)
    model.fit(Xtr, labels)
    test = model.predict_proba(Xte)[:, 1]
    try:
        auc = float(roc_auc_score(labels, oof))
    except ValueError:
        auc = 0.5
    return oof, test, auc


def make_gate_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000, random_state=42)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
    test_diag: pd.DataFrame | None,
    branch: str,
    gate_model: str,
    gate_auc: float,
    branch_delta: float,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, best_oof, base_rmse, best, best_increment, test_species)
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "branch": branch,
            "gate_model": gate_model,
            "gate_auc": gate_auc,
            "branch_delta": branch_delta,
            "bad_new_overlap_count": bad_new_overlap(changed, test_diag),
            "clip_like_count": int(np.sum(np.abs(diff[changed]) > 0.5)) if np.any(changed) else 0,
            "diff_p95": float(np.quantile(np.abs(diff[changed]), 0.95)) if np.any(changed) else 0.0,
        }
    )
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.30:
        reasons.append("weak_oof_for_direct")
    if float(row["gate_auc"]) < 0.58:
        reasons.append("weak_gate_auc")
    if not (0.12 <= float(row["anchor_diff_rmse"]) <= 1.20):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) < 0.80:
        reasons.append("max_too_small_for_12s")
    if float(row["anchor_diff_max"]) > 8.0:
        reasons.append("max_too_large")
    if int(row["changed_count"]) < 20 or int(row["changed_count"]) > 110:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.25:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.40:
        reasons.append("species_shift")
    if int(row["species_max"]) > 35:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 4:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 3:
        reasons.append("bad_new_overlap")
    if abs(float(row["best_increment_corr"])) > 0.70:
        reasons.append("too_correlated_best")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def load_public_delta_diag() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "stage5_public_delta_diagnostics" / "row_public_delta_diagnostics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def bad_new_overlap(changed: np.ndarray, test_diag: pd.DataFrame | None) -> int:
    if test_diag is None:
        return 0
    mask = np.zeros(len(changed), dtype=bool)
    for col in ["failed_new_vs_cand2"]:
        if col in test_diag.columns:
            mask |= test_diag[col].astype(bool).to_numpy()
    if "broad_changed" in test_diag.columns and "cand2_changed" in test_diag.columns:
        mask |= test_diag["broad_changed"].astype(bool).to_numpy() & ~test_diag["cand2_changed"].astype(bool).to_numpy()
    return int(np.sum(changed & mask))


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(str(row["branch"]) == str(prev["branch"]) for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.7, 0.0) + abs(float(row["anchor_diff_rmse"]) - 0.45),
        -float(row["gate_auc"]),
        float(row["bad_new_overlap_count"]),
        float(row["species_shift"]),
    )


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    target = max(1, int(round(len(score) * frac)))
    order = np.argsort(score, kind="mergesort")[::-1]
    out = np.zeros(len(score), dtype=bool)
    out[order[:target]] = True
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    return ppg.rank01(values)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(
        family=cand.family,
        experiment=cand.experiment,
        pred=cand.pred,
        corr_test=cand.corr_test,
        corr_oof=cand.corr_oof,
        signal_corr=cand.signal_corr,
        beta=cand.beta,
        memo=cand.memo,
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max']:.4f} "
        f"changed={row['changed_count']} auc={row['gate_auc']:.3f} "
        f"sp={row['species_shift']:.4f} bad={row['bad_new_overlap_count']}"
    )


if __name__ == "__main__":
    main()
