#!/usr/bin/env python3
"""Broader branch search after next5 #2 became the current best."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nir_next5_followup_search as fu
import nir_next5_diverse_candidate_builder as n5
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_branch_broad"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BEST = fu.BEST
CAND2 = fu.CAND2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--save-top", type=int, default=8)
    parser.add_argument("--max-branches", type=int, default=None)
    parser.add_argument("--focus-near", action="store_true")
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
    base_rmse = fu.rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    train_clusters, test_clusters = n5.cluster_labels(F_train, F_test, n_clusters=64)

    rows: list[dict[str, object]] = []
    candidates: list[fu.Candidate] = []
    branches = build_focus_near_grid() if args.focus_near else build_branch_grid()
    if args.max_branches is not None:
        branches = branches[: args.max_branches]

    print(f"branches={len(branches)}", flush=True)
    for i, branch in enumerate(branches, start=1):
        print(f"[{i}/{len(branches)}] {branch.name}", flush=True)
        try:
            branch_candidates = branch_candidates_for(
                branch,
                data,
                best_oof,
                best,
                residual,
                F_train,
                F_test,
                train_clusters,
                test_clusters,
            )
        except Exception as exc:
            print(f"  skip {type(exc).__name__}: {exc}", flush=True)
            continue
        for cand in branch_candidates:
            row = fu.diagnostics(cand, data["y"], best_oof, base_rmse, best, best - cand2, test_species)
            candidates.append(cand)
            rows.append(row)
            if row["submit_gate"] == "pass":
                print_one(row)

    n5.add_diversity_columns(rows, [fu.to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=fu.rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "branch_broad_summary.csv", index=False)
    (out_dir / "branch_broad_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, candidates, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_branch_broad_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "branch_broad_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "branch_broad_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "branch_broad_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:40]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_branch_grid() -> list[stn.BranchSpec]:
    branches: list[stn.BranchSpec] = []
    # Keep the grid broad enough to leave the k75/q1.6/c5 pocket, but bounded
    # enough that the OOF proxy remains interpretable.
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for score_mode in ["knn", "knn_cluster"]:
            for keep_frac in [0.55, 0.65, 0.75, 0.85]:
                for quality_power in [0.7, 1.0, 1.3, 1.6]:
                    for comp in [3, 4, 5, 6]:
                        branches.append(
                            stn.BranchSpec(
                                name=(
                                    f"bb_{preprocess}_{score_mode}_"
                                    f"k{stn.pct_tag(keep_frac)}_q{stn.tag(quality_power)}_c{comp}"
                                ),
                                preprocess=preprocess,
                                model="pls_raw",
                                score_mode=score_mode,
                                keep_frac=keep_frac,
                                n_components=comp,
                                quality_power=quality_power,
                            )
                        )
    # Add a few ridge/YJ contrast branches. They often fail, but if they pass
    # the low-correlation gate they are more informative than another PLS tweak.
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for score_mode in ["knn", "knn_cluster"]:
            for keep_frac in [0.65, 0.75, 0.85]:
                for quality_power in [1.0, 1.6]:
                    branches.append(
                        stn.BranchSpec(
                            name=(
                                f"bb_ridgeyj_{preprocess}_{score_mode}_"
                                f"k{stn.pct_tag(keep_frac)}_q{stn.tag(quality_power)}_p20"
                            ),
                            preprocess=preprocess,
                            model="ridge_yj",
                            score_mode=score_mode,
                            keep_frac=keep_frac,
                            n_components=20,
                            quality_power=quality_power,
                        )
                    )
    return branches


def build_focus_near_grid() -> list[stn.BranchSpec]:
    branches: list[stn.BranchSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for score_mode in ["knn_cluster"]:
            for keep_frac in [0.65, 0.75, 0.85]:
                for quality_power in [1.3, 1.6]:
                    for comp in [4, 5, 6]:
                        branches.append(
                            stn.BranchSpec(
                                name=(
                                    f"bn_{preprocess}_{score_mode}_"
                                    f"k{stn.pct_tag(keep_frac)}_q{stn.tag(quality_power)}_c{comp}"
                                ),
                                preprocess=preprocess,
                                model="pls_raw",
                                score_mode=score_mode,
                                keep_frac=keep_frac,
                                n_components=comp,
                                quality_power=quality_power,
                            )
                        )
    return branches


def branch_candidates_for(
    branch: stn.BranchSpec,
    data: dict[str, np.ndarray],
    best_oof: np.ndarray,
    best: np.ndarray,
    residual: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
) -> list[fu.Candidate]:
    branch_oof = stn.make_quality_branch_oof(branch, data, residual)
    branch_test = stn.fit_quality_branch(branch, data["X_train"], data["y"], data["X_test"], residual, data["groups"])
    signal_oof = branch_oof - best_oof
    signal_test = branch_test - best
    beta = n5.fit_beta(signal_oof, residual)
    signal_corr = stn.safe_corr(signal_oof, residual)
    if signal_corr < 0.05 and abs(beta) < 0.02:
        return []

    out: list[fu.Candidate] = []
    if branch.name.startswith("bn_"):
        gate_modes = ["abs_signal_cluster1", "abs_signal_cluster2", "abs_signal_cluster1_space02"]
        fracs = [0.032, 0.035, 0.038]
        shrinks = [0.022, 0.028, 0.034]
        clips = [0.14, 0.16]
    else:
        gate_modes = gate_modes_for(signal_corr)
        fracs = [0.032, 0.035, 0.040, 0.045]
        shrinks = [0.018, 0.022, 0.028, 0.034]
        clips = [0.12, 0.14, 0.16]
    for gate_mode in gate_modes:
        for frac in fracs:
            for shrink in shrinks:
                for clip in clips:
                    corr_oof, pred = stn.make_candidate(
                        stn.CandidateSpec(branch, gate_mode, frac, shrink, clip, False),
                        signal_oof,
                        signal_test,
                        best_oof,
                        best,
                        residual,
                        beta,
                        train_clusters,
                        test_clusters,
                        F_train,
                        F_test,
                    )
                    corr_test = pred - best
                    if np.count_nonzero(np.abs(corr_test) > 1e-12) == 0:
                        continue
                    out.append(
                        fu.Candidate(
                            family="branch_broad",
                            experiment=(
                                f"{branch.name}_{gate_mode}_f{stn.tag(frac)}_"
                                f"s{stn.tag(shrink)}_c{stn.tag(clip)}"
                            ),
                            pred=pred,
                            corr_oof=corr_oof,
                            corr_test=corr_test,
                            beta=beta,
                            signal_corr=signal_corr,
                            memo=(
                                f"#2 anchor; broad branch search; {branch.name}; "
                                f"{gate_mode}; frac={frac}; shrink={shrink}; clip={clip}; "
                                f"signal_corr={signal_corr:.4f}; beta={beta:.4f}."
                            ),
                        )
                    )
    return out


def gate_modes_for(signal_corr: float) -> list[str]:
    modes = [
        "abs_signal_cluster1",
        "abs_signal_cluster2",
        "abs_signal_cluster1_space02",
        "abs_signal_cluster2_space02",
        "abs_signal_cluster1_space05",
    ]
    if signal_corr >= 0.12:
        modes.extend(["negative_signal_cluster1", "negative_signal_cluster1_space02"])
    return modes


def select_submit_worthy(
    rows: list[dict[str, object]],
    candidates: list[fu.Candidate],
    limit: int,
) -> list[dict[str, object]]:
    passes = [
        row
        for row in rows
        if row["submit_gate"] == "pass"
        and float(row["oof_delta"]) <= -0.0045
        and 0.010 <= float(row["anchor_diff_rmse"]) <= 0.020
        and int(row["top10_species_max"]) <= 3
        and abs(float(row["best_increment_corr"])) <= 0.70
    ]
    selected: list[dict[str, object]] = []
    for row in passes:
        if any(similar(row, prev) for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def similar(row: dict[str, object], prev: dict[str, object]) -> bool:
    return (
        str(row["experiment"]).split("_abs_signal")[0] == str(prev["experiment"]).split("_abs_signal")[0]
        and abs(float(row["anchor_diff_rmse"]) - float(prev["anchor_diff_rmse"])) < 0.0015
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.6f} "
        f"diff={row['anchor_diff_rmse']:.6f} max={row['anchor_diff_max']:.6f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.6f} "
        f"top10={row['top10_species_max']} corr={row['best_increment_corr']:.3f}"
    )


if __name__ == "__main__":
    main()
