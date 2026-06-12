#!/usr/bin/env python3
"""Asymmetric golden-expert attack after next5 #2 best.

The direct golden expert has the strongest OOF evidence, but naive direct
replacement is too concentrated.  This probe raises the expert-supported
positive hard cases and uses smaller opposite-sign rows only to control mean
shift and species concentration.
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

import nir_next5_direct_blend_search as db
import nir_next5_diverse_candidate_builder as n5
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_asym_attack"
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
    diag = load_public_delta_diag()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building #2 anchor OOF", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    print("building golden direct expert", flush=True)
    experts = db.build_experts(data, best_oof, best, residual, base_rmse)
    expert = [e for e in experts if e.name == "golden_sg9_keep80_c3"][0]
    signal_oof = expert.oof - best_oof
    signal_test = expert.test - best

    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    for cand in candidate_grid(
        best,
        residual,
        signal_oof,
        signal_test,
        expert.signal_corr,
        expert.oof_delta,
        data["groups"],
        test_species,
        diag,
    ):
        row = diagnostics(cand, data["y"], best_oof, base_rmse, best, best - cand2, test_species, diag)
        rows.append(row)
        candidates.append(cand)
        if row["submit_gate"] == "pass":
            print_one(row)

    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "asym_attack_summary.csv", index=False)
    (out_dir / "asym_attack_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_asym_attack_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "asym_attack_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "asym_attack_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "asym_attack_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:80]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def candidate_grid(
    best: np.ndarray,
    residual: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    signal_corr: float,
    expert_delta: float,
    train_species: np.ndarray,
    test_species: np.ndarray,
    diag: pd.DataFrame | None,
) -> list[Candidate]:
    out: list[Candidate] = []
    bad_train = np.zeros(len(signal_oof), dtype=bool)
    bad_test = bad_mask(diag, len(signal_test))
    for pos_frac in [0.035, 0.05, 0.065, 0.08, 0.10]:
        for neg_ratio in [0.25, 0.50, 0.75, 1.00]:
            for blend in [0.12, 0.18, 0.25, 0.35]:
                for clip in [1.2, 1.8, 2.5, 3.5, 4.5]:
                    for species_cap in [2, 3, 4]:
                        corr_oof = asym_corr(
                            signal_oof,
                            residual,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            train_species,
                            species_cap,
                            bad_train,
                        )
                        corr_test = asym_corr(
                            signal_test,
                            None,
                            pos_frac,
                            neg_ratio,
                            blend,
                            clip,
                            test_species,
                            species_cap,
                            bad_test,
                        )
                        exp = (
                            f"asym_gold_pos{n5.safe_name(str(pos_frac))}_nr{n5.safe_name(str(neg_ratio))}_"
                            f"b{n5.safe_name(str(blend))}_c{n5.safe_name(str(clip))}_sp{species_cap}"
                        )
                        out.append(
                            Candidate(
                                family="asym_attack",
                                experiment=exp,
                                pred=np.clip(best + corr_test, 0, None),
                                corr_oof=corr_oof,
                                corr_test=corr_test,
                                beta=blend,
                                signal_corr=signal_corr,
                                memo=(
                                    f"#2 anchor; asymmetric golden_sg9_keep80_c3 correction; "
                                    f"pos_frac={pos_frac}; neg_ratio={neg_ratio}; blend={blend}; "
                                    f"clip={clip}; species_cap={species_cap}; expert_delta={expert_delta:.4f}; "
                                    f"signal_corr={signal_corr:.4f}; public-failed rows excluded from test gate."
                                ),
                            )
                        )
    return out


def asym_corr(
    signal: np.ndarray,
    residual: np.ndarray | None,
    pos_frac: float,
    neg_ratio: float,
    blend: float,
    clip: float,
    labels: np.ndarray,
    species_cap: int,
    banned: np.ndarray,
) -> np.ndarray:
    n_pos = max(1, int(round(len(signal) * pos_frac)))
    n_neg = max(1, int(round(n_pos * neg_ratio)))
    if residual is None:
        pos_score = np.maximum(signal, 0)
        neg_score = np.maximum(-signal, 0)
    else:
        # Require the expert signal to agree with residual direction in OOF.
        pos_score = np.where((signal > 0) & (residual > 0), signal * np.abs(residual), 0.0)
        neg_score = np.where((signal < 0) & (residual < 0), -signal * np.abs(residual), 0.0)
    pos = select_capped(pos_score, labels, species_cap, n_pos, banned)
    neg = select_capped(neg_score, labels, species_cap, n_neg, banned)
    corr = np.zeros(len(signal), dtype=float)
    selected = np.concatenate([pos, neg])
    if len(selected) == 0:
        return corr
    raw = np.clip(blend * signal[selected], -clip, clip)
    raw = raw - float(np.mean(raw))
    corr[selected] = raw
    return corr


def select_capped(score: np.ndarray, labels: np.ndarray, cap: int, n: int, banned: np.ndarray) -> np.ndarray:
    counts: dict[int, int] = {}
    selected: list[int] = []
    order = np.argsort(score, kind="mergesort")[::-1]
    for idx in order:
        if banned[idx] or score[idx] <= 0:
            continue
        label = int(labels[idx])
        if counts.get(label, 0) >= cap:
            continue
        selected.append(int(idx))
        counts[label] = counts.get(label, 0) + 1
        if len(selected) >= n:
            break
    return np.asarray(selected, dtype=int)


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
    diag: pd.DataFrame | None,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, best_oof, base_rmse, best, best_increment, test_species)
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "bad_new_overlap_count": db.bad_new_overlap(changed, diag),
            "failed_any_overlap_count": failed_any_overlap(changed, diag),
            "diff_p95": float(np.quantile(np.abs(diff[changed]), 0.95)) if np.any(changed) else 0.0,
            "tail_low_shift": float(np.mean(diff[best <= np.quantile(best, 0.10)])),
            "tail_high_shift": float(np.mean(diff[best >= np.quantile(best, 0.90)])),
        }
    )
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "review"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(row["oof_delta"]) > -0.08:
        reasons.append("weak_oof")
    if float(row["signal_corr"]) < 0.25:
        reasons.append("weak_signal_corr")
    if not (0.25 <= float(row["anchor_diff_rmse"]) <= 1.35):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) < 1.0:
        reasons.append("max_too_small")
    if float(row["anchor_diff_max"]) > 4.5:
        reasons.append("max_too_large")
    if int(row["changed_count"]) < 24 or int(row["changed_count"]) > 80:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.08:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.30:
        reasons.append("species_shift")
    if int(row["species_max"]) > 12:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 0:
        reasons.append("bad_new_overlap")
    if int(row["failed_any_overlap_count"]) > 2:
        reasons.append("failed_any_overlap")
    if abs(float(row["best_increment_corr"])) > 0.65:
        reasons.append("too_correlated_best")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    return (
        penalty + max(float(row["oof_delta"]) + 0.20, 0.0) + 0.12 * abs(float(row["anchor_diff_rmse"]) - 0.75),
        float(row["bad_new_overlap_count"]),
        float(row["failed_any_overlap_count"]),
        float(row["species_shift"]),
        -float(row["oof_delta"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        if any(abs(float(row["anchor_diff_rmse"]) - float(prev["anchor_diff_rmse"])) < 0.03 for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def load_public_delta_diag() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "stage5_public_delta_diagnostics" / "row_public_delta_diagnostics.csv"
    return pd.read_csv(path) if path.exists() else None


def bad_mask(diag: pd.DataFrame | None, n: int) -> np.ndarray:
    if diag is None:
        return np.zeros(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for col in ["failed_new_vs_cand2", "failed_overlap_cand2"]:
        if col in diag.columns:
            mask |= diag[col].astype(bool).to_numpy()
    return mask


def failed_any_overlap(changed: np.ndarray, diag: pd.DataFrame | None) -> int:
    if diag is None or "failed_any_changed" not in diag.columns:
        return 0
    return int(np.sum(changed & diag["failed_any_changed"].astype(bool).to_numpy()))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2)))


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(cand.family, cand.experiment, cand.pred, cand.corr_test, cand.corr_oof, cand.signal_corr, cand.beta, cand.memo)


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max']:.4f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.4f} "
        f"bad={row['bad_new_overlap_count']} failed={row['failed_any_overlap_count']}"
    )


if __name__ == "__main__":
    main()
