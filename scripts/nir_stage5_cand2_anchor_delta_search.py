#!/usr/bin/env python3
"""Build small cand2-anchored delta candidates from prior Stage5 signals."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "stage5_cand2_anchor_delta"
SUBMISSION_DIR = ROOT / "data" / "submissions"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SEARCH_ROOT = ROOT / "outputs" / "nir_slot1_testnear_branch"

Q5 = SUBMISSION_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv"
CAND2 = SUBMISSION_DIR / "nir_stage5_q5cand2_knn_k6_q1_c4_cl1_f0p045_s0p015_c0p18_b0p48411_20260609.csv"


def main() -> None:
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE_PATH, header=None)
    q5 = read_submission(Q5, sample)
    anchor = read_submission(CAND2, sample)
    test_species = pd.read_csv(TEST_PATH, encoding="cp932")["species number"].to_numpy()
    cand2_increment = anchor - q5

    source_rows = load_source_rows()
    rows: list[dict[str, object]] = []
    saved: list[dict[str, object]] = []

    for source in source_rows:
        pred_path = source["path"]
        source_pred = read_submission(pred_path, sample)
        signal = source_pred - anchor
        nonzero = np.flatnonzero(np.abs(signal) > 1e-12)
        if len(nonzero) < 8:
            continue

        order = nonzero[np.argsort(-np.abs(signal[nonzero]))]
        for top_n in [8, 10, 12, 14, 16, 18]:
            selected = order[: min(top_n, len(order))]
            if len(selected) < 8:
                continue
            for shrink in [0.55, 0.70, 0.85, 1.00]:
                for clip in [0.08, 0.10, 0.12, 0.13]:
                    corr = np.zeros_like(anchor)
                    corr[selected] = np.clip(shrink * signal[selected], -clip, clip)
                    pred = np.clip(anchor + corr, 0, None)
                    row = diagnose(
                        source,
                        pred,
                        anchor,
                        q5,
                        corr,
                        cand2_increment,
                        test_species,
                        top_n,
                        shrink,
                        clip,
                    )
                    rows.append(row)

    rows_sorted = sorted(rows, key=rank_key)
    serializable_rows = [{k: v for k, v in row.items() if k != "_pred"} for row in rows_sorted]
    pd.DataFrame(serializable_rows).to_csv(out_dir / "cand2_delta_summary.csv", index=False)
    (out_dir / "cand2_delta_summary.json").write_text(
        json.dumps(serializable_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = select_diverse(rows_sorted, limit=5)
    for i, row in enumerate(selected, start=1):
        pred = np.asarray(row.pop("_pred"), dtype=float)
        name = (
            f"nir_stage5_cand2delta{i}_{row['source_short']}_"
            f"n{row['top_n']}_s{tag(float(row['shrink']))}_c{tag(float(row['clip']))}_20260610.csv"
        )
        path = SUBMISSION_DIR / name
        pd.DataFrame({0: sample[0].to_numpy(), 1: pred}).to_csv(path, header=False, index=False)
        row["file"] = str(path)
        saved.append(row)

    pd.DataFrame(saved).to_csv(out_dir / "cand2_delta_selected.csv", index=False)
    pd.DataFrame(saved).to_csv(OUTPUT_ROOT / "cand2_delta_selected_latest.csv", index=False)
    print(f"saved summary: {out_dir / 'cand2_delta_summary.csv'}")
    print(f"passes: {sum(1 for r in rows_sorted if r['submit_gate'] == 'pass')}")
    print("selected:")
    for row in saved:
        print(
            f"  {row['file']} gate={row['submit_gate']} diff={row['cand2_diff_rmse']:.6f} "
            f"max={row['cand2_diff_max']:.6f} changed={row['changed_count']} "
            f"src_oof={row['source_oof_delta']:.6f} sp={row['species_shift']:.6f} "
            f"corr={row['cand2_increment_corr']:.3f} reasons={row['reject_reasons']}"
        )


def read_submission(path: Path, sample: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    if df.shape != sample.shape:
        raise ValueError(f"shape mismatch: {path}")
    if not df[0].equals(sample[0]):
        raise ValueError(f"sample order mismatch: {path}")
    return df[1].to_numpy(float)


def load_source_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    manifest = ROOT / "outputs" / "stage5_q5_candidate_manifest_20260609.csv"
    if manifest.exists():
        dfm = pd.read_csv(manifest)
        for _, row in dfm.iterrows():
            path = ROOT / str(row["file"])
            if not path.exists() or "q5cand2" in path.name:
                continue
            experiment = str(row["experiment"])
            out.append(
                {
                    "experiment": experiment,
                    "path": path,
                    "source_short": short_name(path.stem),
                    "source_oof_delta": float(row["oof_delta_vs_q5"]),
                    "source_groups": int(str(row["groups"]).split("/")[0]),
                    "source_signal_corr": float(row["signal_corr_gate"]),
                    "source_q5corr": float(row["q5_increment_corr"]),
                    "source_species_shift": float(row["species_shift"]),
                }
            )
    for summary_path in sorted(SEARCH_ROOT.glob("*/slot1_testnear_summary.csv")):
        run_dir = summary_path.parent
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            experiment = str(row["experiment"])
            path = run_dir / "candidates" / f"{experiment}.csv"
            if not path.exists():
                continue
            if "knn_cluster_k55" in experiment:
                continue
            oof = float(row.get("oof_delta_vs_slot1", row.get("oof_delta_vs_q5", 0.0)))
            groups = int(row.get("improved_group_count", 0))
            signal_corr = float(row.get("signal_residual_corr_on_gate", row.get("signal_corr_gate", 0.0)))
            if str(row.get("submit_gate", "")).lower() not in {"pass", "true"}:
                if not (oof <= -0.0045 and groups >= 9 and signal_corr >= 0.30):
                    continue
            out.append(
                {
                    "experiment": experiment,
                    "path": path,
                    "source_short": short_name(experiment),
                    "source_oof_delta": oof,
                    "source_groups": groups,
                    "source_signal_corr": signal_corr,
                    "source_q5corr": float(row.get("q5_increment_corr", 0.0)),
                    "source_species_shift": float(row.get("max_abs_species_mean_shift", row.get("species_shift", 0.0))),
                }
            )
    return out


def diagnose(
    source: dict[str, object],
    pred: np.ndarray,
    anchor: np.ndarray,
    q5: np.ndarray,
    corr: np.ndarray,
    cand2_increment: np.ndarray,
    test_species: np.ndarray,
    top_n: int,
    shrink: float,
    clip: float,
) -> dict[str, object]:
    diff = pred - anchor
    changed = np.abs(diff) > 1e-12
    q5_inc = pred - q5
    species_shift = max_species_shift(diff, test_species)
    species_counts = pd.Series(test_species[changed]).value_counts() if changed.any() else pd.Series(dtype=int)
    top_idx = np.argsort(-np.abs(diff))[:10]
    top_species_counts = pd.Series(test_species[top_idx[np.abs(diff[top_idx]) > 1e-12]]).value_counts()
    row: dict[str, object] = {
        "source_experiment": source["experiment"],
        "source_short": source["source_short"],
        "top_n": top_n,
        "shrink": shrink,
        "clip": clip,
        "changed_count": int(changed.sum()),
        "cand2_diff_rmse": rmse(pred, anchor),
        "cand2_diff_max": float(np.max(np.abs(diff))),
        "cand2_diff_mean": float(np.mean(diff)),
        "q5_diff_rmse": rmse(pred, q5),
        "q5_diff_max": float(np.max(np.abs(q5_inc))),
        "cand2_increment_corr": safe_corr(q5_inc, cand2_increment),
        "species_shift": species_shift,
        "species_max": int(species_counts.iloc[0]) if len(species_counts) else 0,
        "top10_species_max": int(top_species_counts.iloc[0]) if len(top_species_counts) else 0,
        "positive_count": int((diff > 1e-12).sum()),
        "negative_count": int((diff < -1e-12).sum()),
        "clip_frac": float(np.mean(np.abs(corr[changed]) >= clip - 1e-12)) if changed.any() else 0.0,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "source_oof_delta": source["source_oof_delta"],
        "source_groups": source["source_groups"],
        "source_signal_corr": source["source_signal_corr"],
        "source_q5corr": source["source_q5corr"],
        "source_species_shift": source["source_species_shift"],
        "_pred": pred,
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "reject"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if not (8 <= int(row["changed_count"]) <= 18):
        reasons.append("changed_count")
    if not (0.004 <= float(row["cand2_diff_rmse"]) <= 0.0125):
        reasons.append("diff_rmse")
    if float(row["cand2_diff_max"]) > 0.13:
        reasons.append("max_diff")
    if abs(float(row["cand2_diff_mean"])) > 0.002:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.0065:
        reasons.append("species_shift")
    if int(row["species_max"]) > 5:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if float(row["clip_frac"]) > 0.35:
        reasons.append("clip_frac")
    if abs(float(row["cand2_increment_corr"])) > 0.90:
        reasons.append("cand2_corr")
    if float(row["source_oof_delta"]) > -0.0045:
        reasons.append("weak_source_oof")
    if int(row["source_groups"]) < 9:
        reasons.append("weak_source_groups")
    if float(row["source_signal_corr"]) < 0.30:
        reasons.append("weak_source_signal")
    if int(row["positive_count"]) == 0 or int(row["negative_count"]) == 0:
        reasons.append("one_sided")
    return reasons


def select_diverse(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    passes = [r for r in rows if r["submit_gate"] == "pass"]
    selected: list[dict[str, object]] = []
    used_sources: set[str] = set()
    for row in passes:
        source = str(row["source_short"])
        if source in used_sources:
            continue
        selected.append(dict(row))
        used_sources.add(source)
        if len(selected) >= limit:
            break
    return selected


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 100.0
    return (
        penalty,
        -float(row["source_oof_delta"]),
        abs(float(row["cand2_increment_corr"]) - 0.55),
        float(row["species_shift"]),
    )


def max_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    vals = []
    for sp in np.unique(species):
        mask = species == sp
        vals.append(abs(float(np.mean(diff[mask]))))
    return max(vals) if vals else 0.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d * d)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def short_name(name: str) -> str:
    for prefix in ["nir_s1tn_pls_sg9_snv_", "knn_cluster_"]:
        name = name.replace(prefix, "")
    return name.split("_abs_signal")[0].split("_positive_signal")[0].replace(".", "p")


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
