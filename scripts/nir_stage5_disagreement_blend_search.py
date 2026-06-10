#!/usr/bin/env python3
"""Disagreement-based soft model-selection search around the cand2 public best."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "stage5_disagreement_blend"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"

CAND2 = SUBMISSION_DIR / "nir_stage5_q5cand2_knn_k6_q1_c4_cl1_f0p045_s0p015_c0p18_b0p48411_20260609.csv"


@dataclass(frozen=True)
class Family:
    name: str
    paths: tuple[Path, ...]
    role: str


@dataclass(frozen=True)
class Spec:
    name: str
    source_roles: tuple[str, ...]
    min_agree: int
    top_n: int
    shrink: float
    clip: float
    use_median: bool
    require_two_sided: bool


def main() -> None:
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE_PATH, header=None)
    species = pd.read_csv(TEST_PATH, encoding="cp932")["species number"].to_numpy()
    anchor = read_submission(CAND2, sample)
    families = load_families(sample)
    family_pred = {fam.name: family_average(fam, sample) for fam in families}
    family_diff = {name: pred - anchor for name, pred in family_pred.items()}

    specs = build_specs()
    rows: list[dict[str, object]] = []
    for spec in specs:
        pred, diag = build_candidate(spec, families, family_diff, anchor)
        row = diagnose(spec, pred, anchor, species, family_diff, diag)
        path = cand_dir / f"{spec.name}.csv"
        pd.DataFrame({0: sample[0].to_numpy(), 1: pred}).to_csv(path, header=False, index=False)
        row["path"] = str(path)
        rows.append(row)

    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(rows_sorted).to_csv(out_dir / "disagreement_blend_summary.csv", index=False)
    (out_dir / "disagreement_blend_summary.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = select_diverse(rows_sorted, limit=5)
    saved: list[dict[str, object]] = []
    for i, row in enumerate(selected, start=1):
        src = Path(str(row["path"]))
        dst = SUBMISSION_DIR / f"nir_stage5_disagree{i}_{row['experiment']}_20260610.csv"
        dst.write_bytes(src.read_bytes())
        out = dict(row)
        out["file"] = str(dst)
        saved.append(out)

    pd.DataFrame(saved).to_csv(out_dir / "disagreement_blend_selected.csv", index=False)
    pd.DataFrame(saved).to_csv(OUTPUT_ROOT / "disagreement_blend_selected_latest.csv", index=False)

    print(f"families: {', '.join(f.name for f in families)}")
    print(f"saved summary: {out_dir / 'disagreement_blend_summary.csv'}")
    print(f"passes: {sum(1 for r in rows_sorted if r['submit_gate'] == 'pass')}/{len(rows_sorted)}")
    print("selected:")
    for row in saved:
        print(
            f"  {row['file']} gate={row['submit_gate']} diff={row['cand2_diff_rmse']:.6f} "
            f"max={row['max_abs_diff']:.6f} changed={row['changed_count']} mean={row['mean_shift']:.6f} "
            f"posfrac={row['positive_frac']:.3f} broadcorr={row['broad_corr']:.3f} "
            f"poscorr={row['positive_corr']:.3f} reasons={row['reject_reasons']}"
        )


def load_families(sample: pd.DataFrame) -> list[Family]:
    candidates = [
        Family(
            "q_queue",
            tuple(
                p
                for p in [
                    SUBMISSION_DIR / "nir_tomorrow_q1_s4tn_safe_s4_k65_q1_c5_cl1_f0p035_s0p02_c0p18_b0p102886_20260608.csv",
                    SUBMISSION_DIR / "nir_tomorrow_q2_s4tn_safe_diverse_k75_k75_q1p6_c5_cl1_f0p04_s0p02_c0p18_b0p167011_20260608.csv",
                    SUBMISSION_DIR / "nir_tomorrow_q3_s4tn_safe_diverse_k55_k55_q1_c5_cl1_f0p04_s0p025_c0p18_b0p115147_20260608.csv",
                    SUBMISSION_DIR / "nir_tomorrow_q4_s4tn_attack_lite_lowcorr_k55_q1p3_c4_cl1_f0p04_s0p012_c0p18_b0p442354_20260608.csv",
                    SUBMISSION_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv",
                ]
                if p.exists()
            ),
            "trusted",
        ),
        Family(
            "q5_cand1",
            (SUBMISSION_DIR / "nir_stage5_q5cand1_knn_k6_q1_c3_cl1_f0p045_s0p015_c0p18_b0p536402_20260609.csv",),
            "trusted",
        ),
        Family(
            "q5_cand3",
            (SUBMISSION_DIR / "nir_stage5_q5cand3_knn_k65_q1_c3_cl1_f0p045_s0p012_c0p18_b0p632875_20260609.csv",),
            "trusted",
        ),
        Family(
            "foundation_yj",
            tuple(
                p
                for p in [
                    SUBMISSION_DIR / "nir_ext_base_yj_pca20_ridge3500.csv",
                    SUBMISSION_DIR / "nir_ext_base_yj_pca20_ridge3600.csv",
                    SUBMISSION_DIR / "nir_ext_orthopcrem1_pca20_yj_ridge3500.csv",
                    SUBMISSION_DIR / "nir_ext_orthopcrem2_pca20_yj_ridge3500.csv",
                ]
                if p.exists()
            ),
            "diverse",
        ),
        Family(
            "foundation_msc",
            tuple(
                p
                for p in [
                    SUBMISSION_DIR / "nir_ms_emsc1_raw_target_yeojohnson_pca20_ridge3500.csv",
                    SUBMISSION_DIR / "nir_msc_pca20_ridge2500.csv",
                    SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge2500.csv",
                ]
                if p.exists()
            ),
            "diverse",
        ),
        Family(
            "failed_broad",
            (SUBMISSION_DIR / "nir_stage5_broad_k55q13c4_attack_from_q5_after_cand2_20260610.csv",),
            "failed",
        ),
        Family(
            "failed_positive",
            (SUBMISSION_DIR / "nir_stage5_cand2_positive_attack_k55_after_broadfail_20260610.csv",),
            "failed",
        ),
    ]
    out = []
    for fam in candidates:
        valid_paths = []
        for path in fam.paths:
            if path.exists():
                read_submission(path, sample)
                valid_paths.append(path)
        if valid_paths:
            out.append(Family(fam.name, tuple(valid_paths), fam.role))
    return out


def build_specs() -> list[Spec]:
    specs: list[Spec] = []
    role_sets = [
        ("trusted",),
        ("trusted", "diverse"),
        ("trusted", "diverse", "failed"),
    ]
    for roles in role_sets:
        role_tag = "".join(role[0] for role in roles)
        for min_agree in [2, 3]:
            for top_n in [8, 12, 16, 22, 30, 40]:
                for shrink in [0.08, 0.12, 0.18, 0.25, 0.35]:
                    for clip in [0.04, 0.06, 0.08, 0.10, 0.13]:
                        for use_median in [False, True]:
                            for require_two_sided in [False, True]:
                                name = (
                                    f"disagree_{role_tag}_a{min_agree}_n{top_n}_"
                                    f"s{tag(shrink)}_c{tag(clip)}_"
                                    f"{'med' if use_median else 'mean'}_"
                                    f"{'twoside' if require_two_sided else 'free'}"
                                )
                                specs.append(Spec(name, roles, min_agree, top_n, shrink, clip, use_median, require_two_sided))
    return specs


def build_candidate(
    spec: Spec,
    families: list[Family],
    family_diff: dict[str, np.ndarray],
    anchor: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source_families = [fam for fam in families if fam.role in spec.source_roles]
    diffs = np.vstack([family_diff[fam.name] for fam in source_families])
    signs = np.sign(diffs)
    pos_votes = np.sum(signs > 1e-12, axis=0)
    neg_votes = np.sum(signs < -1e-12, axis=0)
    agree_votes = np.maximum(pos_votes, neg_votes)
    agree_sign = np.where(pos_votes >= neg_votes, 1.0, -1.0)
    signed = diffs * agree_sign
    aligned = np.where(signed > 0, signed, np.nan)
    signal = np.nanmedian(aligned, axis=0) if spec.use_median else np.nanmean(aligned, axis=0)
    signal = np.nan_to_num(signal, nan=0.0) * agree_sign
    eligible = agree_votes >= spec.min_agree
    score = np.abs(signal) * (agree_votes / max(1, len(source_families)))
    if spec.require_two_sided:
        pos_idx = np.flatnonzero(eligible & (signal > 1e-12))
        neg_idx = np.flatnonzero(eligible & (signal < -1e-12))
        half = max(1, spec.top_n // 2)
        chosen = np.concatenate(
            [
                pos_idx[np.argsort(-score[pos_idx])[:half]],
                neg_idx[np.argsort(-score[neg_idx])[: spec.top_n - half]],
            ]
        )
    else:
        idx = np.flatnonzero(eligible)
        chosen = idx[np.argsort(-score[idx])[: spec.top_n]]
    corr = np.zeros_like(anchor)
    corr[chosen] = np.clip(spec.shrink * signal[chosen], -spec.clip, spec.clip)
    pred = np.clip(anchor + corr, 0, None)
    return pred, {
        "corr": corr,
        "agree_votes": agree_votes,
        "score": score,
        "source_count": np.full_like(anchor, len(source_families), dtype=float),
    }


def diagnose(
    spec: Spec,
    pred: np.ndarray,
    anchor: np.ndarray,
    species: np.ndarray,
    family_diff: dict[str, np.ndarray],
    diag: dict[str, np.ndarray],
) -> dict[str, object]:
    diff = pred - anchor
    changed = np.abs(diff) > 1e-12
    top_idx = np.argsort(-np.abs(diff))[:10]
    species_counts = pd.Series(species[changed]).value_counts() if changed.any() else pd.Series(dtype=int)
    top_species_counts = pd.Series(species[top_idx[np.abs(diff[top_idx]) > 1e-12]]).value_counts()
    row: dict[str, object] = {
        "experiment": spec.name,
        "roles": "+".join(spec.source_roles),
        "min_agree": spec.min_agree,
        "top_n": spec.top_n,
        "shrink": spec.shrink,
        "clip": spec.clip,
        "use_median": spec.use_median,
        "two_sided": spec.require_two_sided,
        "changed_count": int(changed.sum()),
        "cand2_diff_rmse": rmse(pred, anchor),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_shift": float(np.mean(diff)),
        "p95_abs_diff": float(np.quantile(np.abs(diff), 0.95)),
        "p99_abs_diff": float(np.quantile(np.abs(diff), 0.99)),
        "positive_frac": float(np.mean(diff[changed] > 0)) if changed.any() else 0.0,
        "species_shift": max_species_shift(diff, species),
        "species_max": int(species_counts.iloc[0]) if len(species_counts) else 0,
        "top10_species_max": int(top_species_counts.iloc[0]) if len(top_species_counts) else 0,
        "negative_count": int((pred < 0).sum()),
        "broad_corr": safe_corr(diff, family_diff.get("failed_broad", np.zeros_like(diff))),
        "positive_corr": safe_corr(diff, family_diff.get("failed_positive", np.zeros_like(diff))),
        "q_queue_corr": safe_corr(diff, family_diff.get("q_queue", np.zeros_like(diff))),
        "cand1_corr": safe_corr(diff, family_diff.get("q5_cand1", np.zeros_like(diff))),
        "cand3_corr": safe_corr(diff, family_diff.get("q5_cand3", np.zeros_like(diff))),
        "mean_agree_votes": float(np.mean(diag["agree_votes"][changed])) if changed.any() else 0.0,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons)
    row["submit_gate"] = "pass" if not reasons else "reject"
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    if not (6 <= int(row["changed_count"]) <= 40):
        reasons.append("changed")
    if not (0.002 <= float(row["cand2_diff_rmse"]) <= 0.018):
        reasons.append("diff_rmse")
    if float(row["max_abs_diff"]) > 0.13:
        reasons.append("max_diff")
    if abs(float(row["mean_shift"])) > 0.004:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.010:
        reasons.append("species_shift")
    if int(row["species_max"]) > 9:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 5:
        reasons.append("top10_species")
    if float(row["positive_frac"]) in {0.0, 1.0} and int(row["changed_count"]) > 10:
        reasons.append("one_sided")
    if abs(float(row["broad_corr"])) > 0.92:
        reasons.append("broad_clone")
    if abs(float(row["positive_corr"])) > 0.92:
        reasons.append("positive_clone")
    return reasons


def select_diverse(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    passes = [row for row in rows if row["submit_gate"] == "pass"]
    selected: list[dict[str, object]] = []
    used_roles: set[str] = set()
    for row in passes:
        key = f"{row['roles']}:{row['two_sided']}:{row['use_median']}"
        if key in used_roles:
            continue
        selected.append(row)
        used_roles.add(key)
        if len(selected) >= limit:
            break
    return selected


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 100.0
    return (
        penalty,
        abs(float(row["positive_frac"]) - 0.5),
        float(row["cand2_diff_rmse"]),
        abs(float(row["broad_corr"])) + abs(float(row["positive_corr"])),
        float(row["species_shift"]),
    )


def family_average(fam: Family, sample: pd.DataFrame) -> np.ndarray:
    preds = [read_submission(path, sample) for path in fam.paths]
    return np.mean(np.vstack(preds), axis=0)


def read_submission(path: Path, sample: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    if df.shape != sample.shape:
        raise ValueError(f"shape mismatch: {path}")
    if not df[0].equals(sample[0]):
        raise ValueError(f"sample order mismatch: {path}")
    arr = df[1].to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite prediction: {path}")
    return arr


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d * d)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def max_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return max(abs(float(diff[species == sp].mean())) for sp in np.unique(species))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
