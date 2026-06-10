#!/usr/bin/env python3
"""Public-delta diagnostics after cand2 saturation.

This script does not infer row-wise truth. It compares rows changed by the
public-improving cand2 submission against rows touched by later public-worse
candidates, then attaches unsupervised spectral diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SUB_DIR = ROOT / "data" / "submissions"
OUT_DIR = ROOT / "outputs" / "stage5_public_delta_diagnostics"

Q5 = SUB_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv"
CAND2 = SUB_DIR / "nir_stage5_q5cand2_knn_k6_q1_c4_cl1_f0p045_s0p015_c0p18_b0p48411_20260609.csv"
FAILED = {
    "broad": SUB_DIR / "nir_stage5_broad_k55q13c4_attack_from_q5_after_cand2_20260610.csv",
    "positive": SUB_DIR / "nir_stage5_cand2_positive_attack_k55_after_broadfail_20260610.csv",
    "disagree": SUB_DIR / "nir_stage5_disagreement_td_a2_n8_c0p10_twoside_20260610.csv",
}
REFERENCE = {
    "cand1": SUB_DIR / "nir_stage5_q5cand1_knn_k6_q1_c3_cl1_f0p045_s0p015_c0p18_b0p536402_20260609.csv",
    "cand3": SUB_DIR / "nir_stage5_q5cand3_knn_k65_q1_c3_cl1_f0p045_s0p012_c0p18_b0p632875_20260609.csv",
}

PUBLIC_SCORES = {
    "q5": 13.922966797992677,
    "cand2": 13.919389953803133,
    "broad": 13.919834478573494,
    "positive": 13.919930017586724,
    "disagree": 13.920448701428276,
}


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)


def read_submission(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["sample_number", "pred"])
    if ids is not None and not np.array_equal(df["sample_number"].to_numpy(), ids):
        raise ValueError(f"sample order mismatch: {path}")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv", encoding="cp932")
    test = pd.read_csv(RAW_DIR / "test.csv", encoding="cp932")
    sample = pd.read_csv(RAW_DIR / "sample_submit.csv", header=None)
    ids = sample[0].to_numpy()

    q5_df = read_submission(Q5, ids)
    cand2_df = read_submission(CAND2, ids)
    failed_df = {name: read_submission(path, ids) for name, path in FAILED.items()}
    ref_df = {name: read_submission(path, ids) for name, path in REFERENCE.items()}

    q5_pred = q5_df["pred"].to_numpy(float)
    cand2_pred = cand2_df["pred"].to_numpy(float)
    cand2_diff = cand2_pred - q5_pred
    cand2_changed = np.abs(cand2_diff) > 1e-12

    failed_diff = {
        name: df["pred"].to_numpy(float) - cand2_pred
        for name, df in failed_df.items()
    }
    ref_diff = {
        name: df["pred"].to_numpy(float) - cand2_pred
        for name, df in ref_df.items()
    }

    spectral = spectral_diagnostics(train, test)

    rows = pd.DataFrame(
        {
            "row": np.arange(len(ids)),
            "sample_number": ids,
            "species_number": test["species number"].to_numpy(),
            "species": test["樹種"].to_numpy(),
            "cand2_pred": cand2_pred,
            "q5_pred": q5_pred,
            "cand2_vs_q5_diff": cand2_diff,
            "cand2_changed": cand2_changed,
            "cand2_raise": cand2_diff > 1e-12,
            "cand2_lower": cand2_diff < -1e-12,
        }
    )
    for name, diff in failed_diff.items():
        rows[f"{name}_diff_vs_cand2"] = diff
        rows[f"{name}_changed"] = np.abs(diff) > 1e-12
        rows[f"{name}_same_sign_as_cand2"] = same_nonzero_sign(diff, cand2_diff)
        rows[f"{name}_opposite_sign_to_cand2"] = opposite_nonzero_sign(diff, cand2_diff)
    for name, diff in ref_diff.items():
        rows[f"{name}_diff_vs_cand2"] = diff
        rows[f"{name}_changed"] = np.abs(diff) > 1e-12
        rows[f"{name}_same_sign_as_cand2"] = same_nonzero_sign(diff, cand2_diff)
    for key, value in spectral.items():
        rows[key] = value

    failed_any = np.logical_or.reduce([rows[f"{name}_changed"].to_numpy(bool) for name in FAILED])
    failed_overlap_cand2 = failed_any & cand2_changed
    failed_new = failed_any & ~cand2_changed
    rows["failed_any_changed"] = failed_any
    rows["failed_overlap_cand2"] = failed_overlap_cand2
    rows["failed_new_vs_cand2"] = failed_new
    rows["failed_changed_count"] = sum(rows[f"{name}_changed"].astype(int) for name in FAILED)

    rows["public_proxy_bucket"] = "untouched"
    rows.loc[cand2_changed, "public_proxy_bucket"] = "cand2_success_proxy"
    rows.loc[failed_new, "public_proxy_bucket"] = "failed_new_proxy"
    rows.loc[failed_overlap_cand2, "public_proxy_bucket"] = "failed_overlap_proxy"

    rows.to_csv(OUT_DIR / "row_public_delta_diagnostics.csv", index=False)

    summary = build_summary(rows, failed_diff, ref_diff)
    (OUT_DIR / "public_delta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(summary, OUT_DIR / "public_delta_diagnostic_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def spectral_diagnostics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    meta = {"sample number", "species number", "樹種", "含水率"}
    wave_cols = [c for c in train.columns if c not in meta]
    X_train = train[wave_cols].to_numpy(float)
    X_test = test[wave_cols].to_numpy(float)

    Xtr = snv(X_train)
    Xte = snv(X_test)
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zte = scaler.transform(Xte)
    pca = PCA(n_components=20, random_state=42).fit(Ztr)
    Ttr = pca.transform(Ztr)
    Tte = pca.transform(Zte)
    recon = pca.inverse_transform(Tte)
    pca_q = np.sum((Zte - recon) ** 2, axis=1)
    nn = NearestNeighbors(n_neighbors=20).fit(Ttr)
    dist, _ = nn.kneighbors(Tte)
    knn_dist = dist.mean(axis=1)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Ttr)
    lof_score = -lof.score_samples(Tte)
    iso = IsolationForest(n_estimators=400, contamination="auto", random_state=42).fit(Ttr)
    iso_score = -iso.score_samples(Tte)
    detector = (rank01(pca_q) + rank01(knn_dist) + rank01(lof_score) + rank01(iso_score)) / 4.0
    clusters32 = KMeans(n_clusters=32, random_state=42, n_init=20).fit_predict(Tte)
    clusters64 = KMeans(n_clusters=64, random_state=42, n_init=20).fit_predict(Tte)
    pred_dummy = np.zeros(len(test))
    return {
        "pca_q_resid": pca_q,
        "knn_dist": knn_dist,
        "lof_score": lof_score,
        "isoforest_score": iso_score,
        "detector_score": detector,
        "test_cluster32": clusters32,
        "test_cluster64": clusters64,
        "spectral_pc1": Tte[:, 0],
        "spectral_pc2": Tte[:, 1],
        "spectral_pc3": Tte[:, 2],
        "_unused": pred_dummy,
    }


def same_nonzero_sign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12) & (np.sign(a) == np.sign(b))


def opposite_nonzero_sign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12) & (np.sign(a) != np.sign(b))


def build_summary(
    rows: pd.DataFrame,
    failed_diff: dict[str, np.ndarray],
    ref_diff: dict[str, np.ndarray],
) -> dict[str, object]:
    bucket_summary = rows.groupby("public_proxy_bucket").agg(
        n=("row", "size"),
        pred_mean=("cand2_pred", "mean"),
        pred_min=("cand2_pred", "min"),
        pred_max=("cand2_pred", "max"),
        detector_mean=("detector_score", "mean"),
        detector_p90=("detector_score", lambda x: float(np.quantile(x, 0.90))),
        pca_q_mean=("pca_q_resid", "mean"),
        knn_dist_mean=("knn_dist", "mean"),
        cand2_abs_mean=("cand2_vs_q5_diff", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
    species_bucket = rows.groupby(["public_proxy_bucket", "species_number", "species"]).agg(
        n=("row", "size"),
        pred_mean=("cand2_pred", "mean"),
        detector_mean=("detector_score", "mean"),
        cand2_abs_mean=("cand2_vs_q5_diff", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index().sort_values(["public_proxy_bucket", "n"], ascending=[True, False])
    cluster_bucket = rows.groupby(["public_proxy_bucket", "test_cluster64"]).agg(
        n=("row", "size"),
        pred_mean=("cand2_pred", "mean"),
        detector_mean=("detector_score", "mean"),
    ).reset_index().sort_values(["public_proxy_bucket", "n"], ascending=[True, False])
    pred_bins = pd.qcut(rows["cand2_pred"], q=10, duplicates="drop")
    pred_bin_bucket = rows.assign(pred_bin=pred_bins.astype(str)).groupby(["pred_bin", "public_proxy_bucket"]).agg(
        n=("row", "size"),
        detector_mean=("detector_score", "mean"),
    ).reset_index()
    null_comparison = random_null_comparison(rows, n=int(rows["cand2_changed"].sum()), repeats=2000)

    overlaps = {}
    for name in FAILED:
        changed = rows[f"{name}_changed"].to_numpy(bool)
        overlaps[name] = {
            "changed": int(changed.sum()),
            "overlap_cand2_changed": int((changed & rows["cand2_changed"].to_numpy(bool)).sum()),
            "new_vs_cand2": int((changed & ~rows["cand2_changed"].to_numpy(bool)).sum()),
            "same_sign_as_cand2": int(rows[f"{name}_same_sign_as_cand2"].sum()),
            "opposite_sign_to_cand2": int(rows[f"{name}_opposite_sign_to_cand2"].sum()),
            "mean_abs_diff": float(np.mean(np.abs(failed_diff[name]))),
            "max_abs_diff": float(np.max(np.abs(failed_diff[name]))),
        }

    correlations = {}
    all_diffs = {**failed_diff, **ref_diff, "cand2_vs_q5": rows["cand2_vs_q5_diff"].to_numpy(float)}
    for a_name, a in all_diffs.items():
        correlations[a_name] = {}
        for b_name, b in all_diffs.items():
            correlations[a_name][b_name] = safe_corr(a, b)

    top_rows = {}
    for flag in ["cand2_changed", "failed_new_vs_cand2", "failed_overlap_cand2"]:
        cols = [
            "row", "sample_number", "species_number", "species", "cand2_pred",
            "cand2_vs_q5_diff", "broad_diff_vs_cand2", "positive_diff_vs_cand2",
            "disagree_diff_vs_cand2", "detector_score", "test_cluster64",
        ]
        top_rows[flag] = rows[rows[flag]].sort_values(
            ["detector_score", "cand2_pred"], ascending=[False, False]
        )[cols].head(25).to_dict(orient="records")
    touched_any = rows["cand2_changed"] | rows["failed_any_changed"]
    high_risk_control = rows[~touched_any].sort_values("detector_score", ascending=False).head(40)[
        [
            "row", "sample_number", "species_number", "species", "cand2_pred",
            "detector_score", "test_cluster64", "pca_q_resid", "knn_dist",
            "lof_score", "isoforest_score",
        ]
    ]

    return {
        "public_scores": PUBLIC_SCORES,
        "public_deltas_vs_cand2": {k: v - PUBLIC_SCORES["cand2"] for k, v in PUBLIC_SCORES.items()},
        "counts": {
            "cand2_changed_vs_q5": int(rows["cand2_changed"].sum()),
            "failed_any_changed_vs_cand2": int(rows["failed_any_changed"].sum()),
            "failed_new_vs_cand2": int(rows["failed_new_vs_cand2"].sum()),
            "failed_overlap_cand2": int(rows["failed_overlap_cand2"].sum()),
        },
        "failed_candidate_overlaps": overlaps,
        "bucket_summary": bucket_summary.to_dict(orient="records"),
        "species_bucket_top": species_bucket.groupby("public_proxy_bucket").head(10).to_dict(orient="records"),
        "cluster_bucket_top": cluster_bucket.groupby("public_proxy_bucket").head(10).to_dict(orient="records"),
        "pred_bin_bucket": pred_bin_bucket.to_dict(orient="records"),
        "null_comparison": null_comparison,
        "diff_correlations": correlations,
        "top_rows": top_rows,
        "unchanged_high_risk_control_top40": high_risk_control.to_dict(orient="records"),
        "artifacts": {
            "row_csv": str(OUT_DIR / "row_public_delta_diagnostics.csv"),
            "report_md": str(OUT_DIR / "public_delta_diagnostic_report.md"),
        },
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    counts = summary["counts"]
    deltas = summary["public_deltas_vs_cand2"]
    lines = [
        "# Stage5 Public Delta Diagnostic",
        "",
        "## Score Readout",
        "",
        f"- cand2 is current best: `{summary['public_scores']['cand2']}`.",
        f"- broad delta vs cand2: `{deltas['broad']}`.",
        f"- positive-only delta vs cand2: `{deltas['positive']}`.",
        f"- disagreement delta vs cand2: `{deltas['disagree']}`.",
        "",
        "## Proxy Counts",
        "",
        f"- cand2 changed vs q5: `{counts['cand2_changed_vs_q5']}` rows.",
        f"- later failed candidates changed any vs cand2: `{counts['failed_any_changed_vs_cand2']}` rows.",
        f"- failed candidates introduced new rows outside cand2 set: `{counts['failed_new_vs_cand2']}` rows.",
        f"- failed candidates overlapped cand2 changed rows: `{counts['failed_overlap_cand2']}` rows.",
        "",
        "## Interpretation Guardrail",
        "",
        "These are proxy labels from public submissions, not row-level truth.",
        "Use them to avoid repeating known-bad movement patterns, not to claim a row is truly correct or incorrect.",
        "",
        "## Bucket Summary",
        "",
        pd.DataFrame(summary["bucket_summary"]).to_markdown(index=False),
        "",
        "## Null Comparison",
        "",
        pd.DataFrame(summary["null_comparison"]).to_markdown(index=False),
        "",
        "## Failed Candidate Overlaps",
        "",
        pd.DataFrame(summary["failed_candidate_overlaps"]).T.to_markdown(),
        "",
        "## Next Direction",
        "",
        "- Stop broadening the cand2 hard residual set unless the new rows are clearly outside the failed-new proxy pattern.",
        "- Treat positive-only correction as known-bad for now; require two-sided movement or a stronger independent model signal.",
        "- Prefer diagnostics that isolate cand2-overlap rows from failed-new rows using spectral clusters and prediction bands.",
        "- Inspect unchanged high-risk rows separately; they are candidate detector targets but must not be corrected by detector alone.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def random_null_comparison(rows: pd.DataFrame, *, n: int, repeats: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(42)
    metrics = {
        "detector_mean": rows["detector_score"].to_numpy(float),
        "pred_mean": rows["cand2_pred"].to_numpy(float),
        "pca_q_mean": rows["pca_q_resid"].to_numpy(float),
        "knn_dist_mean": rows["knn_dist"].to_numpy(float),
    }
    masks = {
        "cand2_changed": rows["cand2_changed"].to_numpy(bool),
        "failed_new_vs_cand2": rows["failed_new_vs_cand2"].to_numpy(bool),
        "failed_overlap_cand2": rows["failed_overlap_cand2"].to_numpy(bool),
    }
    null_stats: dict[str, list[float]] = {key: [] for key in metrics}
    idx_all = np.arange(len(rows))
    for _ in range(repeats):
        idx = rng.choice(idx_all, size=n, replace=False)
        for key, values in metrics.items():
            null_stats[key].append(float(np.mean(values[idx])))
    out = []
    for label, mask in masks.items():
        if not np.any(mask):
            continue
        for key, values in metrics.items():
            observed = float(np.mean(values[mask]))
            null = np.asarray(null_stats[key])
            out.append(
                {
                    "bucket": label,
                    "metric": key,
                    "observed": observed,
                    "null_mean": float(np.mean(null)),
                    "null_p05": float(np.quantile(null, 0.05)),
                    "null_p95": float(np.quantile(null, 0.95)),
                    "z_approx": float((observed - np.mean(null)) / max(np.std(null), 1e-12)),
                }
            )
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    main()
