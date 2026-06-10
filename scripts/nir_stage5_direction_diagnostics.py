#!/usr/bin/env python3
"""Diagnostics for choosing the next Stage5 direction after q5 public win."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SUB_DIR = ROOT / "data" / "submissions"
OUT_DIR = ROOT / "outputs" / "stage5_direction"

STAGE3 = SUB_DIR / "nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv"
QUEUE = [
    SUB_DIR / "nir_tomorrow_q1_s4tn_safe_s4_k65_q1_c5_cl1_f0p035_s0p02_c0p18_b0p102886_20260608.csv",
    SUB_DIR / "nir_tomorrow_q2_s4tn_safe_diverse_k75_k75_q1p6_c5_cl1_f0p04_s0p02_c0p18_b0p167011_20260608.csv",
    SUB_DIR / "nir_tomorrow_q3_s4tn_safe_diverse_k55_k55_q1_c5_cl1_f0p04_s0p025_c0p18_b0p115147_20260608.csv",
    SUB_DIR / "nir_tomorrow_q4_s4tn_attack_lite_lowcorr_k55_q1p3_c4_cl1_f0p04_s0p012_c0p18_b0p442354_20260608.csv",
    SUB_DIR / "nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv",
]


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=["sample number", "pred"])


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW_DIR / "train.csv", encoding="cp932")
    test = pd.read_csv(RAW_DIR / "test.csv", encoding="cp932")
    wave_cols = [c for c in train.columns if c not in {"sample number", "species number", "樹種", "含水率"}]
    X_train = train[wave_cols].to_numpy(float)
    X_test = test[wave_cols].to_numpy(float)

    stage3 = read_submission(STAGE3)
    qdfs = [read_submission(p) for p in QUEUE]
    for q in qdfs:
        if not stage3["sample number"].equals(q["sample number"]):
            raise ValueError("sample order mismatch")

    ids = stage3["sample number"].to_numpy()
    base_pred = stage3["pred"].to_numpy(float)
    qpreds = [q["pred"].to_numpy(float) for q in qdfs]
    qdiffs = [pred - base_pred for pred in qpreds]
    q5diff = qdiffs[-1]
    changed = [np.abs(d) > 1e-12 for d in qdiffs]

    # Unsupervised spectral/test-nearness diagnostics.
    Xtr = snv(X_train)
    Xte = snv(X_test)
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zte = scaler.transform(Xte)

    pca = PCA(n_components=20, random_state=42).fit(Ztr)
    Ttr = pca.transform(Ztr)
    Tte = pca.transform(Zte)
    Xte_recon = pca.inverse_transform(Tte)
    q_resid = np.sum((Zte - Xte_recon) ** 2, axis=1)

    nn = NearestNeighbors(n_neighbors=20).fit(Ttr)
    dist, _ = nn.kneighbors(Tte)
    knn_dist = dist.mean(axis=1)

    lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Ttr)
    lof_score = -lof.score_samples(Tte)
    iso = IsolationForest(n_estimators=400, contamination="auto", random_state=42).fit(Ttr)
    iso_score = -iso.score_samples(Tte)
    test_clusters = KMeans(n_clusters=64, random_state=42, n_init=20).fit_predict(Tte)
    detector = (
        rank01(q_resid)
        + rank01(knn_dist)
        + rank01(lof_score)
        + rank01(iso_score)
    ) / 4.0

    rows = pd.DataFrame(
        {
            "row": np.arange(len(ids)),
            "sample_number": ids,
            "species_number": test["species number"].to_numpy(),
            "species": test["樹種"].to_numpy(),
            "stage3_pred": base_pred,
            "q1_diff": qdiffs[0],
            "q2_diff": qdiffs[1],
            "q3_diff": qdiffs[2],
            "q4_diff": qdiffs[3],
            "q5_diff": qdiffs[4],
            "q5_abs_diff": np.abs(q5diff),
            "changed_any": np.logical_or.reduce(changed),
            "changed_count": np.sum(np.vstack(changed), axis=0),
            "q5_changed": changed[-1],
            "pca_q_resid": q_resid,
            "knn_dist": knn_dist,
            "lof_score": lof_score,
            "isoforest_score": iso_score,
            "detector_score": detector,
            "test_cluster64": test_clusters,
        }
    )
    rows["detector_rank"] = rows["detector_score"].rank(ascending=False, method="first").astype(int)
    rows["q5_abs_rank"] = rows["q5_abs_diff"].rank(ascending=False, method="first").astype(int)
    rows.to_csv(OUT_DIR / "stage5_row_diagnostics.csv", index=False)

    q5_rows = rows[rows["q5_changed"]].sort_values("q5_abs_diff", ascending=False)
    uncorrected_risky = rows[~rows["changed_any"]].sort_values("detector_score", ascending=False).head(50)
    corrected_summary = rows.groupby(["changed_count", "q5_changed"]).agg(
        n=("row", "size"),
        pred_mean=("stage3_pred", "mean"),
        detector_mean=("detector_score", "mean"),
        detector_max=("detector_score", "max"),
        q5_abs_mean=("q5_abs_diff", "mean"),
    ).reset_index()
    species_summary = rows[rows["changed_any"]].groupby(["species_number", "species"]).agg(
        n_changed=("row", "size"),
        n_q5=("q5_changed", "sum"),
        mean_detector=("detector_score", "mean"),
        mean_q5_abs=("q5_abs_diff", "mean"),
    ).reset_index().sort_values(["n_q5", "n_changed", "mean_q5_abs"], ascending=False)
    pred_bins = pd.qcut(rows["stage3_pred"], q=10, duplicates="drop")
    pred_bin_summary = rows.assign(pred_bin=pred_bins.astype(str)).groupby("pred_bin").agg(
        n=("row", "size"),
        changed_any_rate=("changed_any", "mean"),
        q5_changed_rate=("q5_changed", "mean"),
        detector_mean=("detector_score", "mean"),
        detector_max=("detector_score", "max"),
        pred_min=("stage3_pred", "min"),
        pred_max=("stage3_pred", "max"),
    ).reset_index()
    changed_detector_summary = rows.groupby(["changed_any", "q5_changed"]).agg(
        n=("row", "size"),
        detector_mean=("detector_score", "mean"),
        detector_median=("detector_score", "median"),
        detector_p90=("detector_score", lambda x: float(np.quantile(x, 0.90))),
        q5_abs_mean=("q5_abs_diff", "mean"),
    ).reset_index()
    component_corr = rows[
        ["q5_abs_diff", "pca_q_resid", "knn_dist", "lof_score", "isoforest_score", "detector_score"]
    ].corr()
    cluster_summary = rows.groupby("test_cluster64").agg(
        n=("row", "size"),
        n_changed_any=("changed_any", "sum"),
        n_q5=("q5_changed", "sum"),
        detector_mean=("detector_score", "mean"),
        detector_max=("detector_score", "max"),
        pred_mean=("stage3_pred", "mean"),
    ).reset_index().sort_values(["detector_max", "detector_mean"], ascending=False)
    top_detector_species = uncorrected_risky.head(50).groupby(["species_number", "species"]).size().reset_index(name="n")
    top_detector_clusters = uncorrected_risky.head(50).groupby("test_cluster64").size().reset_index(name="n")

    q5_rows.to_csv(OUT_DIR / "q5_changed_rows.csv", index=False)
    uncorrected_risky.to_csv(OUT_DIR / "uncorrected_high_detector_rows.csv", index=False)
    corrected_summary.to_csv(OUT_DIR / "correction_overlap_summary.csv", index=False)
    species_summary.to_csv(OUT_DIR / "corrected_species_summary.csv", index=False)
    pred_bin_summary.to_csv(OUT_DIR / "pred_bin_detector_summary.csv", index=False)
    changed_detector_summary.to_csv(OUT_DIR / "changed_detector_summary.csv", index=False)
    component_corr.to_csv(OUT_DIR / "detector_component_corr.csv")
    cluster_summary.to_csv(OUT_DIR / "test_cluster64_detector_summary.csv", index=False)
    top_detector_species.to_csv(OUT_DIR / "top_uncorrected_detector_species.csv", index=False)
    top_detector_clusters.to_csv(OUT_DIR / "top_uncorrected_detector_clusters.csv", index=False)

    summary = {
        "q5_changed_count": int(rows["q5_changed"].sum()),
        "changed_any_count": int(rows["changed_any"].sum()),
        "never_changed_count": int((~rows["changed_any"]).sum()),
        "q5_changed_detector_mean": float(rows.loc[rows["q5_changed"], "detector_score"].mean()),
        "never_changed_detector_top_mean_20": float(uncorrected_risky.head(20)["detector_score"].mean()),
        "q5_changed_species_counts": q5_rows["species_number"].value_counts().to_dict(),
        "detector_abs_q5_corr": float(rows[["detector_score", "q5_abs_diff"]].corr().iloc[0, 1]),
        "component_corr_with_detector": component_corr["detector_score"].to_dict(),
        "top_uncorrected_species_counts_top50": top_detector_species.to_dict(orient="records"),
        "top_uncorrected_cluster_counts_top50": top_detector_clusters.sort_values("n", ascending=False).head(10).to_dict(orient="records"),
        "changed_detector_summary": changed_detector_summary.to_dict(orient="records"),
        "top_pred_bins_by_detector": pred_bin_summary.sort_values("detector_mean", ascending=False).head(5).to_dict(orient="records"),
        "top_uncorrected_detector_rows": uncorrected_risky.head(15)[
            ["row", "sample_number", "species_number", "species", "stage3_pred", "detector_score", "test_cluster64", "pca_q_resid", "knn_dist", "lof_score", "isoforest_score"]
        ].to_dict(orient="records"),
        "top_q5_rows": q5_rows.head(15)[
            ["row", "sample_number", "species_number", "species", "stage3_pred", "q5_diff", "detector_score", "changed_count"]
        ].to_dict(orient="records"),
    }
    (OUT_DIR / "stage5_direction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
