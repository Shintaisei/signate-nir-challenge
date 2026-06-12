#!/usr/bin/env python3
"""Sample-wise local calibration search after next5 #2 best.

This is a LOCAL / memory-based chemometrics probe: for each prediction point,
select nearby reliable train samples, fit a compact local latent model, then
distill only the OOF-supported residual signal into the current public-best
anchor.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import nir_next5_diverse_candidate_builder as n5
import nir_next5_followup_search as fu
import nir_operator_branch_distill_search as op
import nir_post_public_gate_search as ppg
import nir_slot1_testnear_branch_search as stn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_next5_local_memory"
SUBMISSION_DIR = ROOT / "data" / "submissions"
BEST = fu.BEST
CAND2 = fu.CAND2


@dataclass(frozen=True)
class LocalSpec:
    name: str
    preprocess: str
    latent_dim: int
    model: str
    n_components: int
    alpha: float
    neighbors: int
    pool_mult: float
    quality_power: float


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
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--spec-contains", action="append", default=None)
    args = parser.parse_args()

    data = op.load_data()
    best = n5.read_submission(BEST, data["test_ids"])
    cand2 = n5.read_submission(CAND2, data["test_ids"])
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    public_diag = load_public_delta_diag()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reconstructing current-best OOF proxy", flush=True)
    q5_oof = n5.build_q5_oof(data)
    best_corr_oof = fu.build_best2_corr_oof(data, q5_oof, cand2)
    best_oof = np.clip(q5_oof + best_corr_oof, 0, None)
    base_rmse = rmse(data["y"], best_oof)
    residual = data["y"] - best_oof
    print(f"best_oof_rmse={base_rmse:.6f}", flush=True)

    F_train, F_test = ppg.hmoe.make_current_features(data["X_train"], data["X_test"])
    cluster_train, cluster_test = n5.cluster_labels(F_train, F_test, n_clusters=64)
    detector_oof, detector_test = stn.build_spectral_detector_scores(data)

    specs = build_specs()
    if args.spec_contains:
        specs = [spec for spec in specs if any(token in spec.name for token in args.spec_contains)]
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"local specs={len(specs)}", flush=True)

    branch_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    candidates: list[Candidate] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            local_oof = local_memory_oof(spec, data, residual)
            local_test = local_memory_predict(
                spec,
                data["X_train"],
                data["y"],
                data["X_test"],
                residual,
            )
        except Exception as exc:
            print(f"  skip {type(exc).__name__}: {exc}", flush=True)
            continue

        signal_oof = local_oof - best_oof
        signal_test = local_test - best
        beta = n5.fit_beta(signal_oof, residual)
        signal_corr = n5.safe_corr(signal_oof, residual)
        branch_rmse = rmse(data["y"], local_oof)
        branch_rows.append(
            {
                "spec": spec.name,
                "preprocess": spec.preprocess,
                "latent_dim": spec.latent_dim,
                "model": spec.model,
                "n_components": spec.n_components,
                "alpha": spec.alpha,
                "neighbors": spec.neighbors,
                "pool_mult": spec.pool_mult,
                "quality_power": spec.quality_power,
                "branch_oof_rmse": branch_rmse,
                "branch_delta": branch_rmse - base_rmse,
                "signal_corr": signal_corr,
                "beta": beta,
                "test_diff_rmse": float(np.sqrt(np.mean(signal_test**2))),
                "test_diff_max": float(np.max(np.abs(signal_test))),
            }
        )
        if signal_corr < 0.03 and abs(beta) < 0.03:
            continue

        spec_candidates = candidate_variants(
            spec,
            best,
            best_oof,
            residual,
            signal_oof,
            signal_test,
            beta,
            signal_corr,
            cluster_train,
            cluster_test,
            F_train,
            F_test,
            detector_oof,
            detector_test,
            branch_rmse - base_rmse,
            data["groups"],
            test_species,
        )
        for cand in spec_candidates:
            row = diagnostics(cand, data["y"], best_oof, base_rmse, best, best - cand2, test_species, public_diag, spec)
            rows.append(row)
            candidates.append(cand)
            if row["submit_gate"] == "pass":
                print_one(row)

    n5.add_diversity_columns(rows, [to_n5_candidate(cand) for cand in candidates])
    rows_sorted = sorted(rows, key=rank_key)
    pd.DataFrame(branch_rows).sort_values(["signal_corr", "branch_delta"], ascending=[False, True]).to_csv(
        out_dir / "local_memory_branches.csv", index=False
    )
    pd.DataFrame(rows_sorted).to_csv(out_dir / "local_memory_summary.csv", index=False)
    (out_dir / "local_memory_summary.json").write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = select_submit_worthy(rows_sorted, args.save_top)
    by_name = {cand.experiment: cand for cand in candidates}
    saved: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        cand = by_name[str(row["experiment"])]
        path = SUBMISSION_DIR / f"nir_local_memory_{rank}_{n5.safe_name(cand.experiment)}_{args.date_tag}.csv"
        pd.DataFrame({0: data["test_ids"], 1: cand.pred}).to_csv(path, index=False, header=False)
        saved_row = dict(row)
        saved_row["rank"] = rank
        saved_row["file"] = str(path)
        saved_row["submission_memo"] = cand.memo
        saved.append(saved_row)

    manifest = pd.DataFrame(saved)
    manifest.to_csv(out_dir / "local_memory_selected_manifest.csv", index=False)
    manifest.to_csv(OUTPUT_ROOT / "local_memory_selected_manifest_latest.csv", index=False)
    (OUTPUT_ROOT / "local_memory_selected_manifest_latest.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nBranches:")
    print(pd.DataFrame(branch_rows).sort_values(["signal_corr", "branch_delta"], ascending=[False, True]).head(40).to_string(index=False))
    print("\nTop rows:")
    print(pd.DataFrame(rows_sorted[:80]).to_string(index=False))
    print("\nSelected:")
    print(manifest.to_string(index=False) if len(manifest) else "none")
    print(f"saved diagnostics: {out_dir}")


def build_specs() -> list[LocalSpec]:
    specs: list[LocalSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9", "sg11_snv", "sg9_snv_stack_d1"]:
        for latent_dim in [20, 35]:
            for neighbors in [45, 65, 90]:
                for quality_power in [0.0, 0.8, 1.4]:
                    specs.append(
                        LocalSpec(
                            name=(
                                f"lm_pls_{preprocess}_p{latent_dim}_"
                                f"k{neighbors}_q{stn.tag(quality_power)}_c5"
                            ),
                            preprocess=preprocess,
                            latent_dim=latent_dim,
                            model="pls",
                            n_components=5,
                            alpha=0.0,
                            neighbors=neighbors,
                            pool_mult=2.5,
                            quality_power=quality_power,
                        )
                    )
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for latent_dim in [20, 35]:
            for neighbors in [65, 90, 120]:
                for quality_power in [0.8, 1.4]:
                    specs.append(
                        LocalSpec(
                            name=(
                                f"lm_ridge_{preprocess}_p{latent_dim}_"
                                f"k{neighbors}_q{stn.tag(quality_power)}_a300"
                            ),
                            preprocess=preprocess,
                            latent_dim=latent_dim,
                            model="ridge",
                            n_components=0,
                            alpha=300.0,
                            neighbors=neighbors,
                            pool_mult=2.0,
                            quality_power=quality_power,
                        )
                    )
    return specs


def local_memory_oof(spec: LocalSpec, data: dict[str, np.ndarray], residual: np.ndarray) -> np.ndarray:
    pred = np.empty(len(data["y"]), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(data["groups"]))))
    for train_idx, valid_idx in splitter.split(data["X_train"], data["y"], data["groups"]):
        pred[valid_idx] = local_memory_predict(
            spec,
            data["X_train"][train_idx],
            data["y"][train_idx],
            data["X_train"][valid_idx],
            residual[train_idx],
        )
    return np.clip(pred, 0, None)


def local_memory_predict(
    spec: LocalSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pred_s = scaler.transform(X_pred)
    pca = PCA(n_components=min(spec.latent_dim, X_train_s.shape[0] - 1, X_train_s.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train_s)
    Z_pred = pca.transform(X_pred_s)
    pool = min(len(y), max(spec.neighbors + 5, int(round(spec.neighbors * spec.pool_mult))))
    nn = NearestNeighbors(n_neighbors=pool, metric="euclidean")
    nn.fit(Z_train)
    quality = np.power(1.0 - ppg.rank01(np.abs(residual)), spec.quality_power)
    out = np.empty(len(X_pred_raw), dtype=float)
    n_proto = min(max(8, len(Z_pred) // 14), 32, len(Z_pred))
    labels = KMeans(n_clusters=n_proto, random_state=42, n_init=10).fit_predict(Z_pred)
    for label in np.unique(labels):
        pred_idx = np.flatnonzero(labels == label)
        center = np.mean(Z_pred[pred_idx], axis=0, keepdims=True)
        distances, indices = nn.kneighbors(center, return_distance=True)
        idx = choose_local_indices(indices[0], distances[0], quality, spec.neighbors)
        out[pred_idx] = fit_local_model(spec, Z_train[idx], y[idx], Z_pred[pred_idx])
    return np.clip(out, 0, None)


def choose_local_indices(indices: np.ndarray, distances: np.ndarray, quality: np.ndarray, neighbors: int) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    d_rank = ppg.rank01(d)
    q = quality[indices]
    score = d_rank - 0.35 * ppg.rank01(q)
    order = np.argsort(score, kind="mergesort")
    return np.asarray(indices[order[: min(neighbors, len(order))]], dtype=int)


def fit_local_model(spec: LocalSpec, Z_train: np.ndarray, y: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    if spec.model == "pls":
        n_comp = min(spec.n_components, Z_train.shape[0] - 2, Z_train.shape[1])
        if n_comp < 1:
            return np.repeat(float(np.mean(y)), len(Z_pred))
        model = PLSRegression(n_components=n_comp, scale=True)
        model.fit(Z_train, y)
        return model.predict(Z_pred).ravel()
    if spec.model == "ridge":
        model = Ridge(alpha=spec.alpha)
        model.fit(Z_train, y)
        return model.predict(Z_pred)
    raise ValueError(spec.model)


def candidate_variants(
    spec: LocalSpec,
    best: np.ndarray,
    best_oof: np.ndarray,
    residual: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    beta: float,
    signal_corr: float,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    branch_delta: float,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> list[Candidate]:
    out: list[Candidate] = []
    for gate_mode in [
        "abs_signal_cluster1_space02",
        "abs_signal_cluster2_space02",
        "abs_signal_det15_cluster2",
        "abs_signal_det20_cluster2",
        "positive_signal_cluster1_space02",
        "negative_signal_cluster1_space02",
    ]:
        for frac in [0.03, 0.035, 0.05, 0.065, 0.08]:
            for shrink in [0.012, 0.018, 0.026, 0.038, 0.055]:
                for clip in [0.10, 0.14, 0.16, 0.22]:
                    for species_cap in [0, 2, 3]:
                        out.append(
                            make_gated_candidate(
                                spec,
                                gate_mode,
                                frac,
                                shrink,
                                clip,
                                species_cap,
                                best,
                                residual,
                                signal_oof,
                                signal_test,
                                beta,
                                signal_corr,
                                cluster_train,
                                cluster_test,
                                F_train,
                                F_test,
                                detector_oof,
                                detector_test,
                                branch_delta,
                                train_species,
                                test_species,
                            )
                        )
    for frac in [0.08, 0.12, 0.18]:
        for blend in [0.08, 0.14, 0.22]:
            for clip in [0.50, 0.80, 1.20]:
                out.append(
                    make_balanced_candidate(spec, frac, blend, clip, best, signal_oof, signal_test, signal_corr, branch_delta)
                )
    return out


def make_gated_candidate(
    spec: LocalSpec,
    gate_mode: str,
    frac: float,
    shrink: float,
    clip: float,
    species_cap: int,
    best: np.ndarray,
    residual: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    beta: float,
    signal_corr: float,
    cluster_train: np.ndarray,
    cluster_test: np.ndarray,
    F_train: np.ndarray,
    F_test: np.ndarray,
    detector_oof: np.ndarray,
    detector_test: np.ndarray,
    branch_delta: float,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> Candidate:
    gate_oof = stn.gate_from_signal(signal_oof, gate_mode, frac, cluster_train, F_train, detector_oof)
    gate_test = stn.gate_from_signal(signal_test, gate_mode, frac, cluster_test, F_test, detector_test)
    if species_cap > 0:
        gate_oof = cap_gate_by_label(signal_oof, gate_oof, train_species, species_cap)
        gate_test = cap_gate_by_label(signal_test, gate_test, test_species, species_cap)
    corr_oof = np.zeros(len(signal_oof), dtype=float)
    corr_test = np.zeros(len(signal_test), dtype=float)
    raw_oof = np.clip(shrink * beta * signal_oof, -clip, clip)
    raw_test = np.clip(shrink * beta * signal_test, -clip, clip)
    corr_oof[gate_oof] = raw_oof[gate_oof]
    corr_test[gate_test] = raw_test[gate_test]
    if np.any(gate_oof) and n5.safe_corr(corr_oof[gate_oof], residual[gate_oof]) < 0:
        corr_oof *= -1.0
        corr_test *= -1.0
    experiment = (
        f"lm_{spec.name}_{gate_mode}_f{stn.tag(frac)}_"
        f"s{stn.tag(shrink)}_c{stn.tag(clip)}_sp{species_cap}"
    )
    return Candidate(
        family="local_memory",
        experiment=experiment,
        pred=np.clip(best + corr_test, 0, None),
        corr_oof=corr_oof,
        corr_test=corr_test,
        beta=beta,
        signal_corr=signal_corr,
        memo=(
            f"#2 anchor; sample-wise LOCAL latent {spec.model}; {spec.preprocess}; "
            f"latent={spec.latent_dim}; k={spec.neighbors}; q={spec.quality_power}; "
            f"gate={gate_mode}; frac={frac}; shrink={shrink}; clip={clip}; "
            f"species_cap={species_cap}; branch_delta={branch_delta:.4f}; "
            f"signal_corr={signal_corr:.4f}; beta={beta:.4f}."
        ),
    )


def cap_gate_by_label(signal: np.ndarray, gate: np.ndarray, labels: np.ndarray, cap: int) -> np.ndarray:
    selected = np.flatnonzero(gate)
    if len(selected) == 0:
        return gate
    order = selected[np.argsort(np.abs(signal[selected]), kind="mergesort")[::-1]]
    counts: dict[int, int] = {}
    kept: list[int] = []
    for idx in order:
        label = int(labels[idx])
        if counts.get(label, 0) >= cap:
            continue
        kept.append(int(idx))
        counts[label] = counts.get(label, 0) + 1
    out = np.zeros(len(gate), dtype=bool)
    out[np.asarray(kept, dtype=int)] = True
    return out


def make_balanced_candidate(
    spec: LocalSpec,
    frac: float,
    blend: float,
    clip: float,
    best: np.ndarray,
    signal_oof: np.ndarray,
    signal_test: np.ndarray,
    signal_corr: float,
    branch_delta: float,
) -> Candidate:
    corr_oof = balanced_correction(signal_oof, blend, clip, frac)
    corr_test = balanced_correction(signal_test, blend, clip, frac)
    return Candidate(
        family="local_memory",
        experiment=f"lm_{spec.name}_balanced_f{stn.tag(frac)}_b{stn.tag(blend)}_c{stn.tag(clip)}",
        pred=np.clip(best + corr_test, 0, None),
        corr_oof=corr_oof,
        corr_test=corr_test,
        beta=blend,
        signal_corr=signal_corr,
        memo=(
            f"#2 anchor; balanced sample-wise LOCAL latent {spec.model}; {spec.preprocess}; "
            f"latent={spec.latent_dim}; k={spec.neighbors}; q={spec.quality_power}; "
            f"frac={frac}; blend={blend}; clip={clip}; branch_delta={branch_delta:.4f}; "
            f"signal_corr={signal_corr:.4f}."
        ),
    )


def balanced_correction(signal: np.ndarray, blend: float, clip: float, frac: float) -> np.ndarray:
    n_each = max(1, int(round(len(signal) * frac / 2.0)))
    pos_order = np.argsort(signal, kind="mergesort")[::-1]
    neg_order = np.argsort(signal, kind="mergesort")
    selected: list[int] = []
    selected.extend(int(i) for i in pos_order[:n_each] if signal[i] > 0)
    selected.extend(int(i) for i in neg_order[:n_each] if signal[i] < 0)
    corr = np.zeros(len(signal), dtype=float)
    if selected:
        idx = np.asarray(sorted(set(selected)), dtype=int)
        raw = np.clip(blend * signal[idx], -clip, clip)
        raw = raw - float(np.mean(raw))
        corr[idx] = raw
    return corr


def diagnostics(
    cand: Candidate,
    y: np.ndarray,
    best_oof: np.ndarray,
    base_rmse: float,
    best: np.ndarray,
    best_increment: np.ndarray,
    test_species: np.ndarray,
    public_diag: pd.DataFrame | None,
    spec: LocalSpec,
) -> dict[str, object]:
    row = fu.diagnostics(cand, y, best_oof, base_rmse, best, best_increment, test_species)
    diff = cand.pred - best
    changed = np.abs(diff) > 1e-12
    row.update(
        {
            "local_spec": spec.name,
            "bad_new_overlap_count": bad_new_overlap(changed, public_diag),
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
    if float(row["oof_delta"]) > -0.004:
        reasons.append("weak_oof")
    if float(row["signal_corr"]) < 0.30:
        reasons.append("weak_signal_corr")
    if not (0.006 <= float(row["anchor_diff_rmse"]) <= 0.035):
        reasons.append("diff_range")
    if float(row["anchor_diff_max"]) < 0.04:
        reasons.append("max_too_small")
    if float(row["anchor_diff_max"]) > 0.22:
        reasons.append("max_too_large")
    if int(row["changed_count"]) < 18 or int(row["changed_count"]) > 35:
        reasons.append("changed_count")
    if abs(float(row["anchor_diff_mean"])) > 0.015:
        reasons.append("mean_shift")
    if float(row["species_shift"]) > 0.06:
        reasons.append("species_shift")
    if int(row["species_max"]) > 6:
        reasons.append("species_concentration")
    if int(row["top10_species_max"]) > 3:
        reasons.append("top10_species")
    if int(row["bad_new_overlap_count"]) > 1:
        reasons.append("bad_new_overlap")
    if abs(float(row["best_increment_corr"])) > 0.80:
        reasons.append("too_correlated_best")
    return reasons


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    penalty = 0.0 if row["submit_gate"] == "pass" else 10.0
    target_diff = 0.018
    return (
        penalty + max(float(row["oof_delta"]) + 0.20, 0.0) + 0.35 * abs(float(row["anchor_diff_rmse"]) - target_diff),
        float(row["bad_new_overlap_count"]),
        float(row["species_shift"]),
        -float(row["signal_corr"]),
        -float(row["anchor_diff_rmse"]),
    )


def select_submit_worthy(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_specs: set[str] = set()
    for row in rows:
        if row["submit_gate"] != "pass":
            continue
        spec = str(row["local_spec"])
        if spec in used_specs:
            continue
        selected.append(row)
        used_specs.add(spec)
        if len(selected) >= limit:
            break
    return selected


def load_public_delta_diag() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "stage5_public_delta_diagnostics" / "row_public_delta_diagnostics.csv"
    return pd.read_csv(path) if path.exists() else None


def bad_new_overlap(changed: np.ndarray, diag: pd.DataFrame | None) -> int:
    if diag is None:
        return 0
    mask = np.zeros(len(changed), dtype=bool)
    if "failed_new_vs_cand2" in diag.columns:
        mask |= diag["failed_new_vs_cand2"].astype(bool).to_numpy()
    if "broad_changed" in diag.columns and "cand2_changed" in diag.columns:
        mask |= diag["broad_changed"].astype(bool).to_numpy() & ~diag["cand2_changed"].astype(bool).to_numpy()
    return int(np.sum(changed & mask))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2)))


def to_n5_candidate(cand: Candidate) -> n5.Candidate:
    return n5.Candidate(cand.family, cand.experiment, cand.pred, cand.corr_test, cand.corr_oof, cand.signal_corr, cand.beta, cand.memo)


def print_one(row: dict[str, object]) -> None:
    print(
        f"  PASS {row['experiment']}: oof={row['oof_delta']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max']:.4f} "
        f"changed={row['changed_count']} sp={row['species_shift']:.4f} "
        f"top10={row['top10_species_max']} bad={row['bad_new_overlap_count']}"
    )


if __name__ == "__main__":
    main()
