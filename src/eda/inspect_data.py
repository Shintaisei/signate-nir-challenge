#!/usr/bin/env python3
"""データ分布を確認し、評価設計用のプロファイルを出力する。"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.config import load_config  # noqa: E402


def main() -> None:
    config = load_config()
    train = _read_dicts(config.train_path, config.encoding)
    test = _read_dicts(config.test_path, config.encoding)

    target = [float(row[config.target_col]) for row in train]
    wavelength_cols = [col for col in train[0] if col not in set(config.meta_cols) | {config.target_col}]
    train_species = Counter(row["樹種"] for row in train)
    test_species = Counter(row["樹種"] for row in test)
    train_species_numbers = Counter(row["species number"] for row in train)
    test_species_numbers = Counter(row["species number"] for row in test)

    profile = {
        "n_train": len(train),
        "n_test": len(test),
        "n_train_columns": len(train[0]),
        "n_test_columns": len(test[0]),
        "target": _numeric_summary(target),
        "species": {
            "train_unique": len(train_species),
            "test_unique": len(test_species),
            "overlap": sorted(set(train_species) & set(test_species)),
            "train_counts": dict(train_species),
            "test_counts": dict(test_species),
            "target_by_train_species": _target_by_group(train, group_col="樹種", target_col=config.target_col),
        },
        "species_number": {
            "train_unique": len(train_species_numbers),
            "test_unique": len(test_species_numbers),
            "overlap": sorted(set(train_species_numbers) & set(test_species_numbers), key=int),
            "train_counts": dict(train_species_numbers),
            "test_counts": dict(test_species_numbers),
            "target_by_train_species_number": _target_by_group(
                train, group_col="species number", target_col=config.target_col
            ),
        },
        "wavelengths": {
            "n_columns": len(wavelength_cols),
            "first": wavelength_cols[0],
            "last": wavelength_cols[-1],
        },
    }

    out_path = config.outputs_dir / "logs" / "data_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(
        "species overlap train/test:",
        len(profile["species"]["overlap"]),
        "target mean:",
        f"{profile['target']['mean']:.4f}",
    )


def _read_dicts(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def _numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
        "std": statistics.pstdev(values),
    }


def _target_by_group(rows: list[dict[str, str]], *, group_col: str, target_col: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[group_col]].append(float(row[target_col]))
    return {
        group: {
            "count": len(values),
            **_numeric_summary(values),
        }
        for group, values in sorted(grouped.items())
    }


if __name__ == "__main__":
    main()
