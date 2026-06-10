#!/usr/bin/env python3
"""Focused test-like golden subset search.

This narrows the broad golden-subset experiment around the only branch that
looked structurally useful: MSC+SG9, kNN test-likeness, raw-target PLS.
"""

from __future__ import annotations

import nir_testlike_golden_distill_search as base


def focused_specs() -> list[base.TestLikeSpec]:
    specs: list[base.TestLikeSpec] = []
    for keep_frac in [0.75, 0.80, 0.85]:
        for n_components in [2, 3, 4]:
            specs.append(
                base.TestLikeSpec(
                    name=(
                        "pls_raw_msc_sg9_knn_"
                        f"keep{base.pct_tag(keep_frac)}_c{n_components}_focused"
                    ),
                    preprocess="msc_sg9",
                    model="pls_raw",
                    score_mode="knn",
                    keep_frac=keep_frac,
                    n_components=n_components,
                )
            )
    return specs


def focused_distills() -> list[base.DistillSpec]:
    return [
        base.DistillSpec(0.015, 0.08),
        base.DistillSpec(0.020, 0.08),
        base.DistillSpec(0.025, 0.08),
        base.DistillSpec(0.030, 0.08),
        base.DistillSpec(0.015, 0.10),
        base.DistillSpec(0.020, 0.10),
        base.DistillSpec(0.025, 0.10),
        base.DistillSpec(0.030, 0.10),
    ]


base.build_specs = focused_specs
base.build_distills = focused_distills


if __name__ == "__main__":
    base.main()
