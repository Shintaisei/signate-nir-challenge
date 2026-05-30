#!/usr/bin/env python3
"""後方互換: baseline_mean 実験を実行。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.registry import resolve_experiment
from pipeline.runner import run_experiment


def main() -> None:
    preprocessor, predictor, memo = resolve_experiment("baseline_mean")
    run_experiment("baseline_mean", preprocessor, predictor, memo=memo)


if __name__ == "__main__":
    main()
