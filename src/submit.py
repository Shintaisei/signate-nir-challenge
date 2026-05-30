#!/usr/bin/env python3
"""後方互換: 提出 CSV を SIGNATE にアップロード。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.config import load_config
from pipeline.signate_submit import submit_to_signate


def main() -> None:
    config = load_config()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.submissions_dir / "baseline_mean.csv"
    memo = sys.argv[2] if len(sys.argv) > 2 else "manual submit"
    submit_to_signate(path, memo, config=config)


if __name__ == "__main__":
    main()
