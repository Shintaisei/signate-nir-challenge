"""生データの読み込み。"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from pipeline.config import CompetitionConfig
from pipeline.types import Rows


@dataclass
class RawData:
    train: Rows
    test: Rows


def load_raw_data(config: CompetitionConfig) -> RawData:
    train = _read_csv(config.train_path, config.encoding)
    test = _read_csv(config.test_path, config.encoding)
    return RawData(train=train, test=test)


def _read_csv(path, encoding: str) -> Rows:
    with path.open(encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))
