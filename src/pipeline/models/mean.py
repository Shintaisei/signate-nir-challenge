"""学習データの平均値を予測するベースラインモデル。"""

from __future__ import annotations

from statistics import mean

from pipeline.types import Matrix, Vector


class MeanPredictor:
    name = "mean"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X: Matrix, y: Vector) -> None:
        self._mean = float(mean(y))

    def predict(self, X: Matrix) -> Vector:
        if self._mean is None:
            raise RuntimeError("fit() を先に呼んでください")
        return [self._mean for _ in X]
