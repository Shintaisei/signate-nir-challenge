"""topk_sgd_linearの予測を学習y範囲でクリップする安定版。"""

from __future__ import annotations

from pipeline.models.topk_sgd_linear import TopKSgdLinearPredictor
from pipeline.types import Matrix, Vector


class TopKSgdLinearClippedPredictor:
    name = "topk_sgd_linear_clipped"

    def __init__(self) -> None:
        self._base = TopKSgdLinearPredictor(top_k=40, epochs=35, lr=0.01, seed=42)
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._base.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        pred = self._base.predict(X)
        return [min(max(v, self._y_min), self._y_max) for v in pred]
