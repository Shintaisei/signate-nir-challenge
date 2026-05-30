"""パイプライン各段のインターフェース。"""

from __future__ import annotations

from typing import Protocol

Rows = list[dict[str, str]]
Matrix = list[list[float]]
Vector = list[float]


class Preprocessor(Protocol):
    """前処理: train で fit し、train / test を変換する。"""

    name: str

    def fit(self, train: Rows, *, target_col: str, meta_cols: tuple[str, ...]) -> None:
        ...

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        """X_train, y_train を返す。"""
        ...

    def transform_test(self, test: Rows) -> Matrix:
        ...


class Predictor(Protocol):
    """学習・推論モデル。"""

    name: str

    def fit(self, X: Matrix, y: Vector) -> None:
        ...

    def predict(self, X: Matrix) -> Vector:
        ...
