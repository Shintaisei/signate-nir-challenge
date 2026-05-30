"""スペクトル列のみを特徴量として抽出する前処理。"""

from __future__ import annotations

from pipeline.types import Matrix, Rows, Vector


class SpectralPreprocessor:
    name = "spectral"

    def __init__(self) -> None:
        self._feature_cols: list[str] = []
        self._target_col: str = ""

    def fit(self, train: Rows, *, target_col: str, meta_cols: tuple[str, ...]) -> None:
        self._target_col = target_col
        exclude = set(meta_cols) | {target_col}
        self._feature_cols = [c for c in train[0].keys() if c not in exclude]

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        if not self._feature_cols:
            raise RuntimeError("fit() を先に呼んでください")
        X = [[float(row[col]) for col in self._feature_cols] for row in train]
        y = [float(row[self._target_col]) for row in train]
        return X, y

    def transform_test(self, test: Rows) -> Matrix:
        if not self._feature_cols:
            raise RuntimeError("fit() を先に呼んでください")
        missing = set(self._feature_cols) - set(test[0].keys())
        if missing:
            raise ValueError(f"test に欠けている特徴量列: {sorted(missing)[:5]} ...")
        return [[float(row[col]) for col in self._feature_cols] for row in test]
