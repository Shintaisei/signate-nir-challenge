"""新しいモデルを追加するときのテンプレート。

1. このファイルをコピーして新クラスを実装
2. models/__init__.py の PREDICTOR_REGISTRY に登録
3. experiments/registry.py で experiment を定義
"""

from __future__ import annotations

from pipeline.types import Matrix, Vector


class PredictorTemplate:
    name = "template"

    def fit(self, X: Matrix, y: Vector) -> None:
        raise NotImplementedError

    def predict(self, X: Matrix) -> Vector:
        raise NotImplementedError
