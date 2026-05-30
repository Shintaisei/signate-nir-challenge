"""新しい前処理を追加するときのテンプレート。

1. このファイルをコピーして新クラスを実装
2. preprocessors/__init__.py の PREPROCESSOR_REGISTRY に登録
3. experiments/registry.py で experiment を定義
"""

from __future__ import annotations

from pipeline.types import Matrix, Rows, Vector


class PreprocessorTemplate:
    name = "template"

    def fit(self, train: Rows, *, target_col: str, meta_cols: tuple[str, ...]) -> None:
        raise NotImplementedError

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        raise NotImplementedError

    def transform_test(self, test: Rows) -> Matrix:
        raise NotImplementedError
