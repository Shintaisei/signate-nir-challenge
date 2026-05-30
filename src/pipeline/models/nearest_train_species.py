"""test樹種をスペクトル類似のtrain樹種に写像し、その樹種専用1波長モデルで予測する。"""

from __future__ import annotations

import math

from pipeline.models.single_feature_linear import _fit_1d_linear, _select_top1_index
from pipeline.types import Matrix, Rows, Vector


class NearestTrainSpeciesLinearPredictor:
    name = "nearest_train_species_linear"

    def __init__(self) -> None:
        self._train_rows: Rows | None = None
        self._models: dict[str, tuple[int, float, float]] = {}
        self._train_centroids: dict[str, Vector] = {}
        self._test_to_train: dict[str, str] = {}
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_train_context(self, train_rows: Rows, config) -> None:
        self._train_rows = train_rows

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._train_rows is None:
            raise RuntimeError("set_train_context() を fit 前に呼んでください")
        self._y_min = min(y)
        self._y_max = max(y)
        self._models = {}
        self._train_centroids = {}

        by_species: dict[str, list[int]] = {}
        for i, row in enumerate(self._train_rows):
            by_species.setdefault(str(row["樹種"]), []).append(i)

        for species, indices in by_species.items():
            X_sub = [X[i] for i in indices]
            y_sub = [y[i] for i in indices]
            feature_idx = _select_top1_index(X_sub, y_sub)
            x_col = [row[feature_idx] for row in X_sub]
            slope, intercept = _fit_1d_linear(x_col, y_sub)
            self._models[species] = (feature_idx, slope, intercept)
            self._train_centroids[species] = _mean_row(X_sub)

    def set_test_rows(self, test_rows: Rows, X_test: Matrix) -> None:
        grouped: dict[str, list[int]] = {}
        for i, row in enumerate(test_rows):
            grouped.setdefault(str(row["樹種"]), []).append(i)
        centroids: dict[str, Vector] = {}
        for species, indices in grouped.items():
            centroids[species] = _mean_row([X_test[i] for i in indices])
        for test_species, centroid in centroids.items():
            self._test_to_train[test_species] = _nearest_train_species(centroid, self._train_centroids)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        self.set_test_rows(test_rows, X_test)
        out: Vector = []
        for row, features in zip(test_rows, X_test):
            test_species = str(row["樹種"])
            train_species = self._test_to_train.get(test_species)
            if train_species is None or train_species not in self._models:
                train_species = next(iter(self._models))
            feature_idx, slope, intercept = self._models[train_species]
            value = intercept + slope * features[feature_idx]
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("predict_test(test_rows, X_test) を使用してください")


def _mean_row(rows: Matrix) -> Vector:
    if not rows:
        return []
    dim = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dim)]


def _nearest_train_species(centroid: Vector, train_centroids: dict[str, Vector]) -> str:
    best_name = next(iter(train_centroids))
    best_sim = -1.0
    for name, train_vec in train_centroids.items():
        sim = _cosine(centroid, train_vec)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-12)


class NearestTrainSpeciesBiasCorrectedPredictor:
    """1f 線形 + test樹種の最近傍 train 樹種における平均残差補正。"""

    name = "nearest_train_species_bias_corrected"

    def __init__(self) -> None:
        from pipeline.models.single_feature_linear import SingleFeatureLinearPredictor

        self._linear = SingleFeatureLinearPredictor()
        self._train_rows: Rows | None = None
        self._train_centroids: dict[str, Vector] = {}
        self._species_bias: dict[str, float] = {}
        self._test_to_train: dict[str, str] = {}
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_train_context(self, train_rows: Rows, config) -> None:
        self._train_rows = train_rows

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._train_rows is None:
            raise RuntimeError("set_train_context() を fit 前に呼んでください")
        self._y_min = min(y)
        self._y_max = max(y)
        self._linear.fit(X, y)
        preds = self._linear.predict(X)

        by_species: dict[str, list[int]] = {}
        for i, row in enumerate(self._train_rows):
            by_species.setdefault(str(row["樹種"]), []).append(i)

        self._species_bias = {}
        self._train_centroids = {}
        for species, indices in by_species.items():
            residuals = [y[i] - preds[i] for i in indices]
            self._species_bias[species] = sum(residuals) / len(residuals)
            self._train_centroids[species] = _mean_row([X[i] for i in indices])

    def set_test_rows(self, test_rows: Rows, X_test: Matrix) -> None:
        grouped: dict[str, list[int]] = {}
        for i, row in enumerate(test_rows):
            grouped.setdefault(str(row["樹種"]), []).append(i)
        for test_species, indices in grouped.items():
            centroid = _mean_row([X_test[i] for i in indices])
            self._test_to_train[test_species] = _nearest_train_species(centroid, self._train_centroids)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        self.set_test_rows(test_rows, X_test)
        preds = self._linear.predict(X_test)
        out: Vector = []
        for row, pred in zip(test_rows, preds):
            train_sp = self._test_to_train.get(str(row["樹種"]), "")
            bias = self._species_bias.get(train_sp, 0.0)
            value = pred + bias
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("predict_test(test_rows, X_test) を使用してください")
