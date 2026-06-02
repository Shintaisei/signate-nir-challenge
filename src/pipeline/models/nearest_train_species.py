"""Nearest-train-species based predictors."""

from __future__ import annotations

import math

from pipeline.models.single_feature_linear import _fit_1d_linear, _select_top1_index
from pipeline.types import Matrix, Rows, Vector


def _species_key(row: dict[str, str]) -> str:
    return str(row.get("species", row.get("樹種", row.get("讓ｹ遞ｮ", ""))))


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
            raise RuntimeError("set_train_context() must be called before fit")
        self._y_min = min(y)
        self._y_max = max(y)
        self._models = {}
        self._train_centroids = {}

        by_species: dict[str, list[int]] = {}
        for i, row in enumerate(self._train_rows):
            by_species.setdefault(_species_key(row), []).append(i)

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
            grouped.setdefault(_species_key(row), []).append(i)
        centroids: dict[str, Vector] = {}
        for species, indices in grouped.items():
            centroids[species] = _mean_row([X_test[i] for i in indices])
        for test_species, centroid in centroids.items():
            self._test_to_train[test_species] = _nearest_train_species(centroid, self._train_centroids)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        self.set_test_rows(test_rows, X_test)
        out: Vector = []
        for row, features in zip(test_rows, X_test):
            test_species = _species_key(row)
            train_species = self._test_to_train.get(test_species)
            if train_species is None or train_species not in self._models:
                train_species = next(iter(self._models))
            feature_idx, slope, intercept = self._models[train_species]
            value = intercept + slope * features[feature_idx]
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("use predict_test(test_rows, X_test)")


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
    """Single-feature linear baseline with nearest-train-species residual bias correction."""

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
            raise RuntimeError("set_train_context() must be called before fit")
        self._y_min = min(y)
        self._y_max = max(y)
        self._linear.fit(X, y)
        preds = self._linear.predict(X)

        by_species: dict[str, list[int]] = {}
        for i, row in enumerate(self._train_rows):
            by_species.setdefault(_species_key(row), []).append(i)

        self._species_bias = {}
        self._train_centroids = {}
        for species, indices in by_species.items():
            residuals = [y[i] - preds[i] for i in indices]
            self._species_bias[species] = sum(residuals) / len(residuals)
            self._train_centroids[species] = _mean_row([X[i] for i in indices])

    def set_test_rows(self, test_rows: Rows, X_test: Matrix) -> None:
        grouped: dict[str, list[int]] = {}
        for i, row in enumerate(test_rows):
            grouped.setdefault(_species_key(row), []).append(i)
        for test_species, indices in grouped.items():
            centroid = _mean_row([X_test[i] for i in indices])
            self._test_to_train[test_species] = _nearest_train_species(centroid, self._train_centroids)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        self.set_test_rows(test_rows, X_test)
        preds = self._linear.predict(X_test)
        out: Vector = []
        for row, pred in zip(test_rows, preds):
            train_sp = self._test_to_train.get(_species_key(row), "")
            bias = self._species_bias.get(train_sp, 0.0)
            value = pred + bias
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("use predict_test(test_rows, X_test)")


class _SingleFeatureAntiNnBiasBlendPredictor:
    """baseline_1f + w * (nn_bias - baseline_1f) with configurable w."""

    name = "single_feature_anti_nn_bias_blend"
    blend_weight = -0.25

    def __init__(self) -> None:
        from pipeline.models.single_feature_linear import SingleFeatureLinearPredictor

        self._baseline = SingleFeatureLinearPredictor()
        self._nn_bias = NearestTrainSpeciesBiasCorrectedPredictor()
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_train_context(self, train_rows: Rows, config) -> None:
        self._nn_bias.set_train_context(train_rows, config)

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._baseline.fit(X, y)
        self._nn_bias.fit(X, y)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        baseline_pred = self._baseline.predict(X_test)
        nn_bias_pred = self._nn_bias.predict_test(test_rows, X_test)
        out: Vector = []
        for base, nn in zip(baseline_pred, nn_bias_pred):
            value = base + self.blend_weight * (nn - base)
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("use predict_test(test_rows, X_test)")


class SingleFeatureAntiNnBiasBlendWm025Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm025"
    blend_weight = -0.25


class SingleFeatureAntiNnBiasBlendWm035Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm035"
    blend_weight = -0.35


class SingleFeatureAntiNnBiasBlendWm040Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm040"
    blend_weight = -0.40


class SingleFeatureAntiNnBiasBlendWm045Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm045"
    blend_weight = -0.45


class SingleFeatureAntiNnBiasBlendWm050Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm050"
    blend_weight = -0.5


class SingleFeatureAntiNnBiasBlendWm075Predictor(_SingleFeatureAntiNnBiasBlendPredictor):
    name = "single_feature_anti_nn_bias_blend_wm075"
    blend_weight = -0.75


class _SingleFeatureAntiNnBiasAntiSnvBlendPredictor:
    """Combine the validated anti-nn-bias direction with anti-SNV25."""

    name = "single_feature_anti_nn_bias_anti_snv_blend"
    nn_weight = -0.40
    snv_weight = -0.25

    def __init__(self) -> None:
        from pipeline.models.single_feature_linear import SingleFeatureLinearPredictor

        self._baseline = SingleFeatureLinearPredictor()
        self._nn_bias = NearestTrainSpeciesBiasCorrectedPredictor()
        self._snv25 = SingleFeatureLinearPredictor()
        self._prep_snv25 = None
        self._train_rows = None
        self._config = None

    def set_train_context(self, train_rows: Rows, config) -> None:
        self._train_rows = train_rows
        self._config = config
        self._nn_bias.set_train_context(train_rows, config)

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._train_rows is None or self._config is None:
            raise RuntimeError("set_train_context() must be called before fit")
        from pipeline.preprocessors import get_preprocessor

        self._baseline.fit(X, y)
        self._nn_bias.fit(X, y)
        self._prep_snv25 = get_preprocessor("spectral_blend_snv25")
        self._prep_snv25.fit(
            self._train_rows,
            target_col=self._config.target_col,
            meta_cols=self._config.meta_cols,
        )
        X_snv25, y_snv25 = self._prep_snv25.transform_train(self._train_rows)
        self._snv25.fit(X_snv25, y_snv25)

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        if self._prep_snv25 is None:
            raise RuntimeError("fit() must be called before predict_test")
        baseline_pred = self._baseline.predict(X_test)
        nn_bias_pred = self._nn_bias.predict_test(test_rows, X_test)
        X_snv25 = self._prep_snv25.transform_test(test_rows)
        snv25_pred = self._snv25.predict(X_snv25)
        out: Vector = []
        for base, nn, snv in zip(baseline_pred, nn_bias_pred, snv25_pred):
            value = base + self.nn_weight * (nn - base) + self.snv_weight * (snv - base)
            out.append(value)
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("use predict_test(test_rows, X_test)")


class SingleFeatureAntiNnWm040AntiSnvWm025Predictor(_SingleFeatureAntiNnBiasAntiSnvBlendPredictor):
    name = "single_feature_anti_nn_wm040_anti_snv_wm025"
    nn_weight = -0.40
    snv_weight = -0.25


class SingleFeatureAntiNnWm040AntiSnvWm050Predictor(_SingleFeatureAntiNnBiasAntiSnvBlendPredictor):
    name = "single_feature_anti_nn_wm040_anti_snv_wm050"
    nn_weight = -0.40
    snv_weight = -0.50
