"""相関で特徴選択してから安定した線形モデルを当てる。"""

from __future__ import annotations

import math

from pipeline.types import Matrix, Vector


class TopKFeatureRidgePredictor:
    name = "topk_ridge"

    def __init__(
        self,
        top_k: int = 8,
        alpha: float = 100.0,
        clip: bool = True,
        selection: str = "correlation",
    ) -> None:
        self.top_k = top_k
        self.alpha = alpha
        self.clip = clip
        self.selection = selection
        self._indices: list[int] = []
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        k = min(self.top_k, len(X[0]))
        if self.selection == "stable":
            self._indices = _select_stable_topk_indices(X, y, k)
        else:
            self._indices = _select_topk_indices(X, y, k)
        self._y_min = min(y)
        self._y_max = max(y)
        X_sel = _select_columns(X, self._indices)
        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(X_sel, y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        pred = [float(value) for value in self._model.predict(_select_columns(X, self._indices))]
        if not self.clip:
            return pred
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKFeatureRidge3Predictor(TopKFeatureRidgePredictor):
    name = "topk3_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=3, alpha=100.0)


class TopKFeatureRidge4Predictor(TopKFeatureRidgePredictor):
    name = "topk4_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=4, alpha=100.0)


class TopKFeatureRidge5Predictor(TopKFeatureRidgePredictor):
    name = "topk5_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=5, alpha=100.0)


class TopKFeatureRidge6Predictor(TopKFeatureRidgePredictor):
    name = "topk6_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0)


class TopKFeatureRidge8Predictor(TopKFeatureRidgePredictor):
    name = "topk8_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=8, alpha=100.0)


class TopKFeatureRidge12Predictor(TopKFeatureRidgePredictor):
    name = "topk12_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=12, alpha=100.0)


class TopKFeatureRidge5Alpha30Predictor(TopKFeatureRidgePredictor):
    name = "topk5_ridge_a30"

    def __init__(self) -> None:
        super().__init__(top_k=5, alpha=30.0)


class TopKFeatureRidge5Alpha300Predictor(TopKFeatureRidgePredictor):
    name = "topk5_ridge_a300"

    def __init__(self) -> None:
        super().__init__(top_k=5, alpha=300.0)


class TopKFeatureStableRidge5Predictor(TopKFeatureRidgePredictor):
    name = "topk5_stable_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=5, alpha=100.0, selection="stable")


class TopKFeatureRidgeCVPredictor:
    name = "topk_ridge_cv"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._indices: list[int] = []
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._indices = _select_topk_indices(X, y, min(self.top_k, len(X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        self._model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=[3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]),
        )
        self._model.fit(_select_columns(X, self._indices), y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        pred = [float(value) for value in self._model.predict(_select_columns(X, self._indices))]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKFeatureElasticNetPredictor:
    name = "topk_elastic"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._indices: list[int] = []
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._indices = _select_topk_indices(X, y, min(self.top_k, len(X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        self._model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=10000),
        )
        self._model.fit(_select_columns(X, self._indices), y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        pred = [float(value) for value in self._model.predict(_select_columns(X, self._indices))]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKRidgeEnsemblePredictor:
    name = "topk_ridge_ensemble"

    def __init__(self) -> None:
        self._models = [
            TopKFeatureRidgePredictor(top_k=3, alpha=100.0),
            TopKFeatureRidgePredictor(top_k=5, alpha=30.0),
            TopKFeatureRidgePredictor(top_k=5, alpha=100.0),
            TopKFeatureRidgePredictor(top_k=6, alpha=100.0),
            TopKFeatureRidgePredictor(top_k=8, alpha=300.0),
        ]

    def fit(self, X: Matrix, y: Vector) -> None:
        for model in self._models:
            model.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        predictions = [model.predict(X) for model in self._models]
        return [sum(values) / len(values) for values in zip(*predictions)]


class TopKWeightedRidgePredictor(TopKFeatureRidgePredictor):
    name = "topk_weighted_ridge"

    def __init__(
        self,
        top_k: int = 6,
        alpha: float = 100.0,
        high_weight: float = 2.0,
        quantile: float = 0.7,
    ) -> None:
        super().__init__(top_k=top_k, alpha=alpha)
        self.high_weight = high_weight
        self.quantile = quantile

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._indices = _select_topk_indices(X, y, min(self.top_k, len(X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        threshold = _quantile(y, self.quantile)
        sample_weight = [self.high_weight if value >= threshold else 1.0 for value in y]
        X_sel = _select_columns(X, self._indices)
        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(X_sel, y, ridge__sample_weight=sample_weight)


class TopKWeightedRidge6Q70Predictor(TopKWeightedRidgePredictor):
    name = "topk6_weighted_q70"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0, high_weight=2.0, quantile=0.7)


class TopKWeightedRidge6Q80Predictor(TopKWeightedRidgePredictor):
    name = "topk6_weighted_q80"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0, high_weight=3.0, quantile=0.8)


class TopKWeightedRidge8Q70Predictor(TopKWeightedRidgePredictor):
    name = "topk8_weighted_q70"

    def __init__(self) -> None:
        super().__init__(top_k=8, alpha=100.0, high_weight=2.0, quantile=0.7)


class TopKQuantileBlendPredictor:
    name = "topk_quantile_blend"

    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k
        self._indices: list[int] = []
        self._median_model = None
        self._upper_model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import QuantileRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._indices = _select_topk_indices(X, y, min(self.top_k, len(X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        X_sel = _select_columns(X, self._indices)
        self._median_model = make_pipeline(
            StandardScaler(),
            QuantileRegressor(quantile=0.5, alpha=0.01, solver="highs"),
        )
        self._upper_model = make_pipeline(
            StandardScaler(),
            QuantileRegressor(quantile=0.75, alpha=0.01, solver="highs"),
        )
        self._median_model.fit(X_sel, y)
        self._upper_model.fit(X_sel, y)

    def predict(self, X: Matrix) -> Vector:
        if self._median_model is None or self._upper_model is None:
            raise RuntimeError("fit() を先に呼んでください")
        X_sel = _select_columns(X, self._indices)
        median_pred = [float(value) for value in self._median_model.predict(X_sel)]
        upper_pred = [float(value) for value in self._upper_model.predict(X_sel)]
        pred = [0.65 * med + 0.35 * upper for med, upper in zip(median_pred, upper_pred)]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKRidgeUpperCalibratedPredictor:
    name = "topk_ridge_upper_cal"

    def __init__(self, top_k: int = 6, alpha: float = 100.0, upper_q: float = 0.7, boost: float = 0.12) -> None:
        self.top_k = top_k
        self.alpha = alpha
        self.upper_q = upper_q
        self.boost = boost
        self._base = TopKFeatureRidgePredictor(top_k=top_k, alpha=alpha)
        self._pred_threshold: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._base.fit(X, y)
        fitted = self._base.predict(X)
        self._pred_threshold = _quantile(fitted, self.upper_q)
        self._y_min = min(y)
        self._y_max = max(y)

    def predict(self, X: Matrix) -> Vector:
        pred = self._base.predict(X)
        out: Vector = []
        for value in pred:
            if value > self._pred_threshold:
                value = value + self.boost * (value - self._pred_threshold)
            out.append(min(max(value, self._y_min), self._y_max))
        return out


class TopKRidgeUpperCalQ70Predictor(TopKRidgeUpperCalibratedPredictor):
    name = "topk6_upper_cal_q70"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0, upper_q=0.7, boost=0.15)


class TopKRidgeUpperCalQ80Predictor(TopKRidgeUpperCalibratedPredictor):
    name = "topk6_upper_cal_q80"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0, upper_q=0.8, boost=0.25)


class TopKRidgeMeanShiftPredictor:
    name = "topk6_mean_shift"

    def __init__(self, shift_ratio: float = 0.15) -> None:
        self.shift_ratio = shift_ratio
        self._base = TopKFeatureRidgePredictor(top_k=6, alpha=100.0)
        self._train_mean: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._base.fit(X, y)
        self._train_mean = sum(y) / len(y)
        self._y_min = min(y)
        self._y_max = max(y)

    def predict(self, X: Matrix) -> Vector:
        pred = self._base.predict(X)
        out: Vector = []
        for value in pred:
            if value > self._train_mean:
                value += self.shift_ratio * (value - self._train_mean)
            else:
                value -= 0.05 * (self._train_mean - value)
            out.append(min(max(value, self._y_min), self._y_max))
        return out


class MetaMonotonicTopKRidgePredictor:
    name = "meta_monotonic_topk_ridge"

    def __init__(self, top_k: int = 6, alpha: float = 100.0) -> None:
        self._base = TopKFeatureRidgePredictor(top_k=top_k, alpha=alpha)
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._base.fit(_strip_meta(X), y)

    def predict(self, X: Matrix) -> Vector:
        pred = self._base.predict(_strip_meta(X))
        pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class MetaMonotonicTopK6RidgePredictor(MetaMonotonicTopKRidgePredictor):
    name = "meta_monotonic_topk6_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=6, alpha=100.0)


class MetaMonotonicTopK5RidgePredictor(MetaMonotonicTopKRidgePredictor):
    name = "meta_monotonic_topk5_ridge"

    def __init__(self) -> None:
        super().__init__(top_k=5, alpha=100.0)


class MetaMonotonicQuantileBlendPredictor(TopKQuantileBlendPredictor):
    name = "meta_monotonic_quantile_blend"

    def fit(self, X: Matrix, y: Vector) -> None:
        super().fit(_strip_meta(X), y)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(_strip_meta(X))
        pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class GroupCurveTemplatePredictor:
    name = "group_curve_template"

    def __init__(self, base: str = "quantile", range_floor_ratio: float = 0.6, blend: float = 0.65) -> None:
        self.base = base
        self.range_floor_ratio = range_floor_ratio
        self.blend = blend
        self._base_model = TopKQuantileBlendPredictor() if base == "quantile" else TopKFeatureRidgePredictor(top_k=6)
        self._curve_model = None
        self._median_group_range: float = 0.0
        self._upper_group_range: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.isotonic import IsotonicRegression

        self._base_model.fit(_strip_meta(X), y)
        self._y_min = min(y)
        self._y_max = max(y)
        groups = _group_indices(X)
        ranges: list[float] = []
        positions: list[float] = []
        normalized_y: list[float] = []

        for indices in groups.values():
            group_y = [y[i] for i in indices]
            group_min = min(group_y)
            group_max = max(group_y)
            group_range = max(group_max - group_min, 1e-12)
            ranges.append(group_range)
            for i in indices:
                positions.append(X[i][-1])
                normalized_y.append((y[i] - group_min) / group_range)

        self._median_group_range = _quantile(ranges, 0.5) if ranges else 0.0
        self._upper_group_range = _quantile(ranges, 0.75) if ranges else 0.0
        self._curve_model = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self._curve_model.fit(positions, normalized_y)

    def predict(self, X: Matrix) -> Vector:
        if self._curve_model is None:
            raise RuntimeError("fit() を先に呼んでください")

        base_pred = self._base_model.predict(_strip_meta(X))
        out = base_pred[:]
        for indices in _group_indices(X).values():
            group_base = [base_pred[i] for i in indices]
            base_min = min(group_base)
            base_max = max(group_base)
            base_range = max(base_max - base_min, 1e-12)
            target_range = max(base_range, self._median_group_range * self.range_floor_ratio)
            center = sum(group_base) / len(group_base)
            curve_values = [float(self._curve_model.predict([X[i][-1]])[0]) for i in indices]
            curve_mean = sum(curve_values) / len(curve_values)
            curve_pred = [center + (value - curve_mean) * target_range for value in curve_values]

            for i, value in zip(indices, curve_pred):
                out[i] = self.blend * value + (1.0 - self.blend) * base_pred[i]

        out = _monotonic_decreasing_by_group(out, X)
        return [min(max(value, self._y_min), self._y_max) for value in out]


class GroupCurveTemplateQuantilePredictor(GroupCurveTemplatePredictor):
    name = "group_curve_template_quantile"

    def __init__(self) -> None:
        super().__init__(base="quantile", range_floor_ratio=0.6, blend=0.65)


class GroupCurveTemplateRidgePredictor(GroupCurveTemplatePredictor):
    name = "group_curve_template_ridge"

    def __init__(self) -> None:
        super().__init__(base="ridge", range_floor_ratio=0.6, blend=0.65)


class GroupCurveTemplateWidePredictor(GroupCurveTemplatePredictor):
    name = "group_curve_template_wide"

    def __init__(self) -> None:
        super().__init__(base="quantile", range_floor_ratio=0.9, blend=0.75)


class GroupCurveUpperAnchorPredictor(GroupCurveTemplatePredictor):
    name = "group_curve_upper_anchor"

    def __init__(self) -> None:
        super().__init__(base="quantile", range_floor_ratio=0.75, blend=0.65)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        out = pred[:]
        for indices in _group_indices(X).values():
            group_pred = [pred[i] for i in indices]
            pred_range = max(group_pred) - min(group_pred)
            missing_range = max(self._upper_group_range * 0.8 - pred_range, 0.0)
            for i in indices:
                position = X[i][-1] if len(X[i]) >= 2 else 0.0
                if position <= 0.45:
                    out[i] += 0.35 * missing_range * (1.0 - position / 0.45)
        out = _monotonic_decreasing_by_group(out, X)
        return [min(max(value, self._y_min), self._y_max) for value in out]


class GroupCurveRangeCalibratedPredictor(GroupCurveTemplatePredictor):
    name = "group_curve_range_calibrated"

    def __init__(self) -> None:
        super().__init__(base="ridge", range_floor_ratio=0.8, blend=0.75)

    def predict(self, X: Matrix) -> Vector:
        if self._curve_model is None:
            raise RuntimeError("fit() を先に呼んでください")

        base_pred = self._base_model.predict(_strip_meta(X))
        out = base_pred[:]
        for indices in _group_indices(X).values():
            group_base = [base_pred[i] for i in indices]
            base_range = max(group_base) - min(group_base)
            target_range = max(base_range, 0.7 * self._upper_group_range)
            center = sum(group_base) / len(group_base)
            curve_values = [float(self._curve_model.predict([X[i][-1]])[0]) for i in indices]
            curve_mean = sum(curve_values) / len(curve_values)
            curve_pred = [center + (value - curve_mean) * target_range for value in curve_values]
            for i, value in zip(indices, curve_pred):
                out[i] = self.blend * value + (1.0 - self.blend) * base_pred[i]

        out = _monotonic_decreasing_by_group(out, X)
        return [min(max(value, self._y_min), self._y_max) for value in out]


class GroupCurveBlendPredictor:
    name = "group_curve_blend"

    def __init__(self) -> None:
        self._ridge = GroupCurveTemplateRidgePredictor()
        self._quantile = GroupCurveTemplateQuantilePredictor()
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._ridge.fit(X, y)
        self._quantile.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        ridge_pred = self._ridge.predict(X)
        quantile_pred = self._quantile.predict(X)
        pred = [0.55 * ridge + 0.45 * quantile for ridge, quantile in zip(ridge_pred, quantile_pred)]
        pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class GroupCurveResidualPredictor:
    name = "group_curve_residual"

    def __init__(self) -> None:
        self._curve = GroupCurveTemplatePredictor(base="quantile", range_floor_ratio=0.6, blend=1.0)
        self._residual_model = TopKFeatureRidgePredictor(top_k=6, alpha=300.0)
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._curve.fit(X, y)
        curve_pred = self._curve.predict(X)
        residual = [actual - pred for actual, pred in zip(y, curve_pred)]
        self._residual_model.fit(_strip_meta(X), residual)

    def predict(self, X: Matrix) -> Vector:
        curve_pred = self._curve.predict(X)
        residual = self._residual_model.predict(_strip_meta(X))
        pred = [base + 0.5 * res for base, res in zip(curve_pred, residual)]
        pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKHistGradientBoostingPredictor:
    name = "topk_hgbdt"

    def __init__(self, top_k: int = 24) -> None:
        self.top_k = top_k
        self._indices: list[int] = []
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        spectral_X = _strip_meta(X) if _has_meta(X) else X
        self._indices = _select_topk_indices(spectral_X, y, min(self.top_k, len(spectral_X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        self._model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=7,
            l2_regularization=3.0,
            min_samples_leaf=20,
            random_state=42,
        )
        self._model.fit(_select_columns(spectral_X, self._indices), y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        spectral_X = _strip_meta(X) if _has_meta(X) else X
        pred = [float(value) for value in self._model.predict(_select_columns(spectral_X, self._indices))]
        if _has_meta(X):
            pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class TopKHistGradientBoosting12Predictor(TopKHistGradientBoostingPredictor):
    name = "topk12_hgbdt"

    def __init__(self) -> None:
        super().__init__(top_k=12)


class TopKHistGradientBoosting24Predictor(TopKHistGradientBoostingPredictor):
    name = "topk24_hgbdt"

    def __init__(self) -> None:
        super().__init__(top_k=24)


class PlsHistGradientBoostingPredictor:
    name = "pls_hgbdt"

    def __init__(self, n_components: int = 6) -> None:
        self.n_components = n_components
        self._scaler = None
        self._pls = None
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        spectral_X = _strip_meta(X) if _has_meta(X) else X
        self._y_min = min(y)
        self._y_max = max(y)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(spectral_X)
        self._pls = PLSRegression(n_components=self.n_components)
        X_pls = self._pls.fit_transform(X_scaled, y)[0]
        self._model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=7,
            l2_regularization=3.0,
            min_samples_leaf=20,
            random_state=42,
        )
        self._model.fit(X_pls, y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None or self._scaler is None or self._pls is None:
            raise RuntimeError("fit() を先に呼んでください")
        spectral_X = _strip_meta(X) if _has_meta(X) else X
        X_scaled = self._scaler.transform(spectral_X)
        X_pls = self._pls.transform(X_scaled)
        pred = [float(value) for value in self._model.predict(X_pls)]
        if _has_meta(X):
            pred = _monotonic_decreasing_by_group(pred, X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class PlsHistGradientBoosting4Predictor(PlsHistGradientBoostingPredictor):
    name = "pls4_hgbdt"

    def __init__(self) -> None:
        super().__init__(n_components=4)


class PlsHistGradientBoosting6Predictor(PlsHistGradientBoostingPredictor):
    name = "pls6_hgbdt"

    def __init__(self) -> None:
        super().__init__(n_components=6)


class TopKFeatureHuberPredictor:
    name = "topk_huber"

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._indices: list[int] = []
        self._model = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._indices = _select_topk_indices(X, y, min(self.top_k, len(X[0])))
        self._y_min = min(y)
        self._y_max = max(y)
        self._model = make_pipeline(
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000),
        )
        self._model.fit(_select_columns(X, self._indices), y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        pred = [float(value) for value in self._model.predict(_select_columns(X, self._indices))]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


def _select_columns(X: Matrix, indices: list[int]) -> Matrix:
    return [[row[j] for j in indices] for row in X]


def _strip_meta(X: Matrix) -> Matrix:
    return [row[:-2] for row in X]


def _has_meta(X: Matrix) -> bool:
    return bool(X and len(X[0]) >= 2)


def _monotonic_decreasing_by_group(predictions: Vector, X: Matrix) -> Vector:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return predictions

    out = predictions[:]
    groups: dict[float, list[tuple[int, float, float]]] = {}
    for i, row in enumerate(X):
        if len(row) < 2:
            return predictions
        group = row[-2]
        position = row[-1]
        groups.setdefault(group, []).append((i, position, predictions[i]))

    for items in groups.values():
        if len(items) < 3:
            continue
        items.sort(key=lambda item: item[1])
        x = [item[1] for item in items]
        y = [item[2] for item in items]
        fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(x, y)
        for (i, _, _), value in zip(items, fitted):
            out[i] = float(value)
    return out


def _group_indices(X: Matrix) -> dict[float, list[int]]:
    groups: dict[float, list[int]] = {}
    for i, row in enumerate(X):
        if len(row) < 2:
            groups.setdefault(0.0, []).append(i)
        else:
            groups.setdefault(row[-2], []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: X[i][-1] if len(X[i]) >= 2 else float(i))
    return groups


def _select_topk_indices(X: Matrix, y: Vector, k: int) -> list[int]:
    n = len(y)
    y_mean = sum(y) / n
    y_var = sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1)
    y_std = math.sqrt(max(y_var, 1e-12))
    scores: list[tuple[float, int]] = []
    for j in range(len(X[0])):
        col = [row[j] for row in X]
        x_mean = sum(col) / n
        x_var = sum((xi - x_mean) ** 2 for xi in col) / max(n - 1, 1)
        x_std = math.sqrt(max(x_var, 1e-12))
        cov = sum((col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(n - 1, 1)
        corr = cov / max(x_std * y_std, 1e-12)
        scores.append((abs(corr), j))
    scores.sort(reverse=True)
    return [j for _, j in scores[:k]]


def _select_stable_topk_indices(X: Matrix, y: Vector, k: int) -> list[int]:
    n = len(y)
    d = len(X[0]) if X else 0
    if n < 5 or d == 0:
        return _select_topk_indices(X, y, k)

    aggregate: dict[int, float] = {}
    n_chunks = 5
    for chunk in range(n_chunks):
        indices = [i for i in range(n) if i % n_chunks != chunk]
        X_part = [X[i] for i in indices]
        y_part = [y[i] for i in indices]
        for rank, feature_idx in enumerate(_select_topk_indices(X_part, y_part, min(k * 3, d))):
            aggregate[feature_idx] = aggregate.get(feature_idx, 0.0) + 1.0 / (rank + 1)

    ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
    return [feature_idx for feature_idx, _ in ranked[:k]]


def _quantile(values: Vector, q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("空データのquantileは計算できません")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight
