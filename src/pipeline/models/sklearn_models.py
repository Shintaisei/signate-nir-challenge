"""scikit-learn系モデル。"""

from __future__ import annotations

from pipeline.types import Matrix, Vector


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn が必要です。`python -m pip install scikit-learn numpy scipy` を実行してください。"
        ) from exc


class SklearnPredictor:
    name = "sklearn"

    def __init__(self) -> None:
        self._model = None

    def fit(self, X: Matrix, y: Vector) -> None:
        _require_sklearn()
        self._model = self._build_model()
        self._model.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() を先に呼んでください")
        pred = self._model.predict(X)
        return [float(value) for value in _flatten(pred)]

    def _build_model(self):
        raise NotImplementedError


class RidgeAlpha10Predictor(SklearnPredictor):
    name = "ridge_alpha10"

    def _build_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


class RidgeAlpha1Predictor(SklearnPredictor):
    name = "ridge_alpha1"

    def _build_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


class RidgeAlpha100Predictor(SklearnPredictor):
    name = "ridge_alpha100"

    def _build_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


class RidgeAlpha1000Predictor(SklearnPredictor):
    name = "ridge_alpha1000"

    def _build_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))


class LogRidgeAlpha100Predictor(SklearnPredictor):
    name = "log_ridge_alpha100"

    def _build_model(self):
        import numpy as np
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        regressor = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


class ElasticNetPredictor(SklearnPredictor):
    name = "elastic_net"

    def _build_model(self):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=10000))


class BayesianRidgePredictor(SklearnPredictor):
    name = "bayesian_ridge"

    def _build_model(self):
        from sklearn.linear_model import BayesianRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), BayesianRidge())


class HuberPredictor(SklearnPredictor):
    name = "huber"

    def _build_model(self):
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=1000))


class PcaRidgeC50Predictor(SklearnPredictor):
    name = "pca50_ridge_alpha100"

    def _build_model(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PCA(n_components=50, random_state=42), Ridge(alpha=100.0))


class PcaRidgePredictor(SklearnPredictor):
    n_components: int = 50
    alpha: float = 100.0

    def _build_model(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components, random_state=42),
            Ridge(alpha=self.alpha),
        )


class GaussianPcaRidgePredictor(PcaRidgePredictor):
    def _build_model(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer, StandardScaler

        return make_pipeline(
            QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                random_state=42,
            ),
            StandardScaler(),
            PCA(n_components=self.n_components, random_state=42),
            Ridge(alpha=self.alpha),
        )


class Pca25RidgeAlpha100Predictor(PcaRidgePredictor):
    name = "pca25_ridge_alpha100"
    n_components = 25
    alpha = 100.0


class Pca25RidgeAlpha300Predictor(PcaRidgePredictor):
    name = "pca25_ridge_alpha300"
    n_components = 25
    alpha = 300.0


class Pca25RidgeAlpha1000Predictor(PcaRidgePredictor):
    name = "pca25_ridge_alpha1000"
    n_components = 25
    alpha = 1000.0


class Pca50RidgeAlpha300Predictor(PcaRidgePredictor):
    name = "pca50_ridge_alpha300"
    n_components = 50
    alpha = 300.0


class Pca50RidgeAlpha1000Predictor(PcaRidgePredictor):
    name = "pca50_ridge_alpha1000"
    n_components = 50
    alpha = 1000.0


class Pca100RidgeAlpha100Predictor(PcaRidgePredictor):
    name = "pca100_ridge_alpha100"
    n_components = 100
    alpha = 100.0


class Pca100RidgeAlpha300Predictor(PcaRidgePredictor):
    name = "pca100_ridge_alpha300"
    n_components = 100
    alpha = 300.0


class Pca100RidgeAlpha1000Predictor(PcaRidgePredictor):
    name = "pca100_ridge_alpha1000"
    n_components = 100
    alpha = 1000.0


class Pca150RidgeAlpha100Predictor(PcaRidgePredictor):
    name = "pca150_ridge_alpha100"
    n_components = 150
    alpha = 100.0


class Pca150RidgeAlpha300Predictor(PcaRidgePredictor):
    name = "pca150_ridge_alpha300"
    n_components = 150
    alpha = 300.0


class Pca150RidgeAlpha1000Predictor(PcaRidgePredictor):
    name = "pca150_ridge_alpha1000"
    n_components = 150
    alpha = 1000.0


class GaussianPca25RidgeAlpha300Predictor(GaussianPcaRidgePredictor):
    name = "gauss_pca25_ridge_alpha300"
    n_components = 25
    alpha = 300.0


class GaussianPca50RidgeAlpha300Predictor(GaussianPcaRidgePredictor):
    name = "gauss_pca50_ridge_alpha300"
    n_components = 50
    alpha = 300.0


class GaussianPca100RidgeAlpha1000Predictor(GaussianPcaRidgePredictor):
    name = "gauss_pca100_ridge_alpha1000"
    n_components = 100
    alpha = 1000.0


class WindowPcaRidgePredictor:
    name = "window_pca_ridge"

    window: int = 10
    components_per_window: int = 1
    alpha: float = 300.0

    def __init__(self) -> None:
        self._blocks: list[tuple[int, int, object]] = []
        self._model = None
        self._y_min = 0.0
        self._y_max = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        _require_sklearn()
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if not X:
            raise ValueError("WindowPcaRidgePredictor requires non-empty X")
        n_features = len(X[0])
        self._blocks = []
        transformed_blocks: list[list[list[float]]] = []
        for start in range(0, n_features, self.window):
            end = min(start + self.window, n_features)
            width = end - start
            if width <= 0:
                continue
            n_components = min(self.components_per_window, width)
            block_model = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=42),
            )
            X_block = [row[start:end] for row in X]
            block_features = block_model.fit_transform(X_block)
            self._blocks.append((start, end, block_model))
            transformed_blocks.append(_matrix_from_array(block_features))

        X_transformed = _hstack_blocks(transformed_blocks)
        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(X_transformed, y)
        self._y_min = min(y)
        self._y_max = max(y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        transformed_blocks: list[list[list[float]]] = []
        for start, end, block_model in self._blocks:
            transformed_blocks.append(_matrix_from_array(block_model.transform([row[start:end] for row in X])))
        pred = [float(value) for value in self._model.predict(_hstack_blocks(transformed_blocks))]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class WindowPca10C1RidgeAlpha300Predictor(WindowPcaRidgePredictor):
    name = "window_pca10_c1_ridge_alpha300"
    window = 10
    components_per_window = 1
    alpha = 300.0


class WindowPca10C1RidgeAlpha1000Predictor(WindowPcaRidgePredictor):
    name = "window_pca10_c1_ridge_alpha1000"
    window = 10
    components_per_window = 1
    alpha = 1000.0


class WindowPca10C2RidgeAlpha1000Predictor(WindowPcaRidgePredictor):
    name = "window_pca10_c2_ridge_alpha1000"
    window = 10
    components_per_window = 2
    alpha = 1000.0


class WindowPca20C1RidgeAlpha1000Predictor(WindowPcaRidgePredictor):
    name = "window_pca20_c1_ridge_alpha1000"
    window = 20
    components_per_window = 1
    alpha = 1000.0


class SvrRbfC10Predictor(SklearnPredictor):
    name = "svr_rbf_c10"

    def _build_model(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=2.0, gamma="scale"))


class SvrLinearC1Predictor(SklearnPredictor):
    name = "svr_linear_c1"

    def _build_model(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVR

        return make_pipeline(StandardScaler(), LinearSVR(C=1.0, epsilon=1.0, random_state=42, max_iter=20000))


class PlsC2Predictor(SklearnPredictor):
    name = "pls_c2"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=2))


class PlsC4Predictor(SklearnPredictor):
    name = "pls_c4"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=4))


class PlsC6Predictor(SklearnPredictor):
    name = "pls_c6"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=6))


class PlsC8Predictor(SklearnPredictor):
    name = "pls_c8"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=8))


class PlsC12Predictor(SklearnPredictor):
    name = "pls_c12"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=12))


class PlsC16Predictor(SklearnPredictor):
    name = "pls_c16"

    def _build_model(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), PLSRegression(n_components=16))


class BandWindowPlsC2Predictor:
    name = "band_window_pls_c2"

    center_idx: int = 616
    radius: int = 5

    def __init__(self, *, center_idx: int | None = None, radius: int | None = None) -> None:
        self.center_idx = self.center_idx if center_idx is None else center_idx
        self.radius = self.radius if radius is None else radius
        self._model = None
        self._start = 0
        self._end = 0

    def fit(self, X: Matrix, y: Vector) -> None:
        _require_sklearn()
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_features = len(X[0]) if X else 0
        self._start = max(0, self.center_idx - self.radius)
        self._end = min(n_features, self.center_idx + self.radius + 1)
        if self._end - self._start < 2:
            raise ValueError("BandWindowPlsC2Predictor requires at least 2 features")
        X_band = self._slice(X)
        self._model = make_pipeline(StandardScaler(), PLSRegression(n_components=2))
        self._model.fit(X_band, y)

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        pred = self._model.predict(self._slice(X))
        return [float(value) for value in _flatten(pred)]

    def _slice(self, X: Matrix) -> Matrix:
        return [row[self._start : self._end] for row in X]


class Band616PlsC2Radius2Predictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_r2"
    radius = 2


class Band616PlsC2Radius3Predictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_r3"
    radius = 3


class Band616PlsC2Radius5Predictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_r5"
    radius = 5


class Band616PlsC2Radius8Predictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_r8"
    radius = 8


class Band616PlsC2Radius10Predictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_r10"
    radius = 10


class Band616PlsC2MeanAlignPredictor(BandWindowPlsC2Predictor):
    name = "band616_pls_c2_mean_align"

    align_weight: float = 0.10

    def __init__(self, *, align_weight: float | None = None) -> None:
        super().__init__(radius=5)
        self.align_weight = self.align_weight if align_weight is None else align_weight
        self._train_band_mean: list[float] = []

    def fit(self, X: Matrix, y: Vector) -> None:
        super().fit(X, y)
        self._train_band_mean = _column_means(self._slice(X))

    def predict(self, X: Matrix) -> Vector:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        X_band = self._slice(X)
        X_aligned = self._mean_align(X_band)
        pred = self._model.predict(X_aligned)
        return [float(value) for value in _flatten(pred)]

    def _mean_align(self, X_band: Matrix) -> Matrix:
        if not X_band or not self._train_band_mean:
            return X_band
        target_mean = _column_means(X_band)
        shifts = [
            self.align_weight * (target_mean[j] - self._train_band_mean[j])
            for j in range(len(self._train_band_mean))
        ]
        return [[value - shifts[j] for j, value in enumerate(row)] for row in X_band]


class Band616PlsC2MeanAlignL010Predictor(Band616PlsC2MeanAlignPredictor):
    name = "band616_pls_c2_mean_align_l010"
    align_weight = 0.10


class Band616PlsC2MeanAlignL020Predictor(Band616PlsC2MeanAlignPredictor):
    name = "band616_pls_c2_mean_align_l020"
    align_weight = 0.20


class Band616PlsC2OofAffinePredictor:
    name = "band616_pls_c2_oof_affine"

    shrink: float = 0.10

    def __init__(self, *, shrink: float | None = None) -> None:
        self.shrink = self.shrink if shrink is None else shrink
        self._group_labels: list[str] | None = None
        self._anchor = Band616PlsC2Radius5Predictor()
        self._scale = 1.0
        self._shift = 0.0

    def set_group_labels(self, labels: list[str]) -> None:
        self._group_labels = labels

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() must be called before fit()")

        oof_pred: list[float] = []
        oof_true: list[float] = []
        for group in sorted(set(self._group_labels)):
            train_idx = [i for i, label in enumerate(self._group_labels) if label != group]
            valid_idx = [i for i, label in enumerate(self._group_labels) if label == group]
            if len(train_idx) < 3 or not valid_idx:
                continue
            model = Band616PlsC2Radius5Predictor()
            model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
            pred = model.predict([X[i] for i in valid_idx])
            oof_pred.extend(pred)
            oof_true.extend(y[i] for i in valid_idx)

        self._scale, self._shift = _shrunk_affine(oof_pred, oof_true, self.shrink)
        self._anchor.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        pred = self._anchor.predict(X)
        return [float(self._scale * value + self._shift) for value in pred]


class Band616PlsC2OofAffineL010Predictor(Band616PlsC2OofAffinePredictor):
    name = "band616_pls_c2_oof_affine_l010"
    shrink = 0.10


class Band616PlsC2OofBiasPredictor(Band616PlsC2OofAffinePredictor):
    name = "band616_pls_c2_oof_bias"

    shrink: float = 0.05

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() must be called before fit()")

        residuals: list[float] = []
        for group in sorted(set(self._group_labels)):
            train_idx = [i for i, label in enumerate(self._group_labels) if label != group]
            valid_idx = [i for i, label in enumerate(self._group_labels) if label == group]
            if len(train_idx) < 3 or not valid_idx:
                continue
            model = Band616PlsC2Radius5Predictor()
            model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
            pred = model.predict([X[i] for i in valid_idx])
            residuals.extend(y[i] - value for i, value in zip(valid_idx, pred))

        raw_shift = sum(residuals) / len(residuals) if residuals else 0.0
        self._scale = 1.0
        self._shift = self.shrink * raw_shift
        self._anchor.fit(X, y)


class Band616PlsC2OofBiasL005Predictor(Band616PlsC2OofBiasPredictor):
    name = "band616_pls_c2_oof_bias_l005"
    shrink = 0.05


class Band616PlsC2SmoothAnchorBlendPredictor:
    name = "band616_pls_c2_smooth_anchor_blend"

    target_radius: int = 8
    weight: float = 0.1

    def __init__(
        self,
        *,
        target_radius: int | None = None,
        weight: float | None = None,
    ) -> None:
        self.target_radius = self.target_radius if target_radius is None else target_radius
        self.weight = self.weight if weight is None else weight
        self._anchor = Band616PlsC2Radius5Predictor()
        self._target = BandWindowPlsC2Predictor(radius=self.target_radius)

    def fit(self, X: Matrix, y: Vector) -> None:
        self._anchor.fit(X, y)
        self._target.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        anchor_pred = self._anchor.predict(X)
        target_pred = self._target.predict(X)
        w = self.weight
        return [float(anchor + w * (target - anchor)) for anchor, target in zip(anchor_pred, target_pred)]


class Band616R8BlendW005Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r8_blend_w005"
    target_radius = 8
    weight = 0.05


class Band616R8BlendW010Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r8_blend_w010"
    target_radius = 8
    weight = 0.10


class Band616R8BlendW015Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r8_blend_w015"
    target_radius = 8
    weight = 0.15


class Band616R8BlendW020Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r8_blend_w020"
    target_radius = 8
    weight = 0.20


class Band616R10BlendW003Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r10_blend_w003"
    target_radius = 10
    weight = 0.03


class Band616R10BlendW005Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r10_blend_w005"
    target_radius = 10
    weight = 0.05


class Band616R10BlendW0075Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r10_blend_w0075"
    target_radius = 10
    weight = 0.075


class Band616R10BlendW010Predictor(Band616PlsC2SmoothAnchorBlendPredictor):
    name = "band616_r10_blend_w010"
    target_radius = 10
    weight = 0.10


def _shrunk_affine(pred: Vector, true: Vector, shrink: float) -> tuple[float, float]:
    if len(pred) < 2 or len(pred) != len(true):
        return 1.0, 0.0
    pred_mean = sum(pred) / len(pred)
    true_mean = sum(true) / len(true)
    pred_var = sum((value - pred_mean) ** 2 for value in pred)
    if pred_var <= 1e-12:
        raw_scale = 1.0
        raw_shift = true_mean - pred_mean
    else:
        cov = sum((pred[i] - pred_mean) * (true[i] - true_mean) for i in range(len(pred)))
        raw_scale = cov / pred_var
        raw_shift = true_mean - raw_scale * pred_mean
    scale = 1.0 + shrink * (raw_scale - 1.0)
    shift = shrink * raw_shift
    return float(scale), float(shift)


def _column_means(X: Matrix) -> list[float]:
    if not X:
        return []
    n = len(X)
    n_features = len(X[0])
    return [sum(row[j] for row in X) / n for j in range(n_features)]


def _flatten(values) -> list[float]:
    if hasattr(values, "ravel"):
        return [float(value) for value in values.ravel()]
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.append(float(value[0]))
        else:
            out.append(float(value))
    return out


def _matrix_from_array(values) -> Matrix:
    return [[float(value) for value in row] for row in values]


def _hstack_blocks(blocks: list[Matrix]) -> Matrix:
    if not blocks:
        return []
    out: Matrix = []
    for rows in zip(*blocks):
        merged: list[float] = []
        for row in rows:
            merged.extend(row)
        out.append(merged)
    return out
