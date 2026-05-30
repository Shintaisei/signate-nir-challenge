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
