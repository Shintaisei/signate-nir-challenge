from __future__ import annotations

from pipeline.config import CompetitionConfig
from pipeline.models.nearest_train_species import SingleFeatureAntiNnBiasBlendWm040Predictor
from pipeline.models.single_feature_linear import SingleFeatureLinearPredictor
from pipeline.types import Matrix, Rows, Vector


class SingleFeatureResidualMlpPredictor:
    name = "single_feature_residual_mlp"

    alpha: float = 0.1
    seeds: tuple[int, ...] = (11, 42, 73)

    def __init__(self, *, alpha: float | None = None) -> None:
        self.alpha = self.alpha if alpha is None else alpha
        self._base = SingleFeatureLinearPredictor()
        self._models = []

    def fit(self, X: Matrix, y: Vector) -> None:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._base.fit(X, y)
        base_pred = self._base.predict(X)
        residual = [target - pred for target, pred in zip(y, base_pred)]
        n_components = max(2, min(32, len(X) - 1, len(X[0]) if X else 0))

        self._models = []
        for seed in self.seeds:
            regressor = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=seed),
                MLPRegressor(
                    hidden_layer_sizes=(32, 8),
                    activation="relu",
                    solver="adam",
                    alpha=0.1,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=25,
                    validation_fraction=0.2,
                    random_state=seed,
                ),
            )
            model = TransformedTargetRegressor(
                regressor=regressor,
                transformer=StandardScaler(),
            )
            model.fit(X, residual)
            self._models.append(model)

    def predict(self, X: Matrix) -> Vector:
        if not self._models:
            raise RuntimeError("fit() must be called before predict()")
        base_pred = self._base.predict(X)
        residual_preds = []
        for model in self._models:
            pred = model.predict(X)
            residual_preds.append([float(value) for value in pred])

        out: Vector = []
        for i, base in enumerate(base_pred):
            residual = sum(pred[i] for pred in residual_preds) / len(residual_preds)
            out.append(float(base + self.alpha * residual))
        return out


class SingleFeatureResidualMlpA005Predictor(SingleFeatureResidualMlpPredictor):
    name = "single_feature_residual_mlp_a005"
    alpha = 0.05


class SingleFeatureResidualMlpA010Predictor(SingleFeatureResidualMlpPredictor):
    name = "single_feature_residual_mlp_a010"
    alpha = 0.10


class SingleFeatureResidualMlpA020Predictor(SingleFeatureResidualMlpPredictor):
    name = "single_feature_residual_mlp_a020"
    alpha = 0.20


class AntiNnWm040ResidualMlpPredictor:
    name = "anti_nn_wm040_residual_mlp"

    residual_alpha: float = 0.05

    def __init__(self, *, residual_alpha: float | None = None) -> None:
        self.residual_alpha = self.residual_alpha if residual_alpha is None else residual_alpha
        self._base = SingleFeatureLinearPredictor()
        self._anti = SingleFeatureAntiNnBiasBlendWm040Predictor()
        self._residual = SingleFeatureResidualMlpPredictor(alpha=self.residual_alpha)
        self._train_rows: Rows | None = None
        self._config: CompetitionConfig | None = None

    def set_train_context(self, train_rows: Rows, config: CompetitionConfig) -> None:
        self._train_rows = train_rows
        self._config = config
        if hasattr(self._anti, "set_train_context"):
            self._anti.set_train_context(train_rows, config)

    def fit(self, X: Matrix, y: Vector) -> None:
        self._base.fit(X, y)
        self._anti.fit(X, y)
        self._residual.fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        base_pred = self._base.predict(X)
        anti_pred = self._anti.predict(X)
        residual_pred = self._residual.predict(X)
        return [
            float(anti + (resid - base))
            for base, anti, resid in zip(base_pred, anti_pred, residual_pred)
        ]

    def predict_test(self, test_rows: Rows, X_test: Matrix) -> Vector:
        base_pred = self._base.predict(X_test)
        anti_pred = self._anti.predict_test(test_rows, X_test)
        residual_pred = self._residual.predict(X_test)
        return [
            float(anti + (resid - base))
            for base, anti, resid in zip(base_pred, anti_pred, residual_pred)
        ]


class AntiNnWm040ResidualMlpA005Predictor(AntiNnWm040ResidualMlpPredictor):
    name = "anti_nn_wm040_residual_mlp_a005"
    residual_alpha = 0.05


class AntiNnWm040ResidualMlpA010Predictor(AntiNnWm040ResidualMlpPredictor):
    name = "anti_nn_wm040_residual_mlp_a010"
    residual_alpha = 0.10
