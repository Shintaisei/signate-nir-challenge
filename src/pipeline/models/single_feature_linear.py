"""1次元線形回帰（最も相関が高い1特徴だけ使う）。"""

from __future__ import annotations

import math

from pipeline.types import Matrix, Vector


def _select_columns(X: Matrix, indices: list[int]) -> Matrix:
    return [[row[j] for j in indices] for row in X]


def _column_variance(X: Matrix, j: int) -> float:
    col = [row[j] for row in X]
    x_mean = sum(col) / len(col)
    return sum((xi - x_mean) ** 2 for xi in col) / max(len(col) - 1, 1)


def _select_topk_indices(X: Matrix, y: Vector, k: int, *, min_col_var: float = 1e-6) -> list[int]:
    n = len(y)
    y_mean = sum(y) / n
    y_var = sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1)
    y_std = math.sqrt(max(y_var, 1e-12))
    scores: list[tuple[float, int]] = []
    for j in range(len(X[0])):
        if _column_variance(X, j) < min_col_var:
            continue
        col = [row[j] for row in X]
        x_mean = sum(col) / n
        x_var = sum((xi - x_mean) ** 2 for xi in col) / max(n - 1, 1)
        x_std = math.sqrt(max(x_var, 1e-12))
        cov = sum((col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(n - 1, 1)
        corr = cov / max(x_std * y_std, 1e-12)
        scores.append((abs(corr), j))
    scores.sort(reverse=True)
    if not scores:
        return list(range(min(k, len(X[0]))))
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


class SingleFeatureLinearPredictor:
    name = "single_feature_linear"

    def __init__(self) -> None:
        self._feature_idx: int | None = None
        self._slope: float = 0.0
        self._intercept: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        n = len(y)
        n_features = len(X[0]) if X else 0
        if n == 0 or n_features == 0:
            raise ValueError("空データでは学習できません")

        y_mean = sum(y) / n
        y_var = sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1)
        y_std = math.sqrt(max(y_var, 1e-12))

        best_idx = _select_top1_index(X, y)
        self._feature_idx = best_idx
        x_best = [row[best_idx] for row in X]
        x_mean = sum(x_best) / n
        x_var = sum((xi - x_mean) ** 2 for xi in x_best)
        if x_var <= 1e-12:
            self._slope = 0.0
            self._intercept = y_mean
            return

        cov_num = sum((x_best[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        self._slope = cov_num / x_var
        self._intercept = y_mean - self._slope * x_mean

    def predict(self, X: Matrix) -> Vector:
        if self._feature_idx is None:
            raise RuntimeError("fit() を先に呼んでください")
        return [self._intercept + self._slope * row[self._feature_idx] for row in X]


class SingleFeatureFixedIndexPredictor:
    """波長 index を固定（EDA: raw 1f = 616）して 1 次元線形回帰。"""

    name = "single_feature_fixed_index"

    def __init__(self, feature_idx: int = 616) -> None:
        self._feature_idx = feature_idx
        self._slope: float = 0.0
        self._intercept: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        idx = min(self._feature_idx, len(X[0]) - 1) if X and X[0] else self._feature_idx
        x_col = [row[idx] for row in X]
        self._slope, self._intercept = _fit_1d_linear(x_col, y)
        self._feature_idx = idx

    def predict(self, X: Matrix) -> Vector:
        return [self._intercept + self._slope * row[self._feature_idx] for row in X]


def _fit_1d_linear(x: Vector, y: Vector) -> tuple[float, float]:
    """OLS slope/intercept（`SingleFeatureLinearPredictor` と同じ SS 定義）。"""
    n = len(y)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    x_ss = sum((xi - x_mean) ** 2 for xi in x)
    if x_ss <= 1e-12:
        return 0.0, y_mean
    cov_num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    slope = cov_num / x_ss
    return slope, y_mean - slope * x_mean


def _loo_species_rmse_for_feature(
    X: Matrix,
    y: Vector,
    groups: list[str],
    feature_idx: int,
) -> float:
    unique_groups = sorted(set(groups))
    sq_errors: list[float] = []
    for group in unique_groups:
        train_idx = [i for i, label in enumerate(groups) if label != group]
        valid_idx = [i for i, label in enumerate(groups) if label == group]
        if len(train_idx) < 2 or not valid_idx:
            continue
        x_train = [X[i][feature_idx] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        slope, intercept = _fit_1d_linear(x_train, y_train)
        for i in valid_idx:
            pred = intercept + slope * X[i][feature_idx]
            err = pred - y[i]
            sq_errors.append(err * err)
    if not sq_errors:
        return float("inf")
    return math.sqrt(sum(sq_errors) / len(sq_errors))


def _select_top1_index(X: Matrix, y: Vector) -> int:
    n = len(y)
    n_features = len(X[0]) if X else 0
    y_mean = sum(y) / n
    y_var = sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1)
    y_std = math.sqrt(max(y_var, 1e-12))

    best_idx = 0
    best_abs_corr = -1.0
    for j in range(n_features):
        col = [row[j] for row in X]
        x_mean = sum(col) / n
        x_var = sum((xi - x_mean) ** 2 for xi in col) / max(n - 1, 1)
        x_std = math.sqrt(max(x_var, 1e-12))
        cov = sum((col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(n - 1, 1)
        corr = cov / max(x_std * y_std, 1e-12)
        if abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_idx = j
    return best_idx


class SingleFeatureLooConsensusWavelengthPredictor(SingleFeatureLinearPredictor):
    """各樹種LOO foldで選ばれた top-1 波長の多数決で最終波長を決める。"""

    name = "single_feature_loo_consensus_wavelength"

    def __init__(self) -> None:
        super().__init__()
        self._group_labels: list[str] | None = None

    def set_group_labels(self, labels: list[str]) -> None:
        self._group_labels = labels

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() を fit 前に呼んでください")

        groups = sorted(set(self._group_labels))
        votes: dict[int, int] = {}
        for group in groups:
            train_idx = [i for i, label in enumerate(self._group_labels) if label != group]
            if len(train_idx) < 2:
                continue
            X_part = [X[i] for i in train_idx]
            y_part = [y[i] for i in train_idx]
            feature_idx = _select_top1_index(X_part, y_part)
            votes[feature_idx] = votes.get(feature_idx, 0) + 1

        if not votes:
            feature_idx = _select_top1_index(X, y)
        else:
            feature_idx = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]

        self._feature_idx = feature_idx
        x_best = [row[feature_idx] for row in X]
        n = len(y)
        y_mean = sum(y) / n
        x_mean = sum(x_best) / n
        x_var = sum((xi - x_mean) ** 2 for xi in x_best)
        if x_var <= 1e-12:
            self._slope = 0.0
            self._intercept = y_mean
            return
        cov_num = sum((x_best[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        self._slope = cov_num / x_var
        self._intercept = y_mean - self._slope * x_mean


def _ridge_fit(X: Matrix, y: Vector, alpha: float) -> tuple[list[float], float]:
    """標準化 + 閉形式 Ridge（特徴数が少ない前提）。"""
    n = len(y)
    if n == 0:
        return [], 0.0
    d = len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / max(n - 1, 1)
        stds.append(math.sqrt(max(var, 1e-12)))
    Xs = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]
    dim = d + 1
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for row, target in zip(Xs, y):
        extended = row + [1.0]
        for i, xi in enumerate(extended):
            xty[i] += xi * target
            for j, xj in enumerate(extended):
                xtx[i][j] += xi * xj
    for j in range(d):
        xtx[j][j] += alpha
    coef = _solve_linear_system(xtx, xty)
    weights = [coef[j] / stds[j] for j in range(d)]
    bias = coef[d] - sum(weights[j] * means[j] for j in range(d))
    return weights, bias


def _predict_row(row: list[float], weights: list[float], bias: float) -> float:
    return bias + sum(w * x for w, x in zip(weights, row))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[pivot][col]):
                pivot = row
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        if abs(pivot_val) < 1e-12:
            continue
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot_val
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    solution = [0.0] * n
    for col in reversed(range(n)):
        if abs(aug[col][col]) < 1e-12:
            solution[col] = 0.0
            continue
        solution[col] = (aug[col][n] - sum(aug[col][j] * solution[j] for j in range(col + 1, n))) / aug[col][col]
    return solution


class DualStableWavelengthRidgePredictor:
    """stable top-2 波長 + 強正則化 Ridge（保守的2波長拡張）。"""

    name = "dual_stable_wavelength_ridge"

    def __init__(self, alpha: float = 1000.0) -> None:
        self.alpha = alpha
        self._indices: list[int] = []
        self._weights: list[float] = []
        self._bias: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        n_features = len(X[0]) if X else 0
        k = min(2, n_features)
        self._indices = _select_stable_topk_indices(X, y, k)
        self._y_min = min(y)
        self._y_max = max(y)
        X_sel = _select_columns(X, self._indices)
        self._weights, self._bias = _ridge_fit(X_sel, y, self.alpha)

    def predict(self, X: Matrix) -> Vector:
        if not self._indices:
            raise RuntimeError("fit() を先に呼んでください")
        X_sel = _select_columns(X, self._indices)
        pred = [_predict_row(row, self._weights, self._bias) for row in X_sel]
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class DualPrepAnchorRawBlendPredictor:
    """アンカー(snv+diff1)と raw 1f の予測をブレンド（実測Public同点の1fを微量混合）。"""

    name = "dual_prep_anchor_raw_blend"

    def __init__(self, raw_weight: float = 0.02) -> None:
        self.raw_weight = raw_weight
        self._anchor_model = SingleFeatureLinearPredictor()
        self._raw_model = SingleFeatureLinearPredictor()
        self._prep_anchor = None
        self._prep_raw = None
        self._train_rows = None
        self._config = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_train_context(self, train_rows, config) -> None:
        self._train_rows = train_rows
        self._config = config

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._train_rows is None or self._config is None:
            raise RuntimeError("set_train_context() を fit 前に呼んでください")
        from pipeline.preprocessors import get_preprocessor

        self._y_min = min(y)
        self._y_max = max(y)
        self._prep_anchor = get_preprocessor("spectral_snv_diff1")
        self._prep_raw = get_preprocessor("spectral")
        meta = self._config.meta_cols
        target = self._config.target_col
        self._prep_anchor.fit(self._train_rows, target_col=target, meta_cols=meta)
        self._prep_raw.fit(self._train_rows, target_col=target, meta_cols=meta)
        X_anchor, y_anchor = self._prep_anchor.transform_train(self._train_rows)
        X_raw, _y_raw = self._prep_raw.transform_train(self._train_rows)
        self._anchor_model.fit(X_anchor, y_anchor)
        self._raw_model.fit(X_raw, y_anchor)

    def predict_test(self, test_rows) -> Vector:
        if self._prep_anchor is None or self._prep_raw is None:
            raise RuntimeError("fit() を先に呼んでください")
        X_anchor = self._prep_anchor.transform_test(test_rows)
        X_raw = self._prep_raw.transform_test(test_rows)
        pred_anchor = self._anchor_model.predict(X_anchor)
        pred_raw = self._raw_model.predict(X_raw)
        w = self.raw_weight
        out: Vector = []
        for pa, pr in zip(pred_anchor, pred_raw):
            value = (1.0 - w) * pa + w * pr
            out.append(min(max(value, self._y_min), self._y_max))
        return out

    def predict(self, X: Matrix) -> Vector:
        raise RuntimeError("predict_test() を使用してください")


class SingleFeatureTop3MedianPredictor:
    """相関上位3波長の1次元線形予測のメディアン（外れ値に強い単純アンサンブル）。"""

    name = "single_feature_top3_median"

    def __init__(self) -> None:
        self._models: list[tuple[int, float, float]] = []
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        if not X:
            raise ValueError("空データでは学習できません")
        self._y_min = min(y)
        self._y_max = max(y)
        self._models = []
        for feature_idx in _select_topk_indices(X, y, 3):
            x_col = [row[feature_idx] for row in X]
            slope, intercept = _fit_1d_linear(x_col, y)
            self._models.append((feature_idx, slope, intercept))

    def predict(self, X: Matrix) -> Vector:
        if not self._models:
            raise RuntimeError("fit() を先に呼んでください")
        out: Vector = []
        for row in X:
            values = [intercept + slope * row[feature_idx] for feature_idx, slope, intercept in self._models]
            values.sort()
            median = values[len(values) // 2]
            out.append(min(max(median, self._y_min), self._y_max))
        return out


class SingleFeatureLinearOofInterceptPredictor(SingleFeatureLinearPredictor):
    """アンカーと同じ波長 + LOO樹種OOFの平均バイアス補正（全サンプルへ一定シフト）。"""

    name = "single_feature_linear_oof_intercept"

    def __init__(self) -> None:
        super().__init__()
        self._group_labels: list[str] | None = None
        self._bias_correction: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_group_labels(self, labels: list[str]) -> None:
        self._group_labels = labels

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() を fit 前に呼んでください")
        self._y_min = min(y)
        self._y_max = max(y)
        super().fit(X, y)
        if self._feature_idx is None:
            return
        residuals: list[float] = []
        for group in sorted(set(self._group_labels)):
            train_idx = [i for i, label in enumerate(self._group_labels) if label != group]
            valid_idx = [i for i, label in enumerate(self._group_labels) if label == group]
            if len(train_idx) < 2 or not valid_idx:
                continue
            x_train = [X[i][self._feature_idx] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            slope, intercept = _fit_1d_linear(x_train, y_train)
            for i in valid_idx:
                pred = intercept + slope * X[i][self._feature_idx]
                residuals.append(y[i] - pred)
        self._bias_correction = sum(residuals) / len(residuals) if residuals else 0.0

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        out = [value + self._bias_correction for value in pred]
        return [min(max(value, self._y_min), self._y_max) for value in out]


class SingleFeatureSecondWavelengthPredictor(SingleFeatureLinearPredictor):
    """相関2位の波長で1次元線形回帰（1位と異なる仮説）。"""

    name = "single_feature_second_wavelength"

    def __init__(self) -> None:
        super().__init__()
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        n = len(y)
        if n == 0:
            raise ValueError("空データでは学習できません")
        self._y_min = min(y)
        self._y_max = max(y)
        top2 = _select_topk_indices(X, y, 2)
        rank1 = top2[0]
        rank2 = top2[1] if len(top2) > 1 else rank1
        self._feature_idx = rank2 if rank2 != rank1 else rank2
        x_best = [row[self._feature_idx] for row in X]
        self._slope, self._intercept = _fit_1d_linear(x_best, y)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class SingleFeatureLooRmseWavelengthPredictor(SingleFeatureLinearPredictor):
    """相関上位候補のうち LOO樹種RMSE 最小の波長を選ぶ。"""

    name = "single_feature_loo_rmse_wavelength"

    def __init__(self, candidate_k: int = 30) -> None:
        super().__init__()
        self.candidate_k = candidate_k
        self._group_labels: list[str] | None = None
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_group_labels(self, labels: list[str]) -> None:
        self._group_labels = labels

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() を fit 前に呼んでください")
        self._y_min = min(y)
        self._y_max = max(y)
        candidates = _select_topk_indices(X, y, min(self.candidate_k, len(X[0]) if X else 0))
        best_idx = candidates[0]
        best_rmse = float("inf")
        for feature_idx in candidates:
            rmse = _loo_species_rmse_for_feature(X, y, self._group_labels, feature_idx)
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = feature_idx
        self._feature_idx = best_idx
        x_best = [row[best_idx] for row in X]
        self._slope, self._intercept = _fit_1d_linear(x_best, y)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class SingleFeatureStableWavelengthPredictor(SingleFeatureLinearPredictor):
    """5-foldチャンクで相関上位を集約し、最も安定した1波長を選ぶ。"""

    name = "single_feature_stable_wavelength"

    def fit(self, X: Matrix, y: Vector) -> None:
        n = len(y)
        n_features = len(X[0]) if X else 0
        if n == 0 or n_features == 0:
            raise ValueError("空データでは学習できません")

        stable = _select_stable_topk_indices(X, y, 1)
        self._feature_idx = stable[0] if stable else 0

        x_best = [row[self._feature_idx] for row in X]
        y_mean = sum(y) / n
        x_mean = sum(x_best) / n
        x_var = sum((xi - x_mean) ** 2 for xi in x_best)
        if x_var <= 1e-12:
            self._slope = 0.0
            self._intercept = y_mean
            return

        cov_num = sum((x_best[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        self._slope = cov_num / x_var
        self._intercept = y_mean - self._slope * x_mean


class SingleFeatureLinearClippedPredictor(SingleFeatureLinearPredictor):
    name = "single_feature_linear_clipped"

    def __init__(self) -> None:
        super().__init__()
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        super().fit(X, y)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        return [min(max(value, self._y_min), self._y_max) for value in pred]


class SingleFeatureLinearLowTailPredictor(SingleFeatureLinearPredictor):
    name = "single_feature_linear_low_tail"

    def __init__(self, quantile: float = 0.1, factor: float = 1.25) -> None:
        super().__init__()
        self.quantile = quantile
        self.factor = factor
        self._threshold: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        super().fit(X, y)
        train_pred = super().predict(X)
        self._threshold = _quantile(train_pred, self.quantile)

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        out: Vector = []
        for value in pred:
            if value < self._threshold:
                value = self._threshold + self.factor * (value - self._threshold)
            out.append(value)
        return out


class SingleFeatureLinearLowTailMildPredictor(SingleFeatureLinearLowTailPredictor):
    name = "single_feature_linear_low_tail_mild"

    def __init__(self) -> None:
        super().__init__(quantile=0.1, factor=1.15)


class SingleFeatureLinearLowTailStrongPredictor(SingleFeatureLinearLowTailPredictor):
    name = "single_feature_linear_low_tail_strong"

    def __init__(self) -> None:
        super().__init__(quantile=0.1, factor=1.35)


class SingleFeatureLinearRangeCalibratedPredictor(SingleFeatureLinearPredictor):
    """LOO樹種OOFで学習した global アフィン補正 pred' = a * pred + b。"""

    name = "single_feature_linear_range_calibrated"

    def __init__(self) -> None:
        super().__init__()
        self._group_labels: list[str] | None = None
        self._scale: float = 1.0
        self._shift: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_group_labels(self, labels: list[str]) -> None:
        self._group_labels = labels

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._group_labels is None or len(self._group_labels) != len(y):
            raise RuntimeError("set_group_labels() を fit 前に呼んでください")
        self._y_min = min(y)
        self._y_max = max(y)
        super().fit(X, y)
        if self._feature_idx is None:
            return

        oof_pred: list[float] = []
        oof_true: list[float] = []
        groups = sorted(set(self._group_labels))
        for group in groups:
            train_idx = [i for i, label in enumerate(self._group_labels) if label != group]
            valid_idx = [i for i, label in enumerate(self._group_labels) if label == group]
            if len(train_idx) < 2 or not valid_idx:
                continue
            x_train = [X[i][self._feature_idx] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            slope, intercept = _fit_1d_linear(x_train, y_train)
            for i in valid_idx:
                oof_pred.append(intercept + slope * X[i][self._feature_idx])
                oof_true.append(y[i])

        if len(oof_pred) < 2:
            self._scale = 1.0
            self._shift = 0.0
            return

        pred_mean = sum(oof_pred) / len(oof_pred)
        true_mean = sum(oof_true) / len(oof_true)
        pred_var = sum((p - pred_mean) ** 2 for p in oof_pred)
        if pred_var <= 1e-12:
            self._scale = 1.0
            self._shift = true_mean - pred_mean
            return
        cov = sum((oof_pred[i] - pred_mean) * (oof_true[i] - true_mean) for i in range(len(oof_pred)))
        self._scale = cov / pred_var
        self._shift = true_mean - self._scale * pred_mean

    def predict(self, X: Matrix) -> Vector:
        pred = super().predict(X)
        out = [self._scale * value + self._shift for value in pred]
        return [min(max(value, self._y_min), self._y_max) for value in out]


class MoistureBinWavelengthLinearPredictor:
    """含水率四分位ビンごとに1波長線形。testは global 1f 予測でビン割当。"""

    name = "moisture_bin_wavelength_linear"

    def __init__(self, n_bins: int = 4, min_bin_samples: int = 40) -> None:
        self.n_bins = max(2, n_bins)
        self.min_bin_samples = min_bin_samples
        self._global: SingleFeatureLinearPredictor = SingleFeatureLinearPredictor()
        self._bin_edges: list[float] = []
        self._bin_models: list[tuple[float, float]] = []
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._global.fit(X, y)
        sorted_y = sorted(y)
        self._bin_edges = [
            sorted_y[0] - 1e-6,
            *[_quantile(sorted_y, i / self.n_bins) for i in range(1, self.n_bins)],
            sorted_y[-1] + 1e-6,
        ]

        global_idx = self._global._feature_idx or 0
        global_slope = self._global._slope
        global_intercept = self._global._intercept
        self._bin_models = []

        for b in range(self.n_bins):
            lo, hi = self._bin_edges[b], self._bin_edges[b + 1]
            if b == self.n_bins - 1:
                idx = [i for i, yi in enumerate(y) if yi >= lo]
            else:
                idx = [i for i, yi in enumerate(y) if lo <= yi < hi]
            if len(idx) < self.min_bin_samples:
                self._bin_models.append((global_slope, global_intercept))
                continue
            x_col = [X[i][global_idx] for i in idx]
            y_bin = [y[i] for i in idx]
            slope, intercept = _fit_1d_linear(x_col, y_bin)
            self._bin_models.append((slope, intercept))

    def predict(self, X: Matrix) -> Vector:
        feature_idx = self._global._feature_idx or 0
        base_preds = self._global.predict(X)
        out: Vector = []
        for row, moisture_proxy in zip(X, base_preds):
            b = _moisture_bin_index(moisture_proxy, self._bin_edges)
            slope, intercept = self._bin_models[b]
            value = intercept + slope * row[feature_idx]
            out.append(min(max(value, self._y_min), self._y_max))
        return out


class MoistureBinFixedWavelengthPredictor:
    """global 波長固定 + 含水率ビン別 slope（縮小付き）。"""

    name = "moisture_bin_fixed_wavelength"

    def __init__(self, n_bins: int = 4, min_bin_samples: int = 40, shrink_lambda: float = 80.0) -> None:
        self.n_bins = n_bins
        self.min_bin_samples = min_bin_samples
        self.shrink_lambda = shrink_lambda
        self._global = SingleFeatureLinearPredictor()
        self._feature_idx: int = 0
        self._global_slope: float = 0.0
        self._global_intercept: float = 0.0
        self._bin_edges: list[float] = []
        self._bin_models: list[tuple[float, float]] = []
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        self._y_min = min(y)
        self._y_max = max(y)
        self._global.fit(X, y)
        self._feature_idx = self._global._feature_idx or 0
        self._global_slope = self._global._slope
        self._global_intercept = self._global._intercept

        train_base = self._global.predict(X)
        sorted_proxy = sorted(train_base)
        self._bin_edges = [
            sorted_proxy[0] - 1e-6,
            *[_quantile(sorted_proxy, i / self.n_bins) for i in range(1, self.n_bins)],
            sorted_proxy[-1] + 1e-6,
        ]
        self._bin_models = []
        for b in range(self.n_bins):
            lo, hi = self._bin_edges[b], self._bin_edges[b + 1]
            if b == self.n_bins - 1:
                idx = [i for i, proxy in enumerate(train_base) if proxy >= lo]
            else:
                idx = [i for i, proxy in enumerate(train_base) if lo <= proxy < hi]
            if len(idx) < self.min_bin_samples:
                self._bin_models.append((self._global_slope, self._global_intercept))
                continue
            x_col = [X[i][self._feature_idx] for i in idx]
            y_bin = [y[i] for i in idx]
            slope, intercept = _fit_1d_linear(x_col, y_bin)
            w = len(idx) / (len(idx) + self.shrink_lambda)
            slope = (1 - w) * self._global_slope + w * slope
            intercept = (1 - w) * self._global_intercept + w * intercept
            self._bin_models.append((slope, intercept))

    def predict(self, X: Matrix) -> Vector:
        base_preds = self._global.predict(X)
        out: Vector = []
        for row, moisture_proxy in zip(X, base_preds):
            b = _moisture_bin_index(moisture_proxy, self._bin_edges)
            slope, intercept = self._bin_models[b]
            value = intercept + slope * row[self._feature_idx]
            out.append(min(max(value, self._y_min), self._y_max))
        return out


class MscSelectRawLinearPredictor:
    """MSC で波長選択し、raw スペクトルで 1 次元線形回帰。"""

    name = "msc_select_raw_linear"

    def __init__(self) -> None:
        self._train_rows: Rows | None = None
        self._config = None
        self._feature_idx: int | None = None
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._y_min: float = 0.0
        self._y_max: float = 0.0

    def set_train_context(self, train_rows, config) -> None:
        self._train_rows = train_rows
        self._config = config

    def fit(self, X: Matrix, y: Vector) -> None:
        if self._train_rows is None or self._config is None:
            raise RuntimeError("set_train_context() を fit 前に呼んでください")
        from pipeline.preprocessors import get_preprocessor

        self._y_min = min(y)
        self._y_max = max(y)
        msc = get_preprocessor("spectral_msc")
        msc.fit(self._train_rows, target_col=self._config.target_col, meta_cols=self._config.meta_cols)
        X_msc, _ = msc.transform_train(self._train_rows)
        self._feature_idx = _select_top1_index(X_msc, y)
        x_raw = [row[self._feature_idx] for row in X]
        self._slope, self._intercept = _fit_1d_linear(x_raw, y)

    def predict(self, X: Matrix) -> Vector:
        if self._feature_idx is None:
            raise RuntimeError("fit() を先に呼んでください")
        out: Vector = []
        for row in X:
            value = self._intercept + self._slope * row[self._feature_idx]
            out.append(min(max(value, self._y_min), self._y_max))
        return out


def _moisture_bin_index(value: float, edges: list[float]) -> int:
    n_bins = len(edges) - 1
    for b in range(n_bins - 1):
        if value < edges[b + 1]:
            return b
    return n_bins - 1


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
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight
