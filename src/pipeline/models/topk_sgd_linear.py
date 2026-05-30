"""上位相関特徴量を使う簡易SGD線形回帰。"""

from __future__ import annotations

import math
import random

from pipeline.types import Matrix, Vector


class TopKSgdLinearPredictor:
    name = "topk_sgd_linear"

    def __init__(self, top_k: int = 40, epochs: int = 35, lr: float = 0.01, seed: int = 42) -> None:
        self.top_k = top_k
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._indices: list[int] = []
        self._means: list[float] = []
        self._stds: list[float] = []
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, X: Matrix, y: Vector) -> None:
        n = len(y)
        d = len(X[0]) if X else 0
        if n == 0 or d == 0:
            raise ValueError("空データでは学習できません")

        self._indices = self._select_topk_indices(X, y, min(self.top_k, d))
        X_sel = [[row[j] for j in self._indices] for row in X]
        X_std = self._fit_transform_standardize(X_sel)

        self._w = [0.0] * len(self._indices)
        self._b = sum(y) / n
        idxs = list(range(n))
        rng = random.Random(self.seed)

        for _ in range(self.epochs):
            rng.shuffle(idxs)
            for i in idxs:
                xi = X_std[i]
                yi = y[i]
                pred = self._b + sum(self._w[j] * xi[j] for j in range(len(self._w)))
                err = pred - yi
                for j in range(len(self._w)):
                    self._w[j] -= self.lr * err * xi[j]
                self._b -= self.lr * err

    def predict(self, X: Matrix) -> Vector:
        if not self._indices:
            raise RuntimeError("fit() を先に呼んでください")
        X_sel = [[row[j] for j in self._indices] for row in X]
        X_std = self._transform_standardize(X_sel)
        return [self._b + sum(self._w[j] * row[j] for j in range(len(self._w))) for row in X_std]

    def _select_topk_indices(self, X: Matrix, y: Vector, k: int) -> list[int]:
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

    def _fit_transform_standardize(self, X: Matrix) -> Matrix:
        n = len(X)
        p = len(X[0]) if X else 0
        self._means = []
        self._stds = []
        out: Matrix = [[0.0] * p for _ in range(n)]
        for j in range(p):
            col = [X[i][j] for i in range(n)]
            mean = sum(col) / n
            var = sum((v - mean) ** 2 for v in col) / max(n - 1, 1)
            std = math.sqrt(max(var, 1e-12))
            self._means.append(mean)
            self._stds.append(std)
            for i in range(n):
                out[i][j] = (X[i][j] - mean) / std
        return out

    def _transform_standardize(self, X: Matrix) -> Matrix:
        n = len(X)
        p = len(X[0]) if X else 0
        out: Matrix = [[0.0] * p for _ in range(n)]
        for j in range(p):
            mean = self._means[j]
            std = self._stds[j]
            for i in range(n):
                out[i][j] = (X[i][j] - mean) / std
        return out
