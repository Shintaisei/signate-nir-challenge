"""ローカル評価指標。"""

from __future__ import annotations

import math

from pipeline.types import Vector


def rmse(y_true: Vector, y_pred: Vector) -> float:
    """Root Mean Squared Error。小さいほど良い。"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true: Vector, y_pred: Vector) -> float:
    """Mean Absolute Error。小さいほど良い。"""
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def r2(y_true: Vector, y_pred: Vector) -> float:
    """Coefficient of determination。大きいほど良い。"""
    y_mean = sum(y_true) / len(y_true)
    denominator = sum((value - y_mean) ** 2 for value in y_true)
    if denominator == 0:
        return 0.0
    numerator = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - numerator / denominator


def regression_scores(y_true: Vector, y_pred: Vector) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
