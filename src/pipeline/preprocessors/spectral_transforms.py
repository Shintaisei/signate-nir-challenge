"""スペクトル向けの前処理。"""

from __future__ import annotations

import math

from pipeline.preprocessors.spectral import SpectralPreprocessor
from pipeline.types import Matrix, Rows, Vector


class SpectralSnvPreprocessor(SpectralPreprocessor):
    name = "spectral_snv"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _snv(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _snv(super().transform_test(test))


class SpectralDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(super().transform_test(test))


class SpectralDiff2Preprocessor(SpectralPreprocessor):
    name = "spectral_diff2"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_diff(X)), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_diff(super().transform_test(test)))


class SpectralSnvDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_snv_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_snv(X)), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_snv(super().transform_test(test)))


class SpectralMetaPreprocessor(SpectralPreprocessor):
    name = "spectral_meta"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _append_group_position(X, train), y

    def transform_test(self, test: Rows) -> Matrix:
        return _append_group_position(super().transform_test(test), test)


class SpectralSnvDiff1MetaPreprocessor(SpectralPreprocessor):
    name = "spectral_snv_diff1_meta"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _append_group_position(_diff(_snv(X)), train), y

    def transform_test(self, test: Rows) -> Matrix:
        return _append_group_position(_diff(_snv(super().transform_test(test))), test)


class SpectralSnvDiff2Preprocessor(SpectralPreprocessor):
    name = "spectral_snv_diff2"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_diff(_snv(X))), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_diff(_snv(super().transform_test(test))))


class SpectralSmoothPreprocessor(SpectralPreprocessor):
    name = "spectral_smooth"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _moving_average(X, window=9), y

    def transform_test(self, test: Rows) -> Matrix:
        return _moving_average(super().transform_test(test), window=9)


class SpectralSmoothDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_smooth_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_moving_average(X, window=9)), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_moving_average(super().transform_test(test), window=9))


class SpectralSnvSmoothDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_snv_smooth_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_moving_average(_snv(X), window=9)), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_moving_average(_snv(super().transform_test(test)), window=9))


class SpectralSgDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_sg_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_sg_smooth5(X)), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_sg_smooth5(super().transform_test(test)))


class SpectralSnvSgDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_snv_sg_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_sg_smooth5(_snv(X))), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_sg_smooth5(_snv(super().transform_test(test))))


class SpectralSnvSgDiff1MetaPreprocessor(SpectralPreprocessor):
    name = "spectral_snv_sg_diff1_meta"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _append_group_position(_diff(_sg_smooth5(_snv(X))), train), y

    def transform_test(self, test: Rows) -> Matrix:
        return _append_group_position(_diff(_sg_smooth5(_snv(super().transform_test(test)))), test)


class SpectralAreaPreprocessor(SpectralPreprocessor):
    name = "spectral_area"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _area_normalize(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _area_normalize(super().transform_test(test))


class SpectralSnvAreaDiff1Preprocessor(SpectralPreprocessor):
    name = "spectral_snv_area_diff1"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _diff(_area_normalize(_snv(X))), y

    def transform_test(self, test: Rows) -> Matrix:
        return _diff(_area_normalize(_snv(super().transform_test(test))))


class SpectralDetrendPreprocessor(SpectralPreprocessor):
    name = "spectral_detrend"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _detrend(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _detrend(super().transform_test(test))


class SpectralCenterPreprocessor(SpectralPreprocessor):
    """行平均のみ除去（SNV より弱いスケール補正）。"""

    name = "spectral_center"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _center(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _center(super().transform_test(test))


class SpectralL2NormPreprocessor(SpectralPreprocessor):
    """行ベクトルの L2 正規化（形状は保持、強度のみ除去）。"""

    name = "spectral_l2norm"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _l2_normalize(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _l2_normalize(super().transform_test(test))


class SpectralBlendSnv25Preprocessor(SpectralPreprocessor):
    """raw 75% + SNV 25% の線形ブレンド。"""

    name = "spectral_blend_snv25"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _blend_raw_snv(X, 0.25), y

    def transform_test(self, test: Rows) -> Matrix:
        return _blend_raw_snv(super().transform_test(test), 0.25)


class SpectralBlendSnv50Preprocessor(SpectralPreprocessor):
    """raw 50% + SNV 50% の線形ブレンド。"""

    name = "spectral_blend_snv50"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _blend_raw_snv(X, 0.5), y

    def transform_test(self, test: Rows) -> Matrix:
        return _blend_raw_snv(super().transform_test(test), 0.5)


class SpectralSmooth3Preprocessor(SpectralPreprocessor):
    name = "spectral_smooth3"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _moving_average(X, window=3), y

    def transform_test(self, test: Rows) -> Matrix:
        return _moving_average(super().transform_test(test), window=3)


class SpectralSmooth5Preprocessor(SpectralPreprocessor):
    name = "spectral_smooth5"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _moving_average(X, window=5), y

    def transform_test(self, test: Rows) -> Matrix:
        return _moving_average(super().transform_test(test), window=5)


class SpectralMscPreprocessor(SpectralPreprocessor):
    """Multiplicative scatter correction using the train mean spectrum."""

    name = "spectral_msc"

    def __init__(self) -> None:
        super().__init__()
        self._reference: list[float] = []

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        self._reference = _column_means(X)
        return _msc(X, self._reference), y

    def transform_test(self, test: Rows) -> Matrix:
        if not self._reference:
            raise RuntimeError("fit() を先に呼んでください")
        return _msc(super().transform_test(test), self._reference)


class SpectralMultiviewPreprocessor(SpectralPreprocessor):
    name = "spectral_multiview"

    def transform_train(self, train: Rows) -> tuple[Matrix, Vector]:
        X, y = super().transform_train(train)
        return _multiview(X), y

    def transform_test(self, test: Rows) -> Matrix:
        return _multiview(super().transform_test(test))


def _center(X: Matrix) -> Matrix:
    out: Matrix = []
    for row in X:
        mean = sum(row) / len(row)
        out.append([value - mean for value in row])
    return out


def _l2_normalize(X: Matrix) -> Matrix:
    out: Matrix = []
    for row in X:
        norm = math.sqrt(sum(value * value for value in row))
        out.append([value / max(norm, 1e-12) for value in row])
    return out


def _blend_raw_snv(X: Matrix, snv_weight: float) -> Matrix:
    snv = _snv(X)
    raw_weight = 1.0 - snv_weight
    return [
        [raw_weight * raw + snv_weight * snv for raw, snv in zip(raw_row, snv_row)]
        for raw_row, snv_row in zip(X, snv)
    ]


def _snv(X: Matrix) -> Matrix:
    out: Matrix = []
    for row in X:
        mean = sum(row) / len(row)
        var = sum((value - mean) ** 2 for value in row) / max(len(row) - 1, 1)
        std = math.sqrt(max(var, 1e-12))
        out.append([(value - mean) / std for value in row])
    return out


def _diff(X: Matrix) -> Matrix:
    return [[row[i + 1] - row[i] for i in range(len(row) - 1)] for row in X]


def _diff_padded(X: Matrix) -> Matrix:
    return [[0.0, *[row[i + 1] - row[i] for i in range(len(row) - 1)]] for row in X]


def _diff2_padded(X: Matrix) -> Matrix:
    return [
        [0.0, 0.0, *[row[i + 2] - 2.0 * row[i + 1] + row[i] for i in range(len(row) - 2)]]
        for row in X
    ]


def _multiview(X: Matrix) -> Matrix:
    views = [
        X,
        _moving_average(X, window=3),
        _moving_average(X, window=5),
        _center(X),
        _snv(X),
        _diff_padded(X),
        _diff2_padded(X),
    ]
    out: Matrix = []
    for rows in zip(*views):
        merged: list[float] = []
        for row in rows:
            merged.extend(row)
        out.append(merged)
    return out


def _append_group_position(X: Matrix, rows: Rows) -> Matrix:
    positions = _group_positions(rows)
    out: Matrix = []
    for values, row, position in zip(X, rows, positions):
        group = float(row.get("species number", 0) or 0)
        out.append([*values, group, position])
    return out


def _group_positions(rows: Rows) -> list[float]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for i, row in enumerate(rows):
        group = row.get("species number", row.get("樹種", ""))
        sample_number = float(row.get("sample number", i))
        grouped.setdefault(str(group), []).append((i, sample_number))

    positions = [0.0] * len(rows)
    for items in grouped.values():
        samples = [sample for _, sample in items]
        start = min(samples)
        width = max(max(samples) - start, 1.0)
        for i, sample in items:
            positions[i] = (sample - start) / width
    return positions


def _moving_average(X: Matrix, *, window: int) -> Matrix:
    radius = window // 2
    out: Matrix = []
    for row in X:
        smoothed: list[float] = []
        for i in range(len(row)):
            start = max(0, i - radius)
            end = min(len(row), i + radius + 1)
            smoothed.append(sum(row[start:end]) / (end - start))
        out.append(smoothed)
    return out


def _sg_smooth5(X: Matrix) -> Matrix:
    weights = (-3.0, 12.0, 17.0, 12.0, -3.0)
    scale = 35.0
    out: Matrix = []
    for row in X:
        if len(row) < 5:
            out.append(row[:])
            continue
        smoothed = row[:]
        for i in range(2, len(row) - 2):
            smoothed[i] = sum(weights[j] * row[i + j - 2] for j in range(5)) / scale
        smoothed[0] = (row[0] + row[1]) / 2
        smoothed[1] = (row[0] + row[1] + row[2]) / 3
        smoothed[-2] = (row[-3] + row[-2] + row[-1]) / 3
        smoothed[-1] = (row[-2] + row[-1]) / 2
        out.append(smoothed)
    return out


def _area_normalize(X: Matrix) -> Matrix:
    out: Matrix = []
    for row in X:
        area = sum(abs(value) for value in row) / len(row)
        out.append([value / max(area, 1e-12) for value in row])
    return out


def _detrend(X: Matrix) -> Matrix:
    out: Matrix = []
    for row in X:
        n = len(row)
        x_mean = (n - 1) / 2
        y_mean = sum(row) / n
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum((i - x_mean) * (row[i] - y_mean) for i in range(n)) / max(denom, 1e-12)
        intercept = y_mean - slope * x_mean
        out.append([row[i] - (intercept + slope * i) for i in range(n)])
    return out


def _column_means(X: Matrix) -> list[float]:
    n = len(X)
    p = len(X[0]) if X else 0
    return [sum(row[j] for row in X) / n for j in range(p)]


def _msc(X: Matrix, reference: list[float]) -> Matrix:
    ref_mean = sum(reference) / len(reference)
    ref_centered = [value - ref_mean for value in reference]
    ref_var = sum(value * value for value in ref_centered)

    out: Matrix = []
    for row in X:
        row_mean = sum(row) / len(row)
        row_centered = [value - row_mean for value in row]
        slope = sum(row_centered[i] * ref_centered[i] for i in range(len(row))) / max(ref_var, 1e-12)
        if abs(slope) <= 1e-12:
            slope = 1e-12
        intercept = row_mean - slope * ref_mean
        out.append([(value - intercept) / slope for value in row])
    return out
