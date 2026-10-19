# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Typed model access and shared feature-importance aggregation."""
import numojo as nm
from std.math import isfinite, isnan, inf, floor, sqrt
from moxie.base import BaseEstimator
from moxie._dense import sum_axis2d
from ._contracts import ImportanceProvider

trait SelectionEstimator(BaseEstimator, ImportanceProvider, Copyable, Deinitable):
    """A real estimator adapter; numerical fitting belongs to the wrapped model."""
    comptime Target: Copyable & Deinitable
    def fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], sample_weight: Optional[nm.NDArray[Self.dtype]]) raises:
        ...
    def n_features(self) raises -> Int:
        ...
    def uses_l1_threshold(self) -> Bool:
        ...
    def allows_nan(self) -> Bool:
        return False
    def importances(self, getter: String) raises -> nm.NDArray[Self.dtype]:
        if getter != "auto" and getter != "feature_importances_":
            raise Error("ValueError: unsupported importance_getter; use a typed estimator adapter")
        return self.feature_importances()


trait IncrementalSelectionEstimator(SelectionEstimator):
    """Optional incremental fitting; update arguments belong to each adapter."""
    comptime PartialFitParams: Copyable & Deinitable

    def partial_fit_selection(
        mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target],
        params: Optional[Self.PartialFitParams] = None,
    ) raises:
        """Update learned state without resetting it; never emulate with fit."""
        ...


def aggregate_importances[dtype: DType](raw: nm.NDArray[dtype], norm_order: Float64 = 1, square: Bool = False) raises -> nm.NDArray[dtype]:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if isnan(norm_order) or norm_order == 0 or (isfinite(norm_order) and norm_order != floor(norm_order)):
        raise Error("ValueError: norm_order must be a nonzero integer or infinity")
    if raw.ndim != 1 and raw.ndim != 2:
        raise Error("ValueError: importances must have rank one or two")
    var columns = raw.shape[raw.ndim - 1]
    var rows = 1 if raw.ndim == 1 else raw.shape[0]
    if columns == 0 or rows == 0:
        raise Error("ValueError: importances must be nonempty")
    var values = nm.zeros[dtype](nm.Shape(rows, columns))
    var result = nm.zeros[dtype](nm.Shape(columns))
    for j in range(columns):
        var extreme = SIMD[dtype, 1](0) if norm_order > 0 else inf[dtype]()
        for i in range(rows):
            var offset = j * Int(raw.strides[raw.ndim - 1])
            if raw.ndim == 2:
                offset += i * Int(raw.strides[0])
            var value = abs(raw.unsafe_get(offset))
            if isnan(value):
                extreme = value
            elif not isnan(extreme):
                extreme = max(extreme, value) if norm_order > 0 else min(extreme, value)
            if isnan(value):
                values.unsafe_set(i * columns + j, value)
            elif square:
                values.unsafe_set(i * columns + j, value * value)
            elif raw.ndim == 1:
                values.unsafe_set(j, value)
            elif isfinite(norm_order):
                if value == 0 and norm_order < 0:
                    values.unsafe_set(i * columns + j, inf[dtype]())
                elif norm_order == 1:
                    values.unsafe_set(i * columns + j, value)
                elif norm_order == 2:
                    values.unsafe_set(i * columns + j, value * value)
                elif norm_order == -1:
                    values.unsafe_set(i * columns + j, 1 / value)
                else:
                    values.unsafe_set(i * columns + j, value ** SIMD[dtype, 1](norm_order))
        result.unsafe_set(j, extreme)
    if not square and raw.ndim == 2 and not isfinite(norm_order):
        return result^
    var totals = sum_axis2d(values, 0)
    for j in range(columns):
        var value = totals.unsafe_get(j)
        if raw.ndim == 2 and not square and norm_order != 1 and not isnan(value):
            if norm_order == 2:
                value = sqrt(value)
            elif norm_order == -1:
                value = 1 / value
            else:
                value = value ** SIMD[dtype, 1](1 / norm_order)
        result.unsafe_set(j, value)
    return result^


trait SelectionClassLabels(SelectionEstimator):
    """Optional owned class-label access for a classifier adapter."""
    comptime Classes: Copyable & Deinitable
    def classes_(self) raises -> Self.Classes:
        ...
