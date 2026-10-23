# Adapted from scikit-learn sklearn/cross_decomposition/_pls.py (PLSSVD) at
# 4358542a1bd39af2e7e1cbace8b9fed067473c5b; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense PLSSVD: one SVD of the centered/scaled cross-covariance X.T @ y.

Reuses `PLSEngine.center_scale` and `projection_weights`
(mojo-linalg economy SVD + u-based `svd_flip`); no NIPALS state, deflation,
rotations or prediction.
"""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isnan
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie.cross_decomposition._base import CrossDecompositionTransformer
from moxie.cross_decomposition._contracts import Matrix, PairedScores
from moxie.cross_decomposition._pls import (
    PLSEngine,
    as_target_matrix,
    check_consistent_length,
    check_matrix,
    check_n_components_bound,
    check_n_features,
    check_positive_int,
    check_target,
    compact_float64,
    not_fitted_error,
    prefixed_feature_names,
    write_back_float64,
)


@fieldwise_init
struct PLSSVDParams(Copyable, Movable, Writable):
    """PLSSVD constructor configuration (`copy` is stored as `copy_input`)."""

    var n_components: Int
    var scale: Bool
    var copy_input: Bool


def _standardize(x: Matrix, mean: Matrix, std: Matrix) raises -> Matrix:
    """(x - mean) / std in Float64 on an (n, d) Float64 matrix."""
    var rows = x.shape[0]
    var cols = x.shape[1]
    var out = nm.zeros[DType.float64](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            out.unsafe_set(
                i * cols + j,
                (x.unsafe_get(i * cols + j) - mean.unsafe_get(j)) / std.unsafe_get(j),
            )
    return out^


struct PLSSVD[dtype: DType = DType.float64](
    Copyable, CrossDecompositionTransformer, FeatureNamesOut, InputFeatureCount, Movable
):
    """Partial Least Square SVD over NuMojo arrays of `dtype`.

    X is projected on the left and y on the right singular vectors of the
    cross-covariance. All fit and transform arithmetic is Float64; like
    scikit-learn, transform converts inputs to Float64 before centering.
    """

    comptime Params = PLSSVDParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = Matrix
    comptime PairedTransform = PairedScores

    var n_components: Int
    var scale: Bool
    var copy_input: Bool
    var x_weights_: Matrix
    """(n_features, n_components) left singular vectors."""
    var y_weights_: Matrix
    """(n_targets, n_components) right singular vectors."""
    var n_features_in_: Int
    var _x_mean: Matrix
    var _y_mean: Matrix
    var _x_std: Matrix
    var _y_std: Matrix
    var _fitted: Bool

    def __init__(
        out self, n_components: Int = 2, *, scale: Bool = True, copy: Bool = True
    ) raises:
        """scikit-learn defaults: n_components=2, scale=True, copy=True."""
        self = Self(params=PLSSVDParams(n_components, scale, copy))

    def __init__(out self, *, params: PLSSVDParams) raises:
        """Construct from a typed parameter record (used by clone)."""
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "PLSSVD accepts Float32 or Float64 inputs"
        self.n_components = params.n_components
        self.scale = params.scale
        self.copy_input = params.copy_input
        self.x_weights_ = nm.zeros[DType.float64](Shape(0, 0))
        self.y_weights_ = nm.zeros[DType.float64](Shape(0, 0))
        self.n_features_in_ = 0
        self._x_mean = nm.zeros[DType.float64](Shape(0))
        self._y_mean = nm.zeros[DType.float64](Shape(0))
        self._x_std = nm.zeros[DType.float64](Shape(0))
        self._y_std = nm.zeros[DType.float64](Shape(0))
        self._fitted = False

    @staticmethod
    def name() -> String:
        return "PLSSVD"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return PLSSVDParams(self.n_components, self.scale, self.copy_input)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.scale = params.scale
        self.copy_input = params.copy_input
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[Int] = None,
        scale: Optional[Bool] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update only the supplied parameters (validated at fit)."""
        if n_components:
            self.n_components = n_components.value()
        if scale:
            self.scale = scale.value()
        if copy:
            self.copy_input = copy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(params=self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        """Fit on copies; the caller's arrays are never modified."""
        self._validate(X, y)
        var xk = compact_float64(X)
        var yk = as_target_matrix(compact_float64(y))
        self._fit_workspace(xk, yk)
        return self

    def fit_inplace[origin: MutOrigin](
        ref[origin] self, mut X: Self.Input, mut y: Self.Target
    ) raises -> ref[self] Self:
        """scikit-learn `fit` honouring `copy`: with copy=False and Float64
        inputs, X and y end centered (and scaled), as in scikit-learn."""
        self._validate(X, y)
        var xk = compact_float64(X)
        var yk = as_target_matrix(compact_float64(y))
        self._fit_workspace(xk, yk)
        comptime if Self.dtype == DType.float64:
            if not self.copy_input:
                write_back_float64(X, xk)
                if y.ndim == 1:
                    write_back_float64(y, yk.reshape(Shape(yk.shape[0])))
                else:
                    write_back_float64(y, yk)
        return self

    def _validate(self, X: Self.Input, y: Self.Target) raises:
        check_positive_int(self.n_components, "n_components", Self.name())
        check_consistent_length(X.shape[0], y.shape[0])
        check_matrix(X, "X", Self.name(), 2)
        check_target(y, Self.name())
        var q = 1 if y.ndim == 1 else y.shape[1]
        check_n_components_bound(
            self.n_components, min(X.shape[0], min(X.shape[1], q))
        )

    def _fit_workspace(mut self, mut xk: Matrix, mut yk: Matrix) raises:
        var engine = PLSEngine()
        var scaling = engine.center_scale(xk, yk, self.scale)
        var weights = engine.projection_weights(xk, yk, self.n_components)
        self.x_weights_ = weights.x_weights.copy()
        self.y_weights_ = weights.y_weights.copy()
        self._x_mean = scaling.x_mean.copy()
        self._y_mean = scaling.y_mean.copy()
        self._x_std = scaling.x_std.copy()
        self._y_std = scaling.y_std.copy()
        self.n_features_in_ = xk.shape[1]
        self._fitted = True

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """((X - x_mean) / x_std) @ x_weights_, shape (n, n_components)."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        check_matrix(X, "X", Self.name(), 1)
        check_n_features(X.shape[1], self.n_features_in_, Self.name())
        return _standardize(compact_float64(X), self._x_mean, self._x_std) @ self.x_weights_

    def transform(self, X: Self.Input, y: Self.Target) raises -> Self.PairedTransform:
        """Paired scores; y may be (n,) or (n, q)."""
        var x_scores = self.transform(X)
        check_target(y, Self.name())
        var y2 = as_target_matrix(compact_float64(y))
        if y2.shape[1] != self._y_mean.shape[0]:
            raise Error(
                "ValueError: y has "
                + String(y2.shape[1])
                + " targets, but PLSSVD was fitted with "
                + String(self._y_mean.shape[0])
            )
        var y_scores = _standardize(y2, self._y_mean, self._y_std) @ self.y_weights_
        return PairedScores(x_scores^, y_scores^)

    def fit_transform(mut self, X: Self.Input, y: Self.Target) raises -> Self.PairedTransform:
        """fit(X, y) followed by the paired transform(X, y)."""
        _ = self.fit(X, y)
        return self.transform(X, y)

    def get_n_features_in(self) raises -> Int:
        """scikit-learn's `n_features_in_` (InputFeatureCount); raises before fit."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        return self.n_features_in_

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        """['plssvd0', 'plssvd1', ...]."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        return prefixed_feature_names(
            Self.name(), self.x_weights_.shape[1], self.n_features_in_, input_features
        )
