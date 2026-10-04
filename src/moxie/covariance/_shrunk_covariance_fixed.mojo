# Adapted from scikit-learn sklearn/covariance/_shrunk_covariance.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Fixed-shrinkage covariance estimator (sklearn `ShrunkCovariance`).

The `shrunk_covariance` kernel lives with the family engine in
`_empirical_covariance`; adaptive Ledoit-Wolf/OAS shrinkage is owned by the
shrinkage child. Both are re-exported from `moxie.covariance`.
"""
import numojo as nm
from numojo import NDArray, Shape
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._empirical_covariance import (
    CovarianceModel, column_means, empirical_covariance, empirical_covariance_float64, shrunk_covariance, to_array, validate_samples,
)


def check_shrinkage(shrinkage: Float64, owner: String) raises:
    if not (shrinkage >= 0.0 and shrinkage <= 1.0):
        raise Error(String("InvalidParameterError: The 'shrinkage' parameter of ", owner,
                           " must be a float in the range [0, 1]. Got ", shrinkage, " instead."))


@fieldwise_init
struct ShrunkCovarianceParams(Copyable, Movable, Writable):
    """Constructor configuration of ShrunkCovariance."""

    var store_precision: Bool
    var assume_centered: Bool
    var shrinkage: Float64


struct ShrunkCovariance[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Covariance estimator with fixed shrinkage toward a scaled identity.

    Defaults: store_precision=True, assume_centered=False, shrinkage=0.1.
    covariance_ = (1 - shrinkage) S + shrinkage tr(S)/p I, with S the
    empirical covariance. Learned attributes are accessor methods.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = ShrunkCovarianceParams
    comptime NAME = "ShrunkCovariance"

    var store_precision: Bool
    var assume_centered: Bool
    var shrinkage: Float64
    var _model: CovarianceModel[Self.dtype]

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False, shrinkage: Float64 = 0.1):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage
        self._model = CovarianceModel[Self.dtype]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered, self.shrinkage)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        self.shrinkage = params.shrinkage
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, store_precision: Optional[Bool] = None, assume_centered: Optional[Bool] = None,
                   shrinkage: Optional[Float64] = None) raises -> ref[self] Self:
        """Update only the supplied parameters (validated at fit, as in sklearn)."""
        if store_precision:
            self.store_precision = store_precision.value()
        if assume_centered:
            self.assume_centered = assume_centered.value()
        if shrinkage:
            self.shrinkage = shrinkage.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered, shrinkage=self.shrinkage)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        check_shrinkage(self.shrinkage, Self.NAME)
        validate_samples(X, Self.NAME + ".fit")
        var p = X.shape[1]
        var location: NDArray[Self.dtype]
        if self.assume_centered:
            location = nm.zeros[Self.dtype](Shape(p))
        else:
            location = to_array[Self.dtype](column_means(X))
        self._model.update_location(location)
        if self.assume_centered:
            var covariance = shrunk_covariance(empirical_covariance(X, assume_centered=True), self.shrinkage)
            self._model.install(location, covariance, self.store_precision, Self.NAME)
        else:
            # sklearn: np.cov gives Float64, and shrunk_covariance keeps it.
            var covariance = shrunk_covariance(empirical_covariance_float64(X, False), self.shrinkage)
            self._model.install_float64(location, covariance, self.store_precision, Self.NAME)
        return self

    def get_precision(self) raises -> Self.Precision:
        return self._model.get_precision(Self.NAME)

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        return self._model.mahalanobis(X, Self.NAME)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        return self._model.score(X, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return self._model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._model.n_features_in(Self.NAME)
