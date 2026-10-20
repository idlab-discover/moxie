# Adapted from scikit-learn sklearn/linear_model/_base.py (LinearRegression);
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Dense ordinary least squares and nonnegative least squares (`LinearRegression`)."""
from numojo import NDArray, Shape, zeros
from std.math import isfinite
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie.linear_model._base import LinearRegressor
from moxie.linear_model._linear_base import (
    preprocess_data,
    scalar_weights,
    set_intercept,
    linear_decision,
    public_coef,
    canonical_coef,
    public_prediction,
    check_predict_input,
    r2_score,
)
from moxie._linear_linalg import svd, NumericalError
from mojo_optimize import nnls


@fieldwise_init
struct LinearRegressionParams(Copyable, Movable):
    """Typed `get_params`/`set_params` record; field names follow sklearn."""

    var fit_intercept: Bool
    var copy_X: Bool
    var tol: Float64
    var n_jobs: Optional[Int]
    var positive: Bool


@fieldwise_init
struct _Solution[dtype: DType](Movable):
    var coef: NDArray[Self.dtype]  # (outputs, features)
    var rank: Int
    var singular: NDArray[Self.dtype]


def _lstsq[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], tol: Float64) raises NumericalError -> _Solution[dtype]:
    """scipy.linalg.lstsq(X, Y, cond=tol): singular values <= tol * s_max are dropped."""
    var factor = svd(X)
    var solved = factor.solve(Y, rtol=tol, atol=0.0)
    var p = X.shape[1]
    var k = Y.shape[1]
    var coef = zeros[dtype](Shape(k, p))
    for j in range(p):
        for t in range(k):
            coef.unsafe_set(t * p + j, solved.solution.unsafe_get(j * k + t))
    return _Solution[dtype](coef^, solved.rank, factor.singular_values())


struct LinearRegression[dtype: DType = DType.float64](Copyable, LinearRegressor, WeightedSupervisedEstimator):
    """Ordinary least squares Linear Regression (sklearn.linear_model.LinearRegression).

    Minimizes ||y - X w - b||^2 (optionally weighted). The dense path is a
    rank-revealing minimum-norm SVD solve with cutoff `tol` (scipy
    `lstsq(cond=tol)`), exposing `rank_` and `singular_`. `positive=True`
    solves each target by Lawson-Hanson NNLS (scipy.optimize.nnls). Sample
    weights are applied by sqrt-weight rescaling of centered data. Float32 and
    Float64 are compile-time choices; vector and matrix targets keep their rank.

    Learned attributes: `coef_` ((features,) or (targets, features)),
    `intercept_` ((targets,), length one for a vector target: a Mojo adaptation
    of sklearn's scalar), `rank_`/`singular_` (None when `positive=True`, where
    sklearn leaves them unset), `n_features_in_`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = LinearRegressionParams

    var fit_intercept: Bool
    var copy_X: Bool
    var tol: Float64
    var n_jobs: Optional[Int]
    var positive: Bool
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var rank_: Optional[Int]
    var singular_: Optional[NDArray[Self.dtype]]
    var n_features_in_: Int
    var _vector_target: Bool
    var _fitted: Bool

    def __init__(
        out self,
        *,
        fit_intercept: Bool = True,
        copy_X: Bool = True,
        tol: Float64 = 1e-6,
        n_jobs: Optional[Int] = None,
        positive: Bool = False,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "LinearRegression supports Float32 and Float64"
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.rank_ = None
        self.singular_ = None
        self.n_features_in_ = 0
        self._vector_target = True
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.fit_intercept, self.copy_X, self.tol, self.n_jobs, self.positive)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.fit_intercept = params.fit_intercept
        self.copy_X = params.copy_X
        self.tol = params.tol
        self.n_jobs = params.n_jobs
        self.positive = params.positive
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        fit_intercept: Optional[Bool] = None,
        copy_X: Optional[Bool] = None,
        tol: Optional[Float64] = None,
        positive: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update only the supplied parameters (use the Params record to reset n_jobs)."""
        if fit_intercept:
            self.fit_intercept = fit_intercept.value()
        if copy_X:
            self.copy_X = copy_X.value()
        if tol:
            self.tol = tol.value()
        if positive:
            self.positive = positive.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(fit_intercept=self.fit_intercept, copy_X=self.copy_X, tol=self.tol, n_jobs=self.n_jobs, positive=self.positive)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _validate_params(self) raises:
        if not isfinite(self.tol) or self.tol < 0:
            raise Error(String("ValueError: LinearRegression: tol must be a finite float >= 0, got ", self.tol))
        if self.n_jobs:
            if self.n_jobs.value() == 0:
                raise Error("ValueError: LinearRegression: n_jobs == 0 has no meaning")

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        self._fit(X, y, Optional(scalar_weights(X, sample_weight)))
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[NDArray[Self.dtype]]) raises:
        self._validate_params()
        var data = preprocess_data(X, y, sample_weight, self.fit_intercept, True, "LinearRegression")
        var n_features = data.X.shape[1]
        var n_outputs = data.y.shape[1]
        var coef: NDArray[Self.dtype]
        var rank = Optional[Int](None)
        var singular = Optional[NDArray[Self.dtype]](None)
        if self.positive:
            coef = zeros[Self.dtype](Shape(n_outputs, n_features))
            var column = zeros[Self.dtype](Shape(data.y.shape[0]))
            for t in range(n_outputs):
                for i in range(data.y.shape[0]):
                    column.unsafe_set(i, data.y.unsafe_get(i * n_outputs + t))
                var solved = nnls(data.X, column)
                if not solved.converged():
                    raise Error(String("RuntimeError: LinearRegression(positive=True): NNLS reached the maximum number of iterations (", solved.iterations, ") for target ", t))
                for j in range(n_features):
                    coef.unsafe_set(t * n_features + j, solved.x.unsafe_get(j))
        else:
            try:
                var solution = _lstsq(data.X, data.y, self.tol)
                rank = solution.rank
                coef = solution.coef.copy()
                singular = solution.singular.copy()
            except error:
                raise Error(String("LinAlgError: LinearRegression least-squares solve failed (", error.kind, " in ", error.operation, ", index ", error.index, ")"))
        var intercept = set_intercept(coef, data.X_offset, data.y_offset, data.X_scale, self.fit_intercept)
        # Commit learned state only after every step has succeeded.
        self.coef_ = public_coef(coef, data.vector_target)
        self.intercept_ = intercept^
        self.rank_ = rank
        self.singular_ = singular^
        self.n_features_in_ = n_features
        self._vector_target = data.vector_target
        self._fitted = True

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "LinearRegression")
        return public_prediction(linear_decision(X, canonical_coef(self.coef_), self.intercept_), self._vector_target)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        """Coefficient of determination R2 of predict(X) against y (uniform average)."""
        return r2_score(y, self.predict(X), sample_weight)
