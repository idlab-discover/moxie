# Adapted from sklearn/linear_model/_coordinate_descent.py, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense elastic-net estimators using the family coordinate engine."""
from numojo import NDArray, Shape, zeros
from std.math import isfinite
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie.linear_model._base import LinearRegressor
from moxie.linear_model._contracts import validate_weights
from moxie.linear_model._linear_base import preprocess_data, set_intercept, canonical_coef, public_coef, linear_decision, public_prediction, check_predict_input, r2_score, check_finite
from moxie.linear_model._cd_fast import coordinate_descent


@fieldwise_init
struct _FitResult[dtype: DType](Movable):
    var coef: NDArray[Self.dtype]
    var intercept: NDArray[Self.dtype]
    var gaps: NDArray[Self.dtype]
    var iterations: List[Int]
    var converged: List[Bool]


def _fit_coordinate[dtype: DType](X: NDArray[dtype], y: NDArray[dtype], sample_weight: Optional[NDArray[dtype]], initial: Optional[NDArray[dtype]], alpha: Float64, l1_ratio: Float64, fit_intercept: Bool, precompute: Bool, max_iter: Int, tol: Float64, positive: Bool, random_state: Optional[Int], selection: String) raises -> _FitResult[dtype]:
    if not isfinite(alpha) or alpha < 0 or not isfinite(l1_ratio) or l1_ratio < 0 or l1_ratio > 1:
        raise Error("ValueError: alpha must be finite and nonnegative; l1_ratio must be in [0, 1]")
    if max_iter < 1 or not isfinite(tol) or tol < 0:
        raise Error("ValueError: max_iter must be positive and tol finite and nonnegative")
    if selection != "cyclic" and selection != "random":
        raise Error("ValueError: selection must be cyclic or random")
    if selection == "random":
        raise Error("NotImplementedError: random selection requires shared our_rand_r")
    if random_state:
        if random_state.value() < 0 or random_state.value() > 4294967295:
            raise Error("ValueError: random_state must be in [0, 2**32 - 1]")
    if precompute:
        raise Error("NotImplementedError: Gram coordinate descent is not implemented")
    var normalized = Optional[NDArray[dtype]](None)
    if sample_weight:
        if X.ndim != 2:
            raise Error("ValueError: X must be a matrix")
        ref sw = sample_weight.value()
        validate_weights(sw, X.shape[0])
        var total = Float64(0)
        for i in range(X.shape[0]):
            total += Float64(sw.unsafe_get(i * Int(sw.strides[0])))
        var weights = zeros[dtype](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            weights.unsafe_set(i, Scalar[dtype](Float64(sw.unsafe_get(i * Int(sw.strides[0]))) * Float64(X.shape[0]) / total))
        normalized = weights^
    var data = preprocess_data(X, y, normalized, fit_intercept)
    var n = data.X.shape[0]
    var p = data.X.shape[1]
    var k = data.y.shape[1]
    var coef = zeros[dtype](Shape(k, p))
    if initial:
        coef = canonical_coef(initial.value())
        if coef.shape[0] != k or coef.shape[1] != p:
            raise Error("ValueError: warm_start coefficients have incompatible target/feature shape")
    var gaps = zeros[dtype](Shape(k))
    var iterations = List[Int]()
    var converged = List[Bool]()
    for t in range(k):
        var target = zeros[dtype](Shape(n))
        var start = zeros[dtype](Shape(p))
        for i in range(n):
            target.unsafe_set(i, data.y.unsafe_get(i * k + t))
        for j in range(p):
            start.unsafe_set(j, coef.unsafe_get(t * p + j))
        var solved = coordinate_descent(data.X, target, start, alpha * l1_ratio * Float64(n), alpha * (1 - l1_ratio) * Float64(n), max_iter, tol, positive)
        for j in range(p):
            coef.unsafe_set(t * p + j, solved.coef.unsafe_get(j))
        gaps.unsafe_set(t, Scalar[dtype](solved.dual_gap / Float64(n)))
        iterations.append(solved.iterations)
        converged.append(solved.converged)
    var intercept = set_intercept(coef, data.X_offset, data.y_offset, data.X_scale, fit_intercept)
    check_finite(intercept, "intercept", "ElasticNet")
    return _FitResult[dtype](public_coef(coef, k == 1), intercept^, gaps^, iterations^, converged^)

@fieldwise_init
struct ElasticNetParams(Copyable, Movable):
    var alpha: Float64
    var l1_ratio: Float64
    var fit_intercept: Bool
    var precompute: Bool
    var max_iter: Int
    var copy_X: Bool
    var tol: Float64
    var warm_start: Bool
    var positive: Bool
    var random_state: Optional[Int]
    var selection: String


struct ElasticNet[dtype: DType = DType.float64](Copyable, LinearRegressor, WeightedSupervisedEstimator):
    """ElasticNet with cyclic dense fitting; see docs/api.md for typed diagnostics."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = ElasticNetParams
    var alpha: Float64
    var l1_ratio: Float64
    var fit_intercept: Bool
    var precompute: Bool
    var max_iter: Int
    var copy_X: Bool
    var tol: Float64
    var warm_start: Bool
    var positive: Bool
    var random_state: Optional[Int]
    var selection: String
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var dual_gap_: NDArray[Self.dtype]
    var n_iter_: List[Int]
    var converged_: List[Bool]
    var n_features_in_: Int
    var _fitted: Bool

    def __init__(out self, alpha: Float64 = 1.0, *,
        l1_ratio: Float64 = 0.5,
        fit_intercept: Bool = True,
        precompute: Bool = False,
        max_iter: Int = 1000,
        copy_X: Bool = True,
        tol: Float64 = 1e-4,
        warm_start: Bool = False,
        positive: Bool = False,
        random_state: Optional[Int] = None,
        var selection: String = "cyclic",
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 required"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection^
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.dual_gap_ = zeros[Self.dtype](Shape(0))
        self.n_iter_ = List[Int]()
        self.converged_ = List[Bool]()
        self.n_features_in_ = 0
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alpha, self.l1_ratio, self.fit_intercept, self.precompute, self.max_iter, self.copy_X, self.tol, self.warm_start, self.positive, self.random_state, self.selection.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alpha = params.alpha
        self.l1_ratio = params.l1_ratio
        self.fit_intercept = params.fit_intercept
        self.precompute = params.precompute
        self.max_iter = params.max_iter
        self.copy_X = params.copy_X
        self.tol = params.tol
        self.warm_start = params.warm_start
        self.positive = params.positive
        self.random_state = params.random_state
        self.selection = params.selection.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, alpha: Optional[Float64] = None, tol: Optional[Float64] = None, max_iter: Optional[Int] = None, warm_start: Optional[Bool] = None) raises -> ref[self] Self:
        if alpha:
            self.alpha = alpha.value()
        if tol:
            self.tol = tol.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if warm_start:
            self.warm_start = warm_start.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, precompute=self.precompute, max_iter=self.max_iter, copy_X=self.copy_X, tol=self.tol, warm_start=self.warm_start, positive=self.positive, random_state=self.random_state, selection=self.selection.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Float64) raises -> ref[self] Self:
        # The reference explicitly ignores scalar weights (normalization cancels).
        self._fit(X, y, None)
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        var initial = Optional[NDArray[Self.dtype]](None)
        if self.warm_start and self._fitted:
            initial = self.coef_.copy()
        var solved = _fit_coordinate(X, y, sample_weight, initial, self.alpha, self.l1_ratio, self.fit_intercept, self.precompute, self.max_iter, self.tol, self.positive, self.random_state, self.selection)
        self.coef_ = solved.coef.copy()
        self.intercept_ = solved.intercept.copy()
        self.dual_gap_ = solved.gaps.copy()
        self.n_iter_ = solved.iterations.copy()
        self.converged_ = solved.converged.copy()
        self.n_features_in_ = X.shape[1]
        self._fitted = True

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "ElasticNet")
        return public_prediction(linear_decision(X, canonical_coef(self.coef_), self.intercept_), self.coef_.ndim == 1)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return r2_score(y, self.predict(X), sample_weight)


@fieldwise_init
struct LassoParams(Copyable, Movable):
    var alpha: Float64
    var fit_intercept: Bool
    var precompute: Bool
    var max_iter: Int
    var copy_X: Bool
    var tol: Float64
    var warm_start: Bool
    var positive: Bool
    var random_state: Optional[Int]
    var selection: String


struct Lasso[dtype: DType = DType.float64](Copyable, LinearRegressor, WeightedSupervisedEstimator):
    """Lasso with cyclic dense fitting; see docs/api.md for typed diagnostics."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = LassoParams
    var alpha: Float64
    var fit_intercept: Bool
    var precompute: Bool
    var max_iter: Int
    var copy_X: Bool
    var tol: Float64
    var warm_start: Bool
    var positive: Bool
    var random_state: Optional[Int]
    var selection: String
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var dual_gap_: NDArray[Self.dtype]
    var n_iter_: List[Int]
    var converged_: List[Bool]
    var n_features_in_: Int
    var _fitted: Bool

    def __init__(out self, alpha: Float64 = 1.0, *,
        fit_intercept: Bool = True,
        precompute: Bool = False,
        max_iter: Int = 1000,
        copy_X: Bool = True,
        tol: Float64 = 1e-4,
        warm_start: Bool = False,
        positive: Bool = False,
        random_state: Optional[Int] = None,
        var selection: String = "cyclic",
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 required"
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection^
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.dual_gap_ = zeros[Self.dtype](Shape(0))
        self.n_iter_ = List[Int]()
        self.converged_ = List[Bool]()
        self.n_features_in_ = 0
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alpha, self.fit_intercept, self.precompute, self.max_iter, self.copy_X, self.tol, self.warm_start, self.positive, self.random_state, self.selection.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alpha = params.alpha
        self.fit_intercept = params.fit_intercept
        self.precompute = params.precompute
        self.max_iter = params.max_iter
        self.copy_X = params.copy_X
        self.tol = params.tol
        self.warm_start = params.warm_start
        self.positive = params.positive
        self.random_state = params.random_state
        self.selection = params.selection.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, alpha: Optional[Float64] = None, tol: Optional[Float64] = None, max_iter: Optional[Int] = None, warm_start: Optional[Bool] = None) raises -> ref[self] Self:
        if alpha:
            self.alpha = alpha.value()
        if tol:
            self.tol = tol.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if warm_start:
            self.warm_start = warm_start.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(alpha=self.alpha, fit_intercept=self.fit_intercept, precompute=self.precompute, max_iter=self.max_iter, copy_X=self.copy_X, tol=self.tol, warm_start=self.warm_start, positive=self.positive, random_state=self.random_state, selection=self.selection.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Float64) raises -> ref[self] Self:
        # The reference explicitly ignores scalar weights (normalization cancels).
        self._fit(X, y, None)
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        var initial = Optional[NDArray[Self.dtype]](None)
        if self.warm_start and self._fitted:
            initial = self.coef_.copy()
        var solved = _fit_coordinate(X, y, sample_weight, initial, self.alpha, 1.0, self.fit_intercept, self.precompute, self.max_iter, self.tol, self.positive, self.random_state, self.selection)
        self.coef_ = solved.coef.copy()
        self.intercept_ = solved.intercept.copy()
        self.dual_gap_ = solved.gaps.copy()
        self.n_iter_ = solved.iterations.copy()
        self.converged_ = solved.converged.copy()
        self.n_features_in_ = X.shape[1]
        self._fitted = True

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "Lasso")
        return public_prediction(linear_decision(X, canonical_coef(self.coef_), self.intercept_), self.coef_.ndim == 1)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return r2_score(y, self.predict(X), sample_weight)
