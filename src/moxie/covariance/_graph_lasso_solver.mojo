# Adapted from scikit-learn sklearn/covariance/_graph_lasso.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Sparse inverse covariance by the graphical lasso (Friedman, Hastie, Tibshirani 2008).

`GraphicalLasso` and `graphical_lasso` with coordinate-descent and LARS
solvers, sklearn's warm start (0.95-shrunk covariance), dual gap and objective.
Work is Float64 (sklearn's emp_cov is Float64 from np.cov); results keep the
estimator dtype. LARS uses the private least-angle Gram path.
"""
import numojo as nm
from moxie.covariance._param_update import NullableUpdate
from numojo import NDArray, Shape
from std.math import abs, isfinite
from std.sys import stderr
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._empirical_covariance import (
    CovarianceModel, column_means, empirical_covariance_float64, fast_logdet, log_likelihood, pinvh, to_array,
    to_float64, validate_samples,
)
from moxie.covariance._enet_gram import enet_coordinate_descent_gram
from moxie.covariance._lars_gram import lars_gram
from mojo_linalg import cholesky, symmetric_indefinite

comptime _EPS64: Float64 = 2.220446049250313e-16
comptime _LOG_TWO_PI: Float64 = 1.8378770664093453


def _off_diagonal_l1(precision: NDArray[DType.float64]) raises -> Float64:
    var p = precision.shape[0]
    var total: Float64 = 0
    for a in range(p):
        for b in range(p):
            if a != b:
                total += abs(precision.unsafe_get(a * p + b))
    return total


def _objective(mle: NDArray[DType.float64], precision: NDArray[DType.float64], alpha: Float64) raises -> Float64:
    """-2 log_likelihood + p log(2 pi) + alpha * off-diagonal |precision|_1."""
    var p = precision.shape[0]
    var cost = -2.0 * log_likelihood(mle, precision) + Float64(p) * _LOG_TWO_PI
    return cost + alpha * _off_diagonal_l1(precision)


def _dual_gap(emp_cov: NDArray[DType.float64], precision: NDArray[DType.float64], alpha: Float64) raises -> Float64:
    var p = precision.shape[0]
    var gap: Float64 = 0
    for k in range(p * p):
        gap += emp_cov.unsafe_get(k) * precision.unsafe_get(k)
    gap -= Float64(p)
    return gap + alpha * _off_diagonal_l1(precision)


def alpha_max(emp_cov: NDArray[DType.float64]) raises -> Float64:
    """Largest absolute off-diagonal entry: the smallest alpha giving a diagonal precision."""
    return _off_diagonal_l1_max(emp_cov)


def _off_diagonal_l1_max(a: NDArray[DType.float64]) raises -> Float64:
    var p = a.shape[0]
    var result: Float64 = 0
    for i in range(p):
        for j in range(p):
            if i != j:
                result = max(result, abs(a.unsafe_get(i * p + j)))
    return result


@fieldwise_init
struct GraphicalLassoResult(Copyable, Movable):
    """covariance (p, p), precision (p, p), costs [(objective, dual gap)], n_iter (Float64 arrays)."""

    var covariance: NDArray[DType.float64]
    var precision: NDArray[DType.float64]
    var costs: List[Tuple[Float64, Float64]]
    var n_iter: Int


def _inverse(a: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """Invert symmetric covariance; try SPD factorization, then indefinite LDLᵀ.

    SciPy's structure-detecting inv uses Cholesky for positive-definite inputs.
    Preserve indefinite inversion and exact singular failures without jitter.
    """
    var p = a.shape[0]
    var identity = nm.zeros[DType.float64](Shape(p, p))
    for k in range(p):
        identity.unsafe_set(k * p + k, 1.0)
    try:
        return cholesky(a, lower=False).solve(identity)
    except e:
        if e.kind != "not_positive_definite":
            raise Error(String("LinAlgError: covariance inverse failed (", e.kind, ")"))
    try:
        return symmetric_indefinite(a).solve(identity)
    except e:
        raise Error(String("LinAlgError: singular matrix (", e.kind, ")"))


def graphical_lasso_solve(emp_cov: NDArray[DType.float64], alpha: Float64, mode: String = "cd", tol: Float64 = 1e-4,
                          enet_tol: Float64 = 1e-4, max_iter: Int = 100, eps: Float64 = _EPS64,
                          cov_init: Optional[NDArray[DType.float64]] = None, *, verbose: Bool = False,
                          emit_warnings: Bool = True) raises -> GraphicalLassoResult:
    """sklearn `_graphical_lasso` on a contiguous Float64 empirical covariance."""
    var p = emp_cov.shape[1]
    if alpha == 0:
        var precision = _inverse(emp_cov)
        var cost = -2.0 * log_likelihood(emp_cov, precision) + Float64(p) * _LOG_TWO_PI
        var d_gap: Float64 = -Float64(p)
        for k in range(p * p):
            d_gap += emp_cov.unsafe_get(k) * precision.unsafe_get(k)
        var costs = List[Tuple[Float64, Float64]]()
        costs.append((cost, d_gap))
        return GraphicalLassoResult(emp_cov.copy(), precision^, costs^, 0)
    var covariance = cov_init.value().copy() if cov_init else emp_cov.copy()
    for a in range(p):
        for b in range(p):
            covariance.unsafe_set(a * p + b, covariance.unsafe_get(a * p + b) * 0.95 if a != b else emp_cov.unsafe_get(a * p + a))
    var precision = pinvh(covariance)
    var m = p - 1
    var sub = nm.zeros[DType.float64](Shape(m, m))
    var costs = List[Tuple[Float64, Float64]]()
    var d_gap = Float64.MAX * 2.0
    var iterations = 0
    var others = List[Int]()
    for i in range(max_iter):
        iterations = i
        for idx in range(p):
            others = List[Int]()
            for k in range(p):
                if k != idx:
                    others.append(k)
            # sub_covariance = covariance_ without row/column idx (sklearn updates it incrementally).
            for r in range(m):
                for c in range(m):
                    sub.unsafe_set(r * m + c, covariance.unsafe_get(others[r] * p + others[c]))
            var row = nm.zeros[DType.float64](Shape(m))
            var coefs = nm.zeros[DType.float64](Shape(m))
            var pivot = precision.unsafe_get(idx * p + idx) + 1000.0 * eps
            for r in range(m):
                row.unsafe_set(r, emp_cov.unsafe_get(idx * p + others[r]))
                coefs.unsafe_set(r, -(precision.unsafe_get(others[r] * p + idx) / pivot))
            var w: NDArray[DType.float64]
            if mode == "cd":
                var solved = enet_coordinate_descent_gram(coefs^, alpha, 0.0, sub, row, row, max(20, max_iter), enet_tol, emit_warnings=emit_warnings)
                w = solved.w.copy()
            else:
                w = lars_gram(sub, row, alpha, eps)
            var dot: Float64 = 0
            for r in range(m):
                dot += covariance.unsafe_get(others[r] * p + idx) * w.unsafe_get(r)
            var diagonal = 1.0 / (covariance.unsafe_get(idx * p + idx) - dot)
            precision.unsafe_set(idx * p + idx, diagonal)
            for r in range(m):
                precision.unsafe_set(others[r] * p + idx, -diagonal * w.unsafe_get(r))
                precision.unsafe_set(idx * p + others[r], -diagonal * w.unsafe_get(r))
            for r in range(m):
                var value: Float64 = 0
                for c in range(m):
                    value += sub.unsafe_get(r * m + c) * w.unsafe_get(c)
                covariance.unsafe_set(idx * p + others[r], value)
                covariance.unsafe_set(others[r] * p + idx, value)
        var total: Float64 = 0
        for k in range(p * p):
            total += precision.unsafe_get(k)
        if not isfinite(total):
            raise Error("FloatingPointError: The system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver")
        d_gap = _dual_gap(emp_cov, precision, alpha)
        var cost = _objective(emp_cov, precision, alpha)
        if verbose:
            print("[graphical_lasso] Iteration", i, "cost", cost, "dual gap", d_gap)
        costs.append((cost, d_gap))
        if abs(d_gap) < tol:
            break
        if not isfinite(cost) and i > 0:
            raise Error("FloatingPointError: Non SPD result: the system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver")
    if emit_warnings and not (abs(d_gap) < tol):
        print("ConvergenceWarning: graphical_lasso: did not converge after", max_iter,
              "iteration: dual gap:", d_gap, file=stderr)
    return GraphicalLassoResult(covariance^, precision^, costs^, iterations + 1)


def _check_parameters(alpha: Float64, mode: String, tol: Float64, enet_tol: Float64, max_iter: Int, eps: Float64, owner: String) raises:
    if not (alpha >= 0):
        raise Error(String("InvalidParameterError: The 'alpha' parameter of ", owner, " must be a float in the range [0, inf). Got ", alpha, " instead."))
    if mode != "cd" and mode != "lars":
        raise Error(String("InvalidParameterError: The 'mode' parameter of ", owner, " must be a str among {'cd', 'lars'}. Got '", mode, "' instead."))
    if not (tol > 0):
        raise Error(String("InvalidParameterError: The 'tol' parameter of ", owner, " must be a float in the range (0, inf). Got ", tol, " instead."))
    if not (enet_tol > 0):
        raise Error(String("InvalidParameterError: The 'enet_tol' parameter of ", owner, " must be a float in the range (0, inf). Got ", enet_tol, " instead."))
    if max_iter < 0:
        raise Error(String("InvalidParameterError: The 'max_iter' parameter of ", owner, " must be an int in the range [0, inf). Got ", max_iter, " instead."))
    if not (eps >= 0):
        raise Error(String("InvalidParameterError: The 'eps' parameter of ", owner, " must be a float in the range [0, inf]. Got ", eps, " instead."))


@fieldwise_init
struct GraphicalLassoParams(Copyable, Movable, Writable):
    var alpha: Float64
    var mode: String
    var covariance: Optional[String]
    var tol: Float64
    var enet_tol: Float64
    var max_iter: Int
    var verbose: Bool
    var eps: Float64
    var assume_centered: Bool


struct GraphicalLasso[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Sparse inverse covariance estimation with an l1 penalty (sklearn `GraphicalLasso`).

    Defaults: alpha=0.01, mode='cd', covariance=None ('precomputed' treats X as
    the empirical covariance), tol=1e-4, enet_tol=1e-4, max_iter=100,
    verbose=False (no output), eps=float64 eps, assume_centered=False.
    Learned: `location_()`, `covariance_()`, `precision_()`, `costs_()`, `n_iter_()`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = GraphicalLassoParams
    comptime NAME = "GraphicalLasso"

    var alpha: Float64
    var mode: String
    var covariance: Optional[String]
    var tol: Float64
    var enet_tol: Float64
    var max_iter: Int
    var verbose: Bool
    var eps: Float64
    var assume_centered: Bool
    var _model: CovarianceModel[Self.dtype]
    var _costs: List[Tuple[Float64, Float64]]
    var _n_iter: Int

    def __init__(out self, alpha: Float64 = 0.01, *, var mode: String = "cd", var covariance: Optional[String] = None,
                 tol: Float64 = 1e-4, enet_tol: Float64 = 1e-4, max_iter: Int = 100, verbose: Bool = False,
                 eps: Float64 = _EPS64, assume_centered: Bool = False):
        self.alpha = alpha
        self.mode = mode^
        self.covariance = covariance^
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self._model = CovarianceModel[Self.dtype]()
        self._costs = List[Tuple[Float64, Float64]]()
        self._n_iter = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alpha, self.mode.copy(), self.covariance.copy(), self.tol, self.enet_tol, self.max_iter,
                           self.verbose, self.eps, self.assume_centered)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alpha = params.alpha
        self.mode = params.mode.copy()
        self.covariance = params.covariance.copy()
        self.tol = params.tol
        self.enet_tol = params.enet_tol
        self.max_iter = params.max_iter
        self.verbose = params.verbose
        self.eps = params.eps
        self.assume_centered = params.assume_centered
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        alpha: Optional[Float64] = None,
        mode: Optional[String] = None,
        covariance: NullableUpdate[String] = NullableUpdate[String](),
        tol: Optional[Float64] = None,
        enet_tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None,
        verbose: Optional[Bool] = None,
        eps: Optional[Float64] = None,
        assume_centered: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update supplied constructor parameters without changing fitted state."""
        if alpha:
            self.alpha = alpha.value().copy()
        if mode:
            self.mode = mode.value().copy()
        if covariance.supplied:
            self.covariance = covariance.value.copy()
        if tol:
            self.tol = tol.value().copy()
        if enet_tol:
            self.enet_tol = enet_tol.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if eps:
            self.eps = eps.value().copy()
        if assume_centered:
            self.assume_centered = assume_centered.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.alpha, mode=self.mode.copy(), covariance=self.covariance.copy(), tol=self.tol, enet_tol=self.enet_tol,
                    max_iter=self.max_iter, verbose=self.verbose, eps=self.eps, assume_centered=self.assume_centered)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _check_parameters(self.alpha, self.mode, self.tol, self.enet_tol, self.max_iter, self.eps, Self.NAME)
        if self.covariance and self.covariance.value() != "precomputed":
            raise Error(String("InvalidParameterError: The 'covariance' parameter of GraphicalLasso must be a str among {'precomputed'} or None. Got '",
                               self.covariance.value(), "' instead."))
        validate_samples(X, Self.NAME + ".fit", min_samples=2, min_features=2)
        var p = X.shape[1]
        var emp_cov: NDArray[DType.float64]
        var location: NDArray[Self.dtype]
        if self.covariance:
            emp_cov = to_float64(X)
            location = nm.zeros[Self.dtype](Shape(p))
        else:
            emp_cov = empirical_covariance_float64(X, self.assume_centered)
            if self.assume_centered:
                location = nm.zeros[Self.dtype](Shape(p))
            else:
                location = to_array[Self.dtype](column_means(X))
        self._model.update_location(location)
        var result = graphical_lasso_solve(emp_cov, self.alpha, self.mode, self.tol, self.enet_tol, self.max_iter, self.eps, verbose=self.verbose)
        var candidate = CovarianceModel[Self.dtype]()
        candidate.install_estimate_float64(location, result.covariance, result.precision)
        self._model = candidate^
        self._costs = result.costs.copy()
        self._n_iter = result.n_iter
        return self

    def costs_(self) raises -> List[Tuple[Float64, Float64]]:
        self._model.check_fitted(Self.NAME)
        return self._costs.copy()

    def n_iter_(self) raises -> Int:
        self._model.check_fitted(Self.NAME)
        return self._n_iter

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


def graphical_lasso[dtype: DType](emp_cov: NDArray[dtype], alpha: Float64, *, mode: String = "cd", tol: Float64 = 1e-4,
                                  enet_tol: Float64 = 1e-4, max_iter: Int = 100, verbose: Bool = False,
                                  eps: Float64 = _EPS64) raises -> GraphicalLassoResult:
    """sklearn `graphical_lasso`: covariance, precision, costs and n_iter in one record.

    The return_costs/return_n_iter switches of sklearn become always-present
    fields of the result (Float64 arrays).
    """
    var model = GraphicalLasso[dtype](alpha, mode=mode, covariance=String("precomputed"), tol=tol, enet_tol=enet_tol,
                                      max_iter=max_iter, verbose=verbose, eps=eps, assume_centered=True)
    model.fit(emp_cov)
    return GraphicalLassoResult(to_float64(model.covariance_()), to_float64(model.get_precision()), model.costs_(), model.n_iter_())
