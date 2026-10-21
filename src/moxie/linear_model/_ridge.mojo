# Adapted from scikit-learn sklearn/linear_model/_ridge.py; authors:
# scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Dense Ridge, RidgeClassifier and ridge_regression with sklearn's solver dispatch.

Solvers: 'cholesky' (primal normal equations, or the kernel/dual form when
n_features > n_samples, with sklearn's SVD/lstsq fallbacks), 'svd', 'lsqr'
(damped LSQR), 'sparse_cg' (CG on the normal or dual operator) and 'lbfgs'
(L-BFGS-B with nonnegativity bounds for positive=True). 'sag'/'saga' need the
unported SAG engine and raise NotImplementedError.
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite, sqrt, inf
from std.utils import Variant
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie.linear_model._base import LinearRegressor, LinearClassifier
from moxie.linear_model._linear_base import (
    preprocess_data,
    scalar_weights,
    set_intercept,
    linear_decision,
    public_coef,
    canonical_coef,
    public_prediction,
    check_predict_input,
    check_finite,
    compact_matrix,
    r2_score,
    gram_matrix,
    outer_gram,
    transpose_product,
)
from moxie.linear_model._ridge_labels import (
    ClassWeight,
    LabelEncoding,
    fit_labels,
    binarize,
    class_sample_weight,
    accuracy,
)
from moxie._linear_linalg import (
    cholesky,
    svd,
    least_squares,
    NumericalError,
    LinearOperator,
    DenseOperator,
    conjugate_gradient,
    lsqr,
)
from mojo_optimize import LBFGSB, LBFGSBConfig, Objective


comptime ScalarOrVector = Variant[Float64, List[Float64]]


@fieldwise_init
struct RidgeSolution[dtype: DType](Movable):
    """Canonical (targets, features) coefficients, per-target iterations, solver used."""

    var coef: NDArray[Self.dtype]
    var n_iter: Optional[List[Int]]
    var solver: String


def alpha_values(alpha: ScalarOrVector) -> List[Float64]:
    if alpha.isa[Float64]():
        return [alpha[Float64]]
    return alpha[List[Float64]].copy()


def validate_alpha(alpha: ScalarOrVector, owner: String) raises:
    var values = alpha_values(alpha)
    if len(values) == 0:
        raise Error(String("ValueError: ", owner, ": alpha must not be empty"))
    for value in values:
        if not isfinite(value) or value < 0:
            raise Error(String("ValueError: ", owner, ": alpha must be a finite float >= 0, got ", value))


def expand_alphas(alpha: ScalarOrVector, n_targets: Int) raises -> List[Float64]:
    """sklearn: one penalty or one per target; a single value is broadcast."""
    var values = alpha_values(alpha)
    if len(values) != 1 and len(values) != n_targets:
        raise Error(String("ValueError: Number of targets and number of penalties do not correspond: ", len(values), " != ", n_targets))
    if len(values) == 1 and n_targets > 1:
        var result = List[Float64]()
        for _ in range(n_targets):
            result.append(values[0])
        return result^
    return values^


def resolve_solver(solver: String, positive: Bool, return_intercept: Bool) -> String:
    """sklearn resolve_solver_for_numpy for dense input."""
    if solver != "auto":
        return solver
    if positive:
        return "lbfgs"
    if return_intercept:
        return "sag"
    return "cholesky"


def _check_solver(solver: String, owner: String) raises:
    var solvers: List[String] = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
    for name in solvers:
        if name == solver:
            return
    raise Error(String("ValueError: ", owner, ": solver must be one of auto, svd, cholesky, lsqr, sparse_cg, sag, saga, lbfgs; got '", solver, "'"))


def _column[dtype: DType](Y: NDArray[dtype], t: Int) raises -> NDArray[DType.float64]:
    var n = Y.shape[0]
    var k = Y.shape[1]
    var result = zeros[DType.float64](Shape(n))
    for i in range(n):
        result.unsafe_set(i, Float64(Y.unsafe_get(i * k + t)))
    return result^


def _all_equal(values: List[Float64]) -> Bool:
    for v in values:
        if v != values[0]:
            return False
    return True


def _solve_cholesky[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64]) raises NumericalError -> NDArray[dtype]:
    """w = inv(X^T X + alpha I) X^T y (sklearn _solve_cholesky)."""
    var p = X.shape[1]
    var k = Y.shape[1]
    var A = gram_matrix(X)
    var XY = transpose_product(X, Y)
    var coef = zeros[dtype](Shape(k, p))
    if _all_equal(alphas):
        for j in range(p):
            A.unsafe_set(j * p + j, A.unsafe_get(j * p + j) + Scalar[dtype](alphas[0]))
        var solved = cholesky(A).solve(XY)
        for j in range(p):
            for t in range(k):
                coef.unsafe_set(t * p + j, solved.unsafe_get(j * k + t))
        return coef^
    for t in range(k):
        var At = A.copy()
        for j in range(p):
            At.unsafe_set(j * p + j, At.unsafe_get(j * p + j) + Scalar[dtype](alphas[t]))
        var rhs = zeros[dtype](Shape(p))
        for j in range(p):
            rhs.unsafe_set(j, XY.unsafe_get(j * k + t))
        var solved = cholesky(At).solve(rhs)
        for j in range(p):
            coef.unsafe_set(t * p + j, solved.unsafe_get(j))
    return coef^


def _solve_cholesky_kernel[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64]) raises NumericalError -> NDArray[dtype]:
    """Dual form for n_features > n_samples: coef = X^T inv(X X^T + alpha I) y.

    One shared alpha: a failed Cholesky falls back to lstsq(K + alpha I, y) with
    cutoff eps (sklearn warns); per-target alphas propagate the failure so the
    caller switches to 'svd', as sklearn does.
    """
    var n = X.shape[0]
    var p = X.shape[1]
    var k = Y.shape[1]
    var K = outer_gram(X)
    var dual = zeros[dtype](Shape(n, k))
    if _all_equal(alphas):
        for i in range(n):
            K.unsafe_set(i * n + i, K.unsafe_get(i * n + i) + Scalar[dtype](alphas[0]))
        try:
            dual = cholesky(K).solve(Y)
        except error:
            var eps = 1.1920928955078125e-07 if dtype == DType.float32 else 2.220446049250313e-16
            dual = least_squares(K, Y, rtol=eps, atol=0.0).solution.copy()
    else:
        for t in range(k):
            var Kt = K.copy()
            for i in range(n):
                Kt.unsafe_set(i * n + i, Kt.unsafe_get(i * n + i) + Scalar[dtype](alphas[t]))
            var rhs = zeros[dtype](Shape(n))
            for i in range(n):
                rhs.unsafe_set(i, Y.unsafe_get(i * k + t))
            var solved = cholesky(Kt).solve(rhs)
            for i in range(n):
                dual.unsafe_set(i * k + t, solved.unsafe_get(i))
    var W = transpose_product(X, dual)  # (p, k)
    var coef = zeros[dtype](Shape(k, p))
    for j in range(p):
        for t in range(k):
            coef.unsafe_set(t * p + j, W.unsafe_get(j * k + t))
    return coef^


def _solve_svd[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64]) raises NumericalError -> NDArray[dtype]:
    """sklearn _solve_svd: d = s / (s^2 + alpha) on s > 1e-15; coef = V d U^T y."""
    var p = X.shape[1]
    var n = X.shape[0]
    var k = Y.shape[1]
    var factor = svd(X)
    var U = factor.u()
    var s = factor.singular_values()
    var Vt = factor.vt()
    var m = s.shape[0]
    var coef = zeros[dtype](Shape(k, p))
    for t in range(k):
        for j in range(m):
            var sj = Float64(s.unsafe_get(j))
            if not (sj > 1e-15):
                continue
            var uty = Float64(0)
            for i in range(n):
                uty += Float64(U.unsafe_get(i * m + j)) * Float64(Y.unsafe_get(i * k + t))
            var scale = sj / (sj * sj + alphas[t]) * uty
            for l in range(p):
                coef.unsafe_set(t * p + l, coef.unsafe_get(t * p + l) + Scalar[dtype](scale * Float64(Vt.unsafe_get(j * p + l))))
    return coef^


struct _NormalOperator(LinearOperator):
    """sklearn _solve_sparse_cg operators: X^T X + alpha I or (dual) X X^T + alpha I."""

    var _x: NDArray[DType.float64]
    var _n: Int
    var _p: Int
    var _alpha: Float64
    var _dual: Bool
    var _work: NDArray[DType.float64]

    def __init__[dtype: DType](out self, X: NDArray[dtype], alpha: Float64, dual: Bool) raises:
        self._n = X.shape[0]
        self._p = X.shape[1]
        self._x = zeros[DType.float64](Shape(self._n * self._p))
        for i in range(self._n * self._p):
            self._x.unsafe_set(i, Float64(X.unsafe_get(i)))
        self._alpha = alpha
        self._dual = dual
        self._work = zeros[DType.float64](Shape(self._p if dual else self._n))

    def rows(self) -> Int:
        return self._n if self._dual else self._p

    def columns(self) -> Int:
        return self.rows()

    def _times(self, v: NDArray[DType.float64], mut out: NDArray[DType.float64]):
        """out = X v (v: p, out: n)."""
        for i in range(self._n):
            var total = Float64(0)
            for j in range(self._p):
                total += self._x.unsafe_get(i * self._p + j) * v.unsafe_get(j)
            out.unsafe_set(i, total)

    def _transposed(self, v: NDArray[DType.float64], mut out: NDArray[DType.float64]):
        """out = X^T v (v: n, out: p)."""
        for j in range(self._p):
            out.unsafe_set(j, 0)
        for i in range(self._n):
            var vi = v.unsafe_get(i)
            for j in range(self._p):
                out.unsafe_set(j, out.unsafe_get(j) + self._x.unsafe_get(i * self._p + j) * vi)

    def matvec(self, x: NDArray[DType.float64], mut y: NDArray[DType.float64]) raises:
        var work = self._work.copy()
        if self._dual:
            self._transposed(x, work)
            self._times(work, y)
        else:
            self._times(x, work)
            self._transposed(work, y)
        for i in range(self.rows()):
            y.unsafe_set(i, y.unsafe_get(i) + self._alpha * x.unsafe_get(i))

    def rmatvec(self, x: NDArray[DType.float64], mut y: NDArray[DType.float64]) raises:
        self.matvec(x, y)  # symmetric


def _solve_sparse_cg[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64], max_iter: Optional[Int], tol: Float64) raises -> NDArray[dtype]:
    var n = X.shape[0]
    var p = X.shape[1]
    var k = Y.shape[1]
    var coef = zeros[dtype](Shape(k, p))
    var dual = p > n
    for t in range(k):
        var y = _column(Y, t)
        var C = _NormalOperator(X, alphas[t], dual)
        if dual:
            # sklearn does not pass max_iter in the kernel branch.
            var result = conjugate_gradient(C, y, rtol=tol)
            for j in range(p):
                var total = Float64(0)
                for i in range(n):
                    total += Float64(X.unsafe_get(i * p + j)) * result.solution.unsafe_get(i)
                coef.unsafe_set(t * p + j, Scalar[dtype](total))
        else:
            var rhs = zeros[DType.float64](Shape(p))
            for i in range(n):
                var yi = y.unsafe_get(i)
                for j in range(p):
                    rhs.unsafe_set(j, rhs.unsafe_get(j) + Float64(X.unsafe_get(i * p + j)) * yi)
            var result = conjugate_gradient(C, rhs, rtol=tol, max_iterations=max_iter)
            for j in range(p):
                coef.unsafe_set(t * p + j, Scalar[dtype](result.solution.unsafe_get(j)))
    return coef^


def _solve_lsqr[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64], max_iter: Optional[Int], tol: Float64, mut n_iter: List[Int]) raises -> NDArray[dtype]:
    var p = X.shape[1]
    var k = Y.shape[1]
    var coef = zeros[dtype](Shape(k, p))
    var op: DenseOperator
    try:
        op = DenseOperator(X)
    except error:
        raise Error(String("LinAlgError: ridge lsqr operator: ", error.kind))
    for t in range(k):
        var result = lsqr(op, _column(Y, t), damp=sqrt(alphas[t]), atol=tol, btol=tol, iteration_limit=max_iter)
        n_iter.append(result.iterations)
        for j in range(p):
            coef.unsafe_set(t * p + j, Scalar[dtype](result.solution.unsafe_get(j)))
    return coef^


struct _RidgeObjective(Objective):
    """0.5 ||X w - y||^2 + 0.5 alpha ||w||^2 and its gradient (sklearn _solve_lbfgs)."""

    comptime dtype = DType.float64
    var _x: NDArray[DType.float64]
    var _y: NDArray[DType.float64]
    var _n: Int
    var _p: Int
    var _alpha: Float64

    def __init__[source: DType](out self, X: NDArray[source], var y: NDArray[DType.float64], alpha: Float64) raises:
        self._n = X.shape[0]
        self._p = X.shape[1]
        self._x = zeros[DType.float64](Shape(self._n * self._p))
        for i in range(self._n * self._p):
            self._x.unsafe_set(i, Float64(X.unsafe_get(i)))
        self._y = y^
        self._alpha = alpha

    def evaluate(mut self, w: NDArray[DType.float64], mut gradient: NDArray[DType.float64]) raises -> Float64:
        var f = Float64(0)
        for j in range(self._p):
            gradient.unsafe_set(j, self._alpha * w.unsafe_get(j))
        for i in range(self._n):
            var r = -self._y.unsafe_get(i)
            for j in range(self._p):
                r += self._x.unsafe_get(i * self._p + j) * w.unsafe_get(j)
            f += 0.5 * r * r
            for j in range(self._p):
                gradient.unsafe_set(j, gradient.unsafe_get(j) + self._x.unsafe_get(i * self._p + j) * r)
        var ww = Float64(0)
        for j in range(self._p):
            ww += w.unsafe_get(j) * w.unsafe_get(j)
        return f + 0.5 * self._alpha * ww


def _solve_lbfgs[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype], alphas: List[Float64], max_iter: Optional[Int], tol: Float64, positive: Bool) raises -> NDArray[dtype]:
    var p = X.shape[1]
    var k = Y.shape[1]
    var coef = zeros[dtype](Shape(k, p))
    var config = LBFGSBConfig(max_iterations=max_iter.value() if max_iter else 15000, function_tolerance=tol, projected_gradient_tolerance=tol)
    var lower = zeros[DType.float64](Shape(p))
    var upper = zeros[DType.float64](Shape(p))
    for j in range(p):
        lower.unsafe_set(j, 0.0 if positive else -inf[DType.float64]())
        upper.unsafe_set(j, inf[DType.float64]())
    for t in range(k):
        var objective = _RidgeObjective(X, _column(Y, t), alphas[t])
        var solver = LBFGSB[DType.float64](p, config.copy())
        _ = solver.minimize(objective, zeros[DType.float64](Shape(p)), lower, upper)
        if not solver.state.has_accepted:
            raise Error(String("RuntimeError: ridge lbfgs solver failed: ", solver.message()))
        # sklearn warns (ConvergenceWarning) on non-success and keeps the result.
        for j in range(p):
            coef.unsafe_set(t * p + j, Scalar[dtype](solver.state.x.unsafe_get(j)))
    return coef^


def solve_ridge[
    dtype: DType
](
    X: NDArray[dtype],
    Y: NDArray[dtype],
    alphas: List[Float64],
    solver: String,
    max_iter: Optional[Int],
    tol: Float64,
    positive: Bool,
) raises -> RidgeSolution[dtype]:
    """Dense `_ridge_regression` core on prepared (centered, sqrt-weighted) data.

    X is compact (samples, features), Y (samples, targets), `alphas` has one
    entry per target and `solver` is already resolved (not 'auto').
    """
    var n = X.shape[0]
    var p = X.shape[1]
    var used = solver
    var n_iter = Optional[List[Int]](None)
    var coef = zeros[dtype](Shape(Y.shape[1], p))
    if solver == "sag" or solver == "saga":
        raise Error(String("NotImplementedError: Ridge solver '", solver, "' needs the SAG engine (sag_fast), which is not yet ported; use 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg' or 'lbfgs'"))
    elif solver == "sparse_cg":
        coef = _solve_sparse_cg(X, Y, alphas, max_iter, tol)
    elif solver == "lsqr":
        var counts = List[Int]()
        coef = _solve_lsqr(X, Y, alphas, max_iter, tol, counts)
        n_iter = counts^
    elif solver == "cholesky":
        try:
            if p > n:
                coef = _solve_cholesky_kernel(X, Y, alphas)
            else:
                coef = _solve_cholesky(X, Y, alphas)
        except error:
            used = "svd"  # sklearn: singular matrix -> SVD solver
    elif solver == "lbfgs":
        coef = _solve_lbfgs(X, Y, alphas, max_iter, tol, positive)
    if used == "svd":
        try:
            coef = _solve_svd(X, Y, alphas)
        except error:
            raise Error(String("LinAlgError: ridge svd solver failed (", error.kind, " in ", error.operation, ")"))
    return RidgeSolution[dtype](coef^, n_iter^, used)


def _check_iteration_params(max_iter: Optional[Int], tol: Float64, owner: String, minimum: Int) raises:
    if max_iter:
        if max_iter.value() < minimum:
            raise Error(String("ValueError: ", owner, ": max_iter must be >= ", minimum, " or None, got ", max_iter.value()))
    if not isfinite(tol) or tol < 0:
        raise Error(String("ValueError: ", owner, ": tol must be a finite float >= 0, got ", tol))


@fieldwise_init
struct RidgeRegressionResult[dtype: DType](Movable):
    """ridge_regression output: coef ((features,) or (targets, features)),
    n_iter (per target for lsqr, else None) and intercept (only with
    return_intercept, which requires the unported 'sag')."""

    var coef: NDArray[Self.dtype]
    var n_iter: Optional[List[Int]]
    var intercept: Optional[NDArray[Self.dtype]]


def ridge_regression[
    dtype: DType
](
    X: NDArray[dtype],
    y: NDArray[dtype],
    alpha: ScalarOrVector,
    *,
    sample_weight: Optional[NDArray[dtype]] = None,
    solver: String = "auto",
    max_iter: Optional[Int] = None,
    tol: Float64 = 1e-4,
    verbose: Int = 0,
    positive: Bool = False,
    random_state: Optional[Int] = None,
    return_n_iter: Bool = False,
    return_intercept: Bool = False,
    check_input: Bool = True,
) raises -> RidgeRegressionResult[dtype]:
    """sklearn.linear_model.ridge_regression on dense data (no centering).

    The result always carries every field; `n_iter` is populated only when
    requested (`return_n_iter`) and available, mirroring sklearn's tuple.
    """
    var owner = String("ridge_regression")
    _check_solver(solver, owner)
    validate_alpha(alpha, owner)
    _check_iteration_params(max_iter, tol, owner, 0)
    var resolved = resolve_solver(solver, positive, return_intercept)
    if positive and resolved != "lbfgs":
        raise Error(String("ValueError: When positive=True, only 'lbfgs' solver can be used. Please change solver ", resolved, " to 'lbfgs' or set positive=False."))
    if resolved == "lbfgs" and not positive:
        raise Error("ValueError: 'lbfgs' solver can be used only when positive=True. Please use another solver.")
    if return_intercept and resolved != "sag":
        raise Error("ValueError: In Ridge, only 'sag' solver can directly fit the intercept. Please change solver to 'sag' or set return_intercept=False.")
    var data = preprocess_data(X, y, sample_weight, False, True, owner)
    var alphas = expand_alphas(alpha, data.y.shape[1])
    var solution = solve_ridge(data.X, data.y, alphas, resolved, max_iter, tol, positive)
    var coef = public_coef(solution.coef, data.y.shape[1] == 1)
    var n_iter = Optional[List[Int]](None)
    if return_n_iter:
        n_iter = solution.n_iter.copy()
    return RidgeRegressionResult[dtype](coef^, n_iter^, None)


@fieldwise_init
struct RidgeParams(Copyable, Movable):
    var alpha: ScalarOrVector
    var fit_intercept: Bool
    var copy_X: Bool
    var max_iter: Optional[Int]
    var tol: Float64
    var solver: String
    var positive: Bool
    var random_state: Optional[Int]


def _base_ridge_solver(solver: String, positive: Bool, owner: String) raises -> String:
    """_BaseRidge.fit solver checks for dense input."""
    _check_solver(solver, owner)
    if solver == "lbfgs" and not positive:
        raise Error("ValueError: 'lbfgs' solver can be used only when positive=True. Please use another solver.")
    if positive and solver != "auto" and solver != "lbfgs":
        raise Error(String("ValueError: solver='", solver, "' does not support positive fitting. Please set the solver to 'auto' or 'lbfgs', or set `positive=False`"))
    return resolve_solver(solver, positive, False)


struct Ridge[dtype: DType = DType.float64](Copyable, LinearRegressor, WeightedSupervisedEstimator):
    """Linear least squares with l2 regularization (sklearn.linear_model.Ridge).

    Minimizes ||y - Xw||^2 + alpha ||w||^2 (sample-weighted), with one alpha or
    one per target. Learned: `coef_` ((features,) for one target, else
    (targets, features)), `intercept_` ((targets,), length one for one target),
    `n_iter_` (per-target LSQR iterations, else None), `solver_`, `n_features_in_`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = RidgeParams

    var alpha: ScalarOrVector
    var fit_intercept: Bool
    var copy_X: Bool
    var max_iter: Optional[Int]
    var tol: Float64
    var solver: String
    var positive: Bool
    var random_state: Optional[Int]
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var n_iter_: Optional[List[Int]]
    var solver_: String
    var n_features_in_: Int
    var _vector_target: Bool
    var _fitted: Bool

    def __init__(
        out self,
        var alpha: ScalarOrVector = 1.0,
        *,
        fit_intercept: Bool = True,
        copy_X: Bool = True,
        max_iter: Optional[Int] = None,
        tol: Float64 = 1e-4,
        var solver: String = "auto",
        positive: Bool = False,
        random_state: Optional[Int] = None,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Ridge supports Float32 and Float64"
        self.alpha = alpha^
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver^
        self.positive = positive
        self.random_state = random_state
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.n_iter_ = None
        self.solver_ = String("")
        self.n_features_in_ = 0
        self._vector_target = True
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alpha.copy(), self.fit_intercept, self.copy_X, self.max_iter, self.tol, self.solver.copy(), self.positive, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alpha = params.alpha.copy()
        self.fit_intercept = params.fit_intercept
        self.copy_X = params.copy_X
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.solver = params.solver.copy()
        self.positive = params.positive
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        alpha: Optional[ScalarOrVector] = None,
        fit_intercept: Optional[Bool] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        solver: Optional[String] = None,
        positive: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update only the supplied parameters (use the Params record to reset Optionals to None)."""
        if alpha:
            self.alpha = alpha.value().copy()
        if fit_intercept:
            self.fit_intercept = fit_intercept.value()
        if max_iter:
            self.max_iter = max_iter
        if tol:
            self.tol = tol.value()
        if solver:
            self.solver = solver.value().copy()
        if positive:
            self.positive = positive.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.alpha.copy(), fit_intercept=self.fit_intercept, copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol, solver=self.solver.copy(), positive=self.positive, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Float64) raises -> ref[self] Self:
        self._fit(X, y, Optional(scalar_weights(X, sample_weight)))
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[NDArray[Self.dtype]]) raises:
        var owner = String("Ridge")
        validate_alpha(self.alpha, owner)
        _check_iteration_params(self.max_iter, self.tol, owner, 1)
        var solver = _base_ridge_solver(self.solver, self.positive, owner)
        var data = preprocess_data(X, y, sample_weight, self.fit_intercept, True, owner)
        var alphas = expand_alphas(self.alpha, data.y.shape[1])
        var solution = solve_ridge(data.X, data.y, alphas, solver, self.max_iter, self.tol, self.positive)
        var intercept = set_intercept(solution.coef, data.X_offset, data.y_offset, data.X_scale, self.fit_intercept)
        var vector = data.y.shape[1] == 1  # sklearn ravels coef_ for a single target
        self.coef_ = public_coef(solution.coef, vector)
        self.intercept_ = intercept^
        self.n_iter_ = solution.n_iter.copy()
        self.solver_ = solution.solver
        self.n_features_in_ = data.X.shape[1]
        self._vector_target = vector
        self._fitted = True

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "Ridge")
        return public_prediction(linear_decision(X, canonical_coef(self.coef_), self.intercept_), self._vector_target)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return r2_score(y, self.predict(X), sample_weight)


@fieldwise_init
struct RidgeClassifierParams(Copyable, Movable):
    var alpha: ScalarOrVector
    var fit_intercept: Bool
    var copy_X: Bool
    var max_iter: Optional[Int]
    var tol: Float64
    var class_weight: Optional[ClassWeight]
    var solver: String
    var positive: Bool
    var random_state: Optional[Int]


def fit_classifier_targets[
    dtype: DType, label_dtype: DType
](
    X: NDArray[dtype],
    y: NDArray[label_dtype],
    sample_weight: Optional[NDArray[dtype]],
    class_weight: Optional[ClassWeight],
    owner: String,
) raises -> Tuple[LabelEncoding[label_dtype], NDArray[dtype], NDArray[dtype]]:
    """_RidgeClassifierMixin._prepare_data: encoding, +/-1 targets Y, sample weights."""
    if X.ndim != 2 or (y.ndim != 1 and y.ndim != 2):
        raise Error(String("ValueError: ", owner, ": X must be a matrix and y a vector or matrix"))
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise Error(String("ValueError: ", owner, ": X must be nonempty"))
    if y.shape[0] != X.shape[0]:
        raise Error(String("ValueError: ", owner, ": X has ", X.shape[0], " samples but y has ", y.shape[0]))
    var encoding = fit_labels(y, owner)
    var Y = binarize[dtype](encoding, y)
    var weights = zeros[dtype](Shape(X.shape[0]))
    for i in range(X.shape[0]):
        weights.unsafe_set(i, 1)
    if sample_weight:
        ref sw = sample_weight.value()
        if sw.ndim != 1 or sw.shape[0] != X.shape[0]:
            raise Error(String("ValueError: ", owner, ": sample_weight must have shape (n_samples,)"))
        for i in range(X.shape[0]):
            weights.unsafe_set(i, sw.unsafe_get(i * Int(sw.strides[0])))
    if class_weight:
        var reweighting = class_sample_weight[dtype](class_weight.value(), encoding, y)
        for i in range(X.shape[0]):
            weights.unsafe_set(i, weights.unsafe_get(i) * reweighting.unsafe_get(i))
    return (encoding^, Y^, weights^)


struct RidgeClassifier[dtype: DType = DType.float64, label_dtype: DType = DType.int64](Copyable, LinearClassifier, WeightedSupervisedEstimator):
    """Classifier using Ridge regression on {-1, 1} targets (sklearn RidgeClassifier).

    Labels are integers (`label_dtype`); a rank-2 0/1 y is a multilabel indicator.
    Learned: `classes_`, `coef_` ((features,) for two classes, else
    (classes, features)), `intercept_` ((columns,)), `n_iter_`, `solver_`,
    `n_features_in_`. `decision_function` is (samples,) or (samples, classes).
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label_dtype]
    comptime Prediction = NDArray[Self.label_dtype]
    comptime Decision = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = RidgeClassifierParams

    var alpha: ScalarOrVector
    var fit_intercept: Bool
    var copy_X: Bool
    var max_iter: Optional[Int]
    var tol: Float64
    var class_weight: Optional[ClassWeight]
    var solver: String
    var positive: Bool
    var random_state: Optional[Int]
    var classes_: NDArray[Self.label_dtype]
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var n_iter_: Optional[List[Int]]
    var solver_: String
    var n_features_in_: Int
    var _encoding: Optional[LabelEncoding[Self.label_dtype]]

    def __init__(
        out self,
        var alpha: ScalarOrVector = 1.0,
        *,
        fit_intercept: Bool = True,
        copy_X: Bool = True,
        max_iter: Optional[Int] = None,
        tol: Float64 = 1e-4,
        var class_weight: Optional[ClassWeight] = None,
        var solver: String = "auto",
        positive: Bool = False,
        random_state: Optional[Int] = None,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "RidgeClassifier supports Float32 and Float64"
        comptime assert Self.label_dtype.is_integral() or Self.label_dtype.is_floating_point(), "labels must be numeric"
        self.alpha = alpha^
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight^
        self.solver = solver^
        self.positive = positive
        self.random_state = random_state
        self.classes_ = zeros[Self.label_dtype](Shape(0))
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.n_iter_ = None
        self.solver_ = String("")
        self.n_features_in_ = 0
        self._encoding = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alpha.copy(), self.fit_intercept, self.copy_X, self.max_iter, self.tol, self.class_weight.copy(), self.solver.copy(), self.positive, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alpha = params.alpha.copy()
        self.fit_intercept = params.fit_intercept
        self.copy_X = params.copy_X
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.class_weight = params.class_weight.copy()
        self.solver = params.solver.copy()
        self.positive = params.positive
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        alpha: Optional[ScalarOrVector] = None,
        fit_intercept: Optional[Bool] = None,
        copy_X: Optional[Bool] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        class_weight: Optional[ClassWeight] = None,
        solver: Optional[String] = None,
        positive: Optional[Bool] = None,
        random_state: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Update only the supplied parameters (use the Params record to reset Optionals to None)."""
        if alpha:
            self.alpha = alpha.value().copy()
        if fit_intercept:
            self.fit_intercept = fit_intercept.value()
        if copy_X:
            self.copy_X = copy_X.value()
        if max_iter:
            self.max_iter = max_iter
        if tol:
            self.tol = tol.value()
        if class_weight:
            self.class_weight = class_weight.copy()
        if solver:
            self.solver = solver.value().copy()
        if positive:
            self.positive = positive.value()
        if random_state:
            self.random_state = random_state
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.alpha.copy(), fit_intercept=self.fit_intercept, copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol, class_weight=self.class_weight.copy(), solver=self.solver.copy(), positive=self.positive, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._encoding)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Float64) raises -> ref[self] Self:
        self._fit(X, y, Optional(scalar_weights(X, sample_weight)))
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[NDArray[Self.dtype]]) raises:
        var owner = String("RidgeClassifier")
        validate_alpha(self.alpha, owner)
        _check_iteration_params(self.max_iter, self.tol, owner, 1)
        var solver = _base_ridge_solver(self.solver, self.positive, owner)
        check_finite(X, "X", owner)
        var prepared = fit_classifier_targets(X, y, sample_weight, self.class_weight, owner)
        var data = preprocess_data(X, prepared[1], Optional(prepared[2].copy()), self.fit_intercept, True, owner)
        var alphas = expand_alphas(self.alpha, data.y.shape[1])
        var solution = solve_ridge(data.X, data.y, alphas, solver, self.max_iter, self.tol, self.positive)
        var intercept = set_intercept(solution.coef, data.X_offset, data.y_offset, data.X_scale, self.fit_intercept)
        self.coef_ = public_coef(solution.coef, data.y.shape[1] == 1)
        self.intercept_ = intercept^
        self.n_iter_ = solution.n_iter.copy()
        self.solver_ = solution.solver
        self.n_features_in_ = data.X.shape[1]
        self.classes_ = prepared[0].classes_array()
        self._encoding = prepared[0].copy()

    def _decision_matrix(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "RidgeClassifier")
        return linear_decision(X, canonical_coef(self.coef_), self.intercept_)

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        var decision = self._decision_matrix(X)
        return public_prediction(decision, decision.shape[1] == 1)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var decision = self._decision_matrix(X)  # checks fitted state first
        return self._encoding.value().labels_from_decision(decision)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return accuracy(y, self.predict(X), sample_weight)
