# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""LinearDiscriminantAnalysis: 'svd', 'lsqr' and 'eigen' solvers (sklearn.discriminant_analysis)."""
from std.math import sqrt, abs
from numojo import NDArray, Shape, zeros
from moxie.utils.validation import FittedState
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._discriminant_analysis_contracts import LinearDiscriminantTransformer
from ._labels import ClassLabel, sorted_unique, encode_labels, class_counts
from ._work import (
    Work,
    log,
    get,
    put,
    matrix,
    vector,
    input_matrix,
    cast_array,
    widen,
    matmul,
    matmul_transposed,
    checked_svd,
    checked_least_squares,
)
from ._covariance import (
    Shrinkage,
    CovarianceEstimator,
    NoCovarianceEstimator,
    estimate_covariance,
    class_means,
    class_covariance,
)
from ._generalized_eigh import generalized_symmetric_eigh
from ._prediction import (
    row_argmax,
    softmax_rows,
    binary_probabilities,
    log_of_probabilities,
    select_labels,
    accuracy,
)

comptime ESTIMATOR = "LinearDiscriminantAnalysis"
comptime EPS64: Float64 = 2.220446049250313e-16


@fieldwise_init
struct LinearDiscriminantAnalysisParams[C: CovarianceEstimator](Copyable, Movable):
    """Typed `get_params`/`set_params` record; constructor configuration only."""

    var solver: String
    var shrinkage: Shrinkage
    var priors: Optional[List[Float64]]
    var n_components: Optional[Int]
    var store_covariance: Bool
    var tol: Float64
    var covariance_estimator: Optional[Self.C]


@fieldwise_init
struct _LinearSolution(Movable):
    """Float64 results of one solver before casting to the estimator dtype."""

    var means: Work
    var coef: Work
    var intercept: Work
    var covariance: Optional[Work]
    var explained_variance_ratio: Optional[Work]
    var scalings: Optional[Work]
    var xbar: Optional[Work]


def resolve_priors(
    priors: Optional[List[Float64]], counts: List[Int], n_samples: Int, estimator: String
) raises -> Work:
    """Class priors: sample proportions, or validated user priors renormalized to sum 1.

    sklearn warns (UserWarning) before renormalizing priors whose sum differs from 1
    by more than 1e-5; Mojo has no warnings, so renormalization is silent.
    """
    var k = len(counts)
    var out = vector(k)
    if not priors:
        for c in range(k):
            out.unsafe_set(c, Float64(counts[c]) / Float64(n_samples))
        return out^
    ref given = priors.value()
    if len(given) != k:
        raise Error(
            String("ValueError: ", estimator, " priors has ", len(given), " entries but y has ")
            + String(k, " classes")
        )
    var total: Float64 = 0
    for c in range(k):
        out.unsafe_set(c, given[c])
        total += given[c]
    for c in range(k):
        if given[c] < 0:
            raise Error("ValueError: priors must be non-negative")
    if abs(total - 1.0) > 1e-5:
        for c in range(k):
            out.unsafe_set(c, out.unsafe_get(c) / total)
    return out^


def descending_order(values: Work) raises -> List[Int]:
    """Indices sorting `values` descending (reverse of a stable ascending argsort)."""
    var n = values.shape[0]
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    # Stable insertion sort ascending, then reverse.
    for i in range(1, n):
        var current = order[i]
        var j = i - 1
        while j >= 0 and values.unsafe_get(order[j]) > values.unsafe_get(current):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = current
    order.reverse()
    return order^


def linear_intercept(means: Work, coef: Work, priors: Work) raises -> Work:
    """-0.5 * diag(means @ coef.T) + log(priors)."""
    var k = means.shape[0]
    var p = means.shape[1]
    var out = vector(k)
    for c in range(k):
        var acc: Float64 = 0
        for j in range(p):
            acc += get(means, c, j) * get(coef, c, j)
        out.unsafe_set(c, -0.5 * acc + log(priors.unsafe_get(c)))
    return out^


def solve_lsqr[
    C: CovarianceEstimator
](
    X: Work,
    codes: List[Int],
    counts: List[Int],
    priors: Work,
    shrinkage: Shrinkage,
    estimator: Optional[C],
) raises -> _LinearSolution:
    """Duda et al. 2.6.2: coef_ = lstsq(Sw, means.T).T (gelsd cutoff eps * s_max)."""
    var means = class_means(X, codes, counts)
    var cov = class_covariance(X, codes, counts, priors, shrinkage, estimator)
    var means_t = matrix(X.shape[1], len(counts))
    for c in range(len(counts)):
        for j in range(X.shape[1]):
            put(means_t, j, c, get(means, c, j))
    var solution = checked_least_squares(cov, means_t, rtol=EPS64, context=ESTIMATOR + "(solver='lsqr')")
    var coef = matrix(len(counts), X.shape[1])
    for c in range(len(counts)):
        for j in range(X.shape[1]):
            put(coef, c, j, get(solution, j, c))
    var intercept = linear_intercept(means, coef, priors)
    return _LinearSolution(means^, coef^, intercept^, cov^, None, None, None)


def solve_eigen[
    C: CovarianceEstimator
](
    X: Work,
    codes: List[Int],
    counts: List[Int],
    priors: Work,
    shrinkage: Shrinkage,
    estimator: Optional[C],
    max_components: Int,
) raises -> _LinearSolution:
    """Duda et al. 3.8.3: generalized eigenproblem Sb v = lambda Sw v."""
    var p = X.shape[1]
    var means = class_means(X, codes, counts)
    var within = class_covariance(X, codes, counts, priors, shrinkage, estimator)
    var total = estimate_covariance(X, shrinkage, estimator)
    var between = matrix(p, p)
    for i in range(between.size):
        between.unsafe_set(i, total.unsafe_get(i) - within.unsafe_get(i))
    var eig = generalized_symmetric_eigh(between, within, ESTIMATOR + "(solver='eigen')")
    var evals = eig.values.copy()
    var total_eval: Float64 = 0
    for i in range(p):
        total_eval += evals.unsafe_get(i)
    var ratios = vector(p)
    for i in range(p):
        ratios.unsafe_set(i, evals.unsafe_get(i) / total_eval)
    var ratio_order = descending_order(ratios)
    var evr = vector(max_components)
    for i in range(max_components):
        evr.unsafe_set(i, ratios.unsafe_get(ratio_order[i]))
    var order = descending_order(evals)
    var scalings = matrix(p, p)
    for i in range(p):
        for j in range(p):
            put(scalings, i, j, get(eig.vectors, i, order[j]))
    # coef_ = means @ V @ V.T
    var projected = matmul(means, scalings)
    var coef = matmul_transposed(projected, scalings)
    var intercept = linear_intercept(means, coef, priors)
    return _LinearSolution(means^, coef^, intercept^, within^, evr^, scalings^, None)


def solve_svd(
    X: Work,
    codes: List[Int],
    counts: List[Int],
    priors: Work,
    tol: Float64,
    store_covariance: Bool,
    max_components: Int,
) raises -> _LinearSolution:
    """Two-stage SVD solver: within-class whitening, then between-class SVD."""
    var n = X.shape[0]
    var p = X.shape[1]
    var k = len(counts)
    var means = class_means(X, codes, counts)
    var covariance: Optional[Work] = None
    if store_covariance:
        covariance = class_covariance[NoCovarianceEstimator](X, codes, counts, priors, Shrinkage(), None)
    # Within-class centered data and its population standard deviation.
    var xc = matrix(n, p)
    for s in range(n):
        for j in range(p):
            xc.unsafe_set(s * p + j, X.unsafe_get(s * p + j) - means.unsafe_get(codes[s] * p + j))
    var xbar = vector(p)
    for j in range(p):
        var acc: Float64 = 0
        for c in range(k):
            acc += priors.unsafe_get(c) * get(means, c, j)
        xbar.unsafe_set(j, acc)
    var std = vector(p)
    for j in range(p):
        var mean: Float64 = 0
        for s in range(n):
            mean += xc.unsafe_get(s * p + j)
        mean /= Float64(n)
        var sq: Float64 = 0
        for s in range(n):
            var d = xc.unsafe_get(s * p + j) - mean
            sq += d * d
        var value = sqrt(sq / Float64(n))
        std.unsafe_set(j, 1.0 if value == 0 else value)
    var root = sqrt(1.0 / Float64(n))
    for s in range(n):
        for j in range(p):
            xc.unsafe_set(s * p + j, root * (xc.unsafe_get(s * p + j) / std.unsafe_get(j)))
    var first = checked_svd(xc, ESTIMATOR + "(solver='svd') within-class SVD")
    var s1 = first.singular_values()
    var vt1 = first.vt()
    var rank = 0
    for i in range(s1.shape[0]):
        if s1.unsafe_get(i) > tol:
            rank += 1
    if rank == 0:
        raise Error(
            String("LinAlgError: ", ESTIMATOR, "(solver='svd'): all within-class singular values")
            + String(" are <= tol=", tol, "; the scaled within-class data has rank 0")
        )
    # scalings = (Vt[:rank] / std).T / S[:rank], shape (p, rank)
    var scalings = matrix(p, rank)
    for j in range(p):
        for r in range(rank):
            put(scalings, j, r, get(vt1, r, j) / std.unsafe_get(j) / s1.unsafe_get(r))
    var fac = 1.0 if k == 1 else 1.0 / Float64(k - 1)
    var weighted = matrix(k, p)
    for c in range(k):
        var w = sqrt(Float64(n) * priors.unsafe_get(c) * fac)
        for j in range(p):
            put(weighted, c, j, w * (get(means, c, j) - xbar.unsafe_get(j)))
    var between = matmul(weighted, scalings)
    var second = checked_svd(between, ESTIMATOR + "(solver='svd') between-class SVD")
    var s2 = second.singular_values()
    var vt2 = second.vt()
    var m = s2.shape[0]
    var evr: Work
    if max_components == 0:
        evr = vector(0)
    else:
        var total: Float64 = 0
        for i in range(m):
            total += s2.unsafe_get(i) * s2.unsafe_get(i)
        var length = min(max_components, m)
        evr = vector(length)
        for i in range(length):
            evr.unsafe_set(i, s2.unsafe_get(i) * s2.unsafe_get(i) / total)
    var rank2 = 0
    for i in range(m):
        if s2.unsafe_get(i) > tol * s2.unsafe_get(0):
            rank2 += 1
    # scalings_ = scalings @ Vt2.T[:, :rank2], shape (p, rank2)
    var final_scalings = matrix(p, rank2)
    for j in range(p):
        for r2 in range(rank2):
            var acc: Float64 = 0
            for r in range(rank):
                acc += get(scalings, j, r) * get(vt2, r2, r)
            put(final_scalings, j, r2, acc)
    var centered_means = matrix(k, p)
    for c in range(k):
        for j in range(p):
            put(centered_means, c, j, get(means, c, j) - xbar.unsafe_get(j))
    var coef_small = matmul(centered_means, final_scalings)
    var intercept = vector(k)
    for c in range(k):
        var acc: Float64 = 0
        for r in range(rank2):
            acc += get(coef_small, c, r) * get(coef_small, c, r)
        intercept.unsafe_set(c, -0.5 * acc + log(priors.unsafe_get(c)))
    var coef = matmul_transposed(coef_small, final_scalings)
    for c in range(k):
        var acc: Float64 = 0
        for j in range(p):
            acc += xbar.unsafe_get(j) * get(coef, c, j)
        intercept.unsafe_set(c, intercept.unsafe_get(c) - acc)
    return _LinearSolution(
        means^, coef^, intercept^, covariance^, evr^, final_scalings^, xbar^
    )


struct LinearDiscriminantAnalysis[
    dtype: DType = DType.float64,
    L: ClassLabel = Int,
    C: CovarianceEstimator = NoCovarianceEstimator,
](LinearDiscriminantTransformer, FeatureNamesOut, InputFeatureCount):
    """Linear Discriminant Analysis (sklearn.discriminant_analysis.LinearDiscriminantAnalysis).

    A classifier with a linear decision boundary from class-conditional Gaussian
    densities with a shared covariance, plus supervised dimensionality reduction
    (`transform`, solvers 'svd' and 'eigen').

    Parameters:
        dtype: Float32 or Float64 input/output dtype (internal work is Float64).
        L: Class-label type (e.g. Int or String); `classes_` is sorted ascending.
        C: Native covariance estimator type used when `covariance_estimator` is set.

    Constructor arguments keep sklearn's names and defaults: solver='svd',
    shrinkage=None ('auto' or float), priors=None, n_components=None,
    store_covariance=False, tol=1e-4, covariance_estimator=None.

    Learned attributes (sklearn names): coef_ (1 or n_classes, p), intercept_,
    covariance_ (Optional; svd only with store_covariance), explained_variance_ratio_
    (Optional; svd/eigen), means_, priors_, scalings_ (Optional; svd/eigen),
    xbar_ (Optional; svd), classes_, n_features_in_.
    """

    comptime Params = LinearDiscriminantAnalysisParams[Self.C]
    comptime Input = NDArray[Self.dtype]
    comptime Target = List[Self.L]
    comptime Prediction = List[Self.L]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Probabilities = NDArray[Self.dtype]
    comptime Decision = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var solver: String
    var shrinkage: Shrinkage
    var priors: Optional[List[Float64]]
    var n_components: Optional[Int]
    var store_covariance: Bool
    var tol: Float64
    var covariance_estimator: Optional[Self.C]

    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var covariance_: Optional[NDArray[Self.dtype]]
    var explained_variance_ratio_: Optional[NDArray[Self.dtype]]
    var means_: NDArray[Self.dtype]
    var priors_: NDArray[Self.dtype]
    var scalings_: Optional[NDArray[Self.dtype]]
    var xbar_: Optional[NDArray[Self.dtype]]
    var classes_: List[Self.L]
    var n_features_in_: Int
    var _max_components: Int
    var _fitted_solver: String
    var _state: FittedState

    def __init__(
        out self,
        solver: String = "svd",
        shrinkage: Shrinkage = None,
        priors: Optional[List[Float64]] = None,
        n_components: Optional[Int] = None,
        store_covariance: Bool = False,
        tol: Float64 = 1.0e-4,
        covariance_estimator: Optional[Self.C] = None,
    ) raises:
        """Store configuration only; validation happens in `fit` like sklearn."""
        self.solver = solver
        self.shrinkage = shrinkage.copy()
        self.priors = priors.copy()
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator.copy()
        self.coef_ = zeros[Self.dtype](Shape(0, 0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.covariance_ = None
        self.explained_variance_ratio_ = None
        self.means_ = zeros[Self.dtype](Shape(0, 0))
        self.priors_ = zeros[Self.dtype](Shape(0))
        self.scalings_ = None
        self.xbar_ = None
        self.classes_ = List[Self.L]()
        self.n_features_in_ = 0
        self._max_components = 0
        self._fitted_solver = ""
        self._state = FittedState()

    # ----- configuration -------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        """Return an independent copy of the constructor configuration."""
        return Self.Params(
            self.solver,
            self.shrinkage.copy(),
            self.priors.copy(),
            self.n_components,
            self.store_covariance,
            self.tol,
            self.covariance_estimator.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        """Replace the whole configuration; learned state is left untouched."""
        self.solver = params.solver
        self.shrinkage = params.shrinkage.copy()
        self.priors = params.priors.copy()
        self.n_components = params.n_components
        self.store_covariance = params.store_covariance
        self.tol = params.tol
        self.covariance_estimator = params.covariance_estimator.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        solver: Optional[String] = None,
        shrinkage: Optional[Shrinkage] = None,
        priors: Optional[Optional[List[Float64]]] = None,
        n_components: Optional[Optional[Int]] = None,
        store_covariance: Optional[Bool] = None,
        tol: Optional[Float64] = None,
        covariance_estimator: Optional[Optional[Self.C]] = None,
    ) raises -> ref[self] Self:
        """Keyword partial update; an outer None leaves that parameter unchanged."""
        if solver:
            self.solver = solver.value()
        if shrinkage:
            self.shrinkage = shrinkage.value().copy()
        if priors:
            self.priors = priors.value().copy()
        if n_components:
            self.n_components = n_components.value()
        if store_covariance:
            self.store_covariance = store_covariance.value()
        if tol:
            self.tol = tol.value()
        if covariance_estimator:
            self.covariance_estimator = covariance_estimator.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        """Unfitted estimator with an independent copy of the configuration."""
        return Self(
            self.solver,
            self.shrinkage.copy(),
            self.priors.copy(),
            self.n_components,
            self.store_covariance,
            self.tol,
            self.covariance_estimator.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` through the shared InputFeatureCount contract; raises before fit."""
        if not self._state.is_fitted():
            raise Error(
                "NotFittedError: This LinearDiscriminantAnalysis instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.n_features_in_

    def _validate_params(self) raises:
        if self.solver != "svd" and self.solver != "lsqr" and self.solver != "eigen":
            raise Error(
                "ValueError: The 'solver' parameter of LinearDiscriminantAnalysis must be a str"
                " among {'eigen', 'lsqr', 'svd'}. Got '" + self.solver + "' instead."
            )
        self.shrinkage.validate(ESTIMATOR)
        if self.n_components and self.n_components.value() < 1:
            raise Error(
                String("ValueError: The 'n_components' parameter of LinearDiscriminantAnalysis")
                + String(" must be an int in the range [1, inf) or None. Got ")
                + String(self.n_components.value())
                + " instead."
            )
        if not (self.tol >= 0):
            raise Error(
                String("ValueError: The 'tol' parameter of LinearDiscriminantAnalysis must be")
                + String(" a float in the range [0, inf). Got ", self.tol, " instead.")
            )

    def _reset(mut self) raises:
        self.coef_ = zeros[Self.dtype](Shape(0, 0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.covariance_ = None
        self.explained_variance_ratio_ = None
        self.means_ = zeros[Self.dtype](Shape(0, 0))
        self.priors_ = zeros[Self.dtype](Shape(0))
        self.scalings_ = None
        self.xbar_ = None
        self.classes_ = List[Self.L]()
        self.n_features_in_ = 0
        self._max_components = 0
        self._fitted_solver = ""
        self._state.reset()

    # ----- fitting -------------------------------------------------------

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: NDArray[Self.dtype], ref[y_origin] y: List[Self.L]
    ) raises -> ref[self] Self:
        """Fit the model; replaces learned state (reset to unfitted on failure)."""
        self._reset()
        self._validate_params()
        var work = input_matrix(X, ESTIMATOR, min_samples=2)
        var n = work.shape[0]
        var p = work.shape[1]
        if len(y) != n:
            raise Error(
                String("ValueError: Found input variables with inconsistent numbers of samples: [")
                + String(n, ", ", len(y), "]")
            )
        var classes = sorted_unique(y)
        var codes = encode_labels(y, classes)
        var counts = class_counts(codes, len(classes))
        var k = len(classes)
        if n == k:
            raise Error("ValueError: The number of samples must be more than the number of classes.")
        var priors = resolve_priors(self.priors, counts, n, ESTIMATOR)
        var max_components = min(k - 1, p)
        if self.n_components:
            if self.n_components.value() > max_components:
                raise Error(
                    "ValueError: n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            max_components = self.n_components.value()
        var solution: _LinearSolution
        if self.solver == "svd":
            if not self.shrinkage.is_none():
                raise Error("NotImplementedError: shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator:
                raise Error(
                    "ValueError: covariance estimator is not supported with svd solver."
                    " Try another solver"
                )
            solution = solve_svd(work, codes, counts, priors, self.tol, self.store_covariance, max_components)
        elif self.solver == "lsqr":
            solution = solve_lsqr(work, codes, counts, priors, self.shrinkage, self.covariance_estimator)
        else:
            solution = solve_eigen(
                work, codes, counts, priors, self.shrinkage, self.covariance_estimator, max_components
            )
        if k == 2:
            # Binary case: one row of coefficient differences.
            var coef = matrix(1, p)
            for j in range(p):
                put(coef, 0, j, get(solution.coef, 1, j) - get(solution.coef, 0, j))
            var intercept = vector(1)
            intercept.unsafe_set(0, solution.intercept.unsafe_get(1) - solution.intercept.unsafe_get(0))
            solution.coef = coef^
            solution.intercept = intercept^
        self.coef_ = cast_array[Self.dtype](solution.coef)
        self.intercept_ = cast_array[Self.dtype](solution.intercept)
        self.means_ = cast_array[Self.dtype](solution.means)
        self.priors_ = cast_array[Self.dtype](priors)
        if solution.covariance:
            self.covariance_ = cast_array[Self.dtype](solution.covariance.value())
        if solution.explained_variance_ratio:
            self.explained_variance_ratio_ = cast_array[Self.dtype](solution.explained_variance_ratio.value())
        if solution.scalings:
            self.scalings_ = cast_array[Self.dtype](solution.scalings.value())
        if solution.xbar:
            self.xbar_ = cast_array[Self.dtype](solution.xbar.value())
        self.classes_ = classes^
        self.n_features_in_ = p
        self._max_components = max_components
        self._fitted_solver = self.solver
        self._state.record_fit(n, p, n_targets=len(y))
        return self

    # ----- inference -----------------------------------------------------

    def _check_input(self, X: NDArray[Self.dtype]) raises -> Work:
        if not self._state.is_fitted():
            raise Error(
                "NotFittedError: This LinearDiscriminantAnalysis instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        var work = input_matrix(X, ESTIMATOR)
        if work.shape[1] != self.n_features_in_:
            raise Error(
                String("ValueError: X has ", work.shape[1], " features, but LinearDiscriminantAnalysis")
                + String(" is expecting ", self.n_features_in_, " features as input.")
            )
        return work^

    def _scores(self, X: NDArray[Self.dtype]) raises -> Work:
        """X @ coef_.T + intercept_, shape (n, n_coef_rows)."""
        var work = self._check_input(X)
        var scores = matmul_transposed(work, widen(self.coef_))
        var rows = self.coef_.shape[0]
        for i in range(scores.shape[0]):
            for c in range(rows):
                scores.unsafe_set(i * rows + c, scores.unsafe_get(i * rows + c) + Float64(self.intercept_.unsafe_get(c)))
        return scores^

    def decision_function(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Shape (n,) for one coefficient row (binary), else (n, n_classes)."""
        var scores = self._scores(X)
        if scores.shape[1] == 1:
            var flat = vector(scores.shape[0])
            for i in range(scores.shape[0]):
                flat.unsafe_set(i, get(scores, i, 0))
            return cast_array[Self.dtype](flat)
        return cast_array[Self.dtype](scores)

    def predict(self, X: NDArray[Self.dtype]) raises -> List[Self.L]:
        """Class label per sample: argmax of the decision (binary: decision > 0)."""
        var scores = self._scores(X)
        var indices: List[Int]
        if scores.shape[1] == 1:
            indices = List[Int](capacity=scores.shape[0])
            var top = min(1, len(self.classes_) - 1)
            for i in range(scores.shape[0]):
                indices.append(top if get(scores, i, 0) > 0 else 0)
        else:
            indices = row_argmax(scores)
        return select_labels(self.classes_, indices)

    def _probabilities(self, X: NDArray[Self.dtype]) raises -> Work:
        var scores = self._scores(X)
        if len(self.classes_) == 2:
            var flat = vector(scores.shape[0])
            for i in range(scores.shape[0]):
                flat.unsafe_set(i, get(scores, i, 0))
            return binary_probabilities(flat)
        if len(self.classes_) == 1:
            raise Error(
                "ValueError: predict_proba needs at least two classes; this"
                " LinearDiscriminantAnalysis was fitted on a single class"
            )
        return softmax_rows(scores)

    def predict_proba(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Class probabilities (n, n_classes): expit (binary) or softmax."""
        return cast_array[Self.dtype](self._probabilities(X))

    def predict_log_proba(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Return log(predict_proba) with exact zeros replaced by the smallest normal number."""
        return log_of_probabilities[Self.dtype](self._probabilities(X))

    def score(
        self,
        X: NDArray[Self.dtype],
        y: List[Self.L],
        sample_weight: Optional[NDArray[Self.dtype]] = None,
    ) raises -> Float64:
        """Mean (optionally weighted) accuracy of `predict(X)` against y."""
        return accuracy(self.predict(X), y, sample_weight, ESTIMATOR)

    def transform(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Project to the most discriminative directions, (n, min(rank, n_components)).

        'lsqr' raises NotImplementedError (checked before fitted state, like sklearn).
        """
        if self.solver == "lsqr":
            raise Error(
                "NotImplementedError: transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        var work = self._check_input(X)
        if not self.scalings_:
            raise Error(
                "NotImplementedError: transform not implemented: the model was fitted with"
                " solver='lsqr'; refit with 'svd' or 'eigen'."
            )
        var scalings = widen(self.scalings_.value())
        var width = min(self._max_components, scalings.shape[1])
        var n = work.shape[0]
        var p = work.shape[1]
        if self._fitted_solver == "svd":
            var xbar = self.xbar_.value().copy()
            for s in range(n):
                for j in range(p):
                    work.unsafe_set(s * p + j, work.unsafe_get(s * p + j) - Float64(xbar.unsafe_get(j)))
        var out = matrix(n, width)
        var cols = scalings.shape[1]
        for s in range(n):
            for c in range(width):
                var acc: Float64 = 0
                for j in range(p):
                    acc += work.unsafe_get(s * p + j) * scalings.unsafe_get(j * cols + c)
                put(out, s, c, acc)
        return cast_array[Self.dtype](out)

    def fit_transform(
        mut self, X: NDArray[Self.dtype], y: Optional[List[Self.L]] = None
    ) raises -> NDArray[Self.dtype]:
        """Run fit(X, y) then transform(X); y is required for this supervised transformer."""
        if not y:
            raise Error("ValueError: LinearDiscriminantAnalysis.fit_transform requires y")
        self.fit(X, y.value())
        return self.transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        """['lineardiscriminantanalysis0', ...] for the `_max_components` outputs."""
        if not self._state.is_fitted():
            raise Error(
                "NotFittedError: This LinearDiscriminantAnalysis instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error(
                String("ValueError: input_features should have length equal to number of features (")
                + String(self.n_features_in_, "), got ", len(input_features.value()))
            )
        var names = List[String](capacity=self._max_components)
        for i in range(self._max_components):
            names.append(String("lineardiscriminantanalysis", i))
        return names^
