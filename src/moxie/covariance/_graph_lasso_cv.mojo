# Adapted from scikit-learn sklearn/covariance/_graph_lasso.py.
# Authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Cross-validated graphical lasso, with serial fold execution."""
from moxie.covariance._param_update import NullableUpdate
from std.sys import stderr
from numojo import NDArray, Shape, Item, zeros
from std.math import isfinite, sqrt
from std.sys import llvm_intrinsic
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._empirical_covariance import (
    EmpiricalCovariance, CovarianceModel, column_means, empirical_covariance_float64,
    log_likelihood, to_array, validate_samples,
)
from moxie.covariance._graph_lasso_solver import (
    graphical_lasso_solve, alpha_max, _check_parameters, _EPS64,
)
from moxie.model_selection._indices import IndexArray, IndexSplit
from moxie.model_selection._split import KFold, _rows
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter, cross_val_score


struct AlphaGrid(Copyable, Movable):
    """Implicitly accepts an integer grid size or a List[Float64] of alphas."""
    var count: Int
    var values: Optional[List[Float64]]

    @implicit
    def __init__(out self, count: Int):
        self.count = count
        self.values = None

    @implicit
    def __init__(out self, values: List[Float64]):
        self.count = len(values)
        self.values = values.copy()


@fieldwise_init
struct GraphicalLassoCVResults(Copyable, Movable):
    """Score columns, with split_test_score shaped (n_alphas, n_splits)."""
    var alphas: NDArray[DType.float64]
    var split_test_score: NDArray[DType.float64]
    var mean_test_score: NDArray[DType.float64]
    var std_test_score: NDArray[DType.float64]


struct _EmpiricalScorer[dtype: DType](TypedScorer):
    comptime Estimator = EmpiricalCovariance[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = Int
    comptime Metadata = Int

    def __init__(out self):
        pass

    def score(self, estimator: Self.Estimator, X: Self.Input, y: Int, metadata: Int) raises -> Float64:
        return estimator.score(X)


struct _EmpiricalAdapter[dtype: DType](CVAdapter):
    comptime Scorer = _EmpiricalScorer[Self.dtype]

    def __init__(out self):
        pass

    def n_samples(self, X: NDArray[Self.dtype], y: Int) raises -> Int:
        return X.shape[0]

    def input_rows(self, X: NDArray[Self.dtype], rows: IndexArray, train: IndexArray) raises -> NDArray[Self.dtype]:
        return _rows(X, rows)

    def target_rows(self, y: Int, rows: IndexArray) raises -> Int:
        return y

    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        return metadata

    def fit(self, mut estimator: EmpiricalCovariance[Self.dtype], X: NDArray[Self.dtype], y: Int, metadata: Int) raises:
        estimator.fit(X)


def _logspace(high: Float64, low: Float64, count: Int, interior: Bool = False) raises -> List[Float64]:
    var result = List[Float64]()
    var start = llvm_intrinsic["llvm.log10", Float64, has_side_effect=False](high if interior else low)
    var stop = llvm_intrinsic["llvm.log10", Float64, has_side_effect=False](low if interior else high)
    var total = count + 2 if interior else count
    for i in range(count):
        var index = i + 1 if interior else count - 1 - i
        var step = (stop - start) / Float64(max(1, total - 1))
        var exponent = start + Float64(index) * step
        if not interior and index == total - 1 and total > 1:
            exponent = stop
        result.append(llvm_intrinsic["llvm.pow", Float64, has_side_effect=False](Float64(10), exponent))
    # NumPy's reversed initial logspace with one point picks the low endpoint.
    if count == 1 and not interior:
        result[0] = low
    return result^


@fieldwise_init
struct _PathPoint(Copyable, Movable):
    var alpha: Float64
    var scores: NDArray[DType.float64]


def _mean(scores: NDArray[DType.float64]) -> Float64:
    var total: Float64 = 0
    for i in range(scores.size):
        total += scores.unsafe_get(i)
    return total / Float64(scores.size)


@fieldwise_init
struct GraphicalLassoCVParams(Copyable, Movable):
    var alphas: AlphaGrid
    var n_refinements: Int
    var cv: Optional[Int]
    var tol: Float64
    var enet_tol: Float64
    var max_iter: Int
    var mode: String
    var n_jobs: Optional[Int]
    var verbose: Bool
    var eps: Float64
    var assume_centered: Bool


struct GraphicalLassoCV[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = GraphicalLassoCVParams
    comptime NAME = "GraphicalLassoCV"

    var alphas: AlphaGrid
    var n_refinements: Int
    var cv: Optional[Int]
    var tol: Float64
    var enet_tol: Float64
    var max_iter: Int
    var mode: String
    var n_jobs: Optional[Int]
    var verbose: Bool
    var eps: Float64
    var assume_centered: Bool
    var _model: CovarianceModel[Self.dtype]
    var _results: Optional[GraphicalLassoCVResults]
    var _alpha: Float64
    var _costs: List[Tuple[Float64, Float64]]
    var _n_iter: Int

    def __init__(out self, *, var alphas: AlphaGrid = AlphaGrid(4), n_refinements: Int = 4,
                 cv: Optional[Int] = None, tol: Float64 = 1e-4, enet_tol: Float64 = 1e-4,
                 max_iter: Int = 100, var mode: String = "cd", n_jobs: Optional[Int] = None,
                 verbose: Bool = False, eps: Float64 = _EPS64, assume_centered: Bool = False):
        self.alphas = alphas^
        self.n_refinements = n_refinements
        self.cv = cv
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.mode = mode^
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self._model = CovarianceModel[Self.dtype]()
        self._results = None
        self._alpha = 0
        self._costs = List[Tuple[Float64, Float64]]()
        self._n_iter = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.alphas.copy(), self.n_refinements, self.cv, self.tol, self.enet_tol,
                           self.max_iter, self.mode.copy(), self.n_jobs, self.verbose, self.eps, self.assume_centered)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.alphas = params.alphas.copy()
        self.n_refinements = params.n_refinements
        self.cv = params.cv
        self.tol = params.tol
        self.enet_tol = params.enet_tol
        self.max_iter = params.max_iter
        self.mode = params.mode.copy()
        self.n_jobs = params.n_jobs
        self.verbose = params.verbose
        self.eps = params.eps
        self.assume_centered = params.assume_centered
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        alphas: Optional[AlphaGrid] = None,
        n_refinements: Optional[Int] = None,
        cv: NullableUpdate[Int] = NullableUpdate[Int](),
        tol: Optional[Float64] = None,
        enet_tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None,
        mode: Optional[String] = None,
        n_jobs: NullableUpdate[Int] = NullableUpdate[Int](),
        verbose: Optional[Bool] = None,
        eps: Optional[Float64] = None,
        assume_centered: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update supplied constructor parameters without changing fitted state."""
        if alphas:
            self.alphas = alphas.value().copy()
        if n_refinements:
            self.n_refinements = n_refinements.value().copy()
        if cv.supplied:
            self.cv = cv.value.copy()
        if tol:
            self.tol = tol.value().copy()
        if enet_tol:
            self.enet_tol = enet_tol.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        if mode:
            self.mode = mode.value().copy()
        if n_jobs.supplied:
            self.n_jobs = n_jobs.value.copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if eps:
            self.eps = eps.value().copy()
        if assume_centered:
            self.assume_centered = assume_centered.value().copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        alphas: AlphaGrid,
        n_refinements: Optional[Int] = None,
        cv: NullableUpdate[Int] = NullableUpdate[Int](),
        tol: Optional[Float64] = None,
        enet_tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None,
        mode: Optional[String] = None,
        n_jobs: NullableUpdate[Int] = NullableUpdate[Int](),
        verbose: Optional[Bool] = None,
        eps: Optional[Float64] = None,
        assume_centered: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        self.set_params(
            alphas=Optional[AlphaGrid](alphas.copy()), n_refinements=n_refinements,
            cv=cv.copy(), tol=tol, enet_tol=enet_tol, max_iter=max_iter,
            mode=mode.copy(), n_jobs=n_jobs.copy(), verbose=verbose, eps=eps,
            assume_centered=assume_centered,
        )
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(alphas=self.alphas.copy(), n_refinements=self.n_refinements, cv=self.cv,
                    tol=self.tol, enet_tol=self.enet_tol, max_iter=self.max_iter, mode=self.mode.copy(),
                    n_jobs=self.n_jobs, verbose=self.verbose, eps=self.eps, assume_centered=self.assume_centered)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        var cursor = KFold(self.cv.value() if self.cv else 5).split(X.shape[0] if X.ndim == 2 else 0)
        var splits = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            splits.append(pair.value().copy())
        self.fit(X, splits=splits)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, *, splits: List[IndexSplit]) raises -> ref[self] Self:
        """Use materialized native splitter output in place of Python's cv iterable."""
        _check_parameters(0, self.mode, self.tol, self.enet_tol, self.max_iter, self.eps, Self.NAME)
        validate_samples(X, Self.NAME + ".fit", min_samples=2, min_features=2)
        if self.n_refinements < 1 or self.alphas.count < 1:
            raise Error("ValueError: alphas and n_refinements must be positive and alpha lists nonempty")
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("NotImplementedError: GraphicalLassoCV supports serial n_jobs=None or 1")
        if len(splits) == 0:
            raise Error("ValueError: cross-validation requires at least one split")
        for fold in range(len(splits)):
            splits[fold].validate(X.shape[0])
            if splits[fold].train.size == 0 or splits[fold].test.size == 0:
                raise Error("ValueError: cross-validation train and test sets must be nonempty")
        var location = zeros[Self.dtype](Shape(X.shape[1])) if self.assume_centered else to_array[Self.dtype](column_means(X))
        self._model.update_location(location)
        var empirical = empirical_covariance_float64(X, self.assume_centered)
        var grid: List[Float64]
        var refinements = self.n_refinements
        if self.alphas.values:
            grid = self.alphas.values.value().copy()
            for alpha in grid:
                if not (alpha > 0):
                    raise Error("ValueError: alpha must be > 0")
            refinements = 1
        else:
            var maximum = alpha_max(empirical)
            if not (maximum > 0) or not isfinite(maximum):
                raise Error("ValueError: automatic alpha grid requires positive finite off-diagonal covariance; supply explicit alphas")
            grid = _logspace(maximum, 0.01 * maximum, self.alphas.count)
        var path = List[_PathPoint]()
        var best_index = 0
        for refinement in range(refinements):
            var scores = zeros[DType.float64](Shape(len(grid), len(splits)))
            for fold in range(len(splits)):
                var train = _rows(X, splits[fold].train)
                var test = _rows(X, splits[fold].test)
                # The reference path always centers both folds independently.
                var train_cov = empirical_covariance_float64(train, False)
                var test_cov = empirical_covariance_float64(test, False)
                var warm = train_cov.copy()
                for a in range(len(grid)):
                    var score = -Float64.MAX * 2.0
                    try:
                        var fitted = graphical_lasso_solve(train_cov, grid[a], self.mode, self.tol,
                            self.enet_tol, Int(0.1 * Float64(self.max_iter)), self.eps, warm.copy(), emit_warnings=False)
                        warm = fitted.covariance.copy()
                        score = log_likelihood(test_cov, fitted.precision)
                    except err:
                        if not String(err).startswith("FloatingPointError:"):
                            raise err
                    scores[Item(a, fold)] = score if isfinite(score) else -Float64.MAX * 2.0
            for a in range(len(grid)):
                var column = zeros[DType.float64](Shape(len(splits)))
                for fold in range(len(splits)):
                    column.unsafe_set(fold, scores[Item(a, fold)])
                var point = _PathPoint(grid[a], column^)
                # Stable descending insertion retains duplicate alpha order.
                path.append(point^)
                var j = len(path) - 1
                while j > 0 and path[j].alpha > path[j - 1].alpha:
                    var previous = path[j - 1].copy()
                    path[j - 1] = path[j].copy()
                    path[j] = previous^
                    j -= 1
            var best_score = -Float64.MAX * 2.0
            var last_finite_index = 0
            for i in range(len(path)):
                var score = _mean(path[i].scores)
                if score >= 0.1 / _EPS64:
                    continue
                if isfinite(score):
                    last_finite_index = i
                if score >= best_score:
                    best_score = score
                    best_index = i
            if self.verbose:
                print("[GraphicalLassoCV] Done refinement", refinement + 1, "out of", refinements, file=stderr)
            if refinement + 1 < refinements:
                var high = path[best_index].alpha
                var low = 0.01 * high
                if best_index == 0 and len(path) > 1:
                    low = path[1].alpha
                elif best_index == last_finite_index and best_index != len(path) - 1:
                    low = path[best_index + 1].alpha
                elif best_index != len(path) - 1:
                    high = path[best_index - 1].alpha
                    low = path[best_index + 1].alpha
                grid = _logspace(high, low, self.alphas.count, interior=True)
        # Alpha zero is reported using training-centered empirical scoring,
        # but is deliberately not eligible for selection (sklearn behavior).
        var baseline = cross_val_score[_EmpiricalAdapter[Self.dtype]](
            EmpiricalCovariance[Self.dtype](), X, 0, _EmpiricalAdapter[Self.dtype](),
            _EmpiricalScorer[Self.dtype](), splits, 0, 0)
        var selected_alpha = path[best_index].alpha
        path.append(_PathPoint(0, baseline^))
        var results = GraphicalLassoCVResults(
            zeros[DType.float64](Shape(len(path))), zeros[DType.float64](Shape(len(path), len(splits))),
            zeros[DType.float64](Shape(len(path))), zeros[DType.float64](Shape(len(path))))
        for a in range(len(path)):
            results.alphas.unsafe_set(a, path[a].alpha)
            var average = _mean(path[a].scores)
            var variance: Float64 = 0
            for fold in range(len(splits)):
                var value = path[a].scores.unsafe_get(fold)
                results.split_test_score[Item(a, fold)] = value
                variance += (value - average) * (value - average)
            results.mean_test_score.unsafe_set(a, average)
            results.std_test_score.unsafe_set(a, sqrt(variance / Float64(len(splits))))
        self._results = results^
        self._alpha = selected_alpha
        var fitted = graphical_lasso_solve(empirical, selected_alpha, self.mode, self.tol,
                                          self.enet_tol, self.max_iter, self.eps)
        var candidate = CovarianceModel[Self.dtype]()
        candidate.install_estimate_float64(location, fitted.covariance, fitted.precision)
        self._model = candidate^
        self._costs = fitted.costs.copy()
        self._n_iter = fitted.n_iter
        return self

    def alpha_(self) raises -> Float64:
        self._model.check_fitted(Self.NAME)
        return self._alpha

    def cv_results_(self) raises -> GraphicalLassoCVResults:
        self._model.check_fitted(Self.NAME)
        return self._results.value().copy()

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
