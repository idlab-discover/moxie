# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Serial typed grid/random search, shared candidate evaluation and refitting."""
from moxie.base import SupervisedEstimator, SupervisedScorer, ClassifierMixin, RegressorMixin, BaseEstimator, InputEstimator, clone, Predictor, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction, TransformerMixin, InverseTransformer, SampleScorer, WeightedSupervisedEstimator
from moxie.model_selection._base import SearchEstimator
from moxie.model_selection._evaluation import EstimatorScore, TypedScorer, ParameterAdapter
from moxie.model_selection._validation import DefaultCVAdapter, _cross_validate
from moxie.model_selection._indices import IndexArray, IndexSplit, SplitCursor
from moxie.model_selection._split import KFold, SplitLabels, check_cv, select_rows
from moxie.model_selection._results import ScoreTable, FitStatus, ErrorPolicy, RefitSelector, validate_refit_index, require_refit
from moxie.model_selection._search_parameters import ParameterGrid, ParameterSampler, ParameterDomain, DiscreteDistribution
from moxie.model_selection._candidates import ParameterDistribution
from moxie._dense import sum_axis2d
from moxie._random import RandomState
from numojo import NDArray, Shape, Item, array
from std.math import sqrt, isnan
from std.memory import bitcast
from std.time import perf_counter_ns


trait SearchScorer(TypedScorer, Copyable):
    pass


trait SearchAdapter(DefaultCVAdapter, ParameterAdapter, Copyable, Deinitable):
    """Extend CV with typed partial parameter application."""
    pass


comptime _ParameterApplier[E: SupervisedScorer, V: Copyable & Deinitable] = def(
    mut E, Dict[String, V]
) thin raises


def _collect_splits[C: SplitCursor & Deinitable](var cursor: C) raises -> List[IndexSplit]:
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            return splits^
        splits.append(pair.take())


struct DenseSupervisedAdapter[
    E: SupervisedScorer & Deinitable,
    Y: Copyable & Deinitable = NDArray[DType.float64],
    dtype: DType = DType.float64,
    V: Copyable & Deinitable = Float64,
    weight_dtype: DType = dtype,
](SearchAdapter):
    """Ordinary dense samples with array or list targets and optional weights."""

    comptime X = NDArray[Self.dtype]
    comptime Weight = NDArray[Self.weight_dtype]
    comptime Scorer = EstimatorScore[Self.E, Self.X, Self.Y, Self.Weight]
    comptime Value = Self.V
    comptime _IsArray = (
        Self.Y == NDArray[DType.float32] or Self.Y == NDArray[DType.float64]
        or Self.Y == IndexArray
    )
    comptime _ArrayDType = (
        DType.float32 if Self.Y == NDArray[DType.float32] else
        DType.int64 if Self.Y == IndexArray else DType.float64
    )
    comptime _Label = (
        String if Self.Y == List[String] else
        Int if Self.Y == List[Int] else
        Int64 if Self.Y == List[Int64] else
        Float32 if Self.Y == List[Float32] else Float64
    )

    var _apply_params: Optional[_ParameterApplier[Self.E, Self.V]]
    var _classifier: Bool

    def __init__(
        out self,
        apply_params: Optional[_ParameterApplier[Self.E, Self.V]] = None,
        *,
        classifier: Bool = conforms_to(Self.E, ClassifierMixin),
    ):
        comptime assert Self.X == Self.E.Input, "Dense input must match the estimator"
        comptime assert Self.Y == Self.E.Target, "Target must match the estimator"
        comptime assert Self.Weight == Self.E.SampleWeight, "Weight must match the estimator"
        comptime assert Self._IsArray or Self.Y == List[Self._Label], (
            "Expected Float32, Float64 or Int64 arrays, or String, Int, Int64, Float32 or Float64 lists"
        )
        self._apply_params = apply_params
        self._classifier = classifier

    def n_samples(self, X: Self.X, y: Self.Y) raises -> Int:
        if X.ndim != 2:
            raise Error("ValueError: dense CV expects a rank-two feature matrix")
        var n: Int
        comptime if Self._IsArray:
            ref target = rebind[NDArray[Self._ArrayDType]](y)
            if target.ndim != 1 and target.ndim != 2:
                raise Error("ValueError: dense CV expects rank-one or rank-two targets")
            n = target.shape[0]
        else:
            n = len(rebind[List[Self._Label]](y))
        if X.shape[0] != n:
            raise Error("ValueError: inconsistent sample counts")
        return n

    def input_rows(self, X: Self.X, rows: IndexArray, train: IndexArray) raises -> Self.X:
        return select_rows(X, rows)

    def target_rows(self, y: Self.Y, rows: IndexArray) raises -> Self.Y:
        comptime if Self._IsArray:
            return rebind_var[Self.Y](select_rows(rebind[NDArray[Self._ArrayDType]](y), rows))
        else:
            return rebind_var[Self.Y](select_rows(rebind[List[Self._Label]](y), rows))

    def validate_metadata(self, metadata: Optional[Self.Weight], n_samples: Int) raises:
        if metadata:
            ref weights = metadata.value()
            if weights.ndim != 1 or weights.shape[0] != n_samples:
                raise Error("ValueError: sample weights require one value per sample")

    def metadata_rows(self, metadata: Optional[Self.Weight], rows: IndexArray) raises -> Optional[Self.Weight]:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Self.E, X: Self.X, y: Self.Y, metadata: Optional[Self.Weight]) raises:
        if metadata:
            self.validate_metadata(metadata, self.n_samples(X, y))
            comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                comptime assert Self.Weight == Self.E.SampleWeight
                _ = estimator.fit(
                    rebind[Self.E.Input](X), rebind[Self.E.Target](y),
                    sample_weight=rebind[Self.E.SampleWeight](metadata.value()),
                )
            else:
                raise Error("ValueError: estimator does not support weighted fit")
        else:
            _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))

    def configure(self, mut estimator: Self.E, params: Dict[String, Self.V]) raises:
        if self._apply_params:
            self._apply_params.value()(estimator, params)
        elif len(params) != 0:
            raise Error("ValueError: nonempty parameters require an apply_params function")

    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()

    def default_cv(self, X: Self.X, y: Self.Y) raises -> List[IndexSplit]:
        var n = self.n_samples(X, y)
        if not self._classifier:
            return _collect_splits(KFold(5).split(n))
        var labels = self._split_labels(y)
        return _collect_splits(check_cv(None, labels, classifier=True).split(n, labels))

    @staticmethod
    def _split_labels(y: Self.Y) raises -> SplitLabels:
        comptime if Self._IsArray:
            return SplitLabels._numeric(rebind[NDArray[Self._ArrayDType]](y))
        elif Self._Label == String:
            return SplitLabels(rebind[List[String]](y))
        elif Self._Label == Int:
            ref target = rebind[List[Int]](y)
            var labels = IndexArray(Shape(len(target)))
            for i in range(len(target)):
                labels[Item(i)] = Int64(target[i])
            return SplitLabels(labels)
        else:
            comptime dtype = (
                DType.float32 if Self._Label == Float32 else
                DType.int64 if Self._Label == Int64 else DType.float64
            )
            ref target = rebind[List[Scalar[dtype]]](y)
            return SplitLabels._numeric(array[dtype](target, [len(target)]))


@fieldwise_init
struct SearchResults[V: Copyable & Deinitable](Copyable, Movable):
    var params: List[Dict[String, Self.V]]
    var scores: ScoreTable
    var status: List[FitStatus]
    var mean_test_score: NDArray[DType.float64]
    var std_test_score: NDArray[DType.float64]
    var rank_test_score: NDArray[DType.int64]
    var mean_train_score: Optional[NDArray[DType.float64]]
    var std_train_score: Optional[NDArray[DType.float64]]
    var mean_fit_time: NDArray[DType.float64]
    var std_fit_time: NDArray[DType.float64]
    var mean_score_time: NDArray[DType.float64]
    var std_score_time: NDArray[DType.float64]

    def parameter(self, name: String) raises -> List[Optional[Self.V]]:
        """Missing conditional-grid parameters are None, never fabricated values."""
        var values = List[Optional[Self.V]]()
        for i in range(len(self.params)):
            if name in self.params[i]:
                values.append(self.params[i][name].copy())
            else:
                values.append(None)
        return values^

    def copy(self) -> Self:
        return Self(self.params.copy(), ScoreTable(self.scores.test_scores.copy(), self.scores.train_scores.copy(), self.scores.metric_names.copy()), self.status.copy(), self.mean_test_score.copy(), self.std_test_score.copy(), self.rank_test_score.copy(), self.mean_train_score.copy(), self.std_train_score.copy(), self.mean_fit_time.copy(), self.std_fit_time.copy(), self.mean_score_time.copy(), self.std_score_time.copy())


def _moments(values: NDArray[DType.float64]) raises -> Tuple[NDArray[DType.float64], NDArray[DType.float64]]:
    var means = sum_axis2d(values, 1)
    var residuals = NDArray[DType.float64](values.shape.copy())
    for i in range(values.shape[0]):
        means[Item(i)] = means[Item(i)] / Float64(values.shape[1])
        for j in range(values.shape[1]):
            var delta = values[Item(i, j)] - means[Item(i)]
            residuals[Item(i, j)] = delta * delta
    var deviations = sum_axis2d(residuals, 1)
    for i in range(values.shape[0]):
        deviations[Item(i)] = sqrt(deviations[Item(i)] / Float64(values.shape[1]))
    return (means^, deviations^)


def _summarize(scores: NDArray[DType.float64]) raises -> Tuple[NDArray[DType.float64], NDArray[DType.float64]]:
    var means = NDArray[DType.float64](Shape(scores.shape[0], scores.shape[2]))
    var deviations = NDArray[DType.float64](Shape(scores.shape[0], scores.shape[2]))
    var column = NDArray[DType.float64](Shape(scores.shape[0], scores.shape[1]))
    for m in range(scores.shape[2]):
        for c in range(scores.shape[0]):
            for f in range(scores.shape[1]):
                column[Item(c, f)] = scores[Item(c, f, m)]
        var moments = _moments(column)
        for c in range(scores.shape[0]):
            means[Item(c, m)] = moments[0][Item(c)]
            deviations[Item(c, m)] = moments[1][Item(c)]
    return (means^, deviations^)


def _rank_value(value: Float64, replacement: Float64) -> Float64:
    if isnan(value):
        return replacement
    # xpx.nan_to_num clips original infinities, but not the NaN fill value.
    var largest = bitcast[DType.float64](UInt64(0x7fefffffffffffff))
    return max(-largest, min(largest, value))


def _ranks(means: NDArray[DType.float64]) raises -> NDArray[DType.int64]:
    var ranks = NDArray[DType.int64](means.shape.copy())
    for m in range(means.shape[1]):
        var minimum = bitcast[DType.float64](UInt64(0x7ff0000000000000))
        var all_nan = True
        for c in range(means.shape[0]):
            var value = means[Item(c, m)]
            if not isnan(value):
                all_nan = False
                minimum = min(minimum, value)
        # Match sklearn's actual nanmin - 1 substitution, including +/- infinity.
        var replacement = minimum - 1
        for c in range(means.shape[0]):
            var value = _rank_value(means[Item(c, m)], replacement)
            var rank = 1
            if not all_nan:
                for j in range(means.shape[0]):
                    var other = _rank_value(means[Item(j, m)], replacement)
                    if other > value:
                        rank += 1
            ranks[Item(c, m)] = Int64(rank)
    return ranks^


trait SearchRefitPolicy(RefitSelector, Copyable, Deinitable):
    comptime Value: Copyable & Deinitable
    def __init__(out self):
        ...
    def enabled(self) -> Bool:
        return True
    def is_custom(self) -> Bool:
        return True
    def metric(self, names: List[String], multimetric: Bool) raises -> Int:
        return -1


struct SearchRefit[V: Copyable & Deinitable](SearchRefitPolicy):
    comptime Value = Self.V
    comptime Results = SearchResults[Self.V]
    """Native refit=True/False/name; custom policies implement SearchRefitPolicy."""
    var active: Bool
    var name: String
    def __init__(out self):
        self.active = True
        self.name = String()
    @implicit
    def __init__(out self, enabled: Bool):
        self.active = enabled
        self.name = String()
    @implicit
    def __init__(out self, name: String):
        self.active = True
        self.name = name.copy()
    def enabled(self) -> Bool:
        return self.active
    def is_custom(self) -> Bool:
        return False
    def metric(self, names: List[String], multimetric: Bool) raises -> Int:
        if not multimetric:
            return 0
        if not self.active:
            return -1
        for m in range(len(names)):
            if names[m] == self.name:
                return m
        raise Error("ValueError: multimetric refit requires a metric name, False, or a custom policy")
    def select(self, results: Self.Results) raises -> Int:
        var m = 0
        if self.name.byte_length() > 0:
            for i in range(len(results.scores.metric_names)):
                if results.scores.metric_names[i] == self.name:
                    m = i
        var best = 0
        for c in range(1, len(results.params)):
            if results.rank_test_score[Item(c, m)] < results.rank_test_score[Item(best, m)]:
                best = c
        return best


def _copy_splits(cv: Optional[List[IndexSplit]]) -> Optional[List[IndexSplit]]:
    if not cv:
        return None
    var result = List[IndexSplit]()
    for i in range(len(cv.value())):
        result.append(IndexSplit(cv.value()[i].train.copy(), cv.value()[i].test.copy()))
    return result^


def _copy_scorers[S: TypedScorer](scorers: List[S]) -> List[S]:
    comptime if conforms_to(S, Copyable):
        return scorers.copy()
    else:
        comptime assert False, "SearchAdapter.Scorer must be copyable"


@fieldwise_init
struct SearchOptions[A: SearchAdapter, R: SearchRefitPolicy](Copyable, Movable):
    var adapter: Self.A
    var scoring: List[Self.A.Scorer]
    var metric_names: List[String]
    var multimetric: Bool
    var cv: Optional[List[IndexSplit]]
    var refit: Self.R
    var return_train_score: Bool
    var error_score: ErrorPolicy

    def __init__(out self, *, copy: Self):
        self.adapter = copy.adapter.copy()
        self.scoring = _copy_scorers(copy.scoring)
        self.metric_names = copy.metric_names.copy()
        self.multimetric = copy.multimetric
        self.cv = _copy_splits(copy.cv)
        self.refit = copy.refit.copy()
        self.return_train_score = copy.return_train_score
        self.error_score = copy.error_score.copy()


def _options[A: SearchAdapter, R: SearchRefitPolicy](adapter: A, scoring: Optional[List[A.Scorer]], metric_names: Optional[List[String]], cv: Optional[List[IndexSplit]], refit: R, return_train_score: Bool, error_score: ErrorPolicy) raises -> SearchOptions[A, R]:
    comptime assert conforms_to(A.Scorer, Copyable), "SearchAdapter.Scorer must be copyable"
    var scorers = List[A.Scorer]()
    if scoring:
        scorers = _copy_scorers(scoring.value())
    else:
        scorers.append(adapter.default_scorer())
    var names: List[String] = ["score"]
    if metric_names:
        names = metric_names.value().copy()
    return SearchOptions[A, R](adapter.copy(), scorers^, names^, Bool(metric_names), _copy_splits(cv), refit.copy(), return_train_score, error_score.copy())


@fieldwise_init
struct SearchParams[A: SearchAdapter, S: Copyable & Deinitable, R: SearchRefitPolicy](Copyable, Movable):
    var estimator: Self.A.Scorer.Estimator.Params
    var space: Self.S
    var options: SearchOptions[Self.A, Self.R]


struct _Search[A: SearchAdapter, R: SearchRefitPolicy](Movable, Copyable where conforms_to(A.Scorer.Estimator, Copyable)):
    var estimator: Self.A.Scorer.Estimator
    var options: SearchOptions[Self.A, Self.R]
    var results: Optional[SearchResults[Self.A.Value]]
    var best_estimator: Optional[Self.A.Scorer.Estimator]
    var best_index: Optional[Int]
    var best_score: Optional[Float64]
    var best_params: Optional[Dict[String, Self.A.Value]]
    var refit_time: Optional[Float64]
    var n_splits: Int

    def __init__(out self, estimator: Self.A.Scorer.Estimator, options: SearchOptions[Self.A, Self.R]) raises:
        comptime assert Self.R.Results == SearchResults[Self.A.Value], "refit policy must consume this search result type"
        self.estimator = clone(estimator)
        self.options = options.copy()
        self.results = None
        self.best_estimator = None
        self.best_index = None
        self.best_score = None
        self.best_params = None
        self.refit_time = None
        self.n_splits = 0

    def reset(mut self):
        self.results = None
        self.best_estimator = None
        self.best_index = None
        self.best_score = None
        self.best_params = None
        self.refit_time = None
        self.n_splits = 0

    def check(self) raises:
        if not self.results:
            raise Error("NotFittedError: search has not completed a successful fit")

    def check_refit(self) raises:
        require_refit(self.options.refit.enabled(), Bool(self.best_estimator))

    def fit(mut self, candidates: List[Dict[String, Self.A.Value]], X: Self.A.Scorer.Input, y: Self.A.Scorer.Target, fit_metadata: Self.A.Scorer.Metadata, score_metadata: Self.A.Scorer.Metadata) raises:
        self.reset()
        var metric = self.options.refit.metric(self.options.metric_names, self.options.multimetric)
        if len(candidates) == 0:
            raise Error("ValueError: no search candidates")
        var splits = List[IndexSplit]()
        if self.options.cv:
            for f in range(len(self.options.cv.value())):
                splits.append(self.options.cv.value()[f].copy())
        else:
            splits = self.options.adapter.default_cv(X, y)
        var nc = len(candidates)
        var nf = len(splits)
        var nm = len(self.options.scoring)
        if nf == 0 or nm == 0 or (not self.options.multimetric and nm != 1):
            raise Error("ValueError: nonempty CV and aligned named scorers required")
        var table = ScoreTable(NDArray[DType.float64](Shape(nc, nf, nm)), None, self.options.metric_names.copy())
        if self.options.return_train_score:
            table.train_scores = NDArray[DType.float64](Shape(nc, nf, nm))
        table.validate()
        var status = List[FitStatus]()
        var fit_times = NDArray[DType.float64](Shape(nc, nf))
        var score_times = NDArray[DType.float64](Shape(nc, nf))
        var failures = 0
        for c in range(nc):
            var configured = clone(self.estimator)
            self.options.adapter.configure(configured, candidates[c])
            var result = _cross_validate[Self.A](configured, X, y, self.options.adapter, self.options.scoring, self.options.metric_names, splits, fit_metadata, score_metadata, return_train_score=self.options.return_train_score, error_score=self.options.error_score, allow_all_failed=True)
            for f in range(nf):
                failures += Int(result.status[f].failed)
                fit_times[Item(c, f)] = result.status[f].fit_time
                score_times[Item(c, f)] = result.status[f].score_time
                status.append(result.status[f].copy())
                for m in range(nm):
                    table.test_scores[Item(c, f, m)] = result.scores.test_scores[Item(0, f, m)]
                    if self.options.return_train_score:
                        table.train_scores.value()[Item(c, f, m)] = result.scores.train_scores.value()[Item(0, f, m)]
        if failures == nc * nf:
            raise Error("ValueError: all search fits failed")
        var test = _summarize(table.test_scores)
        var ranks = _ranks(test[0])
        var train_mean: Optional[NDArray[DType.float64]] = None
        var train_std: Optional[NDArray[DType.float64]] = None
        if table.train_scores:
            var train = _summarize(table.train_scores.value())
            train_mean = train[0].copy()
            train_std = train[1].copy()
        var ft = _moments(fit_times)
        var st = _moments(score_times)
        var result = SearchResults[Self.A.Value](candidates.copy(), table^, status^, test[0].copy(), test[1].copy(), ranks^, train_mean^, train_std^, ft[0].copy(), ft[1].copy(), st[0].copy(), st[1].copy())
        var best: Optional[Int] = None
        var score: Optional[Float64] = None
        var params: Optional[Dict[String, Self.A.Value]] = None
        var fitted: Optional[Self.A.Scorer.Estimator] = None
        var elapsed: Optional[Float64] = None
        if self.options.refit.enabled() or not self.options.multimetric:
            best = self.options.refit.select(rebind[Self.R.Results](result))
            validate_refit_index(best.value(), nc)
            params = candidates[best.value()].copy()
            if not self.options.refit.is_custom():
                score = result.mean_test_score[Item(best.value(), metric)]
        if self.options.refit.enabled():
            var model = clone(self.estimator)
            self.options.adapter.configure(model, params.value())
            var start = perf_counter_ns()
            # Deliberately outside numeric error handling: final refit always raises.
            self.options.adapter.fit(model, X, y, fit_metadata)
            elapsed = Float64(perf_counter_ns() - start) / 1.0e9
            fitted = model^
        # Commit only after successful refit; failed/repeated fits expose no stale state.
        self.results = result^
        self.best_index = best
        self.best_score = score
        self.best_params = params^
        self.best_estimator = fitted^
        self.refit_time = elapsed
        self.n_splits = nf


@fieldwise_init
struct RandomSearchSpace[D: ParameterDistribution & Copyable & Deinitable](Copyable, Movable):
    var domains: List[List[ParameterDomain[Self.D]]]
    var n_iter: Int
    var random_state: Optional[Int]


def _domains[D: ParameterDistribution & Copyable & Deinitable](grid: ParameterGrid[D.Value]) raises -> List[List[ParameterDomain[D]]]:
    var spaces = List[List[ParameterDomain[D]]]()
    for i in range(len(grid.param_grid)):
        var axes = List[ParameterDomain[D]]()
        for key in grid.param_grid[i].keys():
            axes.append(ParameterDomain[D](key.copy(), grid.param_grid[i][key]))
        spaces.append(axes^)
    return spaces^


struct GridSearchCV[A: SearchAdapter, R: SearchRefitPolicy = SearchRefit[A.Value]](
    SearchEstimator, InputEstimator,
    Copyable where conforms_to(A.Scorer.Estimator, Copyable),
    SupervisedEstimator where conforms_to(A.Scorer.Metadata, Defaultable),
    SupervisedScorer where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, SupervisedScorer),
    ClassifierMixin where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, ClassifierMixin),
    RegressorMixin where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, RegressorMixin),
    Predictor where conforms_to(A.Scorer.Estimator, Predictor),
    ProbabilityPredictor where conforms_to(A.Scorer.Estimator, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(A.Scorer.Estimator, LogProbabilityPredictor),
    DecisionFunction where conforms_to(A.Scorer.Estimator, DecisionFunction),
    TransformerMixin where conforms_to(A.Scorer.Estimator, TransformerMixin),
    SampleScorer where conforms_to(A.Scorer.Estimator, SampleScorer),
):
    comptime E = Self.A.Scorer.Estimator
    comptime Input = Self.A.Scorer.Input
    comptime Target = Self.A.Scorer.Target
    comptime Metadata = Self.A.Scorer.Metadata
    comptime SampleWeight = Self.E.SampleWeight if conforms_to(Self.E, SupervisedScorer) else Int
    comptime Params = SearchParams[Self.A, ParameterGrid[Self.A.Value], Self.R]
    comptime Prediction where conforms_to(Self.E, Predictor) = Self.E.Prediction
    comptime Probabilities where conforms_to(Self.E, ProbabilityPredictor) = Self.E.Probabilities
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision
    comptime Transformed where conforms_to(Self.E, TransformerMixin) = Self.E.Transformed
    comptime InverseTransformed where conforms_to(Self.E, InverseTransformer) = Self.E.Input
    comptime SampleScores where conforms_to(Self.E, SampleScorer) = Self.E.SampleScores
    var _core: _Search[Self.A, Self.R]
    var _space: ParameterGrid[Self.A.Value]

    def __init__(out self, estimator: Self.E, param_grid: ParameterGrid[Self.A.Value], adapter: Self.A, *, scoring: Optional[List[Self.A.Scorer]] = None, metric_names: Optional[List[String]] = None, cv: Optional[List[IndexSplit]] = None, refit: Self.R = Self.R(), return_train_score: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises:
        self._core = _Search[Self.A, Self.R](estimator, _options[Self.A, Self.R](adapter, scoring, metric_names, cv, refit, return_train_score, error_score))
        self._space = param_grid.copy()

    def __init__(out self, estimator: Self.E, param_grid: Dict[String, List[Self.A.Value]], adapter: Self.A, *, scoring: Optional[List[Self.A.Scorer]] = None, metric_names: Optional[List[String]] = None, cv: Optional[List[IndexSplit]] = None, refit: Self.R = Self.R(), return_train_score: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises:
        self._core = _Search[Self.A, Self.R](estimator, _options[Self.A, Self.R](adapter, scoring, metric_names, cv, refit, return_train_score, error_score))
        self._space = ParameterGrid[Self.A.Value](param_grid)

    def __init__(out self, estimator: Self.E, param_grid: List[Dict[String, List[Self.A.Value]]], adapter: Self.A, *, scoring: Optional[List[Self.A.Scorer]] = None, metric_names: Optional[List[String]] = None, cv: Optional[List[IndexSplit]] = None, refit: Self.R = Self.R(), return_train_score: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises:
        self._core = _Search[Self.A, Self.R](estimator, _options[Self.A, Self.R](adapter, scoring, metric_names, cv, refit, return_train_score, error_score))
        self._space = ParameterGrid[Self.A.Value](param_grid)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._core.estimator.get_params(deep), self._space.copy(), self._core.options.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var estimator = clone(self._core.estimator)
        _ = estimator.set_params(params.estimator)
        self._core = _Search[Self.A, Self.R](estimator, params.options)
        self._space = params.space.copy()
        return self

    def set_params(mut self, *, param_grid: ParameterGrid[Self.A.Value]):
        self._core.reset()
        self._space = param_grid.copy()

    def __sklearn_clone__(self) raises -> Self:
        var copied = Self(self._core.estimator, self._space, self._core.options.adapter)
        copied._core.options = self._core.options.copy()
        return copied^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._core.results)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self where conforms_to(Self.Metadata, Defaultable):
        return self.fit(X, y, Self.Metadata(), Self.Metadata())

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        fit_metadata: Self.Metadata, score_metadata: Self.Metadata,
    ) raises -> ref[self] Self:
        self._core.reset()
        var cursor = ParameterGrid[Self.A.Value](self._space.param_grid)
        var candidates = List[Dict[String, Self.A.Value]]()
        while True:
            var candidate = cursor.next()
            if not candidate:
                break
            candidates.append(candidate.value().copy())
        self._core.fit(candidates, X, y, fit_metadata, score_metadata)
        return self

    def cv_results_(self) raises -> SearchResults[Self.A.Value]:
        self._core.check()
        return self._core.results.value().copy()

    def best_index_(self) raises -> Int:
        self._core.check()
        if not self._core.best_index:
            raise Error("AttributeError: multimetric refit=False has no best index")
        return self._core.best_index.value()

    def best_params_(self) raises -> Dict[String, Self.A.Value]:
        _ = self.best_index_()
        return self._core.best_params.value().copy()

    def best_score_(self) raises -> Float64:
        self._core.check()
        if not self._core.best_score:
            raise Error("AttributeError: best_score_ unavailable for custom refit or multimetric refit=False")
        return self._core.best_score.value()

    def best_estimator_(ref self) raises -> ref[origin_of(self._core.best_estimator.value())] Self.E:
        self._core.check_refit()
        return self._core.best_estimator.value()

    def n_splits_(self) raises -> Int:
        self._core.check()
        return self._core.n_splits

    def multimetric_(self) raises -> Bool:
        self._core.check()
        return self._core.options.multimetric

    def scorer_(self) raises -> List[Self.A.Scorer]:
        self._core.check()
        return _copy_scorers(self._core.options.scoring)

    def refit_time_(self) raises -> Float64:
        self._core.check_refit()
        return self._core.refit_time.value()

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64 where conforms_to(Self.Metadata, Defaultable) and conforms_to(Self.E, SupervisedScorer):
        if sample_weight:
            raise Error("ValueError: search sample weights must be supplied through score(X, y, metadata=...)")
        return self.score(X, y, metadata=Self.Metadata())

    def score(self, X: Self.Input, y: Self.Target, *, metadata: Self.Metadata) raises -> Float64:
        self._core.check_refit()
        var metric = self._core.options.refit.metric(self._core.options.metric_names, self._core.options.multimetric)
        if not self._core.options.multimetric:
            metric = 0
        if metric < 0:
            raise Error("ValueError: custom multimetric refit has no selected scoring metric; use score_metric")
        return self._core.options.scoring[metric].score(self._core.best_estimator.value(), X, y, metadata)

    def score_metric(self, name: String, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        self._core.check_refit()
        for m in range(len(self._core.options.metric_names)):
            if self._core.options.metric_names[m] == name:
                return self._core.options.scoring[m].score(self._core.best_estimator.value(), X, y, metadata)
        raise Error("ValueError: unknown metric name")

    def predict(self, X: Self.Input) raises -> Self.Prediction where conforms_to(Self.E, Predictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict(rebind[Self.E.Input](X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict_proba(rebind[Self.E.Input](X))

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict_log_proba(rebind[Self.E.Input](X))

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().decision_function(rebind[Self.E.Input](X))

    def transform(self, X: Self.Input) raises -> Self.Transformed where conforms_to(Self.E, TransformerMixin):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().transform(rebind[Self.E.Input](X))

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores where conforms_to(Self.E, SampleScorer):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().score_samples(rebind[Self.E.Input](X))

    def inverse_transform(self, X: Self.Transformed) raises -> Self.InverseTransformed where conforms_to(Self.E, InverseTransformer):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().inverse_transform(X)


struct RandomizedSearchCV[A: SearchAdapter, D: ParameterDistribution & Copyable & Deinitable = DiscreteDistribution[A.Value], R: SearchRefitPolicy = SearchRefit[A.Value]](
    SearchEstimator, InputEstimator,
    Copyable where conforms_to(A.Scorer.Estimator, Copyable),
    SupervisedEstimator where conforms_to(A.Scorer.Metadata, Defaultable),
    SupervisedScorer where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, SupervisedScorer),
    ClassifierMixin where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, ClassifierMixin),
    RegressorMixin where conforms_to(A.Scorer.Metadata, Defaultable) and conforms_to(A.Scorer.Estimator, RegressorMixin),
    Predictor where conforms_to(A.Scorer.Estimator, Predictor),
    ProbabilityPredictor where conforms_to(A.Scorer.Estimator, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(A.Scorer.Estimator, LogProbabilityPredictor),
    DecisionFunction where conforms_to(A.Scorer.Estimator, DecisionFunction),
    TransformerMixin where conforms_to(A.Scorer.Estimator, TransformerMixin),
    SampleScorer where conforms_to(A.Scorer.Estimator, SampleScorer),
):
    comptime E = Self.A.Scorer.Estimator
    comptime Input = Self.A.Scorer.Input
    comptime Target = Self.A.Scorer.Target
    comptime Metadata = Self.A.Scorer.Metadata
    comptime SampleWeight = Self.E.SampleWeight if conforms_to(Self.E, SupervisedScorer) else Int
    comptime Params = SearchParams[Self.A, RandomSearchSpace[Self.D], Self.R]
    comptime Prediction where conforms_to(Self.E, Predictor) = Self.E.Prediction
    comptime Probabilities where conforms_to(Self.E, ProbabilityPredictor) = Self.E.Probabilities
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision
    comptime Transformed where conforms_to(Self.E, TransformerMixin) = Self.E.Transformed
    comptime InverseTransformed where conforms_to(Self.E, InverseTransformer) = Self.E.Input
    comptime SampleScores where conforms_to(Self.E, SampleScorer) = Self.E.SampleScores
    var _core: _Search[Self.A, Self.R]
    var _space: RandomSearchSpace[Self.D]

    def __init__(out self, estimator: Self.E, param_distributions: List[List[ParameterDomain[Self.D]]], adapter: Self.A, *, n_iter: Int = 10, random_state: Optional[Int] = None, scoring: Optional[List[Self.A.Scorer]] = None, metric_names: Optional[List[String]] = None, cv: Optional[List[IndexSplit]] = None, refit: Self.R = Self.R(), return_train_score: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises:
        comptime assert Self.D.Value == Self.A.Value, "distribution and adapter values must match"
        self._core = _Search[Self.A, Self.R](estimator, _options[Self.A, Self.R](adapter, scoring, metric_names, cv, refit, return_train_score, error_score))
        self._space = RandomSearchSpace[Self.D](param_distributions.copy(), n_iter, random_state)

    def __init__(out self, estimator: Self.E, param_distributions: ParameterGrid[Self.A.Value], adapter: Self.A, *, n_iter: Int = 10, random_state: Optional[Int] = None, scoring: Optional[List[Self.A.Scorer]] = None, metric_names: Optional[List[String]] = None, cv: Optional[List[IndexSplit]] = None, refit: Self.R = Self.R(), return_train_score: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises:
        comptime assert Self.D.Value == Self.A.Value, "distribution and adapter values must match"
        self._core = _Search[Self.A, Self.R](estimator, _options[Self.A, Self.R](adapter, scoring, metric_names, cv, refit, return_train_score, error_score))
        self._space = RandomSearchSpace[Self.D](_domains[Self.D](rebind[ParameterGrid[Self.D.Value]](param_distributions)), n_iter, random_state)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._core.estimator.get_params(deep), self._space.copy(), self._core.options.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var estimator = clone(self._core.estimator)
        _ = estimator.set_params(params.estimator)
        comptime assert Self.D.Value == Self.A.Value, "distribution and adapter values must match"
        self._core = _Search[Self.A, Self.R](estimator, params.options)
        self._space = params.space.copy()
        return self

    def set_params(mut self, *, n_iter: Int):
        self._core.reset()
        self._space.n_iter = n_iter

    def set_params(mut self, *, random_state: Optional[Int]):
        self._core.reset()
        self._space.random_state = random_state

    def set_params(mut self, *, param_distributions: List[List[ParameterDomain[Self.D]]]):
        self._core.reset()
        self._space.domains = param_distributions.copy()

    def __sklearn_clone__(self) raises -> Self:
        var copied = Self(self._core.estimator, self._space.domains, self._core.options.adapter, n_iter=self._space.n_iter, random_state=self._space.random_state)
        copied._core.options = self._core.options.copy()
        return copied^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._core.results)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self where conforms_to(Self.Metadata, Defaultable):
        return self.fit(X, y, Self.Metadata(), Self.Metadata())

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        fit_metadata: Self.Metadata, score_metadata: Self.Metadata,
    ) raises -> ref[self] Self:
        self._core.reset()
        var cursor = ParameterSampler[Self.D](self._space.domains, self._space.n_iter, random_state=self._space.random_state)
        var candidates = List[Dict[String, Self.A.Value]]()
        while True:
            var candidate = cursor.next()
            if not candidate:
                break
            candidates.append(rebind[Dict[String, Self.A.Value]](candidate.value()).copy())
        self._core.fit(candidates, X, y, fit_metadata, score_metadata)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        fit_metadata: Self.Metadata, score_metadata: Self.Metadata, mut rng: RandomState,
    ) raises -> ref[self] Self:
        self._core.reset()
        if self._space.random_state:
            raise Error("ValueError: use either the configured seed or an application-owned RNG")
        var cursor = ParameterSampler[Self.D](self._space.domains, self._space.n_iter, rng)
        var candidates = List[Dict[String, Self.A.Value]]()
        while True:
            var candidate = cursor.next()
            if not candidate:
                break
            candidates.append(rebind[Dict[String, Self.A.Value]](candidate.value()).copy())
        self._core.fit(candidates, X, y, fit_metadata, score_metadata)
        return self

    def cv_results_(self) raises -> SearchResults[Self.A.Value]:
        self._core.check()
        return self._core.results.value().copy()

    def best_index_(self) raises -> Int:
        self._core.check()
        if not self._core.best_index:
            raise Error("AttributeError: multimetric refit=False has no best index")
        return self._core.best_index.value()

    def best_params_(self) raises -> Dict[String, Self.A.Value]:
        _ = self.best_index_()
        return self._core.best_params.value().copy()

    def best_score_(self) raises -> Float64:
        self._core.check()
        if not self._core.best_score:
            raise Error("AttributeError: best_score_ unavailable for custom refit or multimetric refit=False")
        return self._core.best_score.value()

    def best_estimator_(ref self) raises -> ref[origin_of(self._core.best_estimator.value())] Self.E:
        self._core.check_refit()
        return self._core.best_estimator.value()

    def n_splits_(self) raises -> Int:
        self._core.check()
        return self._core.n_splits

    def multimetric_(self) raises -> Bool:
        self._core.check()
        return self._core.options.multimetric

    def scorer_(self) raises -> List[Self.A.Scorer]:
        self._core.check()
        return _copy_scorers(self._core.options.scoring)

    def refit_time_(self) raises -> Float64:
        self._core.check_refit()
        return self._core.refit_time.value()

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64 where conforms_to(Self.Metadata, Defaultable) and conforms_to(Self.E, SupervisedScorer):
        if sample_weight:
            raise Error("ValueError: search sample weights must be supplied through score(X, y, metadata=...)")
        return self.score(X, y, metadata=Self.Metadata())

    def score(self, X: Self.Input, y: Self.Target, *, metadata: Self.Metadata) raises -> Float64:
        self._core.check_refit()
        var metric = self._core.options.refit.metric(self._core.options.metric_names, self._core.options.multimetric)
        if not self._core.options.multimetric:
            metric = 0
        if metric < 0:
            raise Error("ValueError: custom multimetric refit has no selected scoring metric; use score_metric")
        return self._core.options.scoring[metric].score(self._core.best_estimator.value(), X, y, metadata)

    def score_metric(self, name: String, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        self._core.check_refit()
        for m in range(len(self._core.options.metric_names)):
            if self._core.options.metric_names[m] == name:
                return self._core.options.scoring[m].score(self._core.best_estimator.value(), X, y, metadata)
        raise Error("ValueError: unknown metric name")

    def predict(self, X: Self.Input) raises -> Self.Prediction where conforms_to(Self.E, Predictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict(rebind[Self.E.Input](X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict_proba(rebind[Self.E.Input](X))

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().predict_log_proba(rebind[Self.E.Input](X))

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().decision_function(rebind[Self.E.Input](X))

    def transform(self, X: Self.Input) raises -> Self.Transformed where conforms_to(Self.E, TransformerMixin):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().transform(rebind[Self.E.Input](X))

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores where conforms_to(Self.E, SampleScorer):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().score_samples(rebind[Self.E.Input](X))

    def inverse_transform(self, X: Self.Transformed) raises -> Self.InverseTransformed where conforms_to(Self.E, InverseTransformer):
        self._core.check_refit()
        comptime assert Self.Input == Self.E.Input
        return self._core.best_estimator.value().inverse_transform(X)
