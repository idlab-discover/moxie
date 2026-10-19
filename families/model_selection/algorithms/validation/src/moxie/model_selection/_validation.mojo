"""Serial typed CV orchestration; see docs/scope.md for supported modes."""
from moxie.base import BaseEstimator, clone
from moxie.model_selection._evaluation import FitAdapter, ParameterAdapter
from moxie.model_selection._indices import IndexArray, IndexSplit
from moxie.model_selection._results import ScoreTable, FitStatus, ErrorPolicy
from numojo import NDArray, Shape, Item, zeros, mean
from moxie.model_selection._split import SplitLabels, _rows
from moxie._labels import encode_column, StringTarget
from moxie._random import RandomState
from std.builtin.sort import sort
from std.math import isfinite, min, max
from std.memory import bitcast
from std.time import perf_counter_ns


trait CVAdapter(FitAdapter):
    """Family-specific slicing seam; metadata remains explicitly typed."""
    def n_samples(self, X: Self.Scorer.Input, y: Self.Scorer.Target) raises -> Int:
        ...
    def input_rows(self, X: Self.Scorer.Input, rows: IndexArray, train: IndexArray) raises -> Self.Scorer.Input:
        ...
    def target_rows(self, y: Self.Scorer.Target, rows: IndexArray) raises -> Self.Scorer.Target:
        ...
    def metadata_rows(self, metadata: Self.Scorer.Metadata, rows: IndexArray) raises -> Self.Scorer.Metadata:
        ...


trait IncrementalCVAdapter(CVAdapter):
    """Incremental fit with full target context for classifier class discovery."""
    def partial_fit(self, mut estimator: Self.Scorer.Estimator,
                    X: Self.Scorer.Input, y: Self.Scorer.Target,
                    metadata: Self.Scorer.Metadata,
                    full_target: Self.Scorer.Target) raises:
        ...


trait PredictionAdapter(CVAdapter):
    comptime PredictionDType: DType
    def predict(self, estimator: Self.Scorer.Estimator, X: Self.Scorer.Input) raises -> NDArray[Self.PredictionDType]:
        ...


@fieldwise_init
struct CVResult[E: BaseEstimator & Deinitable](Movable):
    var scores: ScoreTable
    var status: List[FitStatus]
    var estimators: Optional[List[Self.E]]
    var indices: Optional[List[IndexSplit]]


def _checked_splits(pairs: List[IndexSplit], n: Int) raises:
    if len(pairs) == 0:
        raise Error("ValueError: cross-validation requires at least one split")
    for i in range(len(pairs)):
        pairs[i].validate(n)


def _score[A: CVAdapter](scorer: A.Scorer, estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target, metadata: A.Scorer.Metadata, policy: ErrorPolicy) raises -> Tuple[Float64, String]:
    try:
        return (scorer.score(estimator, X, y, metadata), String())
    except err:
        if policy.raise_on_error:
            raise err
        return (policy.error_score, String(err))


def _evaluate_fold[A: CVAdapter](
    mut model: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, scorers: List[A.Scorer], split: IndexSplit,
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    mut scores: ScoreTable, candidate: Int, fold: Int, error_score: ErrorPolicy,
) raises -> FitStatus where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    """Fit a fresh caller-owned clone and fill its score cells; no aggregate policy."""
    var X_train = adapter.input_rows(X, split.train, split.train)
    var X_test = adapter.input_rows(X, split.test, split.train)
    var y_train = adapter.target_rows(y, split.train)
    var y_test = adapter.target_rows(y, split.test)
    var train_meta = adapter.metadata_rows(fit_metadata, split.train)
    var test_meta = adapter.metadata_rows(score_metadata, split.test)
    var record = FitStatus(False, String(), 0, 0)
    var start = perf_counter_ns()
    try:
        adapter.fit(model, X_train, y_train, train_meta)
    except err:
        if error_score.raise_on_error:
            raise err
        record.failed = True
        record.message = String(err)
    record.fit_time = Float64(perf_counter_ns() - start) / 1.0e9
    start = perf_counter_ns()
    for metric in range(len(scorers)):
        var value = error_score.error_score
        if not record.failed:
            var outcome = _score[A](scorers[metric], model, X_test, y_test, test_meta, error_score)
            value = outcome[0]
            if outcome[1].byte_length() > 0:
                record.message += outcome[1] + "\n"
        scores.test_scores[Item(candidate, fold, metric)] = value
    record.score_time = 0 if record.failed else Float64(perf_counter_ns() - start) / 1.0e9
    if scores.train_scores:
        var train_score_meta = adapter.metadata_rows(score_metadata, split.train)
        for metric in range(len(scorers)):
            var value = error_score.error_score
            if not record.failed:
                var outcome = _score[A](scorers[metric], model, X_train, y_train, train_score_meta, error_score)
                value = outcome[0]
                if outcome[1].byte_length() > 0:
                    record.message += outcome[1] + "\n"
            scores.train_scores.value()[Item(candidate, fold, metric)] = value
    return record^


def _cross_validate[A: CVAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, scorers: List[A.Scorer], metric_names: List[String],
    cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    *, return_train_score: Bool = False, return_estimator: Bool = False,
    return_indices: Bool = False, allow_all_failed: Bool = False,
    error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))),
) raises -> CVResult[A.Scorer.Estimator] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    var n = adapter.n_samples(X, y)
    _checked_splits(cv, n)
    var metrics = len(scorers)
    if metrics == 0 or len(metric_names) != metrics:
        raise Error("ValueError: scorers and metric names must be nonempty and aligned")
    var scores = ScoreTable(NDArray[DType.float64](Shape(1, len(cv), metrics)), None, metric_names.copy())
    if return_train_score:
        scores.train_scores = NDArray[DType.float64](Shape(1, len(cv), metrics))
    scores.validate()
    var status = List[FitStatus]()
    var estimators: Optional[List[A.Scorer.Estimator]] = None
    var indices: Optional[List[IndexSplit]] = None
    if return_estimator:
        estimators = List[A.Scorer.Estimator]()
    if return_indices:
        indices = List[IndexSplit]()
    var failed = 0
    for fold in range(len(cv)):
        var model = clone(estimator)
        var record = _evaluate_fold[A](model, X, y, adapter, scorers, cv[fold], fit_metadata, score_metadata, scores, 0, fold, error_score)
        if record.failed:
            failed += 1
        if return_estimator:
            estimators.value().append(model^)
        if return_indices:
            indices.value().append(cv[fold].copy())
        status.append(record^)
    if failed == len(cv) and not allow_all_failed:
        raise Error("ValueError: all cross-validation fits failed")
    return CVResult[A.Scorer.Estimator](scores^, status^, estimators^, indices^)


def cross_validate[A: CVAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, scorers: List[A.Scorer], metric_names: List[String],
    cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    *, return_train_score: Bool = False, return_estimator: Bool = False,
    return_indices: Bool = False,
    error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))),
) raises -> CVResult[A.Scorer.Estimator] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    return _cross_validate[A](estimator, X, y, adapter, scorers, metric_names, cv, fit_metadata, score_metadata, return_train_score=return_train_score, return_estimator=return_estimator, return_indices=return_indices, error_score=error_score)


def cross_val_score[A: CVAdapter](estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target, adapter: A, var scorer: A.Scorer, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata, *, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000)))) raises -> NDArray[DType.float64] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    var scorers = List[A.Scorer]()
    scorers.append(scorer^)
    var result = cross_validate[A](estimator, X, y, adapter, scorers, List[String](["score"]), cv, fit_metadata, score_metadata, error_score=error_score)
    var values = NDArray[DType.float64](Shape(len(cv)))
    for fold in range(len(cv)):
        values[Item(fold)] = result.scores.test_scores[Item(0, fold, 0)]
    return values^


def _checked_prediction_splits(cv: List[IndexSplit], n: Int) raises:
    _checked_splits(cv, n)
    var seen = zeros[DType.int64](Shape(n))
    for fold in range(len(cv)):
        for j in range(cv[fold].test.size):
            var row = Int(cv[fold].test[Item(j)])
            seen[Item(row)] = seen[Item(row)] + 1
    for i in range(n):
        if seen[Item(i)] != 1:
            raise Error("ValueError: cross_val_predict requires test indices to partition samples")


def cross_val_predict[A: PredictionAdapter](estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target, adapter: A, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, *, method: String = "predict") raises -> NDArray[A.PredictionDType] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    if method != "predict":
        raise Error("ValueError: numeric prediction adapter supports method='predict' only; class responses require ClassPredictionAdapter")
    var n = adapter.n_samples(X, y)
    _checked_prediction_splits(cv, n)
    var output: Optional[NDArray[A.PredictionDType]] = None
    for fold in range(len(cv)):
        var X_train = adapter.input_rows(X, cv[fold].train, cv[fold].train)
        var X_test = adapter.input_rows(X, cv[fold].test, cv[fold].train)
        var y_train = adapter.target_rows(y, cv[fold].train)
        var metadata = adapter.metadata_rows(fit_metadata, cv[fold].train)
        var model = clone(estimator)
        adapter.fit(model, X_train, y_train, metadata)
        var predicted = adapter.predict(model, X_test)
        _scatter_prediction(output, predicted, cv[fold].test, n)
    return output.value().copy()


def validation_curve[A: CVAdapter & ParameterAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, var scoring: A.Scorer, cv: List[IndexSplit],
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    *, param_name: String, param_range: List[A.Value],
    error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))),
) raises -> Tuple[NDArray[DType.float64], NDArray[DType.float64]] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    """Return train/test scores (parameter, split), evaluating split-first."""
    _checked_splits(cv, adapter.n_samples(X, y))
    if len(param_range) == 0:
        raise Error("ValueError: param_range must not be empty")
    var scores = ScoreTable(
        NDArray[DType.float64](Shape(len(param_range), len(cv), 1)),
        NDArray[DType.float64](Shape(len(param_range), len(cv), 1)),
        List[String](["score"]),
    )
    var scorers = List[A.Scorer]()
    scorers.append(scoring^)
    for fold in range(len(cv)):
        for candidate in range(len(param_range)):
            var model = clone(estimator)
            var params = Dict[String, A.Value]()
            params[param_name] = param_range[candidate].copy()
            # Configuration errors propagate even with numeric error_score.
            adapter.configure(model, params)
            _ = _evaluate_fold[A](model, X, y, adapter, scorers, cv[fold], fit_metadata, score_metadata, scores, candidate, fold, error_score)
    # Reshape materializes independent arrays in the adopted NuMojo version.
    return (scores.train_scores.value().reshape(Shape(len(param_range), len(cv))),
            scores.test_scores.reshape(Shape(len(param_range), len(cv))))


@fieldwise_init
struct LearningCurveResult(Movable):
    """Owned size vector and (training size, split) score/timing matrices."""
    var train_sizes: IndexArray
    var train_scores: NDArray[DType.float64]
    var test_scores: NDArray[DType.float64]
    var fit_times: Optional[NDArray[DType.float64]]
    var score_times: Optional[NDArray[DType.float64]]


def _learning_sizes[size_dtype: DType](train_sizes: Optional[NDArray[size_dtype]], maximum: Int) raises -> IndexArray:
    """Translate against the FIRST fold, then sort/deduplicate as sklearn does."""
    comptime assert size_dtype == DType.int64 or size_dtype == DType.float64, "train_sizes requires Int64 counts or Float64 fractions"
    if maximum <= 0:
        raise Error("ValueError: learning_curve requires a nonempty first training fold")
    var count = 5
    if train_sizes:
        if train_sizes.value().ndim != 1 or train_sizes.value().size == 0:
            raise Error("ValueError: train_sizes must be a nonempty vector")
        count = train_sizes.value().size
    var sizes = List[Int]()
    for i in range(count):
        var size: Int
        if not train_sizes:
            # np.linspace(0.1, 1.0, 5), including its floating-point rounding.
            var fraction = 0.1 + Float64(i) * 0.225
            if i == 4:
                fraction = 1.0
            size = max(1, Int(fraction * Float64(maximum)))
        else:
            comptime if size_dtype == DType.int64:
                size = Int(train_sizes.value()[Item(i)])
                if size <= 0 or size > maximum:
                    raise Error("ValueError: absolute train_sizes must be in [1, first training fold size]")
            else:
                var fraction = Float64(train_sizes.value()[Item(i)])
                if not isfinite(fraction) or fraction <= 0 or fraction > 1:
                    raise Error("ValueError: fractional train_sizes must be finite and in (0, 1]")
                size = max(1, Int(fraction * Float64(maximum)))
        sizes.append(size)
    sort(Span(sizes))
    var unique = 1
    for i in range(1, len(sizes)):
        if sizes[i] != sizes[i - 1]:
            unique += 1
    var result = IndexArray(Shape(unique))
    var position = 0
    for i in range(len(sizes)):
        if i == 0 or sizes[i] != sizes[i - 1]:
            result[Item(position)] = Int64(sizes[i])
            position += 1
    return result^


def learning_curve[A: CVAdapter, size_dtype: DType = DType.float64](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, var scoring: A.Scorer, cv: List[IndexSplit],
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    mut random_state: RandomState,
    *, train_sizes: Optional[NDArray[size_dtype]] = None,
    shuffle: Bool = False, return_times: Bool = False,
    exploit_incremental_learning: Bool = False,
    error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))),
) raises -> LearningCurveResult where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    """Serial dense learning curve with an explicitly borrowed RNG stream."""
    comptime if not conforms_to(A, IncrementalCVAdapter):
        if exploit_incremental_learning:
            raise Error("ValueError: incremental learning requires IncrementalCVAdapter")
    _checked_splits(cv, adapter.n_samples(X, y))
    var sizes = _learning_sizes[size_dtype](train_sizes, cv[0].train.size)
    var scores = ScoreTable(
        NDArray[DType.float64](Shape(sizes.size, len(cv), 1)),
        NDArray[DType.float64](Shape(sizes.size, len(cv), 1)),
        List[String](["score"]),
    )
    var fit_times: Optional[NDArray[DType.float64]] = None
    var score_times: Optional[NDArray[DType.float64]] = None
    if return_times:
        fit_times = NDArray[DType.float64](Shape(sizes.size, len(cv)))
        score_times = NDArray[DType.float64](Shape(sizes.size, len(cv)))
    # Shuffle each whole training fold exactly once, before any fits. All sizes
    # in that fold then use nested prefixes of the SAME permutation.
    var training = List[IndexArray]()
    if shuffle:
        for fold in range(len(cv)):
            var order = random_state.permutation(cv[fold].train.size)
            var rows = IndexArray(Shape(order.size))
            for i in range(order.size):
                rows[Item(i)] = cv[fold].train[Item(Int(order[Item(i)]))]
            training.append(rows^)
    var scorers = List[A.Scorer]()
    scorers.append(scoring^)
    var failed = 0
    for fold in range(len(cv)):
        if exploit_incremental_learning:
            comptime if conforms_to(A, IncrementalCVAdapter):
                var model = clone(estimator)
                var previous = 0
                for candidate in range(sizes.size):
                    var count = min(Int(sizes[Item(candidate)]), cv[fold].train.size)
                    var rows = IndexArray(Shape(count))
                    for i in range(count):
                        rows[Item(i)] = training[fold][Item(i)] if shuffle else cv[fold].train[Item(i)]
                    var chunk = IndexArray(Shape(count - previous))
                    for i in range(previous, count):
                        chunk[Item(i - previous)] = rows[Item(i)]
                    var X_train = adapter.input_rows(X, rows, rows)
                    var y_train = adapter.target_rows(y, rows)
                    var X_chunk = adapter.input_rows(X, chunk, chunk)
                    var y_chunk = adapter.target_rows(y, chunk)
                    var X_test = adapter.input_rows(X, cv[fold].test, rows)
                    var y_test = adapter.target_rows(y, cv[fold].test)
                    var start = perf_counter_ns()
                    var chunk_meta = adapter.metadata_rows(fit_metadata, chunk)
                    # Reference partial-fit failures propagate even with numeric error_score.
                    adapter.partial_fit(model, X_chunk, y_chunk, chunk_meta, y)
                    var fit_time = Float64(perf_counter_ns() - start) / 1.0e9
                    start = perf_counter_ns()
                    var test_meta = adapter.metadata_rows(score_metadata, cv[fold].test)
                    var train_meta = adapter.metadata_rows(score_metadata, rows)
                    scores.test_scores[Item(candidate, fold, 0)] = _score[A](scorers[0], model, X_test, y_test, test_meta, error_score)[0]
                    scores.train_scores.value()[Item(candidate, fold, 0)] = _score[A](scorers[0], model, X_train, y_train, train_meta, error_score)[0]
                    if return_times:
                        fit_times.value()[Item(candidate, fold)] = fit_time
                        score_times.value()[Item(candidate, fold)] = Float64(perf_counter_ns() - start) / 1.0e9
                    previous = count
                continue
        for candidate in range(sizes.size):
            # Python slicing truncates prefixes when later folds are shorter.
            var count = min(Int(sizes[Item(candidate)]), cv[fold].train.size)
            var rows = IndexArray(Shape(count))
            for i in range(count):
                rows[Item(i)] = training[fold][Item(i)] if shuffle else cv[fold].train[Item(i)]
            var split = IndexSplit(rows^, cv[fold].test.copy())
            var model = clone(estimator)
            var record = _evaluate_fold[A](model, X, y, adapter, scorers, split, fit_metadata, score_metadata, scores, candidate, fold, error_score)
            if record.failed:
                failed += 1
            if return_times:
                fit_times.value()[Item(candidate, fold)] = record.fit_time
                score_times.value()[Item(candidate, fold)] = record.score_time
    if failed == sizes.size * len(cv):
        raise Error("ValueError: all learning_curve fits failed")
    var shape = Shape(sizes.size, len(cv))
    return LearningCurveResult(sizes^, scores.train_scores.value().reshape(shape), scores.test_scores.reshape(shape), fit_times^, score_times^)


def learning_curve[A: CVAdapter, size_dtype: DType = DType.float64](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, var scoring: A.Scorer, cv: List[IndexSplit],
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    *, train_sizes: Optional[NDArray[size_dtype]] = None,
    shuffle: Bool = False, random_state: Optional[Int] = None,
    return_times: Bool = False, exploit_incremental_learning: Bool = False,
    error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))),
) raises -> LearningCurveResult where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
    """An integer seed reconstructs a stream for each call, like check_random_state."""
    if shuffle and not random_state:
        raise Error("ValueError: shuffle=True requires an integer seed or a mutable RandomState")
    var rng = RandomState(random_state.value() if shuffle and random_state else 0)
    return learning_curve[A, size_dtype](estimator, X, y, adapter, scoring^, cv, fit_metadata, score_metadata, rng, train_sizes=train_sizes, shuffle=shuffle, return_times=return_times, exploit_incremental_learning=exploit_incremental_learning, error_score=error_score)


trait PermutationCV:
    """Recompute splits from each permuted target; mutable splitter state is explicit."""
    comptime Adapter: CVAdapter
    def split(mut self, X: Self.Adapter.Scorer.Input, y: Self.Adapter.Scorer.Target, groups: Optional[SplitLabels]) raises -> List[IndexSplit]:
        ...


struct FixedPermutationCV[A: CVAdapter](PermutationCV):
    """A materialized split iterable, independent of the shuffled target."""
    comptime Adapter = Self.A
    var splits: List[IndexSplit]
    def __init__(out self, splits: List[IndexSplit]) raises:
        self.splits = List[IndexSplit]()
        for i in range(len(splits)):
            self.splits.append(splits[i].copy())
    def split(mut self, X: Self.A.Scorer.Input, y: Self.A.Scorer.Target, groups: Optional[SplitLabels]) raises -> List[IndexSplit]:
        var result = List[IndexSplit]()
        for i in range(len(self.splits)):
            result.append(self.splits[i].copy())
        return result^


@fieldwise_init
struct PermutationTestResult(Movable):
    var score: Float64
    var permutation_scores: NDArray[DType.float64]
    var pvalue: Float64


def _permutation_score[C: PermutationCV](
    estimator: C.Adapter.Scorer.Estimator, X: C.Adapter.Scorer.Input, y: C.Adapter.Scorer.Target,
    adapter: C.Adapter, scoring: C.Adapter.Scorer, mut cv: C,
    fit_metadata: C.Adapter.Scorer.Metadata, score_metadata: C.Adapter.Scorer.Metadata,
    groups: Optional[SplitLabels],
) raises -> Float64 where conforms_to(C.Adapter.Scorer.Input, Deinitable) and conforms_to(C.Adapter.Scorer.Target, Deinitable) and conforms_to(C.Adapter.Scorer.Metadata, Deinitable):
    # The reference clones ONCE per dataset, then refits that clone across folds.
    # In particular, replacing this with cross_val_score changes warm-start behavior.
    var model = clone(estimator)
    var splits = cv.split(X, y, groups)
    _checked_splits(splits, adapter.n_samples(X, y))
    var scores = NDArray[DType.float64](Shape(len(splits)))
    for fold in range(len(splits)):
        var X_train = adapter.input_rows(X, splits[fold].train, splits[fold].train)
        var X_test = adapter.input_rows(X, splits[fold].test, splits[fold].train)
        var y_train = adapter.target_rows(y, splits[fold].train)
        var y_test = adapter.target_rows(y, splits[fold].test)
        var fit_meta = adapter.metadata_rows(fit_metadata, splits[fold].train)
        var score_meta = adapter.metadata_rows(score_metadata, splits[fold].test)
        adapter.fit(model, X_train, y_train, fit_meta)
        scores[Item(fold)] = scoring.score(model, X_test, y_test, score_meta)
    return mean[returned_dtype=DType.float64](scores)


def _permutation_rows(n: Int, blocks: List[IndexArray], mut rng: RandomState) raises -> IndexArray:
    if len(blocks) == 0:
        return rng.permutation(n)
    var indices = IndexArray(Shape(n))
    for i in range(n):
        indices[Item(i)] = Int64(i)
    # Blocks are in sorted group order; positions retain original row order.
    for group in range(len(blocks)):
        var order = rng.permutation(blocks[group].size)
        for j in range(order.size):
            indices[Item(Int(blocks[group][Item(j)]))] = blocks[group][Item(Int(order[Item(j)]))]
    return indices^


def permutation_test_score[C: PermutationCV](
    estimator: C.Adapter.Scorer.Estimator, X: C.Adapter.Scorer.Input, y: C.Adapter.Scorer.Target,
    adapter: C.Adapter, scoring: C.Adapter.Scorer, mut cv: C,
    fit_metadata: C.Adapter.Scorer.Metadata, score_metadata: C.Adapter.Scorer.Metadata,
    mut random_state: RandomState, *, groups: Optional[SplitLabels] = None, n_permutations: Int = 100,
) raises -> PermutationTestResult where conforms_to(C.Adapter.Scorer.Input, Deinitable) and conforms_to(C.Adapter.Scorer.Target, Deinitable) and conforms_to(C.Adapter.Scorer.Metadata, Deinitable):
    if n_permutations < 1:
        raise Error("ValueError: n_permutations must be at least one")
    var n = adapter.n_samples(X, y)
    if n < 1:
        raise Error("ValueError: permutation_test_score requires samples")
    var blocks = List[IndexArray]()
    if groups:
        if len(groups.value()) != n:
            raise Error("ValueError: groups must match the sample count")
        var codes = groups.value().group_codes()
        var encoding = encode_column(codes, classification=False)
        for group in range(encoding.classes.size):
            var rows = IndexArray(Shape(Int(encoding.index.counts[Item(group)])))
            var position = 0
            for row in range(n):
                if encoding.index.inverse[Item(row)] == Int64(group):
                    rows[Item(position)] = Int64(row)
                    position += 1
            blocks.append(rows^)
    var score = _permutation_score[C](estimator, X, y, adapter, scoring, cv, fit_metadata, score_metadata, groups)
    var scores = NDArray[DType.float64](Shape(n_permutations))
    var better = 0
    for permutation in range(n_permutations):
        var rows = _permutation_rows(n, blocks, random_state)
        # Only targets move. X, groups and sample-aligned metadata keep their rows.
        var shuffled = adapter.target_rows(y, rows)
        var value = _permutation_score[C](estimator, X, shuffled, adapter, scoring, cv, fit_metadata, score_metadata, groups)
        scores[Item(permutation)] = value
        if value >= score:
            better += 1
    return PermutationTestResult(score, scores^, (Float64(better) + 1.0) / (Float64(n_permutations) + 1.0))


def permutation_test_score[C: PermutationCV](
    estimator: C.Adapter.Scorer.Estimator, X: C.Adapter.Scorer.Input, y: C.Adapter.Scorer.Target,
    adapter: C.Adapter, scoring: C.Adapter.Scorer, mut cv: C,
    fit_metadata: C.Adapter.Scorer.Metadata, score_metadata: C.Adapter.Scorer.Metadata,
    *, groups: Optional[SplitLabels] = None, n_permutations: Int = 100, random_state: Int = 0,
) raises -> PermutationTestResult where conforms_to(C.Adapter.Scorer.Input, Deinitable) and conforms_to(C.Adapter.Scorer.Target, Deinitable) and conforms_to(C.Adapter.Scorer.Metadata, Deinitable):
    var rng = RandomState(random_state)
    return permutation_test_score[C](estimator, X, y, adapter, scoring, cv, fit_metadata, score_metadata, rng, groups=groups, n_permutations=n_permutations)


def permutation_test_score[A: CVAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, scoring: A.Scorer, cv: List[IndexSplit],
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    mut random_state: RandomState, *, groups: Optional[SplitLabels] = None, n_permutations: Int = 100,
) raises -> PermutationTestResult where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable):
    var provider = FixedPermutationCV[A](cv)
    return permutation_test_score[FixedPermutationCV[A]](estimator, X, y, adapter, scoring, provider, fit_metadata, score_metadata, random_state, groups=groups, n_permutations=n_permutations)


def permutation_test_score[A: CVAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, scoring: A.Scorer, cv: List[IndexSplit],
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
    *, groups: Optional[SplitLabels] = None, n_permutations: Int = 100, random_state: Int = 0,
) raises -> PermutationTestResult where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable):
    var rng = RandomState(random_state)
    return permutation_test_score[A](estimator, X, y, adapter, scoring, cv, fit_metadata, score_metadata, rng, groups=groups, n_permutations=n_permutations)


trait StringPredictionAdapter(CVAdapter):
    def predict(self, estimator: Self.Scorer.Estimator, X: Self.Scorer.Input) raises -> StringTarget:
        ...


@fieldwise_init
struct ClassResponse[dtype: DType](Movable):
    """One array/class-code vector per output; is_list preserves sklearn's container."""
    var values: List[NDArray[Self.dtype]]
    var classes: List[IndexArray]
    var is_list: Bool


@fieldwise_init
struct CVClassPrediction[dtype: DType](Movable):
    var values: List[NDArray[Self.dtype]]
    var is_list: Bool

    def array(self) raises -> NDArray[Self.dtype]:
        """Owned snapshot of a single-array response; list results stay explicit."""
        if self.is_list or len(self.values) != 1:
            raise Error("ValueError: prediction contains a list of output arrays")
        return self.values[0].copy()


trait ClassPredictionAdapter(CVAdapter):
    comptime ResponseDType: DType
    def encode_target(self, y: Self.Scorer.Target) raises -> IndexArray:
        """Return globally sorted contiguous class codes, preserving target rank."""
        ...
    def fit_encoded(self, mut estimator: Self.Scorer.Estimator, X: Self.Scorer.Input, y: IndexArray, metadata: Self.Scorer.Metadata) raises:
        ...
    def response(self, estimator: Self.Scorer.Estimator, X: Self.Scorer.Input, method: String) raises -> ClassResponse[Self.ResponseDType]:
        """Classes are fitted global codes, never re-encoded within a fold."""
        ...


def encode_prediction_target[dtype: DType](y: NDArray[dtype]) raises -> IndexArray:
    """Reference LabelEncoder semantics through the shared label owner."""
    if y.ndim == 1:
        return encode_column(y).index.inverse.copy()
    if y.ndim != 2 or y.shape[1] == 0:
        raise Error("ValueError: class prediction requires nonempty target columns")
    var result = IndexArray(Shape(y.shape[0], y.shape[1]))
    for column in range(y.shape[1]):
        var encoding = encode_column(y, column)
        for row in range(y.shape[0]):
            result[Item(row, column)] = encoding.index.inverse[Item(row)]
    return result^


def encode_prediction_target(y: StringTarget) raises -> IndexArray:
    if y.rank() == 1:
        return encode_column(y).index.inverse.copy()
    if y.columns() == 0:
        raise Error("ValueError: class prediction requires nonempty target columns")
    var result = IndexArray(Shape(y.rows(), y.columns()))
    for column in range(y.columns()):
        var encoding = encode_column(y, column)
        for row in range(y.rows()):
            result[Item(row, column)] = encoding.index.inverse[Item(row)]
    return result^


def _scatter_prediction[dtype: DType](mut output: Optional[NDArray[dtype]], predicted: NDArray[dtype], rows: IndexArray, n: Int) raises:
    if (predicted.ndim != 1 and predicted.ndim != 2) or predicted.shape[0] != rows.size:
        raise Error("ValueError: prediction shape does not match held-out rows")
    if not output:
        output = NDArray[dtype](Shape(n) if predicted.ndim == 1 else Shape(n, predicted.shape[1]))
    if output.value().ndim != predicted.ndim:
        raise Error("ValueError: inconsistent prediction rank across folds")
    if predicted.ndim == 2 and output.value().shape[1] != predicted.shape[1]:
        raise Error("ValueError: inconsistent prediction columns across folds")
    for j in range(rows.size):
        var row = Int(rows[Item(j)])
        if predicted.ndim == 1:
            output.value()[Item(row)] = predicted[Item(j)]
        else:
            for col in range(predicted.shape[1]):
                output.value()[Item(row, col)] = predicted[Item(j, col)]


def cross_val_predict[A: StringPredictionAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata,
    *, method: String = "predict",
) raises -> StringTarget where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable):
    if method != "predict":
        raise Error("ValueError: string prediction adapter supports method='predict' only")
    var n = adapter.n_samples(X, y)
    _checked_prediction_splits(cv, n)
    var output: Optional[StringTarget] = None
    for fold in range(len(cv)):
        var X_train = adapter.input_rows(X, cv[fold].train, cv[fold].train)
        var X_test = adapter.input_rows(X, cv[fold].test, cv[fold].train)
        var y_train = adapter.target_rows(y, cv[fold].train)
        var metadata = adapter.metadata_rows(fit_metadata, cv[fold].train)
        var model = clone(estimator)
        adapter.fit(model, X_train, y_train, metadata)
        var predicted = adapter.predict(model, X_test)
        if predicted.rows() != cv[fold].test.size:
            raise Error("ValueError: string prediction shape does not match held-out rows")
        if not output:
            var values = List[String]()
            for i in range(n * predicted.columns()):
                values.append(String())
            if predicted.rank() == 1:
                output = StringTarget(values)
            else:
                output = StringTarget(values, n, predicted.columns())
        if output.value().rank() != predicted.rank() or output.value().columns() != predicted.columns():
            raise Error("ValueError: inconsistent string prediction shape across folds")
        for j in range(predicted.rows()):
            for column in range(predicted.columns()):
                output.value().set(Int(cv[fold].test[Item(j)]), predicted.get(j, column), column)
    return output.value().copy()


def _align_class_response[dtype: DType](classes: IndexArray, values: NDArray[dtype], n_classes: Int, method: String) raises -> NDArray[dtype]:
    comptime assert dtype == DType.float32 or dtype == DType.float64, "class responses require Float32 or Float64"
    if classes.ndim != 1 or (values.ndim != 1 and values.ndim != 2):
        raise Error("ValueError: invalid class response rank")
    for i in range(classes.size):
        var code = Int(classes[Item(i)])
        if code < 0 or code >= n_classes or (i > 0 and classes[Item(i - 1)] >= classes[Item(i)]):
            raise Error("ValueError: fitted classes must be sorted unique global class codes")
    var missing = classes.size != n_classes
    if method != "decision_function" or missing:
        if method == "decision_function" and classes.size <= 2:
            raise Error("ValueError: decision_function cannot align an imbalanced fold with at most two classes")
        if values.ndim != 2 or values.shape[1] != classes.size:
            raise Error("ValueError: irregular class response columns cannot be aligned")
    if not missing:
        # Includes full-class binary vectors and full-class OVO decision matrices.
        return values.copy()
    var fill = Float64(0).cast[dtype]()
    if method != "predict_proba":
        comptime if dtype == DType.float32:
            fill = bitcast[DType.float32](UInt32(0xff7fffff)).cast[dtype]()
        else:
            fill = bitcast[DType.float64](UInt64(0xffefffffffffffff)).cast[dtype]()
    var result = NDArray[dtype](Shape(values.shape[0], n_classes))
    for row in range(values.shape[0]):
        for column in range(n_classes):
            result[Item(row, column)] = fill
        for column in range(classes.size):
            result[Item(row, Int(classes[Item(column)]))] = values[Item(row, column)]
    return result^


def cross_val_predict[A: ClassPredictionAdapter](
    estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target,
    adapter: A, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata,
    *, method: String,
) raises -> CVClassPrediction[A.ResponseDType] where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable):
    if method != "predict_proba" and method != "predict_log_proba" and method != "decision_function":
        raise Error("ValueError: class prediction method must be predict_proba, predict_log_proba or decision_function")
    var n = adapter.n_samples(X, y)
    _checked_prediction_splits(cv, n)
    var encoded = adapter.encode_target(y)
    if (encoded.ndim != 1 and encoded.ndim != 2) or encoded.shape[0] != n:
        raise Error("ValueError: encoded targets must preserve rows and rank one or two")
    var columns = 1 if encoded.ndim == 1 else encoded.shape[1]
    if columns == 0:
        raise Error("ValueError: class prediction requires target columns")
    var counts = IndexArray(Shape(columns))
    for column in range(columns):
        var encoding = encode_column(encoded, column)
        counts[Item(column)] = Int64(encoding.classes.size)
        for i in range(encoding.classes.size):
            if encoding.classes[Item(i)] != Int64(i):
                raise Error("ValueError: encoded target classes must be contiguous from zero")
    var outputs = List[Optional[NDArray[A.ResponseDType]]]()
    var list_output = False
    for fold in range(len(cv)):
        var X_train = adapter.input_rows(X, cv[fold].train, cv[fold].train)
        var X_test = adapter.input_rows(X, cv[fold].test, cv[fold].train)
        var y_train = _rows(encoded, cv[fold].train)
        var metadata = adapter.metadata_rows(fit_metadata, cv[fold].train)
        var model = clone(estimator)
        adapter.fit_encoded(model, X_train, y_train, metadata)
        var predicted = adapter.response(model, X_test, method)
        var length = len(predicted.values)
        if length == 0 or len(predicted.classes) != length:
            raise Error("ValueError: response arrays and class vectors must be nonempty and aligned")
        if predicted.is_list:
            if encoded.ndim != 2 or length != columns:
                raise Error("ValueError: multioutput response count must match target columns")
        elif length != 1:
            raise Error("ValueError: single-array response must contain exactly one array")
        if fold == 0:
            list_output = predicted.is_list
            for _ in range(length):
                outputs.append(None)
        if list_output != predicted.is_list or len(outputs) != length:
            raise Error("ValueError: inconsistent response container across folds")
        for i in range(length):
            # A single matrix for 2-D targets denotes multilabel indicator outputs.
            var n_classes = Int(counts[Item(i)]) if list_output or encoded.ndim == 1 else columns
            var aligned = _align_class_response(predicted.classes[i], predicted.values[i], n_classes, method)
            _scatter_prediction(outputs[i], aligned, cv[fold].test, n)
    var values = List[NDArray[A.ResponseDType]]()
    for i in range(len(outputs)):
        values.append(outputs[i].value().copy())
    return CVClassPrediction[A.ResponseDType](values^, list_output)
