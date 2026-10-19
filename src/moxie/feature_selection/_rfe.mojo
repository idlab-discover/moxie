# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Dense recursive feature elimination; fitting stays in estimator adapters."""
import numojo as nm
from std.utils import Variant
from std.math import isfinite
from moxie.base import InputEstimator, FitTransformer, Predictor, SupervisedScorer, RegressorMixin, ClassifierMixin, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction
from ._base import SupervisedSelector
from ._selection import SelectionState, SupportResult
from ._selector_mixin import SelectorMixin, validate_selector_input
from ._model_importance import SelectionEstimator, SelectionClassLabels, aggregate_importances

comptime FeatureCount = Variant[Int, Float64]

@fieldwise_init
struct RFEParams[E: SelectionEstimator](Copyable, Movable):
    var estimator: Self.E
    var n_features_to_select: Optional[FeatureCount]
    var step: FeatureCount
    var verbose: Int
    var importance_getter: String


def _resolve_count(value: FeatureCount, width: Int, is_step: Bool) raises -> Int:
    if value.isa[Int]():
        var count = value[Int]
        if count <= 0:
            raise Error("ValueError: feature count and step must be positive")
        return count
    var fraction = value[Float64]
    if not isfinite(fraction) or fraction <= 0 or fraction > 1 or (is_step and fraction == 1):
        raise Error("ValueError: feature fraction must be in (0, 1]; step fraction must be in (0, 1)")
    var count = Int(Float64(width) * fraction)
    return max(1, count) if is_step else count


trait _RFEStepObserver(Movable):
    """Internal observation seam for one fitted point on an elimination path."""
    comptime Estimator: SelectionEstimator

    def observe(mut self, estimator: Self.Estimator,
                selection: SelectionState, ranking: List[Int]) raises:
        """Borrow pre-pruning state; copy any metadata retained by the observer."""
        ...


struct _IgnoreRFESteps[E: SelectionEstimator](_RFEStepObserver):
    comptime Estimator = Self.E

    def __init__(out self):
        pass

    def observe(mut self, estimator: Self.Estimator,
                selection: SelectionState, ranking: List[Int]):
        pass


struct RFE[E: SelectionEstimator](
    SupervisedSelector, FitTransformer, SelectorMixin, Copyable,
    Predictor where conforms_to(E, Predictor),
    RegressorMixin where conforms_to(E, RegressorMixin),
    ClassifierMixin where conforms_to(E, ClassifierMixin),
    SupervisedScorer where conforms_to(E, SupervisedScorer),
    ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
    DecisionFunction where conforms_to(E, DecisionFunction),
):
    comptime dtype = Self.E.dtype
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.E.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = RFEParams[Self.E]
    comptime Classes where conforms_to(Self.E, SelectionClassLabels) = Self.E.Classes
    comptime Prediction where conforms_to(Self.E, Predictor) = Self.E.Prediction
    comptime SampleWeight = Self.E.SampleWeight if conforms_to(Self.E, SupervisedScorer) else NoneType
    comptime Probabilities where conforms_to(Self.E, ProbabilityPredictor) = Self.E.Probabilities
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision

    var estimator: Self.E
    var n_features_to_select: Optional[FeatureCount]
    var step: FeatureCount
    var verbose: Int
    var importance_getter: String
    var _selection: SelectionState
    var _estimator: Optional[Self.E]
    var _ranking: List[Int]
    var _names: List[String]

    def __init__(out self, var estimator: Self.E, *, n_features_to_select: Optional[FeatureCount] = None, step: FeatureCount = FeatureCount(1), verbose: Int = 0, var importance_getter: String = "auto"):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime if conforms_to(Self.E, InputEstimator):
            comptime assert Self.E.Input == Self.Input, "RFE adapter inference must accept NDArray[dtype]"
        self.estimator = estimator^
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter^
        self._selection = SelectionState()
        self._estimator = None
        self._ranking = List[Int]()
        self._names = List[String]()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.Input) raises -> ref[self] Self:
        self.fit_named(X, y, List[String](), sample_weight.copy())
        return self

    def fit_named(mut self, X: Self.Input, y: Self.Target, names: List[String], sample_weight: Optional[Self.Input] = None) raises:
        var observer = _IgnoreRFESteps[Self.E]()
        self._fit_with_observer(X, y, observer, names, sample_weight)

    def _fit_with_observer[Observer: _RFEStepObserver](
        mut self, X: Self.Input, y: Self.Target, mut observer: Observer,
        names: List[String] = List[String](), sample_weight: Optional[Self.Input] = None,
    ) raises:
        """Visit fitted steps, including the final fit; commit only on success.

        Observer state is caller-owned and may contain a partial path on error.
        This method does not reset it, choose folds, or aggregate scores.
        """
        comptime assert Observer.Estimator == Self.E, "RFE observer estimator type must match the selector adapter"
        # The adapter validates finite values and its target representation.
        validate_selector_input(X, True)
        var width = X.shape[1]
        if width < 2:
            raise Error("ValueError: RFE requires at least two input features")
        if self.verbose < 0:
            raise Error("ValueError: verbose must be nonnegative")
        var target = width // 2
        if self.n_features_to_select:
            target = _resolve_count(self.n_features_to_select.value(), width, False)
        var stride = _resolve_count(self.step, width, True)
        var mask = List[Bool]()
        var ranking = List[Int]()
        for _ in range(width):
            mask.append(True)
            ranking.append(1)
        var selected = SelectionState()
        selected.commit(mask, names)
        var count = width
        while count > target:
            if self.verbose > 0:
                print("Fitting estimator with " + String(count) + " features.")
            var fitted = self.estimator.__sklearn_clone__()
            fitted.fit_selection(selected.transform(X), y.copy(), sample_weight)
            observer.observe(rebind[Observer.Estimator](fitted), selected, ranking)
            var raw = fitted.importances(self.importance_getter)
            if raw.ndim != 1 and raw.ndim != 2:
                raise Error("ValueError: importances must have rank one or two")
            var rows = 1 if raw.ndim == 1 else raw.shape[0]
            var columns = raw.shape[raw.ndim - 1]
            if columns != count or rows == 0:
                raise Error("ValueError: importance count does not match current feature count")
            for i in range(rows):
                for j in range(columns):
                    var offset = j * Int(raw.strides[raw.ndim - 1])
                    if raw.ndim == 2:
                        offset += i * Int(raw.strides[0])
                    if not isfinite(raw.unsafe_get(offset)):
                        raise Error("ValueError: feature importances must be finite")
            var scores = aggregate_importances(raw, square=True)
            var ordered = nm.sort(scores)
            var remove = min(stride, count - target)
            var cutoff = ordered.unsafe_get(remove - 1)
            var features = selected.get_support(True)[List[Int]].copy()
            var remaining = remove
            for j in range(count):
                if scores.unsafe_get(j) < cutoff:
                    mask[features[j]] = False
                    remaining -= 1
            # Equal squared importances are removed in original feature order.
            for j in range(count):
                if remaining > 0 and scores.unsafe_get(j) == cutoff:
                    mask[features[j]] = False
                    remaining -= 1
            for j in range(width):
                if not mask[j]:
                    ranking[j] += 1
            count -= remove
            selected.commit(mask, names)
        var fitted = self.estimator.__sklearn_clone__()
        fitted.fit_selection(selected.transform(X), y.copy(), sample_weight)
        if fitted.n_features() != count:
            raise Error("ValueError: fitted estimator has an inconsistent feature count")
        observer.observe(rebind[Observer.Estimator](fitted), selected, ranking)
        # Commit only after every fit and observer call succeed.
        self._selection = selected^
        self._estimator = fitted^
        self._ranking = ranking^
        self._names = names.copy()

    def _selection_state(self) raises -> SelectionState:
        self._selection.check()
        return self._selection.copy()

    def _allows_nan(self) -> Bool:
        return self.estimator.allows_nan()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        if not y:
            raise Error("ValueError: RFE requires y")
        _ = self.fit(X, y.value())
        return self.transform(X)

    def support_(self) raises -> List[Bool]:
        return self.get_support()[List[Bool]].copy()

    def ranking_(self) raises -> List[Int]:
        self._selection.check()
        return self._ranking.copy()

    def n_features_(self) raises -> Int:
        return self._selection.selected_count()

    def n_features_in_(self) raises -> Int:
        return self._selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._selection.check()
        return self._names.copy()

    def estimator_(self) raises -> Self.E:
        self._selection.check()
        return self._estimator.value().copy()

    def classes_(self) raises -> Self.Classes where conforms_to(Self.E, SelectionClassLabels):
        self._selection.check()
        return self._estimator.value().classes_()

    def predict(self, X: Self.Input) raises -> Self.Prediction where conforms_to(Self.E, Predictor):
        self._selection.check()
        return self._estimator.value().predict(rebind[Self.E.Input](self.transform(X)))

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64 where conforms_to(Self.E, SupervisedScorer):
        self._selection.check()
        return self._estimator.value().score(rebind[Self.E.Input](self.transform(X)), y, rebind[Optional[Self.E.SampleWeight]](sample_weight))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        self._selection.check()
        return self._estimator.value().predict_proba(rebind[Self.E.Input](self.transform(X)))

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        self._selection.check()
        return self._estimator.value().predict_log_proba(rebind[Self.E.Input](self.transform(X)))

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        self._selection.check()
        return self._estimator.value().decision_function(rebind[Self.E.Input](self.transform(X)))

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self.estimator.copy(), self.n_features_to_select, self.step, self.verbose, self.importance_getter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.n_features_to_select = params.n_features_to_select
        self.step = params.step
        self.verbose = params.verbose
        self.importance_getter = params.importance_getter.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_features_to_select: Optional[FeatureCount]) raises -> ref[self] Self:
        self.n_features_to_select = n_features_to_select
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, step: FeatureCount) raises -> ref[self] Self:
        self.step = step
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__(), n_features_to_select=self.n_features_to_select, step=self.step, verbose=self.verbose, importance_getter=self.importance_getter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._selection.is_fitted()

# RFECV shares the elimination implementation above. Fold generation, slicing,
# scoring contracts and score storage belong to model_selection.
from std.math import isnan, sqrt
from moxie._dense import sum_axis2d, row_squared_norms
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._results import ScoreTable
from ._sequential import SequentialAdapter, SequentialCV

comptime RFECVResultColumn = Variant[nm.NDArray[DType.float64], List[Int], List[List[Bool]], List[List[Int]]]
comptime RFECVResults = Dict[String, RFECVResultColumn]


@fieldwise_init
struct _RFECVStep(Copyable, Movable):
    var count: Int
    var score: Float64
    var support: List[Bool]
    var ranking: List[Int]


@fieldwise_init
struct _RFECVObserver[E: SelectionEstimator, S: TypedScorer & Copyable & Deinitable](_RFEStepObserver) where conforms_to(S.Target, Copyable & Deinitable) and conforms_to(S.Metadata, Copyable & Deinitable):
    comptime Estimator = Self.E
    var X: nm.NDArray[Self.E.dtype]
    var y: Self.S.Target
    var metadata: Self.S.Metadata
    var scorer: Self.S
    var steps: List[_RFECVStep]

    def observe(mut self, estimator: Self.E, selection: SelectionState, ranking: List[Int]) raises:
        var reduced = selection.transform(self.X)
        var value = self.scorer.score(rebind[Self.S.Estimator](estimator), rebind[Self.S.Input](reduced), self.y, self.metadata)
        self.steps.append(_RFECVStep(selection.selected_count(), value, selection.get_support()[List[Bool]].copy(), ranking.copy()))


@fieldwise_init
struct RFECVParams[A: SequentialAdapter, E: SelectionEstimator = A.Scorer.Estimator, S: TypedScorer & Copyable & Deinitable = A.Scorer](Copyable, Movable) where conforms_to(A.Scorer.Estimator, SelectionEstimator) and conforms_to(A.Scorer, TypedScorer & Copyable & Deinitable):
    var estimator: Self.E
    var adapter: Self.A
    var step: FeatureCount
    var min_features_to_select: Int
    var cv: SequentialCV
    var scoring: Optional[Self.S]
    var verbose: Int
    var n_jobs: Optional[Int]
    var importance_getter: String


struct RFECV[A: SequentialAdapter, E: SelectionEstimator = A.Scorer.Estimator, S: TypedScorer & Copyable & Deinitable = A.Scorer](
    SupervisedSelector, FitTransformer, SelectorMixin, SupervisedScorer, Copyable,
    Predictor where conforms_to(E, Predictor),
    RegressorMixin where conforms_to(E, RegressorMixin),
    ClassifierMixin where conforms_to(E, ClassifierMixin),
    ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
    DecisionFunction where conforms_to(E, DecisionFunction),
) where conforms_to(A.Scorer.Estimator, SelectionEstimator) and conforms_to(A.Scorer, TypedScorer & Copyable & Deinitable) and conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(S.Target, Copyable & Deinitable) and conforms_to(S.Metadata, Copyable & Deinitable):
    """Dense RFE paths scored on shared CV splits, followed by a full-data refit."""
    comptime dtype = Self.E.dtype
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.E.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime SampleWeight = Self.Input
    comptime Params = RFECVParams[Self.A, Self.E, Self.S]
    comptime Classes where conforms_to(Self.E, SelectionClassLabels) = Self.E.Classes
    comptime Prediction where conforms_to(Self.E, Predictor) = Self.E.Prediction
    comptime Probabilities where conforms_to(Self.E, ProbabilityPredictor) = Self.E.Probabilities
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision

    var estimator: Self.E
    var adapter: Self.A
    var step: FeatureCount
    var min_features_to_select: Int
    var cv: SequentialCV
    var scoring: Optional[Self.S]
    var verbose: Int
    var n_jobs: Optional[Int]
    var importance_getter: String
    var _fitted: Optional[RFE[Self.E]]
    var _results: RFECVResults

    def __init__(out self, var estimator: Self.E, var adapter: Self.A = Self.A(), *, step: FeatureCount = FeatureCount(1), min_features_to_select: Int = 1, var cv: SequentialCV = SequentialCV(None), scoring: Optional[Self.S] = None, verbose: Int = 0, n_jobs: Optional[Int] = None, var importance_getter: String = "auto"):
        comptime assert Self.S == Self.A.Scorer
        comptime assert Self.E == Self.S.Estimator
        comptime assert Self.Input == Self.S.Input, "RFECV requires dense scorer input"
        comptime assert Self.Target == Self.S.Target
        comptime assert Self.A.dtype == Self.dtype
        comptime assert Self.S.Metadata == Optional[Self.Input], "RFECV metadata is optional sample weights"
        self.estimator = estimator^
        self.adapter = adapter^
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv^
        self.scoring = scoring.copy()
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.importance_getter = importance_getter^
        self._fitted = None
        self._results = RFECVResults()

    def _scorer(self) raises -> Self.S:
        if self.scoring:
            return self.scoring.value().copy()
        var default = self.adapter.default_scorer()
        return rebind[Self.S](default).copy()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.Input) raises -> ref[self] Self:
        self.fit_named(X, y, List[String](), sample_weight=sample_weight.copy())
        return self

    def fit_named(mut self, X: Self.Input, y: Self.Target, names: List[String], *, sample_weight: Optional[Self.Input] = None, score_metadata: Optional[Self.Input] = None) raises:
        validate_selector_input(X, True)
        var width = X.shape[1]
        if width < 2:
            raise Error("ValueError: RFECV requires at least two features")
        if self.min_features_to_select <= 0:
            raise Error("ValueError: min_features_to_select must be positive")
        _ = _resolve_count(self.step, width, True)
        if self.verbose < 0:
            raise Error("ValueError: verbose must be nonnegative")
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs cannot be zero")
        var n = self.adapter.n_samples(rebind[Self.A.Scorer.Input](X), rebind[Self.A.Scorer.Target](y))
        if n != X.shape[0]:
            raise Error("ValueError: adapter sample count differs from X")
        for metadata in [sample_weight.copy(), score_metadata.copy()]:
            if metadata and (metadata.value().ndim != 1 or metadata.value().shape[0] != n):
                raise Error("ValueError: sample weights must be rank one with n_samples entries")
        var name_check = SelectionState()
        var all_features = List[Bool]()
        for _ in range(width):
            all_features.append(True)
        name_check.commit(all_features, names)
        var splits = self.cv.splits(n, self.adapter.cv_labels(rebind[Self.A.Scorer.Target](y)), self.adapter.is_classifier())
        var scorer = self._scorer()
        var paths = List[List[_RFECVStep]]()
        for fold in range(len(splits)):
            splits[fold].validate(n)
            var X_train = self.adapter.input_rows(rebind[Self.A.Scorer.Input](X), splits[fold].train, splits[fold].train)
            var X_test = self.adapter.input_rows(rebind[Self.A.Scorer.Input](X), splits[fold].test, splits[fold].train)
            var y_train = self.adapter.target_rows(rebind[Self.A.Scorer.Target](y), splits[fold].train)
            var y_test = self.adapter.target_rows(rebind[Self.A.Scorer.Target](y), splits[fold].test)
            var train_meta = self.adapter.metadata_rows(rebind[Self.A.Scorer.Metadata](sample_weight), splits[fold].train)
            var test_meta = self.adapter.metadata_rows(rebind[Self.A.Scorer.Metadata](score_metadata), splits[fold].test)
            var observer = _RFECVObserver[Self.E, Self.S](rebind[Self.Input](X_test).copy(), rebind[Self.S.Target](y_test).copy(), rebind[Self.S.Metadata](test_meta).copy(), scorer.copy(), List[_RFECVStep]())
            var path = RFE(self.estimator.__sklearn_clone__(), n_features_to_select=FeatureCount(min(width, self.min_features_to_select)), step=self.step, verbose=self.verbose, importance_getter=self.importance_getter.copy())
            path._fit_with_observer(rebind[Self.Input](X_train), rebind[Self.Target](y_train), observer, sample_weight=rebind[Optional[Self.Input]](train_meta))
            paths.append(observer.steps.copy())
        var count = len(paths[0])
        var table = ScoreTable(nm.zeros[DType.float64](nm.Shape(count, len(splits), 1)), None, [String("score")])
        var results = RFECVResults()
        var counts = List[Int]()
        for j in range(count):
            counts.append(paths[0][count - j - 1].count)
        for fold in range(len(splits)):
            var scores = nm.zeros[DType.float64](nm.Shape(count))
            var supports = List[List[Bool]]()
            var rankings = List[List[Int]]()
            for j in range(count):
                ref point = paths[fold][count - j - 1]
                table.test_scores.unsafe_set(j * len(splits) + fold, point.score)
                scores.unsafe_set(j, point.score)
                supports.append(point.support.copy())
                rankings.append(point.ranking.copy())
            results["split" + String(fold) + "_test_score"] = RFECVResultColumn(scores^)
            results["split" + String(fold) + "_support"] = RFECVResultColumn(supports^)
            results["split" + String(fold) + "_ranking"] = RFECVResultColumn(rankings^)
        var matrix = table.test_scores.reshape(nm.Shape(count, len(splits)))
        var totals = sum_axis2d(matrix, 1)
        var means = nm.zeros[DType.float64](nm.Shape(count))
        var centered = nm.zeros[DType.float64](nm.Shape(count, len(splits)))
        var best = 0
        for j in range(count):
            var total = totals.unsafe_get(j)
            # np.argmax selects the first NaN; ascending counts break ties.
            var previous = totals.unsafe_get(best)
            if not isnan(previous) and (isnan(total) or total > previous):
                best = j
            var mean = total / Float64(len(splits))
            means.unsafe_set(j, mean)
            for fold in range(len(splits)):
                centered.unsafe_set(j * len(splits) + fold, matrix.unsafe_get(j * len(splits) + fold) - mean)
        var deviations = row_squared_norms(centered)
        for j in range(count):
            deviations.unsafe_set(j, sqrt(deviations.unsafe_get(j) / Float64(len(splits))))
        results["mean_test_score"] = RFECVResultColumn(means^)
        results["std_test_score"] = RFECVResultColumn(deviations^)
        results["n_features"] = RFECVResultColumn(counts.copy())
        var fitted = RFE(self.estimator.__sklearn_clone__(), n_features_to_select=FeatureCount(counts[best]), step=self.step, verbose=self.verbose, importance_getter=self.importance_getter.copy())
        fitted.fit_named(X, y, names, sample_weight)
        # sklearn RFECV fits another fresh estimator after the full-data RFE.
        var final_estimator = self.estimator.__sklearn_clone__()
        final_estimator.fit_selection(fitted.transform(X), y.copy(), sample_weight)
        if final_estimator.n_features() != counts[best]:
            raise Error("ValueError: final estimator has an inconsistent feature count")
        fitted._estimator = final_estimator^
        self._fitted = fitted^
        self._results = results^

    def _check(self) raises:
        if not self._fitted:
            raise Error("NotFittedError: RFECV has not been fitted")

    def _selection_state(self) raises -> SelectionState:
        self._check()
        return self._fitted.value()._selection_state()

    def _allows_nan(self) -> Bool:
        return self.estimator.allows_nan()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        if not y:
            raise Error("ValueError: RFECV requires y")
        _ = self.fit(X, y.value())
        return self.transform(X)

    def support_(self) raises -> List[Bool]:
        return self._selection_state().get_support()[List[Bool]].copy()

    def ranking_(self) raises -> List[Int]:
        self._check()
        return self._fitted.value().ranking_()

    def n_features_(self) raises -> Int:
        return self._selection_state().selected_count()

    def n_features_in_(self) raises -> Int:
        return self._selection_state().n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._check()
        return self._fitted.value().feature_names_in_()

    def estimator_(self) raises -> Self.E:
        self._check()
        return self._fitted.value().estimator_()

    def cv_results_(self) raises -> RFECVResults:
        self._check()
        return self._results.copy()

    def classes_(self) raises -> Self.Classes where conforms_to(Self.E, SelectionClassLabels):
        self._check()
        return self._fitted.value().classes_()

    def predict(self, X: Self.Input) raises -> Self.Prediction where conforms_to(Self.E, Predictor):
        self._check()
        return self._fitted.value().predict(X)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        self._check()
        var scorer = self._scorer()
        var reduced = self.transform(X)
        return scorer.score(rebind[Self.S.Estimator](self._fitted.value()._estimator.value()), rebind[Self.S.Input](reduced), rebind[Self.S.Target](y), rebind[Self.S.Metadata](sample_weight))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        self._check()
        return self._fitted.value().predict_proba(X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        self._check()
        return self._fitted.value().predict_log_proba(X)

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        self._check()
        return self._fitted.value().decision_function(X)

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self.estimator.copy(), self.adapter.copy(), self.step, self.min_features_to_select, self.cv.copy(), self.scoring.copy(), self.verbose, self.n_jobs, self.importance_getter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.adapter = params.adapter.copy()
        self.step = params.step
        self.min_features_to_select = params.min_features_to_select
        self.cv = params.cv.copy()
        self.scoring = params.scoring.copy()
        self.verbose = params.verbose
        self.n_jobs = params.n_jobs
        self.importance_getter = params.importance_getter.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__(), self.adapter.copy(), step=self.step, min_features_to_select=self.min_features_to_select, cv=self.cv.copy(), scoring=self.scoring.copy(), verbose=self.verbose, n_jobs=self.n_jobs, importance_getter=self.importance_getter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._fitted)
