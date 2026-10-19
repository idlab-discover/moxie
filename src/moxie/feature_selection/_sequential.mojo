# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Greedy dense feature selection using the shared cross-validation engine."""
import numojo as nm
from std.utils import Variant
from std.math import inf, isfinite
from moxie.base import BaseEstimator, FitTransformer, clone
from moxie.model_selection._evaluation import TypedScorer
from moxie._dense import sum_axis2d
from moxie.model_selection._validation import CVAdapter, cross_val_score
from moxie.model_selection._indices import IndexArray, IndexSplit
from moxie.model_selection._split import SplitLabels, check_cv
from ._base import SupervisedSelector
from ._selection import SelectionState, SupportResult
from ._selector_mixin import SelectorMixin, validate_selector_input

comptime SequentialCount = Variant[String, Int, Float64]


trait SequentialAdapter(CVAdapter, Copyable, Deinitable):
    """Dense feature dtype plus estimator-specific CV defaults; no importances."""
    comptime dtype: DType

    def __init__(out self):
        ...

    def default_scorer(self) raises -> Self.Scorer:
        ...

    def empty_metadata(self) raises -> Self.Scorer.Metadata:
        ...

    def missing_target(self) raises -> Self.Scorer.Target:
        raise Error("ValueError: this estimator requires y")

    def cv_labels(self, y: Self.Scorer.Target) raises -> Optional[SplitLabels]:
        return None

    def is_classifier(self) -> Bool:
        return False

    def allows_nan(self) -> Bool:
        return False


struct SequentialCV(Copyable, Movable):
    """Owned CV configuration: fold count or materialized shared split pairs."""
    var n_splits: Optional[Int]
    # Copyable constructor configuration; IndexSplit itself is only Movable.
    var pairs: Optional[List[Tuple[IndexArray, IndexArray]]]

    @implicit
    def __init__(out self, n_splits: Optional[Int] = 5):
        self.n_splits = n_splits
        self.pairs = None

    def __init__(out self, pairs: List[IndexSplit]):
        self.n_splits = None
        var owned_pairs = List[Tuple[IndexArray, IndexArray]]()
        for i in range(len(pairs)):
            owned_pairs.append((pairs[i].train.copy(), pairs[i].test.copy()))
        self.pairs = owned_pairs^

    def splits(self, n: Int, labels: Optional[SplitLabels], classifier: Bool) raises -> List[IndexSplit]:
        if self.pairs:
            var pairs = List[IndexSplit]()
            for i in range(len(self.pairs.value())):
                pairs.append(IndexSplit(self.pairs.value()[i][0].copy(), self.pairs.value()[i][1].copy()))
            var checked = check_cv(pairs)
            var cursor = checked.split(n)
            var result = List[IndexSplit]()
            while True:
                var pair = cursor.next()
                if not pair:
                    break
                result.append(pair.value().copy())
            if len(result) == 0:
                raise Error("ValueError: cross-validation requires at least one split")
            return result^
        var checked = check_cv(self.n_splits)
        var cursor = checked.split(n)
        if labels:
            checked = check_cv(self.n_splits, labels.value(), classifier=classifier)
            cursor = checked.split(n, labels.value())
        var result = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            result.append(pair.value().copy())
        return result^


@fieldwise_init
struct SequentialParams[A: SequentialAdapter, E: BaseEstimator & Copyable & Deinitable = A.Scorer.Estimator, S: TypedScorer & Copyable & Deinitable = A.Scorer](Copyable, Movable) where conforms_to(A.Scorer, TypedScorer & Copyable & Deinitable) and conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Copyable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer.Estimator, BaseEstimator & Copyable & Deinitable):
    var estimator: Self.E
    var adapter: Self.A
    var n_features_to_select: SequentialCount
    var tol: Optional[Float64]
    var direction: String
    var scoring: Optional[Self.S]
    var cv: SequentialCV
    var n_jobs: Optional[Int]


struct SequentialFeatureSelector[A: SequentialAdapter, E: BaseEstimator & Copyable & Deinitable = A.Scorer.Estimator, S: TypedScorer & Copyable & Deinitable = A.Scorer](SupervisedSelector, FitTransformer, SelectorMixin, Copyable) where conforms_to(A.Scorer, TypedScorer & Copyable & Deinitable) and conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Copyable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer.Estimator, BaseEstimator & Copyable & Deinitable):
    comptime dtype = Self.A.dtype
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.A.Scorer.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SequentialParams[Self.A, Self.E, Self.S]
    var estimator: Self.E
    var adapter: Self.A
    var n_features_to_select: SequentialCount
    var tol: Optional[Float64]
    var direction: String
    var scoring: Optional[Self.S]
    var cv: SequentialCV
    var n_jobs: Optional[Int]
    var _selection: SelectionState
    var _names: List[String]

    def __init__(out self, var estimator: Self.E, var adapter: Self.A = Self.A(), *, n_features_to_select: SequentialCount = SequentialCount(String("auto")), tol: Optional[Float64] = None, direction: String = "forward", scoring: Optional[Self.S] = None, var cv: SequentialCV = SequentialCV(), n_jobs: Optional[Int] = None):
        comptime assert Self.E == Self.A.Scorer.Estimator
        comptime assert Self.S == Self.A.Scorer
        comptime assert Self.A.Scorer.Input == Self.Input, "Sequential selection requires dense NDArray input"
        self.estimator = estimator^
        self.adapter = adapter^
        self.n_features_to_select = n_features_to_select.copy()
        self.tol = tol
        self.direction = direction
        self.scoring = scoring.copy()
        self.cv = cv^
        self.n_jobs = n_jobs
        self._selection = SelectionState()
        self._names = List[String]()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input) raises -> ref[self] Self:
        _ = self.fit(X, self.adapter.missing_target())
        return self

    def fit_named(mut self, X: Self.Input, y: Self.Target, names: List[String], *, fit_metadata: Optional[Self.A.Scorer.Metadata] = None, score_metadata: Optional[Self.A.Scorer.Metadata] = None) raises:
        ref dense = rebind[nm.NDArray[Self.dtype]](X)
        validate_selector_input(dense, self.adapter.allows_nan())
        var width = dense.shape[1]
        if width < 2:
            raise Error("ValueError: SequentialFeatureSelector requires at least two features")
        if self.direction != "forward" and self.direction != "backward":
            raise Error("ValueError: direction must be forward or backward")
        if self.tol and (not isfinite(self.tol.value()) or (self.direction == "forward" and self.tol.value() < 0)):
            raise Error("ValueError: tol must be finite and nonnegative for forward selection")
        var automatic = self.n_features_to_select.isa[String]()
        var count: Int
        if automatic:
            if self.n_features_to_select[String] != "auto":
                raise Error("ValueError: n_features_to_select string must be auto")
            count = width - 1 if self.tol else width // 2
        elif self.n_features_to_select.isa[Int]():
            count = self.n_features_to_select[Int]
            if count <= 0 or count >= width:
                raise Error("ValueError: integer n_features_to_select must be positive and < n_features")
        else:
            var fraction = self.n_features_to_select[Float64]
            if not isfinite(fraction) or fraction <= 0 or fraction > 1:
                raise Error("ValueError: feature fraction must be in (0, 1]")
            count = Int(Float64(width) * fraction)
        var n = self.adapter.n_samples(rebind[Self.A.Scorer.Input](X), y)
        if n != dense.shape[0]:
            raise Error("ValueError: adapter sample count differs from X")
        var iterations = count if automatic or self.direction == "forward" else width - count
        var splits = List[IndexSplit]()
        if iterations > 0:
            if self.n_jobs and self.n_jobs.value() == 0:
                raise Error("ValueError: n_jobs cannot be zero")
            splits = self.cv.splits(n, self.adapter.cv_labels(y), self.adapter.is_classifier())
        elif not self.cv.pairs:
            # check_cv validates the count, but sklearn does not execute unused folds.
            _ = check_cv(self.cv.n_splits)
        var scorer: Self.S
        if self.scoring:
            scorer = self.scoring.value().copy()
        else:
            var default_scorer = self.adapter.default_scorer()
            scorer = rebind[Self.S](default_scorer).copy()
        var fit_meta = self.adapter.empty_metadata() if not fit_metadata else fit_metadata.value().copy()
        var score_meta = self.adapter.empty_metadata() if not score_metadata else score_metadata.value().copy()
        var estimator = clone(self.estimator)
        var mask = List[Bool]()
        for _ in range(width):
            mask.append(False)
        # Validate names before any fit; learned state is committed atomically.
        var selected = SelectionState()
        selected.commit(mask, names)
        var old_score = -inf[DType.float64]()
        for _ in range(iterations):
            var best = -1
            var best_score = -inf[DType.float64]()
            for j in range(width):
                if mask[j]:
                    continue
                var candidate_mask = mask.copy()
                candidate_mask[j] = True
                if self.direction == "backward":
                    for k in range(width):
                        candidate_mask[k] = not candidate_mask[k]
                selected.commit(candidate_mask, names)
                var candidate = selected.transform(dense)
                var scores = cross_val_score[Self.A](rebind[Self.A.Scorer.Estimator](estimator), rebind[Self.A.Scorer.Input](candidate^), y, self.adapter, rebind[Self.A.Scorer](scorer).copy(), splits, fit_meta, score_meta)
                var matrix = scores.reshape(nm.Shape(1, scores.shape[0]))
                var total = sum_axis2d(matrix, 1)
                var score = total.unsafe_get(0) / Float64(scores.shape[0])
                # Python max preserves the first candidate, including a first NaN.
                if best == -1 or score > best_score:
                    best = j
                    best_score = score
            if automatic and self.tol and best_score - old_score < self.tol.value():
                break
            mask[best] = True
            old_score = best_score
        if self.direction == "backward":
            for j in range(width):
                mask[j] = not mask[j]
        selected.commit(mask, names)
        self._selection = selected^
        self._names = names.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        if y:
            _ = self.fit(X, y.value())
        else:
            _ = self.fit(X)
        return self.transform(X)

    def _selection_state(self) raises -> SelectionState:
        self._selection.check()
        return self._selection.copy()

    def _allows_nan(self) -> Bool:
        return self.adapter.allows_nan()

    def support_(self) raises -> List[Bool]:
        return self._selection.get_support()[List[Bool]].copy()

    def n_features_to_select_(self) raises -> Int:
        return self._selection.selected_count()

    def n_features_in_(self) raises -> Int:
        return self._selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._selection.check()
        return self._names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.copy(), self.adapter.copy(), self.n_features_to_select.copy(), self.tol, self.direction, self.scoring.copy(), self.cv.copy(), self.n_jobs)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.adapter = params.adapter.copy()
        self.n_features_to_select = params.n_features_to_select.copy()
        self.tol = params.tol
        self.direction = params.direction.copy()
        self.scoring = params.scoring.copy()
        self.cv = params.cv.copy()
        self.n_jobs = params.n_jobs
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), self.adapter.copy(), n_features_to_select=self.n_features_to_select.copy(), tol=self.tol, direction=self.direction, scoring=self.scoring.copy(), cv=self.cv.copy(), n_jobs=self.n_jobs)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._selection.is_fitted()
