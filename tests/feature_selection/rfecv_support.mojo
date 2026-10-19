"""Typed scorer used only by RFECV tests and generated parity fixtures."""
import numojo as nm
from std.math import nan, inf
from moxie.base import SupervisedScorer
from moxie.feature_selection import SelectionEstimator, DenseSupervisedCV, SequentialAdapter
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._indices import IndexArray
from moxie.model_selection._split import SplitLabels

struct TestScore[E: SelectionEstimator & SupervisedScorer](TypedScorer, Copyable):
    comptime Estimator = Self.E
    comptime Input = nm.NDArray[Self.E.dtype]
    comptime Target = Self.E.Target
    comptime Metadata = Optional[Self.Input]
    var mode: Int
    def __init__(out self, mode: Int = 0):
        self.mode = mode
    def score(self, estimator: Self.E, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        if self.mode == 1:
            return 7
        if self.mode == 2:
            return Float64(X.shape[1])
        if self.mode == 3:
            raise Error("intentional scoring failure")
        if self.mode == 4:
            return nan[DType.float64]() if X.shape[1] == 3 else Float64(X.shape[1])
        if self.mode == 5:
            return inf[DType.float64]()
        if self.mode == 6:
            return -inf[DType.float64]()
        return estimator.score(rebind[Self.E.Input](X), y, rebind[Optional[Self.E.SampleWeight]](metadata))

# Bind the scorer view separately: both estimator traits declare Target, and
# the compiler needs evidence for the SupervisedScorer view used by dense CV.
struct TestCV[E: SelectionEstimator & SupervisedScorer, Scored: SupervisedScorer & Copyable & Deinitable = E](SequentialAdapter) where conforms_to(Scored.Target, Copyable & Deinitable):
    comptime dtype = Self.E.dtype
    comptime Scorer = TestScore[Self.E]
    comptime Dense = nm.NDArray[Self.dtype]
    var dense: DenseSupervisedCV[Self.Scored, Self.dtype]
    def __init__(out self):
        self.dense = DenseSupervisedCV[Self.Scored, Self.dtype]()
    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()
    def empty_metadata(self) -> Optional[Self.Dense]:
        return None
    def is_classifier(self) -> Bool:
        return self.dense.is_classifier()
    def cv_labels(self, y: Self.Scorer.Target) raises -> Optional[SplitLabels]:
        return self.dense.cv_labels(rebind[Self.Scored.Target](y))
    def n_samples(self, X: Self.Dense, y: Self.Scorer.Target) raises -> Int:
        return self.dense.n_samples(X, rebind[Self.Scored.Target](y))
    def input_rows(self, X: Self.Dense, rows: IndexArray, train: IndexArray) raises -> Self.Dense:
        return self.dense.input_rows(X, rows, train)
    def target_rows(self, y: Self.Scorer.Target, rows: IndexArray) raises -> Self.Scorer.Target:
        var result = self.dense.target_rows(rebind[Self.Scored.Target](y), rows)
        return rebind[Self.Scorer.Target](result).copy()
    def metadata_rows(self, metadata: Optional[Self.Dense], rows: IndexArray) raises -> Optional[Self.Dense]:
        return self.dense.metadata_rows(metadata, rows)
    def fit(self, mut estimator: Self.E, X: Self.Dense, y: Self.Scorer.Target, metadata: Optional[Self.Dense]) raises:
        estimator.fit_selection(X, y.copy(), metadata)

from std.testing import assert_equal, assert_true
from std.memory import ArcPointer
from moxie.base import ClassifierMixin, LogProbabilityPredictor, DecisionFunction
from moxie.feature_selection import SelectionClassLabels

struct RoutingSpy(SelectionEstimator, ClassifierMixin, SelectionClassLabels, LogProbabilityPredictor, DecisionFunction):
    """Control-flow spy with a deliberately shared trace; not a statistical model."""
    comptime dtype = DType.float64
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime SampleWeight = Self.Input
    comptime Prediction = Self.Input
    comptime Probabilities = Self.Input
    comptime Decision = Self.Input
    comptime Classes = List[Int]
    comptime Params = Int
    var trace: ArcPointer[Self.Input]
    var width: Int
    var fail_call: Int
    def __init__(out self, var trace: ArcPointer[Self.Input], fail_call: Int = 0):
        self.trace = trace^
        self.width = 0
        self.fail_call = fail_call
    def get_params(self, deep: Bool = True) -> Int:
        return self.fail_call
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.fail_call = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.trace.copy(), self.fail_call)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.width > 0
    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.fit_selection(X, y.copy(), None)
        return self

    def fit_selection(mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: Optional[Self.Input]) raises:
        assert_equal(self.width, 0)
        var call = Int(self.trace[].unsafe_get(0)) + 1
        self.trace[].unsafe_set(0, Float64(call))
        self.trace[].unsafe_set(call * 2 - 1, Float64(X.shape[0]))
        self.trace[].unsafe_set(call * 2, Float64(X.shape[1]))
        if call == self.fail_call:
            raise Error("intentional traced fit failure")
        assert_equal(y.value().shape[0], X.shape[0])
        var class_count = 0
        for i in range(X.shape[0]):
            class_count += Int(y.value().unsafe_get(i) == 42)
            if sample_weight:
                var row = Int(X.unsafe_get(i * X.shape[1])) // 10
                assert_equal(sample_weight.value().unsafe_get(i), Float64(row + 1))
        # With contiguous class blocks, plain KFold would violate this check.
        assert_equal(class_count * 2, X.shape[0])
        self.width = X.shape[1]
    def n_features(self) -> Int:
        return self.width
    def uses_l1_threshold(self) -> Bool:
        return False
    def feature_importances(self) raises -> Self.Input:
        return nm.ones[Self.dtype](nm.Shape(self.width))
    def predict(self, X: Self.Input) raises -> Self.Input:
        assert_equal(X.shape[1], self.width)
        return nm.zeros[Self.dtype](nm.Shape(X.shape[0]))
    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.Input] = None) raises -> Float64:
        assert_equal(X.shape[1], self.width)
        assert_equal(y.shape[0], X.shape[0])
        for i in range(X.shape[0]):
            if sample_weight:
                var row = Int(X.unsafe_get(i * X.shape[1])) // 10
                assert_equal(sample_weight.value().unsafe_get(i), Float64(row + 101))
        return -Float64(self.width)
    def predict_proba(self, X: Self.Input) raises -> Self.Input:
        assert_equal(X.shape[1], self.width)
        return nm.ones[Self.dtype](nm.Shape(X.shape[0], 2))
    def predict_log_proba(self, X: Self.Input) raises -> Self.Input:
        return self.predict_proba(X)
    def decision_function(self, X: Self.Input) raises -> Self.Input:
        return self.predict(X)
    def classes_(self) -> List[Int]:
        return [42, 7]
