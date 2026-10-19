"""Explicit control-flow fixtures; these are not library estimators."""
import numojo as nm
from std.math import nan
from std.testing import assert_equal, assert_false, assert_true
from moxie.base import BaseEstimator
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._indices import IndexArray
from moxie.model_selection._split import _rows, SplitLabels
from moxie.feature_selection._sequential import SequentialAdapter

comptime Dense = nm.NDArray[DType.float64]

struct ProbeEstimator(BaseEstimator, Copyable):
    comptime Params = Int
    var failure: Int
    var fitted: Bool
    var features: Int
    def __init__(out self, failure: Int = 0):
        self.failure = failure
        self.fitted = False
        self.features = 0
    def get_params(self, deep: Bool = True) -> Int:
        return self.failure
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.failure = params
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self.failure)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted

struct ProbeScore(TypedScorer, Copyable):
    comptime Estimator = ProbeEstimator
    comptime Input = Dense
    comptime Target = Optional[Dense]
    comptime Metadata = Optional[Dense]
    var mode: Int
    def __init__(out self, mode: Int = 0):
        self.mode = mode
    def score(self, estimator: ProbeEstimator, X: Dense, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        assert_true(estimator.fitted)
        assert_equal(estimator.features, X.shape[1])
        if y:
            assert_equal(y.value().shape[0], X.shape[0])
        if metadata:
            assert_equal(metadata.value().shape[0], X.shape[0])
            # Routed weights encode their original row number plus one.
            for i in range(X.shape[0]):
                assert_equal(Int(metadata.value().unsafe_get(i)), Int(X.unsafe_get(i * X.shape[1])) % 100 + 1)
        if self.mode == 1:
            return -Float64(X.shape[1])
        if self.mode == 2:
            return Float64(X.shape[1])
        if self.mode == 3 and X.unsafe_get(0) < 100:
            raise Error("intentional score failure for the first candidate")
        if self.mode == 4:
            return nan[DType.float64]()
        return 0

struct ProbeAdapter(SequentialAdapter):
    comptime dtype = DType.float64
    comptime Scorer = ProbeScore
    var classification: Bool
    def __init__(out self):
        self.classification = False
    def is_classifier(self) -> Bool:
        return self.classification
    def default_scorer(self) -> ProbeScore:
        return ProbeScore()
    def empty_metadata(self) -> Optional[Dense]:
        return None
    def missing_target(self) -> Optional[Dense]:
        return None
    def cv_labels(self, y: Optional[Dense]) raises -> Optional[SplitLabels]:
        if self.classification and y:
            return SplitLabels(y.value())
        return None
    def n_samples(self, X: Dense, y: Optional[Dense]) raises -> Int:
        if y and y.value().shape[0] != X.shape[0]:
            raise Error("ValueError: target length mismatch")
        return X.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if y:
            return _rows(y.value(), rows)
        return None
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: ProbeEstimator, X: Dense, y: Optional[Dense], metadata: Optional[Dense]) raises:
        assert_false(estimator.fitted)
        if X.shape[1] == 0:
            raise Error("ValueError: fixture requires at least one feature")
        if estimator.failure == 2 or (estimator.failure == 1 and Int(X.unsafe_get(0)) % 100 == 0):
            raise Error("intentional fit failure")
        estimator.fitted = True
        estimator.features = X.shape[1]
        _ = ProbeScore().score(estimator, X, y, metadata)
