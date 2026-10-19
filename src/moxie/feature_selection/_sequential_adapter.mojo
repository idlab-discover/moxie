"""Dense supervised adapters for sequential selection; no importance contract."""
import numojo as nm
from moxie.base import SupervisedScorer, WeightedSupervisedEstimator, ClassifierMixin
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._indices import IndexArray
from moxie.model_selection._split import _rows, SplitLabels
from ._sequential import SequentialAdapter
from ._univariate_scores import _validate_y
from moxie._labels import StringTarget

struct EstimatorCVScore[E: SupervisedScorer & Copyable & Deinitable, dt: DType = DType.float64](TypedScorer, Copyable) where conforms_to(E.Target, Copyable & Deinitable):
    comptime Estimator = Self.E
    comptime Input = nm.NDArray[Self.dt]
    comptime Target = Self.E.Target
    comptime Metadata = Optional[Self.Input]

    def __init__(out self):
        pass

    def score(self, estimator: Self.E, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        return estimator.score(rebind[Self.E.Input](X), rebind[Self.E.Target](y), rebind[Optional[Self.E.SampleWeight]](metadata))


struct DenseSupervisedCV[E: SupervisedScorer & Copyable & Deinitable, dt: DType = DType.float64](SequentialAdapter) where conforms_to(E.Target, Copyable & Deinitable):
    comptime dtype = Self.dt
    comptime Scorer = EstimatorCVScore[Self.E, Self.dt]

    def __init__(out self):
        comptime assert Self.E.Input == Self.Scorer.Input
        comptime assert Self.E.Target == StringTarget or Self.E.Target == nm.NDArray[DType.int64] or Self.E.Target == nm.NDArray[DType.float32] or Self.E.Target == nm.NDArray[DType.float64], "CV targets require StringTarget or Int64/Float32/Float64 arrays"
        comptime assert Self.E.SampleWeight == Self.Scorer.Input

    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()

    def empty_metadata(self) -> Self.Scorer.Metadata:
        return None

    def is_classifier(self) -> Bool:
        return conforms_to(Self.E, ClassifierMixin)

    def cv_labels(self, y: Self.Scorer.Target) raises -> Optional[SplitLabels]:
        if self.is_classifier():
            comptime if Self.Scorer.Target == StringTarget:
                return SplitLabels(rebind[StringTarget](y))
            elif Self.Scorer.Target == nm.NDArray[DType.int64]:
                return SplitLabels(rebind[nm.NDArray[DType.int64]](y))
            elif Self.Scorer.Target == nm.NDArray[DType.float32]:
                return SplitLabels(rebind[nm.NDArray[DType.float32]](y))
            else:
                return SplitLabels(rebind[nm.NDArray[DType.float64]](y))
        return None

    def n_samples(self, X: Self.Scorer.Input, y: Self.Scorer.Target) raises -> Int:
        comptime if Self.Scorer.Target == StringTarget:
            ref labels = rebind[StringTarget](y)
            if labels.rows() != X.shape[0] or labels.columns() == 0:
                raise Error("ValueError: inconsistent sample counts or empty target")
        else:
            comptime if Self.Scorer.Target == nm.NDArray[DType.int64]:
                _validate_y(rebind[nm.NDArray[DType.int64]](y), X.shape[0], multilabel=True)
            elif Self.Scorer.Target == nm.NDArray[DType.float32]:
                _validate_y(rebind[nm.NDArray[DType.float32]](y), X.shape[0], multilabel=True)
            else:
                _validate_y(rebind[nm.NDArray[DType.float64]](y), X.shape[0], multilabel=True)
        return X.shape[0]

    def input_rows(self, X: Self.Scorer.Input, rows: IndexArray, train: IndexArray) raises -> Self.Scorer.Input:
        return _rows(X, rows)

    def target_rows(self, y: Self.Scorer.Target, rows: IndexArray) raises -> Self.Scorer.Target:
        comptime if Self.Scorer.Target == StringTarget:
            ref labels = rebind[StringTarget](y)
            var values = List[String]()
            for i in range(rows.size):
                for j in range(labels.columns()):
                    values.append(labels.get(Int(rows[nm.Item(i)]), j))
            if labels.rank() == 1:
                var result = StringTarget(values)
                return rebind[Self.Scorer.Target](result).copy()
            var result = StringTarget(values, rows.size, labels.columns())
            return rebind[Self.Scorer.Target](result).copy()
        else:
            comptime if Self.Scorer.Target == nm.NDArray[DType.int64]:
                var result = _rows(rebind[nm.NDArray[DType.int64]](y), rows)
                return rebind[Self.Scorer.Target](result).copy()
            elif Self.Scorer.Target == nm.NDArray[DType.float32]:
                var result = _rows(rebind[nm.NDArray[DType.float32]](y), rows)
                return rebind[Self.Scorer.Target](result).copy()
            else:
                var result = _rows(rebind[nm.NDArray[DType.float64]](y), rows)
                return rebind[Self.Scorer.Target](result).copy()

    def metadata_rows(self, metadata: Self.Scorer.Metadata, rows: IndexArray) raises -> Self.Scorer.Metadata:
        if metadata:
            return _rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Self.E, X: Self.Scorer.Input, y: Self.Scorer.Target, metadata: Self.Scorer.Metadata) raises:
        if metadata:
            comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y), sample_weight=rebind[Self.E.SampleWeight](metadata.value()))
            else:
                raise Error("ValueError: estimator does not support sample weights")
        else:
            _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))
