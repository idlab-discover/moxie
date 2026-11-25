"""Dense supervised adapters for sequential selection; no importance contract."""
import numojo as nm
from moxie.base import SupervisedScorer, WeightedSupervisedEstimator, ClassifierMixin
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._indices import IndexArray
from moxie.model_selection._split import _rows, SplitLabels
from ._sequential import SequentialAdapter
from ._univariate_scores import _validate_y
from moxie._labels import StringTarget


def _split_list_labels[Label: Copyable & Deinitable](y: List[Label]) raises -> SplitLabels:
    comptime if Label == String:
        return SplitLabels(rebind[List[String]](y))
    elif Label == Int:
        ref values = rebind[List[Int]](y)
        var labels = nm.zeros[DType.int64](nm.Shape(len(values)))
        for i in range(len(values)):
            labels[nm.Item(i)] = Int64(values[i])
        return SplitLabels(labels)
    elif Label == Int64:
        return SplitLabels(nm.array[DType.int64](rebind[List[Int64]](y), [len(y)]))
    elif Label == Float32:
        return SplitLabels(nm.array[DType.float32](rebind[List[Float32]](y), [len(y)]))
    else:
        comptime assert Label == Float64
        return SplitLabels(nm.array[DType.float64](rebind[List[Float64]](y), [len(y)]))


def _list_rows[Label: Copyable & Deinitable](y: List[Label], rows: IndexArray) raises -> List[Label]:
    var result = List[Label](capacity=rows.size)
    for i in range(rows.size):
        var row = Int(rows[nm.Item(i)])
        if row < 0 or row >= len(y):
            raise Error("ValueError: target row index is out of bounds")
        result.append(y[row].copy())
    return result^


struct EstimatorCVScore[E: SupervisedScorer & Copyable & Deinitable, dt: DType = DType.float64](TypedScorer, Copyable) where conforms_to(E.Target, Copyable & Deinitable):
    comptime Estimator = Self.E
    comptime Input = nm.NDArray[Self.dt]
    comptime Target: Copyable & Deinitable = Self.E.Target
    comptime Metadata = Optional[Self.Input]

    comptime ScoreFunction = def(Self.Estimator, Self.Input, Self.Target, Self.Metadata) thin raises -> Float64
    var _score: Optional[Self.ScoreFunction]

    def __init__(out self, score: Optional[Self.ScoreFunction] = None):
        self._score = score

    def score(self, estimator: Self.E, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        if self._score:
            var score = self._score.value()
            return score(estimator, X, y, metadata)
        return estimator.score(rebind[Self.E.Input](X), rebind[Self.E.Target](y), rebind[Optional[Self.E.SampleWeight]](metadata))


struct DenseSupervisedCV[E: SupervisedScorer & Copyable & Deinitable, dt: DType = DType.float64](SequentialAdapter) where conforms_to(E.Target, Copyable & Deinitable):
    comptime dtype = Self.dt
    comptime Scorer = EstimatorCVScore[Self.E, Self.dt]
    comptime _ListLabel = (
        String if Self.E.Target == List[String] else
        Int if Self.E.Target == List[Int] else
        Int64 if Self.E.Target == List[Int64] else
        Float32 if Self.E.Target == List[Float32] else
        Float64 if Self.E.Target == List[Float64] else NoneType
    )

    def __init__(out self):
        comptime assert Self.E.Input == Self.Scorer.Input
        comptime assert (
            Self._ListLabel != NoneType
            or Self.E.Target == StringTarget
            or Self.E.Target == nm.NDArray[DType.int64]
            or Self.E.Target == nm.NDArray[DType.float32]
            or Self.E.Target == nm.NDArray[DType.float64]
        ), "CV targets require StringTarget, Int64/Float32/Float64 arrays, or String/Int/Int64/Float32/Float64 lists"
        comptime assert Self.E.SampleWeight == Self.Scorer.Input

    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()

    def empty_metadata(self) -> Self.Scorer.Metadata:
        return None

    def is_classifier(self) -> Bool:
        return conforms_to(Self.E, ClassifierMixin)

    def cv_labels(self, y: Self.Scorer.Target) raises -> Optional[SplitLabels]:
        if self.is_classifier():
            comptime if Self._ListLabel != NoneType:
                return _split_list_labels(rebind[List[Self._ListLabel]](y))
            elif Self.Scorer.Target == StringTarget:
                return SplitLabels(rebind[StringTarget](y))
            elif Self.Scorer.Target == nm.NDArray[DType.int64]:
                return SplitLabels(rebind[nm.NDArray[DType.int64]](y))
            elif Self.Scorer.Target == nm.NDArray[DType.float32]:
                return SplitLabels(rebind[nm.NDArray[DType.float32]](y))
            else:
                return SplitLabels(rebind[nm.NDArray[DType.float64]](y))
        return None

    def n_samples(self, X: Self.Scorer.Input, y: Self.Scorer.Target) raises -> Int:
        comptime if Self._ListLabel != NoneType:
            if len(rebind[List[Self._ListLabel]](y)) != X.shape[0]:
                raise Error("ValueError: inconsistent sample counts")
        elif Self.Scorer.Target == StringTarget:
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
        comptime if Self._ListLabel != NoneType:
            var result = _list_rows(rebind[List[Self._ListLabel]](y), rows)
            return rebind[Self.Scorer.Target](result).copy()
        elif Self.Scorer.Target == StringTarget:
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
