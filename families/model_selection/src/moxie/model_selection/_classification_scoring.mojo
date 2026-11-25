"""Typed adapters for classification label metrics."""
from moxie.base import Predictor
from moxie._labels import StringTarget
from moxie.metrics._classification import accuracy_score
from moxie.metrics._precision_recall import precision_score, recall_score, f1_score
from moxie.model_selection._evaluation import TypedScorer
from numojo import Item, NDArray, Shape, array


def _score_labels[dtype: DType, //, metric: StaticString](
    y: NDArray[dtype], predicted: NDArray[dtype], weights: Optional[NDArray[DType.float64]],
) raises -> Float64:
    comptime if metric == "accuracy":
        return accuracy_score(y, predicted, sample_weight=weights)
    elif metric == "precision":
        return precision_score(y, predicted, average="macro", sample_weight=weights)
    elif metric == "recall":
        return recall_score(y, predicted, average="macro", sample_weight=weights)
    else:
        return f1_score(y, predicted, average="macro", sample_weight=weights)


def _score_labels[Label: Hashable & Comparable & Copyable & Deinitable, //, metric: StaticString](
    y: List[Label], predicted: List[Label], weights: Optional[NDArray[DType.float64]],
) raises -> Float64:
    comptime if metric == "accuracy":
        return accuracy_score(y, predicted, sample_weight=weights)
    elif metric == "precision":
        return precision_score(y, predicted, average="macro", sample_weight=weights)
    elif metric == "recall":
        return recall_score(y, predicted, average="macro", sample_weight=weights)
    else:
        return f1_score(y, predicted, average="macro", sample_weight=weights)


def _score_labels[metric: StaticString](
    y: StringTarget, predicted: StringTarget, weights: Optional[NDArray[DType.float64]],
) raises -> Float64:
    comptime if metric == "accuracy":
        return accuracy_score(y, predicted, sample_weight=weights)
    elif metric == "precision":
        return precision_score(y, predicted, average="macro", sample_weight=weights)
    elif metric == "recall":
        return recall_score(y, predicted, average="macro", sample_weight=weights)
    else:
        return f1_score(y, predicted, average="macro", sample_weight=weights)


struct _ClassificationScorer[
    metric: StaticString,
    E: Predictor & Deinitable,
    XType: Movable & Deinitable,
    Y: Movable & Deinitable,
    weight_dtype: DType = DType.float64,
](TypedScorer, Copyable, Defaultable):
    comptime Estimator = Self.E
    comptime Input = Self.XType
    comptime Target = Self.Y
    comptime Metadata = Optional[NDArray[Self.weight_dtype]]

    def __init__(out self):
        comptime assert Self.Input == Self.E.Input, "Input must match the estimator"
        comptime assert Self.Target == Self.E.Prediction, "Target must match predict output"
        comptime assert Self.weight_dtype == DType.float32 or Self.weight_dtype == DType.float64, (
            "Expected Float32 or Float64 sample weights"
        )

    def __init__(out self, *, copy: Self):
        pass

    def score(
        self, estimator: Self.Estimator, X: Self.Input,
        y: Self.Target, metadata: Self.Metadata,
    ) raises -> Float64:
        var predicted = rebind_var[Self.Target](estimator.predict(rebind[Self.E.Input](X)))
        comptime if Self.weight_dtype == DType.float64:
            return Self._score(y, predicted, rebind[Optional[NDArray[DType.float64]]](metadata))
        else:
            var weights: Optional[NDArray[DType.float64]] = None
            if metadata:
                ref source = metadata.value()
                if source.ndim != 1:
                    raise Error("ValueError: sample weights must be a vector")
                var converted = NDArray[DType.float64](Shape(source.shape[0]))
                for row in range(source.shape[0]):
                    converted[Item(row)] = Float64(source[Item(row)])
                weights = converted^
            return Self._score(y, predicted, weights)

    @staticmethod
    def _score(
        y: Self.Target, predicted: Self.Target, weights: Optional[NDArray[DType.float64]],
    ) raises -> Float64:
        comptime if Self.Target == List[String]:
            return _score_labels[Self.metric](rebind[List[String]](y), rebind[List[String]](predicted), weights)
        elif Self.Target == List[Int]:
            return _score_labels[Self.metric](rebind[List[Int]](y), rebind[List[Int]](predicted), weights)
        elif Self.Target == StringTarget:
            return _score_labels[Self.metric](rebind[StringTarget](y), rebind[StringTarget](predicted), weights)
        else:
            comptime for i in range(3):
                comptime dtype = DType.int64 if i == 0 else DType.float32 if i == 1 else DType.float64
                comptime if Self.Target == NDArray[dtype]:
                    return _score_labels[Self.metric](rebind[NDArray[dtype]](y), rebind[NDArray[dtype]](predicted), weights)
                elif Self.Target == List[Scalar[dtype]]:
                    ref actual = rebind[List[Scalar[dtype]]](y)
                    ref estimates = rebind[List[Scalar[dtype]]](predicted)
                    return _score_labels[Self.metric](
                        array[dtype](actual, [len(actual)]), array[dtype](estimates, [len(estimates)]), weights,
                    )
            raise Error("ValueError: unsupported classification target representation")


comptime Accuracy[
    E: Predictor & Deinitable, X: Movable & Deinitable, Y: Movable & Deinitable,
    weight_dtype: DType = DType.float64,
] = _ClassificationScorer["accuracy", E, X, Y, weight_dtype]


comptime PrecisionMacro[
    E: Predictor & Deinitable, X: Movable & Deinitable, Y: Movable & Deinitable,
    weight_dtype: DType = DType.float64,
] = _ClassificationScorer["precision", E, X, Y, weight_dtype]


comptime RecallMacro[
    E: Predictor & Deinitable, X: Movable & Deinitable, Y: Movable & Deinitable,
    weight_dtype: DType = DType.float64,
] = _ClassificationScorer["recall", E, X, Y, weight_dtype]


comptime F1Macro[
    E: Predictor & Deinitable, X: Movable & Deinitable, Y: Movable & Deinitable,
    weight_dtype: DType = DType.float64,
] = _ClassificationScorer["f1", E, X, Y, weight_dtype]
