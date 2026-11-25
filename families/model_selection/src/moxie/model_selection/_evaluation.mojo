"""Estimator-independent lifecycle seam; no CV evaluation engine."""
from moxie.base import BaseEstimator, Predictor, SupervisedScorer, clone
from numojo import Item, NDArray
from std.math import abs, isfinite, sqrt

trait TypedScorer(Movable, Deinitable):
    comptime Estimator: BaseEstimator & Deinitable
    comptime Input: Movable & Deinitable
    comptime Target: Movable & Deinitable
    comptime Metadata: Movable & Deinitable

    def score(self, estimator: Self.Estimator, X: Self.Input,
              y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        ...


struct EstimatorScore[
    E: SupervisedScorer & Deinitable,
    XType: Movable & Deinitable,
    YType: Movable & Deinitable,
    Weight: Movable & Deinitable = XType,
](TypedScorer, Copyable, Defaultable):
    """Delegate scoring to the estimator's default score method."""

    comptime Estimator = Self.E
    comptime Input = Self.XType
    comptime Target = Self.YType
    comptime Metadata = Optional[Self.Weight]

    def __init__(out self):
        comptime assert Self.Input == Self.E.Input, "Input must match the estimator"
        comptime assert Self.Target == Self.E.Target, "Target must match the estimator"
        comptime assert Self.Weight == Self.E.SampleWeight, "Weight must match the estimator"

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: Self.E, X: Self.Input, y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        return estimator.score(
            rebind[Self.E.Input](X),
            rebind[Self.E.Target](y),
            rebind[Optional[Self.E.SampleWeight]](metadata),
        )


def _dense_prediction[E: Predictor, X: Movable, dtype: DType](
    estimator: E, X_test: X,
) raises -> NDArray[dtype]:
    comptime assert E.Prediction == NDArray[dtype], (
        "Supply a prediction function when predict does not return the scorer's NDArray type"
    )
    return rebind_var[NDArray[dtype]](estimator.predict(rebind[E.Input](X_test)))


struct _NegativeRegressionLoss[
    loss: StaticString,
    E: Predictor & Deinitable,
    XType: Movable & Deinitable,
    dtype: DType = DType.float64,
    predict: def(E, XType) thin raises -> NDArray[dtype] = _dense_prediction[E, XType, dtype],
](TypedScorer, Copyable, Defaultable):
    """Negate a uniform-average loss over dense regression outputs."""

    comptime Estimator = Self.E
    comptime Input = Self.XType
    comptime Target = NDArray[Self.dtype]
    comptime Metadata = Optional[Self.Target]

    def __init__(out self):
        comptime assert Self.Input == Self.E.Input, "Input must match the estimator"
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, (
            "Expected Float32 or Float64 targets and predictions"
        )

    def __init__(out self, *, copy: Self):
        pass

    def score(
        self, estimator: Self.Estimator, X: Self.Input,
        y: Self.Target, metadata: Self.Metadata,
    ) raises -> Float64:
        var predicted = Self.predict(estimator, X)
        if y.ndim < 1 or y.ndim > 2 or predicted.ndim < 1 or predicted.ndim > 2:
            raise Error("ValueError: expected vector or matrix regression targets and predictions")
        var rows = y.shape[0]
        var outputs = 1 if y.ndim == 1 else y.shape[1]
        var predicted_outputs = 1 if predicted.ndim == 1 else predicted.shape[1]
        if rows == 0 or outputs == 0 or rows != predicted.shape[0] or outputs != predicted_outputs:
            raise Error("ValueError: expected matching nonempty regression targets and predictions")

        var weight_sum = Float64(rows)
        if metadata:
            ref weights = metadata.value()
            if weights.ndim != 1 or weights.shape[0] != rows:
                raise Error("ValueError: expected one sample weight per row")
            weight_sum = 0.0
            for row in range(rows):
                var weight = weights[Item(row)]
                if not isfinite(weight):
                    raise Error("ValueError: sample weights must be finite")
                weight_sum += Float64(weight)
            if weight_sum == 0.0:
                raise Error("ZeroDivisionError: weights sum to zero")

        var total: Float64 = 0.0
        for output in range(outputs):
            var error: Float64 = 0.0
            for row in range(rows):
                var observed = y[Item(row)] if y.ndim == 1 else y[Item(row, output)]
                var estimate = predicted[Item(row)] if predicted.ndim == 1 else predicted[Item(row, output)]
                if not isfinite(observed) or not isfinite(estimate):
                    raise Error("ValueError: regression targets and predictions must be finite")
                var residual = observed - estimate
                var term: Scalar[Self.dtype]
                comptime if Self.loss == "absolute":
                    term = abs(residual)
                else:
                    term = residual * residual
                if metadata:
                    term *= metadata.value()[Item(row)]
                error += Float64(term)
            var average = error / weight_sum
            comptime if Self.loss == "root_squared":
                average = sqrt(average)
            total += average
        return -(total / Float64(outputs))


# Negated mean squared error.
comptime NegMeanSquaredError[
    E: Predictor & Deinitable,
    XType: Movable & Deinitable,
    dtype: DType = DType.float64,
    predict: def(E, XType) thin raises -> NDArray[dtype] = _dense_prediction[E, XType, dtype],
] = _NegativeRegressionLoss["squared", E, XType, dtype, predict]


# Negated mean absolute error.
comptime NegMeanAbsoluteError[
    E: Predictor & Deinitable,
    XType: Movable & Deinitable,
    dtype: DType = DType.float64,
    predict: def(E, XType) thin raises -> NDArray[dtype] = _dense_prediction[E, XType, dtype],
] = _NegativeRegressionLoss["absolute", E, XType, dtype, predict]


# Negated root mean squared error, taking the root per output.
comptime NegRootMeanSquaredError[
    E: Predictor & Deinitable,
    XType: Movable & Deinitable,
    dtype: DType = DType.float64,
    predict: def(E, XType) thin raises -> NDArray[dtype] = _dense_prediction[E, XType, dtype],
] = _NegativeRegressionLoss["root_squared", E, XType, dtype, predict]


trait FitAdapter(Movable):
    """Explicit fit capability, including unsupervised/weighted adapters."""
    comptime Scorer: TypedScorer

    def fit(self, mut estimator: Self.Scorer.Estimator,
            X: Self.Scorer.Input, y: Self.Scorer.Target,
            metadata: Self.Scorer.Metadata) raises:
        ...


trait ParameterAdapter(FitAdapter):
    """Apply typed partial configuration; clone nested estimator values."""
    comptime Value: Copyable & Deinitable

    def configure(self, mut estimator: Self.Scorer.Estimator,
                  params: Dict[String, Self.Value]) raises:
        """Reject unknown names and retain no fitted state or caller aliases."""
        ...


def clone_fit_score[A: FitAdapter](
    estimator: A.Scorer.Estimator, adapter: A, scorer: A.Scorer,
    X_train: A.Scorer.Input, y_train: A.Scorer.Target,
    X_test: A.Scorer.Input, y_test: A.Scorer.Target,
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
) raises -> Float64:
    """Borrow inputs; fit an independent clone; propagate all errors."""
    var fitted = clone(estimator)
    adapter.fit(fitted, X_train, y_train, fit_metadata)
    return scorer.score(fitted, X_test, y_test, score_metadata)
