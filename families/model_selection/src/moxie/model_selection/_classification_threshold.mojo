# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed binary thresholds for native dense classifiers."""

from moxie._feature_metadata import InputFeatureCount
from moxie._labels import type_of_target
from moxie.base import (
    DecisionFunction, LogProbabilityPredictor, NumericClassifierClasses,
    ProbabilityPredictor, WeightedSupervisedEstimator, clone,
)
from moxie.metrics._classification import accuracy_score
from moxie.utils.validation import check_is_fitted
from numojo import Item, NDArray, Shape


@fieldwise_init
struct FixedThresholdParams[E: NumericClassifierClasses](Copyable):
    var estimator: Self.E.Params
    var threshold: Optional[Float64]
    var pos_label: Optional[Scalar[Self.E.ClassLabelDType]]
    var response_method: String


struct FixedThresholdClassifier[E: NumericClassifierClasses & Deinitable](
    NumericClassifierClasses,
    ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
    DecisionFunction where conforms_to(E, DecisionFunction),
    InputFeatureCount where conforms_to(E, InputFeatureCount),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
):
    """Apply a fixed threshold to an owned classifier's binary response.

    None represents the automatic threshold. Use a fitted estimator directly,
    or fit a clone of its configuration through this wrapper.
    """

    comptime Input = Self.E.Input
    comptime Target = Self.E.Target
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.E.SampleWeight
    comptime ClassLabelDType = Self.E.ClassLabelDType
    comptime _InputDType = DType.float32 if Self.Input == NDArray[DType.float32] else DType.float64
    comptime _Labels = NDArray[Self.ClassLabelDType]
    comptime _Label = Scalar[Self.ClassLabelDType]
    comptime _UsesList = Self.Target == List[Int]
    comptime Params = FixedThresholdParams[Self.E]
    comptime Probabilities where conforms_to(
        Self.E, ProbabilityPredictor
    ) = Self.E.Probabilities
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision

    var estimator: Self.E
    var estimator_: Optional[Self.E]
    var threshold: Optional[Float64]
    var pos_label: Optional[Self._Label]
    var response_method: String

    def __init__(
        out self, var estimator: Self.E, *,
        threshold: Optional[Float64] = None, pos_label: Optional[Self._Label] = None,
        var response_method: String = "auto",
    ):
        comptime assert Self.Target == Self._Labels or (
            Self._UsesList and Self.ClassLabelDType == DType.int64
        ), "Expected dense numeric targets or List[Int]"
        comptime assert Self.ClassLabelDType in (
            DType.int64, DType.float32, DType.float64,
        ), "Expected Int64, Float32 or Float64 labels"
        comptime assert Self.Input == NDArray[Self._InputDType], (
            "Expected Float32 or Float64 dense features"
        )
        comptime assert Self.SampleWeight == NDArray[DType.float64] or (
            Self.SampleWeight == NDArray[DType.float32]
        ), "Expected Float64 or Float32 array sample weights"
        comptime assert conforms_to(Self.E, ProbabilityPredictor) or conforms_to(
            Self.E, DecisionFunction
        ), "Expected a probability or decision response"
        self.estimator = estimator^
        self.estimator_ = None
        self.threshold = threshold
        self.pos_label = pos_label
        self.response_method = response_method^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.estimator.get_params(deep), self.threshold,
            self.pos_label, self.response_method.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.threshold = params.threshold
        self.pos_label = params.pos_label
        self.response_method = params.response_method.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, threshold: Optional[Float64] = None,
        pos_label: Optional[Self._Label] = None,
        response_method: Optional[String] = None,
    ) -> ref[self] Self:
        if threshold:
            self.threshold = threshold
        if pos_label:
            self.pos_label = pos_label
        if response_method:
            self.response_method = response_method.value().copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, var estimator: Self.E,
    ) -> ref[self] Self:
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            clone(self.estimator), threshold=self.threshold, pos_label=self.pos_label,
            response_method=self.response_method.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.estimator_) or self.estimator.__sklearn_is_fitted__()

    def _fitted_estimator(ref self) raises -> ref[self.estimator, self.estimator_.value()] Self.E:
        check_is_fitted(self)
        if self.estimator_:
            return self.estimator_.value()
        return self.estimator

    def get_classes(self) raises -> Self.Target:
        return self._fitted_estimator().get_classes()

    def classes_(self) raises -> Self.Target:
        return self.get_classes()

    def get_n_features_in(self) raises -> Int where conforms_to(
        Self.E, InputFeatureCount
    ):
        return self._fitted_estimator().get_n_features_in()

    def _validate_fit(self, X: Self.Input, y: Self.Target) raises:
        _ = self._uses_probability()
        var rows: Int
        comptime if Self._UsesList:
            ref values = rebind[List[Int]](y)
            rows = len(values)
            if rows > 0:
                var first = values[0]
                var second = first
                for value in values:
                    if value != first:
                        if second == first:
                            second = value
                        elif value != second:
                            raise Error("ValueError: only binary classification is supported")
        else:
            ref values = rebind[Self._Labels](y)
            if type_of_target(values) != "binary":
                raise Error("ValueError: only binary classification is supported")
            rows = values.shape[0]
        ref features = rebind[NDArray[Self._InputDType]](X)
        if features.ndim != 2 or features.shape[0] != rows:
            raise Error("ValueError: expected a dense matrix with one row per target")

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._validate_fit(X, y)
        var estimator = clone(self.estimator)
        _ = estimator.fit(X, y)
        self.estimator_ = estimator^
        return self

    def fit[
        origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin,
        weight_origin: ImmOrigin,
    ](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._validate_fit(X, y)
        var estimator = clone(self.estimator)
        _ = estimator.fit(X, y, sample_weight=sample_weight)
        self.estimator_ = estimator^
        return self

    def _uses_probability(self) raises -> Bool:
        if self.response_method == "auto":
            return conforms_to(Self.E, ProbabilityPredictor)
        if self.response_method == "predict_proba":
            return True
        if self.response_method == "decision_function":
            return False
        raise Error("ValueError: unknown response_method")

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(
        Self.E, ProbabilityPredictor
    ):
        return self._fitted_estimator().predict_proba(X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(
        Self.E, LogProbabilityPredictor
    ):
        return self._fitted_estimator().predict_log_proba(X)

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(
        Self.E, DecisionFunction
    ):
        return self._fitted_estimator().decision_function(X)

    def _binary_labels(self) raises -> Tuple[Self._Label, Self._Label]:
        comptime if Self._UsesList:
            var labels = rebind_var[List[Int]](self.get_classes())
            if len(labels) == 2:
                return (Self._Label(labels[0]), Self._Label(labels[1]))
        else:
            var labels = rebind_var[Self._Labels](self.get_classes())
            if labels.ndim == 1 and labels.size == 2:
                return (labels[Item(0)], labels[Item(1)])
        raise Error("ValueError: expected a fitted binary classifier with two classes")

    def _threshold_response[
        response_dtype: DType, probability: Bool,
    ](
        self, response: NDArray[response_dtype], positive_column: Int,
        negative_label: Self._Label, positive_label: Self._Label,
    ) raises -> Self.Prediction:
        comptime if probability:
            if response.ndim != 2 or response.shape[1] != 2:
                raise Error("ValueError: expected two probability columns")
        else:
            if response.ndim != 1:
                raise Error("ValueError: expected a rank-one binary decision response")

        var cutoff = Scalar[response_dtype](self.threshold.or_else(0.5 if probability else 0.0))
        var rows = response.shape[0]

        def label_at(row: Int) raises {imm response, cutoff, positive_column, negative_label, positive_label} -> Self._Label:
            var score: Scalar[response_dtype]
            comptime if probability:
                score = response[Item(row, positive_column)]
            else:
                score = response[Item(row)]
                if positive_column == 0:
                    score = -score
            return positive_label if score >= cutoff else negative_label

        comptime if Self._UsesList:
            var predicted = List[Int](capacity=rows)
            for row in range(rows):
                predicted.append(Int(label_at(row)))
            return rebind_var[Self.Prediction](predicted^)
        else:
            var predicted = Self._Labels(Shape(rows))
            for row in range(rows):
                predicted[Item(row)] = label_at(row)
            return rebind_var[Self.Prediction](predicted^)

    def _predict_from[
        R: Movable, probability: Bool,
    ](
        self, var response: R, positive_column: Int,
        negative_label: Self._Label, positive_label: Self._Label,
    ) raises -> Self.Prediction:
        comptime dtype = DType.float32 if R == NDArray[DType.float32] else DType.float64
        comptime assert R == NDArray[dtype], "Expected Float32 or Float64 response"
        return self._threshold_response[dtype, probability](
            rebind_var[NDArray[dtype]](response^), positive_column,
            negative_label, positive_label,
        )

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var first, second = self._binary_labels()
        var positive = self.pos_label.or_else(second)
        if positive != first and positive != second:
            raise Error("ValueError: pos_label is not a fitted class")
        var column = 0 if positive == first else 1
        var negative = second if column == 0 else first

        if self._uses_probability():
            comptime if conforms_to(Self.E, ProbabilityPredictor):
                return self._predict_from[Self.Probabilities, True](
                    self.predict_proba(X), column, negative, positive,
                )
        else:
            comptime if conforms_to(Self.E, DecisionFunction):
                return self._predict_from[Self.Decision, False](
                    self.decision_function(X), column, negative, positive,
                )
        raise Error("ValueError: estimator does not provide the requested response_method")

    def score(
        self, X: Self.Input, y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        comptime if Self.SampleWeight == NDArray[DType.float64]:
            return self._accuracy(
                y, self.predict(X), rebind[Optional[NDArray[DType.float64]]](sample_weight),
            )
        var weights: Optional[NDArray[DType.float64]] = None
        if sample_weight:
            ref weight = rebind[NDArray[DType.float32]](sample_weight.value())
            if weight.ndim != 1:
                raise Error("ValueError: sample_weight must be a vector")
            # NuMojo astype loads linearly, so widen through the logical stride.
            var widened = NDArray[DType.float64](weight.shape)
            for row in range(weight.size):
                widened[Item(row)] = Float64(weight[Item(row)])
            weights = widened^
        return self._accuracy(y, self.predict(X), weights)

    def _accuracy(
        self, y: Self.Target, var predicted: Self.Prediction,
        weights: Optional[NDArray[DType.float64]],
    ) raises -> Float64:
        comptime if Self._UsesList:
            return accuracy_score(
                rebind[List[Int]](y), rebind_var[List[Int]](predicted^), sample_weight=weights,
            )
        else:
            return accuracy_score(
                rebind[Self._Labels](y), rebind_var[Self._Labels](predicted^), sample_weight=weights,
            )
