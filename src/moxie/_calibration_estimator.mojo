# SPDX-License-Identifier: BSD-3-Clause
"""Contract and generic bridge for dense classifiers used by calibration."""
import numojo as nm
from std.memory import ArcPointer
from ._calibration_algorithms import Array, vector
from ._calibration_display import CalibrationDisplayEstimator
from .base import (
    BaseEstimator,
    ClassifierMixin,
    DecisionFunction,
    ProbabilityPredictor,
    WeightedSupervisedEstimator,
    clone,
)
from .frozen._base import FrozenEstimatorAPI
from .svm._linear import LinearSVC
from .utils.validation import check_is_fitted


trait CalibrationEstimator(BaseEstimator, Copyable, Deinitable):
    """Adapter for a classifier's preferred response and class ordering.

    response returns a matrix: one decision column for binary classification,
    otherwise one column per class. Probability responses always have all classes.
    fit accepts weights; adapters without weighted fitting must reject supplied
    weights. class_labels must be sorted numeric labels after fitting.
    """

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        ...

    def response(self, X: Array) raises -> Array:
        ...

    def class_labels(self) raises -> Array:
        ...

    def uses_probabilities(self) -> Bool:
        ...

    def is_frozen(self) -> Bool:
        return False


comptime MAX_INTEGRAL_LABEL = 9007199254740991


def validate_integral_labels(y: Array) raises:
    for i in range(y.size):
        var label = y.item(i)
        if abs(label) > Float64(MAX_INTEGRAL_LABEL) or label != Float64(Int(label)):
            raise Error(
                "ValueError: classification requires integral numeric"
                " labels"
            )


comptime _ClassReader[E: ClassifierMixin] = def(E) thin raises -> Array


struct ClassifierCalibrationAdapter[E: ClassifierMixin & Deinitable](
    CalibrationEstimator,
    CalibrationDisplayEstimator where conforms_to(E, ProbabilityPredictor),
):
    """Adapt a Float64 classifier with numeric list or Int64 array targets.

    Supply a reader that returns the fitted classifier's classes in column
    order. Prefer `adapt_classifier` for classifiers with a built-in reader.
    Copies share a read-only estimator snapshot. Mutations replace the snapshot.
    """
    comptime Params = Self.E.Params

    var _estimator: ArcPointer[Self.E]
    var _read_classes: _ClassReader[Self.E]
    var _name: String
    var _builtin_linear_svc_reader: Bool

    def __init__(
        out self,
        var estimator: Self.E,
        read_classes: _ClassReader[Self.E],
        var name: String,
    ):
        comptime assert Self.E.Input == Array, "Calibration requires Float64 features"
        comptime assert (
            Self.E.Target == List[Int]
            or Self.E.Target == List[Float64]
            or Self.E.Target == nm.NDArray[DType.int64]
        ), "Calibration requires List[Int], List[Float64] or Int64 array targets"
        comptime assert conforms_to(Self.E, DecisionFunction) or conforms_to(
            Self.E, ProbabilityPredictor
        ), "Calibration requires decision_function or predict_proba"
        self._estimator = ArcPointer(estimator^)
        self._read_classes = read_classes
        self._name = name^
        self._builtin_linear_svc_reader = False

    def estimator(self) -> ref[self._estimator] Self.E:
        return self._estimator[]

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._estimator[].get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        var estimator = self._copy_for_update()
        _ = estimator.set_params(params)
        self._estimator = ArcPointer(estimator^)
        return self

    def _copy_for_update(self) raises -> Self.E:
        comptime if conforms_to(Self.E, Copyable):
            return self._estimator[].copy()
        else:
            return clone(self._estimator[])

    def __sklearn_clone__(self) raises -> Self:
        var adapter = Self(
            clone(self._estimator[]), self._read_classes, self._name.copy()
        )
        adapter._builtin_linear_svc_reader = self._builtin_linear_svc_reader
        return adapter^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._estimator[].__sklearn_is_fitted__()

    def is_frozen(self) -> Bool:
        return conforms_to(Self.E, FrozenEstimatorAPI)

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        comptime if conforms_to(Self.E, FrozenEstimatorAPI):
            check_is_fitted(self._estimator[])
        else:
            vector(y)
            validate_integral_labels(y)
            var estimator = self._copy_for_update()
            comptime if Self.E.Target == List[Int]:
                var labels = List[Int](capacity=y.size)
                for i in range(y.size):
                    labels.append(Int(y.item(i)))
                Self._fit_target(
                    estimator, X, rebind[Self.E.Target](labels), sample_weight
                )
            elif Self.E.Target == List[Float64]:
                var labels = List[Float64](capacity=y.size)
                for i in range(y.size):
                    labels.append(y.item(i))
                Self._fit_target(
                    estimator, X, rebind[Self.E.Target](labels), sample_weight
                )
            else:
                var labels = nm.NDArray[DType.int64](y.shape)
                for i in range(y.size):
                    labels[nm.Item(i)] = y.item(i).cast[DType.int64]()
                Self._fit_target(
                    estimator, X, rebind[Self.E.Target](labels), sample_weight
                )
            self._estimator = ArcPointer(estimator^)

    @staticmethod
    def _fit_target(
        mut estimator: Self.E,
        X: Array,
        y: Self.E.Target,
        sample_weight: Optional[Array],
    ) raises:
        ref features = rebind[Self.E.Input](X)
        if not sample_weight:
            _ = estimator.fit(features, y)
            return
        comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
            ref weight = sample_weight.value()
            comptime if Self.E.SampleWeight == Array:
                _ = estimator.fit(
                    features, y, sample_weight=rebind[Self.E.SampleWeight](weight)
                )
            else:
                comptime assert Self.E.SampleWeight == List[Float64], (
                    "Calibration requires Float64 array or list sample weights"
                )
                vector(weight)
                var values = List[Float64](capacity=weight.size)
                for i in range(weight.size):
                    values.append(weight.item(i))
                _ = estimator.fit(
                    features, y, sample_weight=rebind[Self.E.SampleWeight](values)
                )
        else:
            raise Error("ValueError: classifier does not support weighted fitting")

    def class_labels(self) raises -> Array:
        check_is_fitted(self._estimator[])
        return self._read_classes(self._estimator[])

    def uses_probabilities(self) -> Bool:
        return not conforms_to(Self.E, DecisionFunction)

    def response(self, X: Array) raises -> Array:
        comptime if conforms_to(Self.E, DecisionFunction):
            comptime assert Self.E.Decision == Array, (
                "Calibration requires Float64 decision scores"
            )
            var scores = rebind_var[Array](
                self._estimator[].decision_function(rebind[Self.E.Input](X))
            )
            if scores.ndim == 1:
                return scores.reshape(nm.Shape(scores.size, 1))
            return scores^
        elif conforms_to(Self.E, ProbabilityPredictor):
            return self.calibration_probabilities(X)
        else:
            comptime assert False, "Calibration requires decision_function or predict_proba"

    def calibration_probabilities(
        self, X: Array
    ) raises -> Array where conforms_to(Self.E, ProbabilityPredictor):
        comptime assert Self.E.Probabilities == Array, (
            "Calibration requires Float64 probabilities"
        )
        return rebind_var[Array](
            self._estimator[].predict_proba(rebind[Self.E.Input](X))
        )

    def display_name(self) -> String:
        return self._name.copy()


def _adapt_linear_svc(
    var estimator: LinearSVC[],
) -> ClassifierCalibrationAdapter[LinearSVC[]]:
    def classes(model: LinearSVC[]) raises -> Array:
        var result = Array(nm.Shape(len(model.classes_)))
        for i in range(result.size):
            result[nm.Item(i)] = model.classes_[i]
        validate_integral_labels(result)
        return result^

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "LinearSVC")
    adapter._builtin_linear_svc_reader = True
    return adapter^
