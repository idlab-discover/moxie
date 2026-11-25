# SPDX-License-Identifier: BSD-3-Clause
"""Contract and generic bridge for dense classifiers used by calibration."""
import numojo as nm
from std.memory import ArcPointer
from std.math import isfinite
from ._classifier_targets import NumericListTarget
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

    def probability_sum_tolerance(self) -> Float64:
        """Return the absolute tolerance for a probability row to sum to one."""
        return 1e-8

    def is_frozen(self) -> Bool:
        return False


comptime MAX_INTEGRAL_LABEL = 9007199254740991


def validate_integral_labels(y: Array) raises:
    for i in range(y.size):
        var label = y[nm.Item(i)]
        if (
            not isfinite(label)
            or abs(label) > Float64(MAX_INTEGRAL_LABEL)
            or label != Float64(Int(label))
        ):
            raise Error(
                "ValueError: classification requires integral numeric"
                " labels"
            )


comptime _ClassReader[E: ClassifierMixin] = def(E) thin raises -> Array
comptime _Float32Array = nm.NDArray[DType.float32]


def _cast_calibration_array[source: DType, //, target: DType](
    values: nm.NDArray[source],
) raises -> nm.NDArray[target]:
    if values.ndim != 1 and values.ndim != 2:
        raise Error("ValueError: expected a calibration vector or matrix")
    if values.is_c_contiguous() or values.is_f_contiguous():
        return values.astype[target]()
    # NuMojo astype reads consecutive storage. Convert views by logical index.
    var result = nm.NDArray[target](values.shape)
    if values.ndim == 1:
        for i in range(values.size):
            result[nm.Item(i)] = values[nm.Item(i)].cast[target]()
    else:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                result[nm.Item(i, j)] = values.item(i, j).cast[target]()
    return result^


def _calibration_response[R: Movable](var values: R) raises -> Array:
    comptime if R == Array:
        return rebind_var[Array](values^)
    else:
        comptime assert R == _Float32Array, (
            "Calibration requires Float32 or Float64 responses"
        )
        var floats = rebind_var[_Float32Array](values^)
        return _cast_calibration_array[target=DType.float64](floats)


struct ClassifierCalibrationAdapter[E: ClassifierMixin & Deinitable](
    CalibrationEstimator,
    CalibrationDisplayEstimator where conforms_to(E, ProbabilityPredictor),
):
    """Adapt a Float64 classifier or a fitted frozen Float32 classifier.

    Supply a reader that returns the fitted classifier's classes in column
    order. Prefer `adapt_classifier` for classifiers with a built-in reader.
    Copies share a read-only estimator snapshot. Mutations replace the snapshot.
    """
    comptime Params = Self.E.Params
    comptime _domain = NumericListTarget[Self.E.Target]

    var _estimator: ArcPointer[Self.E]
    var _read_classes: _ClassReader[Self.E]
    var _name: String
    var _builtin_reader: Bool

    def __init__(
        out self,
        var estimator: Self.E,
        read_classes: _ClassReader[Self.E],
        var name: String,
    ):
        comptime assert Self.E.Input == Array or (
            conforms_to(Self.E, FrozenEstimatorAPI) and Self.E.Input == _Float32Array
        ), "Calibration requires Float64 features or a frozen Float32 classifier"
        comptime if not conforms_to(Self.E, FrozenEstimatorAPI):
            comptime assert (
                Self._domain.supported
                or Self.E.Target == nm.NDArray[DType.int64]
            ), "Calibration requires a numeric List or Int64 array target"
        comptime assert conforms_to(Self.E, DecisionFunction) or conforms_to(
            Self.E, ProbabilityPredictor
        ), "Calibration requires decision_function or predict_proba"
        self._estimator = ArcPointer(estimator^)
        self._read_classes = read_classes
        self._name = name^
        self._builtin_reader = False

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
        adapter._builtin_reader = self._builtin_reader
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
            if y.ndim != 1 or y.size == 0:
                raise Error("ValueError: expected a nonempty one-dimensional array")
            validate_integral_labels(y)
            var estimator = self._copy_for_update()
            comptime if Self._domain.supported:
                var labels = List[Self._domain.Label](capacity=y.size)
                for i in range(y.size):
                    var value = y[nm.Item(i)]
                    var label = Self._domain.from_array_label(value)
                    comptime if Self._domain.dtype == DType.float32:
                        if Self._domain.as_float64(label) != value:
                            raise Error("ValueError: label cannot be represented exactly as Float32")
                    labels.append(label)
                Self._fit_target(
                    estimator, X, rebind[Self.E.Target](labels), sample_weight
                )
            else:
                var labels = nm.NDArray[DType.int64](y.shape)
                for i in range(y.size):
                    labels[nm.Item(i)] = y[nm.Item(i)].cast[DType.int64]()
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

    def probability_sum_tolerance(self) -> Float64:
        comptime if conforms_to(Self.E, ProbabilityPredictor):
            comptime if Self.E.Probabilities == _Float32Array:
                # The Float32 row was normalized before widening. Match the
                # row-sum tolerance used to recognize probabilities in logits().
                return 1.001e-5
        return 1e-8

    def response(self, X: Array) raises -> Array:
        comptime if conforms_to(Self.E, DecisionFunction):
            var scores: Array
            comptime if Self.E.Input == _Float32Array:
                var features = _cast_calibration_array[target=DType.float32](X)
                scores = _calibration_response(
                    self._estimator[].decision_function(rebind[Self.E.Input](features))
                )
            else:
                scores = _calibration_response(
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
        comptime if Self.E.Input == _Float32Array:
            var features = _cast_calibration_array[target=DType.float32](X)
            return _calibration_response(
                self._estimator[].predict_proba(rebind[Self.E.Input](features))
            )
        else:
            return _calibration_response(
                self._estimator[].predict_proba(rebind[Self.E.Input](X))
            )

    def display_name(self) -> String:
        return self._name.copy()


def _adapt_linear_svc[Label: Copyable & Comparable & Deinitable](
    var estimator: LinearSVC[Label],
) -> ClassifierCalibrationAdapter[LinearSVC[Label]]:
    comptime Domain = NumericListTarget[List[Label]]

    def classes(model: LinearSVC[Label]) raises -> Array:
        var result = Array(nm.Shape(len(model.classes_)))
        for i in range(result.size):
            result[nm.Item(i)] = Domain.as_float64(
                rebind[Domain.Label](model.classes_[i])
            )
        validate_integral_labels(result)
        return result^

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "LinearSVC")
    adapter._builtin_reader = True
    return adapter^
