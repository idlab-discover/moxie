"""Dense binary target bridge for classifiers using numeric List labels."""

import numojo as nm
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    ProbabilityPredictor,
    clone,
)
from moxie._multiclass_contracts import PairwiseInput, pairwise_input


struct BinaryAdapter[E: ClassifierMixin & Copyable & Deinitable](
    ClassifierMixin,
    Copyable,
    Movable,
    PairwiseInput,
    DecisionFunction where conforms_to(E, DecisionFunction),
    ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
):
    """Adapt rank-one Int64 binary codes to the classifier's numeric List labels.

    Use codes 0/1 for class indicators or -1/+1 for signed margins. Input
    features, sample weights, decisions and probabilities use the component's
    native types. The component is available as `estimator`.
    """

    comptime Params = Self.E.Params
    comptime Input = Self.E.Input
    comptime Target = nm.NDArray[DType.int64]
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.E.SampleWeight
    comptime Decision where conforms_to(Self.E, DecisionFunction) = Self.E.Decision
    comptime Probabilities where conforms_to(Self.E, ProbabilityPredictor) = Self.E.Probabilities

    comptime _label = (
        DType.float64 if Self.E.Target == List[Float64] else
        DType.float32 if Self.E.Target == List[Float32] else DType.int64
    )

    var estimator: Self.E

    def __init__(out self, var estimator: Self.E):
        comptime assert Self.E.Target == List[Scalar[Self._label]], (
            "BinaryAdapter requires List[Int64], List[Float32] or List[Float64] targets"
        )
        comptime assert Self.E.Prediction == Self.E.Target, (
            "BinaryAdapter requires prediction and target types to match"
        )
        self.estimator = estimator^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.estimator.get_params(deep=deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _ = self.estimator.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator))

    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()

    def pairwise_input(self) -> Bool:
        return pairwise_input(self.estimator)

    @staticmethod
    def _native_target(y: Self.Target) raises -> List[Scalar[Self._label]]:
        if y.ndim != 1:
            raise Error("ValueError: binary codes must be a rank-one array")
        var labels = List[Scalar[Self._label]](capacity=y.size)
        var signed = False
        var zero = False
        for i in range(y.size):
            var code = y[nm.Item(i)]
            if code < -1 or code > 1:
                raise Error("ValueError: expected binary codes 0/1 or -1/+1")
            signed = signed or code == -1
            zero = zero or code == 0
            labels.append(code.cast[Self._label]())
        if signed and zero:
            raise Error("ValueError: expected binary codes 0/1 or -1/+1")
        return labels^

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        var labels = Self._native_target(y)
        _ = self.estimator.fit(X, rebind[Self.E.Target](labels))
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var labels = rebind_var[List[Scalar[Self._label]]](self.estimator.predict(X))
        var result = Self.Prediction(nm.Shape(len(labels)))
        for i in range(len(labels)):
            var code = labels[i]
            if code != -1 and code != 0 and code != 1:
                raise Error("ValueError: component predicted a value outside the binary codes")
            result[nm.Item(i)] = code.cast[DType.int64]()
        return result^

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var labels = Self._native_target(y)
        return self.estimator.score(X, rebind[Self.E.Target](labels), sample_weight)

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        return self.estimator.decision_function(X)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        return self.estimator.predict_proba(X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        return self.estimator.predict_log_proba(X)
