"""Dense KernelRidge estimator and typed configuration."""
from std.math import isfinite
from std.utils import Variant
from moxie.base import RegressorMixin, WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
import numojo as nm
from ._api import (
    KernelParameters,
    ParameterUpdate,
    WeightRequest,
    KernelRidgeMethodRequest,
    KernelRidgeMetadataRequest,
    validate_request,
    unchanged,
)
from ._validation import validate, dense_cast
from ._kernels import pairwise, pairwise_callable, RowKernel, validate_kernel
from ._promotion import (
    FloatingArray,
    FittedArray,
    solve_promoted,
    prediction_product,
)


trait KernelRidgeRegressor(RegressorMixin):
    pass


comptime ScalarOrVector = Variant[Float64, List[Float64]]


struct NoKernel(Copyable, RowKernel):
    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        raise Error("ValueError: no native kernel supplied")


@fieldwise_init
struct KernelRidgeParams[Metric: RowKernel & Copyable & Deinitable = NoKernel](
    Copyable, Movable
):
    var alpha: ScalarOrVector
    var kernel: String
    var gamma: Optional[Float64]
    var degree: Float64
    var coef0: Float64
    var kernel_params: Optional[KernelParameters]
    var native_kernel: Optional[Self.Metric]


def _weights(
    n: Int,
    weights: Optional[ScalarOrVector],
    nonnegative: Bool = True,
) raises -> List[Float64]:
    var result = List[Float64]()
    var scalar: Float64 = 1
    if weights:
        if weights.value().isa[List[Float64]]():
            result = weights.value()[List[Float64]].copy()
            if len(result) != n:
                raise Error(
                    "ValueError: sample_weight length does not match samples"
                )
        else:
            scalar = weights.value()[Float64]
    if not weights or weights.value().isa[Float64]():
        for _ in range(n):
            result.append(scalar)
    for weight in result:
        if not isfinite(weight) or (nonnegative and weight < 0):
            raise Error(
                "ValueError: sample weights must be finite and nonnegative"
                " for fit"
            )
    return result^


struct KernelRidge[
    dtype: DType = DType.float64,
    Metric: RowKernel & Copyable & Deinitable = NoKernel,
](Copyable, KernelRidgeRegressor, WeightedSupervisedEstimator):
    """Dense mixed-dtype kernel ridge regression with shape-preserving outputs.

    Fit uses Cholesky, with minimum-norm least-squares fallback for shared alpha.
    See docs/family.md for supported kernels and explicit Mojo differences.
    """

    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = FloatingArray
    comptime SampleWeight = ScalarOrVector
    comptime ParameterRecord = KernelRidgeParams[Self.Metric]
    comptime ParameterValue = Variant[
        Float64,
        List[Float64],
        String,
        Optional[Float64],
        Optional[KernelParameters],
        Optional[Self.Metric],
    ]
    comptime ParameterMap = Dict[String, Self.ParameterValue]
    comptime Params = Self.ParameterMap

    var alpha: ScalarOrVector
    var kernel: String
    var gamma: Optional[Float64]
    var degree: Float64
    var coef0: Float64
    var kernel_params: Optional[KernelParameters]
    var native_kernel: Optional[Self.Metric]
    var dual_coef_: FloatingArray
    var X_fit_: FittedArray
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _metadata: KernelRidgeMetadataRequest
    var _fitted: Bool

    def __init__(
        out self,
        var alpha: ScalarOrVector = 1.0,
        *,
        var kernel: String = "linear",
        gamma: Optional[Float64] = None,
        degree: Float64 = 3.0,
        coef0: Float64 = 1.0,
        var kernel_params: Optional[KernelParameters] = None,
        var native_kernel: Optional[Self.Metric] = None,
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        )
        self.alpha = alpha^
        self.kernel = kernel^
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params^
        self.native_kernel = native_kernel^
        self.dual_coef_ = FloatingArray(nm.zeros[Self.dtype](nm.Shape(0)))
        self.X_fit_ = FittedArray(nm.zeros[Self.dtype](nm.Shape(0)))
        self.n_features_in_ = 0
        self._fitted = False
        self.feature_names_in_ = None
        self._metadata = KernelRidgeMetadataRequest(
            KernelRidgeMethodRequest(), KernelRidgeMethodRequest()
        )

    def __init__(
        out self,
        var alpha: ScalarOrVector = 1.0,
        *,
        var kernel: Self.Metric,
        gamma: Optional[Float64] = None,
        degree: Float64 = 3.0,
        coef0: Float64 = 1.0,
        var kernel_params: Optional[KernelParameters] = None,
    ) raises:
        self = Self(
            alpha^,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params^,
            native_kernel=kernel^,
        )

    def get_metadata_routing(self) -> KernelRidgeMetadataRequest:
        return self._metadata.copy()

    def set_fit_request[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        sample_weight: WeightRequest = Variant[Bool, String](
            String("$UNCHANGED$")
        ),
    ) raises -> ref[self] Self:
        validate_request(sample_weight)
        if not unchanged(sample_weight):
            self._metadata.fit.sample_weight = sample_weight.copy()
        return self

    def set_score_request[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        sample_weight: WeightRequest = Variant[Bool, String](
            String("$UNCHANGED$")
        ),
    ) raises -> ref[self] Self:
        validate_request(sample_weight)
        if not unchanged(sample_weight):
            self._metadata.score.sample_weight = sample_weight.copy()
        return self

    def set_fit_request[
        origin: MutOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        *,
        ref[sample_weight_origin] sample_weight: Bool,
    ) raises -> ref[self] Self:
        _ = self.set_fit_request(sample_weight=Variant[Bool, String](sample_weight))
        return self

    def set_fit_request[
        origin: MutOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        *,
        ref[sample_weight_origin] sample_weight: String,
    ) raises -> ref[self] Self:
        _ = self.set_fit_request(
            sample_weight=Variant[Bool, String](sample_weight.copy())
        )
        return self

    def set_score_request[
        origin: MutOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        *,
        ref[sample_weight_origin] sample_weight: Bool,
    ) raises -> ref[self] Self:
        _ = self.set_score_request(
            sample_weight=Variant[Bool, String](sample_weight)
        )
        return self

    def set_score_request[
        origin: MutOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        *,
        ref[sample_weight_origin] sample_weight: String,
    ) raises -> ref[self] Self:
        _ = self.set_score_request(
            sample_weight=Variant[Bool, String](sample_weight.copy())
        )
        return self

    def get_params_record(
        self, deep: Bool = True
    ) raises -> Self.ParameterRecord:
        return Self.ParameterRecord(
            self.alpha.copy(),
            self.kernel.copy(),
            self.gamma,
            self.degree,
            self.coef0,
            self.kernel_params.copy(),
            self.native_kernel.copy(),
        )

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        var result = Self.ParameterMap()
        if self.alpha.isa[Float64]():
            result["alpha"] = Self.ParameterValue(self.alpha[Float64])
        else:
            result["alpha"] = Self.ParameterValue(
                self.alpha[List[Float64]].copy()
            )
        if self.native_kernel and self.kernel == "linear":
            result["kernel"] = Self.ParameterValue(self.native_kernel.copy())
        else:
            result["kernel"] = Self.ParameterValue(self.kernel.copy())
        result["gamma"] = Self.ParameterValue(self.gamma)
        result["degree"] = Self.ParameterValue(self.degree)
        result["coef0"] = Self.ParameterValue(self.coef0)
        result["kernel_params"] = Self.ParameterValue(self.kernel_params.copy())
        result["native_kernel"] = Self.ParameterValue(self.native_kernel.copy())
        return result^

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        var updated = self.get_params_record()
        for entry in params.items():
            var name = entry.key.copy()
            var value = entry.value.copy()
            if name == "alpha" and value.isa[Float64]():
                updated.alpha = ScalarOrVector(value[Float64])
            elif name == "alpha" and value.isa[List[Float64]]():
                updated.alpha = ScalarOrVector(value[List[Float64]].copy())
            elif name == "kernel" and value.isa[String]():
                updated.kernel = value[String].copy()
                updated.native_kernel = None
            elif name == "kernel" and value.isa[Optional[Self.Metric]]():
                if not value[Optional[Self.Metric]]:
                    raise Error("ValueError: kernel must not be None")
                updated.kernel = "linear"
                updated.native_kernel = value[Optional[Self.Metric]].copy()
            elif name == "gamma" and value.isa[Float64]():
                updated.gamma = value[Float64]
            elif name == "gamma" and value.isa[Optional[Float64]]():
                updated.gamma = value[Optional[Float64]]
            elif name == "degree" and value.isa[Float64]():
                updated.degree = value[Float64]
            elif name == "coef0" and value.isa[Float64]():
                updated.coef0 = value[Float64]
            elif (
                name == "kernel_params"
                and value.isa[Optional[KernelParameters]]()
            ):
                updated.kernel_params = value[
                    Optional[KernelParameters]
                ].copy()
            elif name == "native_kernel" and value.isa[Optional[Self.Metric]]():
                updated.native_kernel = value[Optional[Self.Metric]].copy()
            else:
                raise Error(
                    "ValueError: unknown parameter or invalid parameter type: "
                    + name
                )
        _ = self.set_params(updated)
        return self

    def set_params[
        origin: MutOrigin,
        kernel_origin: ImmOrigin,
    ](
        ref[origin] self,
        *,
        ref[kernel_origin] kernel: Self.Metric,
        alpha: Optional[ScalarOrVector] = None,
        gamma: ParameterUpdate[Float64] = ParameterUpdate[Float64](),
        degree: Optional[Float64] = None,
        coef0: Optional[Float64] = None,
        kernel_params: ParameterUpdate[KernelParameters] = ParameterUpdate[
            KernelParameters
        ](),
    ) raises -> ref[self] Self:
        _ = self.set_params(
            alpha=alpha,
            kernel=String("linear"),
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            native_kernel=kernel.copy(),
        )
        return self

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.ParameterRecord,
    ) raises -> ref[self] Self:
        self.alpha = params.alpha.copy()
        self.kernel = params.kernel.copy()
        self.gamma = params.gamma
        self.degree = params.degree
        self.coef0 = params.coef0
        self.kernel_params = params.kernel_params.copy()
        self.native_kernel = params.native_kernel.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        alpha: Optional[ScalarOrVector] = None,
        kernel: Optional[String] = None,
        gamma: ParameterUpdate[Float64] = ParameterUpdate[Float64](),
        degree: Optional[Float64] = None,
        coef0: Optional[Float64] = None,
        kernel_params: ParameterUpdate[KernelParameters] = ParameterUpdate[
            KernelParameters
        ](),
        native_kernel: ParameterUpdate[Self.Metric] = ParameterUpdate[
            Self.Metric
        ](),
    ) raises -> ref[self] Self:
        """Update supplied values; explicit None resets optional parameters."""
        if alpha:
            self.alpha = alpha.value().copy()
        if kernel:
            self.kernel = kernel.value().copy()
            self.native_kernel = None
        if gamma.supplied:
            self.gamma = gamma.value.copy()
        if degree:
            self.degree = degree.value()
        if coef0:
            self.coef0 = coef0.value()
        if kernel_params.supplied:
            self.kernel_params = kernel_params.value.copy()
        if native_kernel.supplied:
            self.native_kernel = native_kernel.value.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(
            self.alpha.copy(),
            kernel=self.kernel.copy(),
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params.copy(),
            native_kernel=self.native_kernel.copy(),
        )

        result._metadata = self._metadata.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _validate_kernel_config(self) raises:
        validate_kernel(self.kernel)
        if self.native_kernel and self.kernel != "linear":
            raise Error(
                "ValueError: native_kernel requires the default kernel name"
            )

    def _validate_params(self) raises:
        self._validate_kernel_config()
        if self.gamma:
            if not isfinite(self.gamma.value()) or self.gamma.value() < 0:
                raise Error("ValueError: gamma must be finite and nonnegative")
        if not isfinite(self.degree) or self.degree < 0:
            raise Error("ValueError: degree must be finite and nonnegative")
        if not isfinite(self.coef0):
            raise Error("ValueError: coef0 must be finite")

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, sample_weight.copy())
        return self

    def fit[
        xdtype: DType,
        ydtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[xdtype],
        ref[y_origin] y: nm.NDArray[ydtype],
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[
        xdtype: DType,
        ydtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[xdtype],
        ref[y_origin] y: nm.NDArray[ydtype],
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, sample_weight.copy())
        return self

    # Borrowed trait inputs cannot implicitly convert scalar/list weights to
    # the variant. Keep the concrete call forms accepted before fluent returns.
    def fit[
        xdtype: DType,
        ydtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[xdtype],
        ref[y_origin] y: nm.NDArray[ydtype],
        *,
        sample_weight: Float64,
    ) raises -> ref[self] Self:
        self._fit(X, y, ScalarOrVector(sample_weight))
        return self

    def fit[
        xdtype: DType,
        ydtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[xdtype],
        ref[y_origin] y: nm.NDArray[ydtype],
        *,
        ref[sample_weight_origin] sample_weight: List[Float64],
    ) raises -> ref[self] Self:
        self._fit(X, y, ScalarOrVector(sample_weight.copy()))
        return self

    def fit[
        xdtype: DType,
        ydtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        names_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[xdtype],
        ref[y_origin] y: nm.NDArray[ydtype],
        *,
        ref[names_origin] feature_names: List[String],
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        self._check_names_width(X, feature_names)
        self._fit(X, y, sample_weight)
        self.feature_names_in_ = feature_names.copy()
        return self

    def _check_names_width[
        xdtype: DType
    ](self, X: nm.NDArray[xdtype], names: List[String]) raises:
        if X.ndim != 2 or len(names) != X.shape[1]:
            raise Error("ValueError: feature_names length must match X columns")

    def _check_names[
        xdtype: DType
    ](self, X: nm.NDArray[xdtype], names: List[String]) raises:
        check_is_fitted(self)
        self._check_names_width(X, names)
        if self.feature_names_in_:
            var fitted = self.feature_names_in_.value().copy()
            if len(fitted) != len(names):
                raise Error(
                    "ValueError: feature names must match those passed"
                    " during fit"
                )
            for i in range(len(names)):
                if fitted[i] != names[i]:
                    raise Error(
                        "ValueError: feature names must match those passed"
                        " during fit, in the same order"
                    )

    def predict[
        xdtype: DType
    ](
        self, X: nm.NDArray[xdtype], *, feature_names: List[String]
    ) raises -> Self.Prediction:
        self._check_names(X, feature_names)
        return self._predict(X)

    def score[
        xdtype: DType, ydtype: DType
    ](
        self,
        X: nm.NDArray[xdtype],
        y: nm.NDArray[ydtype],
        *,
        feature_names: List[String],
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        self._check_names(X, feature_names)
        return self._score(X, y, sample_weight)

    def _fit[
        xdtype: DType, ydtype: DType
    ](
        mut self,
        X: nm.NDArray[xdtype],
        y: nm.NDArray[ydtype],
        weights: Optional[ScalarOrVector],
    ) raises:
        self._validate_params()
        validate(X, "X", True)
        validate(y, "y")
        if X.shape[0] != y.shape[0]:
            raise Error(
                "ValueError: X and y have inconsistent numbers of samples"
            )
        var penalties = List[Float64]()
        if self.alpha.isa[Float64]():
            penalties.append(self.alpha[Float64])
        else:
            penalties = self.alpha[List[Float64]].copy()
        if len(penalties) != 1 and len(penalties) != (
            y.shape[1] if y.ndim == 2 else 1
        ):
            raise Error(
                "ValueError: alpha must be scalar, length one, or one value per"
                " target"
            )
        for penalty in penalties:
            if not isfinite(penalty) or penalty < 0:
                raise Error("ValueError: alpha must be finite and nonnegative")
        var sample_weights = _weights(X.shape[0], weights)
        var K = self._get_kernel(X, X, True)
        var has_weights = False
        if weights:
            has_weights = weights.value().isa[List[Float64]]()
            if not has_weights:
                has_weights = weights.value()[Float64] != 1.0
        var coefficients: FloatingArray
        if K.isa[nm.NDArray[DType.float32]]():
            coefficients = solve_promoted(
                K[nm.NDArray[DType.float32]],
                y,
                penalties,
                sample_weights,
                has_weights,
            )
        else:
            coefficients = solve_promoted(
                K[nm.NDArray[DType.float64]],
                y,
                penalties,
                sample_weights,
                has_weights,
            )
        var snapshot = X.copy()
        self.dual_coef_ = coefficients^
        self.X_fit_ = FittedArray(snapshot^)
        self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        self._fitted = True

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self._predict(X)

    def predict[
        xdtype: DType
    ](self, X: nm.NDArray[xdtype]) raises -> Self.Prediction:
        return self._predict(X)

    def _predict[
        xdtype: DType
    ](self, X: nm.NDArray[xdtype]) raises -> Self.Prediction:
        check_is_fitted(self)
        validate(X, "X", True)
        if X.shape[1] != self.n_features_in_:
            raise Error(
                "ValueError: X feature count does not match fitted data"
            )
        comptime types = (
            DType.float32,
            DType.float64,
            DType.int32,
            DType.int64,
        )
        comptime for i in range(4):
            comptime fitted_dtype = types[i]
            if self.X_fit_.isa[nm.NDArray[fitted_dtype]]():
                return self._predict_fitted(
                    X, self.X_fit_[nm.NDArray[fitted_dtype]]
                )
        raise Error("ValueError: invalid fitted input dtype")

    def _predict_fitted[
        xdtype: DType, ydtype: DType
    ](
        self, X: nm.NDArray[xdtype], fitted: nm.NDArray[ydtype]
    ) raises -> Self.Prediction:
        var K = self._get_kernel(X, fitted)
        comptime types = (DType.float32, DType.float64)
        comptime for k in range(2):
            comptime kd = types[k]
            if K.isa[nm.NDArray[kd]]():
                comptime for c in range(2):
                    comptime cd = types[c]
                    if self.dual_coef_.isa[nm.NDArray[cd]]():
                        return prediction_product(
                            K[nm.NDArray[kd]], self.dual_coef_[nm.NDArray[cd]]
                        )
        raise Error("ValueError: invalid fitted dtype")

    def _get_kernel[
        xdtype: DType, ydtype: DType
    ](
        self,
        X: nm.NDArray[xdtype],
        Y: nm.NDArray[ydtype],
        same_data: Bool = False,
    ) raises -> FloatingArray:
        self._validate_kernel_config()
        comptime kernel_dtype = DType.float32 if xdtype == DType.float32 and ydtype == DType.float32 else DType.float64
        if self.kernel == "precomputed":
            if X.shape[1] != Y.shape[0]:
                raise Error(
                    "ValueError: precomputed columns must match training rows"
                )
            # sklearn's pairwise validation considers both input dtypes here.
            return FloatingArray(dense_cast[kernel_dtype](X))
        if self.kernel == "laplacian":
            return FloatingArray(
                pairwise(
                    dense_cast[DType.float64](X),
                    dense_cast[DType.float64](Y),
                    self.kernel,
                    self.gamma,
                    self.degree,
                    self.coef0,
                    same_data,
                )
            )
        var features = dense_cast[kernel_dtype](X)
        var other = dense_cast[kernel_dtype](Y)
        if self.native_kernel:
            return FloatingArray(
                pairwise_callable(
                    features,
                    other,
                    self.native_kernel.value(),
                    same_data,
                    self.kernel_params,
                )
            )
        return FloatingArray(
            pairwise(
                features,
                other,
                self.kernel,
                self.gamma,
                self.degree,
                self.coef0,
                same_data,
            )
        )

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        return self._score(X, y, sample_weight)

    def score[
        xdtype: DType, ydtype: DType
    ](
        self,
        X: nm.NDArray[xdtype],
        y: nm.NDArray[ydtype],
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        return self._score(X, y, sample_weight)

    def _score[
        xdtype: DType, ydtype: DType
    ](
        self,
        X: nm.NDArray[xdtype],
        y: nm.NDArray[ydtype],
        sample_weight: Optional[Self.SampleWeight],
    ) raises -> Float64:
        var output = self.predict(X)
        if output.isa[nm.NDArray[DType.float32]]():
            return self._score_prediction(
                output[nm.NDArray[DType.float32]], y, sample_weight
            )
        return self._score_prediction(
            output[nm.NDArray[DType.float64]], y, sample_weight
        )

    def _score_prediction[
        pdtype: DType, ydtype: DType
    ](
        self,
        predicted: nm.NDArray[pdtype],
        y: nm.NDArray[ydtype],
        sample_weight: Optional[Self.SampleWeight],
    ) raises -> Float64:
        validate(y, "y")
        validate(predicted, "predictions")
        if y.shape[0] != predicted.shape[0] or (
            y.shape[1] if y.ndim == 2 else 1
        ) != (predicted.shape[1] if predicted.ndim == 2 else 1):
            raise Error("ValueError: y shape does not match predictions")
        var weights = _weights(y.shape[0], sample_weight, False)
        var mass: Float64 = 0
        for weight in weights:
            mass += weight
        # Reject all-zero weights before the single-sample NaN shortcut,
        # matching sklearn's metric validation order.
        if mass == 0:
            raise Error("ValueError: score weights must not sum to zero")
        if y.shape[0] < 2:
            return Float64("nan")
        var total_score: Float64 = 0
        for target in range((y.shape[1] if y.ndim == 2 else 1)):
            var mean: Float64 = 0
            for i in range(y.shape[0]):
                mean += weights[i] * Float64(
                    y.unsafe_get(
                        i * Int(y.strides[0])
                        + (target * Int(y.strides[1]) if y.ndim == 2 else 0)
                    )
                )
            mean /= mass
            var numerator: Float64 = 0
            var denominator: Float64 = 0
            for i in range(y.shape[0]):
                var value = Float64(
                    y.unsafe_get(
                        i * Int(y.strides[0])
                        + (target * Int(y.strides[1]) if y.ndim == 2 else 0)
                    )
                )
                numerator += (
                    weights[i]
                    * (
                        value
                        - Float64(
                            predicted.unsafe_get(
                                i * (y.shape[1] if y.ndim == 2 else 1) + target
                            )
                        )
                    )
                    ** 2
                )
                denominator += weights[i] * (value - mean) ** 2
            if numerator == 0:
                total_score += 1
            elif denominator != 0:
                total_score += 1.0 - numerator / denominator
        return total_score / Float64((y.shape[1] if y.ndim == 2 else 1))
