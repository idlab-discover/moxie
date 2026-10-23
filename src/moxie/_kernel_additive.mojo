"""Dense additive chi-squared feature map, adapted from BSD-licensed sklearn (see LICENSES).

Vedaldi & Zisserman, "Efficient additive kernels via explicit feature maps",
PAMI 2011: each nonnegative entry x maps to 2*sample_steps-1 features, sampled
from the Fourier transform of the kernel signature at spacing `interval`:

    sqrt(x * L),  sqrt(2 x L / cosh(pi j L)) * cos(j L log x),  ... * sin(j L log x)

for j = 1 .. sample_steps-1. Zeros map to zeros (no log(0) is evaluated).
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, cos, sin, isfinite
from std.sys import llvm_intrinsic
from moxie._kernel_features import FeatureMapState, validate_matrix, additive_width
from moxie._kernel_approximation_contracts import KernelFeatureMap
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount

comptime PI = 3.14159265358979323846


# Mojo 1.0's std.math.log (Float64) and cosh are polynomial approximations that
# measured up to ~1.7e7 and ~1e5 ulp away from libm/NumPy; cos, sin and sqrt
# are exact. The LLVM intrinsics lower to the platform libm, matching NumPy.
def _log[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return llvm_intrinsic["llvm.log", Scalar[dtype]](x)


def _cosh(x: Float64) -> Float64:
    var e = llvm_intrinsic["llvm.exp", Float64](x)
    return 0.5 * (e + 1.0 / e)


def default_sample_interval(sample_steps: Int) raises -> Float64:
    """Return sklearn's default interval for 1, 2 or 3 steps (figure 2c of the paper)."""
    if sample_steps == 1:
        return 0.8
    if sample_steps == 2:
        return 0.5
    if sample_steps == 3:
        return 0.4
    raise Error(
        "ValueError: AdditiveChi2Sampler: if sample_steps is not in [1, 2, 3], you need to provide"
        " sample_interval (got sample_steps=" + String(sample_steps) + ")"
    )


struct AdditiveChi2SamplerParams(Copyable, Movable):
    """Constructor configuration of `AdditiveChi2Sampler` (no learned state)."""

    var sample_steps: Int
    var sample_interval: Optional[Float64]

    def __init__(out self, *, sample_steps: Int = 2, sample_interval: Optional[Float64] = None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def validate(self) raises:
        """Apply sklearn's parameter constraints: steps >= 1, interval >= 0 or None."""
        if self.sample_steps < 1:
            raise Error("ValueError: AdditiveChi2Sampler: sample_steps must be >= 1, got " + String(self.sample_steps))
        if self.sample_interval:
            var interval = self.sample_interval.value()
            if not isfinite(interval) or interval < 0:
                raise Error("ValueError: AdditiveChi2Sampler: sample_interval must be a finite value >= 0, got " + String(interval))

    def resolved_interval(self) raises -> Float64:
        """Return the explicit interval or the default for 1-3 steps."""
        self.validate()
        if self.sample_interval:
            return self.sample_interval.value()
        return default_sample_interval(self.sample_steps)


struct AdditiveChi2Sampler[dtype: DType = DType.float64](KernelFeatureMap, FeatureNamesOut, InputFeatureCount):
    """Approximate feature map for the additive chi-squared kernel (dense input).

    Stateless like sklearn: `transform` works without `fit`. `fit` validates
    the parameters and nonnegative input and records `n_features_in_`, after
    which transform also checks the feature count. Output has shape
    `(n_samples, n_features * (2 * sample_steps - 1))` with column blocks
    `[sqrt | cos1 | sin1 | cos2 | sin2 | ...]`, each block `n_features` wide.
    """

    comptime Params = AdditiveChi2SamplerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var params: Self.Params
    var _state: FeatureMapState
    var n_features_in_: Int

    def __init__(out self, *, sample_steps: Int = 2, sample_interval: Optional[Float64] = None):
        self.params = Self.Params(sample_steps=sample_steps, sample_interval=sample_interval)
        self._state = FeatureMapState()
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, sample_steps: Optional[Int] = None, sample_interval: Optional[Float64] = None) raises -> ref[self] Self:
        """Named partial update; pass a full Params record to reset the interval to None."""
        if sample_steps:
            self.params.sample_steps = sample_steps.value()
        if sample_interval:
            self.params.sample_interval = sample_interval
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        """True after `fit`; transform itself never requires it (sklearn requires_fit=False)."""
        return self._state.input_state.is_fitted()

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` attribute, which exists only after fit."""
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: AdditiveChi2Sampler has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        validate_matrix(X, nonnegative=True)
        _ = self.params.resolved_interval()
        var width = additive_width(X.shape[1], self.params.sample_steps)
        self._state.record_fit(X, width)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        comptime assert Self.dtype.is_floating_point(), "AdditiveChi2Sampler requires Float32 or Float64"
        validate_matrix(X, nonnegative=True)
        if self.__sklearn_is_fitted__():
            self._state.input_state.check_features(X.shape[1])
        var steps = self.params.sample_steps
        var interval = self.params.resolved_interval()
        var rows = X.shape[0]
        var features = X.shape[1]
        var width = additive_width(features, steps)
        var result = zeros[Self.dtype](Shape(rows, width))
        var row_stride = Int(X.strides[0])
        var col_stride = Int(X.strides[1])
        # cosh(pi j L) is shared by all entries; a Float64 scalar, as in NumPy,
        # so Float32 inputs get a Float64 factor exactly like the reference.
        var cosh_step = List[Float64](capacity=steps)
        cosh_step.append(1.0)
        for j in range(1, steps):
            cosh_step.append(_cosh(PI * Float64(j) * interval))
        var typed_interval = Scalar[Self.dtype](interval)
        for i in range(rows):
            var out_row = i * width
            for f in range(features):
                var x = X.unsafe_get(i * row_stride + f * col_stride)
                if x == 0:
                    continue
                result.unsafe_set(out_row + f, sqrt(x * typed_interval))
                var log_step = typed_interval * _log(x)
                var step = Float64(2 * x * typed_interval)
                for j in range(1, steps):
                    var factor = sqrt(step / cosh_step[j])
                    var phase = Scalar[Self.dtype](j) * log_step
                    var block = out_row + (2 * j - 1) * features + f
                    result.unsafe_set(block, Scalar[Self.dtype](factor * Float64(cos(phase))))
                    result.unsafe_set(block + features, Scalar[Self.dtype](factor * Float64(sin(phase))))
        return result^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """Names `additivechi2sampler_<name>_{sqrt,cos<j>,sin<j>}` in output column order.

        Requires `fit` (sklearn checks `n_features_in_`). Without input names,
        `x0 .. x{p-1}` are generated.
        """
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: AdditiveChi2Sampler: call fit before get_feature_names_out")
        var names_in = List[String]()
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error(
                    "ValueError: AdditiveChi2Sampler: input_features should have length equal to n_features_in_ ("
                    + String(self.n_features_in_) + "), got " + String(len(input_features.value()))
                )
            names_in = input_features.value().copy()
        else:
            for f in range(self.n_features_in_):
                names_in.append("x" + String(f))
        var names = List[String]()
        for name in names_in:
            names.append("additivechi2sampler_" + name + "_sqrt")
        for j in range(1, self.params.sample_steps):
            for name in names_in:
                names.append("additivechi2sampler_" + name + "_cos" + String(j))
            for name in names_in:
                names.append("additivechi2sampler_" + name + "_sin" + String(j))
        return names^
