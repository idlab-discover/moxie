"""Dense spectral feature maps, adapted from BSD-licensed sklearn (see LICENSES)."""
from numojo import NDArray, Shape, mean
from std.math import sqrt, cos, tan, isfinite
from std.algorithm import vectorize
from std.sys import simd_width_of, llvm_intrinsic
from moxie._random import RandomState
from moxie._kernel_features import FeatureMapState, validate_matrix, component_width
from moxie._kernel_approximation_contracts import KernelFeatureMap
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount

comptime PI = 3.14159265358979323846


def _log(x: Float64) -> Float64:
    """Natural log via the LLVM intrinsic (platform libm, as NumPy uses).

    Mojo 1.0's std.math.log for Float64 measured up to ~1.7e7 ulp off libm.
    """
    return llvm_intrinsic["llvm.log", Float64](x)


def _project[dtype: DType](x: NDArray[dtype], w: NDArray[dtype], b: NDArray[dtype], width: Int, scale64: Float64) raises -> NDArray[dtype]:
    """Return `cos(x @ w + b) * scale64` as a new contiguous `(n, m)` matrix.

    The GEMM is NuMojo's approved rank-2 matmul; the phase shift, cosine and
    scaling are fused in one stdlib SIMD pass over each output row.
    """
    comptime assert dtype.is_floating_point(), "feature maps require Float32 or Float64"
    _ = component_width(width)
    var result = x @ w
    var scale = Scalar[dtype](scale64)
    var cols = result.shape[1]
    debug_assert(b.size == cols, "offset length must equal the projection width")
    for i in range(result.shape[0]):
        var row = i * cols

        def phase[simd_width: Int](j: Int) {mut result, imm b, imm row, imm scale} -> None:
            var value = result.unsafe_load[width=simd_width](row + j) + b.unsafe_load[width=simd_width](j)
            result.unsafe_store[width=simd_width](row + j, val=cos(value) * SIMD[dtype, simd_width](scale))

        vectorize[simd_width_of[dtype]()](cols, phase)
    return result^


def _scale_gamma[dtype: DType](x: NDArray[dtype]) raises -> Float64:
    # Compose accepted operations; no use of NuMojo's excluded variance API.
    var center = mean[returned_dtype=dtype](x)
    var residual = x - center
    var squared = residual * residual
    var variance = mean[returned_dtype=dtype](squared)
    if variance == 0:
        return 1.0
    return Float64(Scalar[dtype](1.0) / (Scalar[dtype](x.shape[1]) * variance))


struct RBFSamplerParams(Copyable, Movable):
    var gamma: Float64
    var n_components: Int
    var random_state: Optional[Int]
    var gamma_mode: String

    def __init__(out self, *, gamma: Float64 = 1.0, n_components: Int = 100, random_state: Optional[Int] = None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.gamma_mode = "numeric"


struct RBFSampler[dtype: DType = DType.float64](KernelFeatureMap, FeatureNamesOut, InputFeatureCount):
    comptime Params = RBFSamplerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var params: Self.Params
    var _state: FeatureMapState
    var random_weights_: NDArray[Self.dtype]
    var random_offset_: NDArray[Self.dtype]
    var n_features_in_: Int
    var _gamma: Float64

    def __init__(out self, *, gamma: Float64 = 1.0, n_components: Int = 100, random_state: Optional[Int] = None) raises:
        self.params = Self.Params(gamma=gamma, n_components=n_components, random_state=random_state)
        self._state = FeatureMapState()
        self.random_weights_ = NDArray[Self.dtype](Shape(0, 0))
        self.random_offset_ = NDArray[Self.dtype](Shape(0))
        self.n_features_in_ = 0
        self._gamma = 0

    def __init__(out self, *, gamma: String, n_components: Int = 100, random_state: Optional[Int] = None) raises:
        self = Self(n_components=n_components, random_state=random_state)
        self.params.gamma_mode = "scale" if gamma == "scale" else "invalid"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, gamma: Optional[Float64] = None, n_components: Optional[Int] = None, random_state: Optional[Int] = None) raises -> ref[self] Self:
        if gamma:
            self.params.gamma = gamma.value()
            self.params.gamma_mode = "numeric"
        if n_components:
            self.params.n_components = n_components.value()
        if random_state:
            self.params.random_state = random_state
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, gamma: String, n_components: Optional[Int] = None, random_state: Optional[Int] = None) raises -> ref[self] Self:
        _ = self.set_params(n_components=n_components, random_state=random_state)
        self.params.gamma_mode = "scale" if gamma == "scale" else "invalid"
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.input_state.is_fitted()

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` attribute, which exists only after fit."""
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: RBFSampler has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.params.random_state:
            raise Error("ValueError: random_state=None requires fit_with_rng(X, context)")
        var rng = RandomState(self.params.random_state.value())
        self.fit_with_rng(X, rng, y)
        return self

    def fit_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises:
        validate_matrix(X)
        var width = component_width(self.params.n_components)
        if width > Int.MAX // (8 if Self.dtype == DType.float64 else 4) // X.shape[1]:
            raise Error("ValueError: projection size overflow")
        if self.params.gamma_mode == "numeric" and not isfinite(self.params.gamma):
            raise Error("ValueError: gamma must be finite")
        if self.params.gamma_mode != "numeric" and self.params.gamma_mode != "scale":
            raise Error("ValueError: gamma must be nonnegative or scale")
        if self.params.gamma_mode == "numeric" and self.params.gamma < 0:
            raise Error("ValueError: gamma must be nonnegative")
        var fitted_gamma = self.params.gamma
        if self.params.gamma_mode == "scale":
            fitted_gamma = _scale_gamma(X)
        var weights = NDArray[Self.dtype](Shape(X.shape[1], width))
        var offsets = NDArray[Self.dtype](Shape(width))
        var spectral_scale = sqrt(2.0 * fitted_gamma)
        if self.params.gamma_mode == "scale":
            spectral_scale = Float64(Scalar[Self.dtype](sqrt(Float64(Scalar[Self.dtype](2.0) * Scalar[Self.dtype](fitted_gamma)))))
        for k in range(weights.size):
            weights.unsafe_set(k, Scalar[Self.dtype](spectral_scale * rng.standard_normal()))
        for k in range(width):
            offsets.unsafe_set(k, Scalar[Self.dtype](rng.uniform(0, 2 * PI)))
        self._state.record_fit(X, width)
        self.n_features_in_ = X.shape[1]
        self.random_weights_ = weights^
        self.random_offset_ = offsets^
        self._gamma = fitted_gamma

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._state.check_transform(X)
        _ = component_width(self.params.n_components)
        # sklearn: projection *= (2.0 / n_components) ** 0.5
        return _project(X, self.random_weights_, self.random_offset_, self.params.n_components, sqrt(2.0 / Float64(self.params.n_components)))

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit_with_rng(X, rng, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before feature names")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length must match n_features_in_")
        var names = List[String]()
        for k in range(self._state.n_features_out_):
            names.append("rbfsampler" + String(k))
        return names^


struct SkewedChi2SamplerParams(Copyable, Movable):
    var skewedness: Float64
    var n_components: Int
    var random_state: Optional[Int]

    def __init__(out self, *, skewedness: Float64 = 1.0, n_components: Int = 100, random_state: Optional[Int] = None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state


struct SkewedChi2Sampler[dtype: DType = DType.float64](KernelFeatureMap, FeatureNamesOut, InputFeatureCount):
    comptime Params = SkewedChi2SamplerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var params: Self.Params
    var _state: FeatureMapState
    var random_weights_: NDArray[Self.dtype]
    var random_offset_: NDArray[Self.dtype]
    var n_features_in_: Int

    def __init__(out self, *, skewedness: Float64 = 1.0, n_components: Int = 100, random_state: Optional[Int] = None) raises:
        self.params = Self.Params(skewedness=skewedness, n_components=n_components, random_state=random_state)
        self._state = FeatureMapState()
        self.random_weights_ = NDArray[Self.dtype](Shape(0, 0))
        self.random_offset_ = NDArray[Self.dtype](Shape(0))
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, skewedness: Optional[Float64] = None, n_components: Optional[Int] = None, random_state: Optional[Int] = None) raises -> ref[self] Self:
        if skewedness:
            self.params.skewedness = skewedness.value()
        if n_components:
            self.params.n_components = n_components.value()
        if random_state:
            self.params.random_state = random_state
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.input_state.is_fitted()

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` attribute, which exists only after fit."""
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: SkewedChi2Sampler has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.params.random_state:
            raise Error("ValueError: random_state=None requires fit_with_rng(X, context)")
        var rng = RandomState(self.params.random_state.value())
        self.fit_with_rng(X, rng, y)
        return self

    def fit_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises:
        validate_matrix(X)
        var width = component_width(self.params.n_components)
        if width > Int.MAX // (8 if Self.dtype == DType.float64 else 4) // X.shape[1]:
            raise Error("ValueError: projection size overflow")
        if not isfinite(self.params.skewedness):
            raise Error("ValueError: skewedness must be finite")
        var weights = NDArray[Self.dtype](Shape(X.shape[1], width))
        var offsets = NDArray[Self.dtype](Shape(width))
        for k in range(weights.size):
            weights.unsafe_set(k, Scalar[Self.dtype](1.0 / PI * _log(tan(PI / 2.0 * rng.random_sample()))))
        for k in range(width):
            offsets.unsafe_set(k, Scalar[Self.dtype](rng.uniform(0, 2 * PI)))
        self._state.record_fit(X, width)
        self.n_features_in_ = X.shape[1]
        self.random_weights_ = weights^
        self.random_offset_ = offsets^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._state.check_transform(X)
        var logged = NDArray[Self.dtype](Shape(X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var value = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                if value <= Scalar[Self.dtype](-self.params.skewedness):
                    raise Error("ValueError: X may not contain entries smaller than -skewedness")
                logged.unsafe_set(i * X.shape[1] + j, Scalar[Self.dtype](_log(Float64(value + Scalar[Self.dtype](self.params.skewedness)))))
        _ = component_width(self.params.n_components)
        # sklearn: projection *= np.sqrt(2.0) / np.sqrt(n_components)
        return _project(logged, self.random_weights_, self.random_offset_, self.params.n_components, sqrt(2.0) / sqrt(Float64(self.params.n_components)))

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit_with_rng(X, rng, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before feature names")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length must match n_features_in_")
        var names = List[String]()
        for k in range(self._state.n_features_out_):
            names.append("skewedchi2sampler" + String(k))
        return names^
