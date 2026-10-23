"""Dense Tensor Sketch polynomial feature map, adapted from BSD-licensed sklearn (see LICENSES).

Pham & Pagh, "Fast and scalable polynomial kernels via explicit feature maps"
(KDD 2013): the count sketch of the degree-fold tensor power of
x' = [sqrt(gamma) x, sqrt(coef0)] is the circular convolution of `degree`
independent count sketches of x', computed as IFFT(prod_d FFT(C_d x')).
Its inner products approximate (gamma <x, y> + coef0)^degree.

As in sklearn, sketches, FFTs and the output are Float64 for both input
dtypes (sklearn's count_sketches buffer is np.zeros, i.e. float64).
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, isfinite
from mojo_linalg.fft import FFTPlan, fft_plan
from moxie._random import RandomState
from moxie._kernel_features import FeatureMapState, validate_matrix, component_width
from moxie._kernel_approximation_contracts import KernelFeatureMap
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount


struct PolynomialCountSketchParams(Copyable, Movable):
    """Constructor configuration of `PolynomialCountSketch` (no learned state)."""

    var gamma: Float64
    var degree: Int
    var coef0: Float64
    var n_components: Int
    var random_state: Optional[Int]

    def __init__(
        out self, *, gamma: Float64 = 1.0, degree: Int = 2, coef0: Float64 = 0.0,
        n_components: Int = 100, random_state: Optional[Int] = None,
    ):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def validate(self) raises:
        """sklearn constraints: gamma >= 0, degree >= 1, finite coef0, n_components >= 1."""
        if not isfinite(self.gamma) or self.gamma < 0:
            raise Error("ValueError: PolynomialCountSketch: gamma must be a finite value >= 0, got " + String(self.gamma))
        if self.degree < 1:
            raise Error("ValueError: PolynomialCountSketch: degree must be >= 1, got " + String(self.degree))
        if not isfinite(self.coef0):
            raise Error("ValueError: PolynomialCountSketch: coef0 must be finite, got " + String(self.coef0))
        if self.n_components < 1:
            raise Error("ValueError: PolynomialCountSketch: n_components must be >= 1, got " + String(self.n_components))


struct PolynomialCountSketch[dtype: DType = DType.float64](KernelFeatureMap, FeatureNamesOut, InputFeatureCount):
    """Polynomial kernel approximation via Tensor Sketch (dense input).

    Learned `indexHash_` and `bitHash_` are Int64 `(degree, p + (coef0 != 0))`
    arrays; `bitHash_` holds -1/+1 (sklearn's `choice([-1, 1])` integers).
    `transform` returns a Float64 `(n_samples, n_components)` matrix for either
    input dtype, matching sklearn.
    """

    comptime Params = PolynomialCountSketchParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[DType.float64]
    var params: Self.Params
    var _state: FeatureMapState
    var indexHash_: NDArray[DType.int64]
    var bitHash_: NDArray[DType.int64]
    var n_features_in_: Int

    def __init__(
        out self, *, gamma: Float64 = 1.0, degree: Int = 2, coef0: Float64 = 0.0,
        n_components: Int = 100, random_state: Optional[Int] = None,
    ) raises:
        self.params = Self.Params(gamma=gamma, degree=degree, coef0=coef0, n_components=n_components, random_state=random_state)
        self._state = FeatureMapState()
        self.indexHash_ = NDArray[DType.int64](Shape(0, 0))
        self.bitHash_ = NDArray[DType.int64](Shape(0, 0))
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, gamma: Optional[Float64] = None, degree: Optional[Int] = None, coef0: Optional[Float64] = None,
        n_components: Optional[Int] = None, random_state: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Named partial update; pass a full Params record to reset random_state to None."""
        if gamma:
            self.params.gamma = gamma.value()
        if degree:
            self.params.degree = degree.value()
        if coef0:
            self.params.coef0 = coef0.value()
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
            raise Error("AttributeError: PolynomialCountSketch has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Draw hashes with a fresh `RandomState(random_state)`; None needs `fit_with_rng`."""
        if not self.params.random_state:
            raise Error("ValueError: PolynomialCountSketch: random_state=None requires fit_with_rng(X, context)")
        var rng = RandomState(self.params.random_state.value())
        self.fit_with_rng(X, rng, y)
        return self

    def fit_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises:
        """Draw all index hashes (C order) and then all sign hashes from `rng`."""
        validate_matrix(X)
        self.params.validate()
        var columns = X.shape[1] + (1 if self.params.coef0 != 0 else 0)
        var degree = self.params.degree
        var index_hash = zeros[DType.int64](Shape(degree, columns))
        var bit_hash = zeros[DType.int64](Shape(degree, columns))
        for k in range(degree * columns):
            index_hash.unsafe_set(k, Int64(rng.randint(0, self.params.n_components)))
        # choice(a=[-1, 1]) draws randint(0, 2) indices into a.
        for k in range(degree * columns):
            bit_hash.unsafe_set(k, Int64(2 * rng.randint(0, 2) - 1))
        self._state.record_fit(X, self.params.n_components)
        self.n_features_in_ = X.shape[1]
        self.indexHash_ = index_hash^
        self.bitHash_ = bit_hash^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Sketch each row per degree, multiply their spectra, inverse FFT, keep the real part."""
        self._state.check_transform(X)
        var width = component_width(self.params.n_components)
        if self.params.coef0 < 0:
            # sklearn evaluates np.sqrt(coef0) -> NaN and returns an all-NaN map.
            raise Error(
                "ValueError: PolynomialCountSketch: coef0 < 0 has no real square root (sklearn returns all-NaN"
                " features); got coef0=" + String(self.params.coef0)
            )
        var columns = X.shape[1] + (1 if self.params.coef0 != 0 else 0)
        if columns != self.indexHash_.shape[1]:
            raise Error("ValueError: PolynomialCountSketch: number of features of test samples does not match that of training samples")
        var degree = self.indexHash_.shape[0]
        var rows = X.shape[0]
        for k in range(self.indexHash_.size):
            if Int(self.indexHash_.unsafe_get(k)) >= width:
                raise Error("ValueError: PolynomialCountSketch: n_components changed after fit; refit before transform")
        var gamma_root = sqrt(self.params.gamma)
        var coef_root = sqrt(self.params.coef0)
        var row_stride = Int(X.strides[0])
        var col_stride = Int(X.strides[1])
        var plan = fft_plan[DType.float64](width)
        var result = zeros[DType.float64](Shape(rows, width))
        var sketch_re = List[Float64](length=width, fill=0)
        var sketch_im = List[Float64](length=width, fill=0)
        var prod_re = List[Float64](length=width, fill=0)
        var prod_im = List[Float64](length=width, fill=0)
        var inverse_n = 1.0 / Float64(width)
        for i in range(rows):
            for d in range(degree):
                for k in range(width):
                    sketch_re[k] = 0
                    sketch_im[k] = 0
                # Same accumulation order as sklearn's loop over j (features) for each d.
                for j in range(columns):
                    var value: Float64
                    if j < X.shape[1]:
                        # sqrt(gamma) is a float64 NumPy scalar, so X_gamma is float64.
                        value = gamma_root * Float64(X.unsafe_get(i * row_stride + j * col_stride))
                    else:
                        value = coef_root
                    var slot = Int(self.indexHash_.unsafe_get(d * columns + j))
                    sketch_re[slot] += Float64(self.bitHash_.unsafe_get(d * columns + j)) * value
                plan.execute(sketch_re, sketch_im)
                if d == 0:
                    for k in range(width):
                        prod_re[k] = sketch_re[k]
                        prod_im[k] = sketch_im[k]
                else:
                    for k in range(width):
                        var re = prod_re[k] * sketch_re[k] - prod_im[k] * sketch_im[k]
                        prod_im[k] = prod_re[k] * sketch_im[k] + prod_im[k] * sketch_re[k]
                        prod_re[k] = re
            plan.execute(prod_re, prod_im, inverse=True)
            for k in range(width):
                result.unsafe_set(i * width + k, prod_re[k] * inverse_n)
        return result^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit_with_rng(X, rng, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """`polynomialcountsketch0 .. polynomialcountsketch{m-1}` using the fitted width."""
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: PolynomialCountSketch: call fit before get_feature_names_out")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: PolynomialCountSketch: input_features length must match n_features_in_")
        var names = List[String]()
        for k in range(self._state.n_features_out_):
            names.append("polynomialcountsketch" + String(k))
        return names^
