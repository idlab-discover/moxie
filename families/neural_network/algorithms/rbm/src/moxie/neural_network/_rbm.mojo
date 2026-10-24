"""Dense Bernoulli RBM kernels. Adapted from sklearn _rbm.py (BSD-3-Clause)."""
from std.sys import llvm_intrinsic
from numojo import NDArray, Shape, zeros
from std.math import exp, log1p, abs, isfinite
from std.memory import bitcast
from std.time import perf_counter_ns
from moxie.neural_network._base import RBMTransformer
from moxie.neural_network._contracts import RBMSampling
from moxie._random import RandomState
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._dense import sum_axis2d
from moxie.neural_network._plumbing import validate_matrix


def _sigmoid[dtype: DType](x: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    comptime assert dtype.is_floating_point()
    # Both branches avoid overflowing the exponential.
    if x >= 0:
        return 1 / (1 + exp(-x))
    var e = exp(x)
    return e / (1 + e)


def _softplus[dtype: DType](x: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    comptime assert dtype.is_floating_point()
    var e = Float64(exp(-abs(x)))
    var y = 1 + e
    var tail = e
    if y != 1:
        tail = llvm_intrinsic["llvm.log.f64", Float64](y) * (e / (y - 1))
    return max(x, 0) + SIMD[dtype, 1](tail)


def _conditional[dtype: DType](X: NDArray[dtype], weights: NDArray[dtype], bias: NDArray[dtype]) raises -> NDArray[dtype]:
    var result = X @ weights
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            var index = i * Int(result.strides[0]) + j * Int(result.strides[1])
            result.unsafe_set(index, _sigmoid(result.unsafe_get(index) + bias.unsafe_get(j * Int(bias.strides[0]))))
    return result^


def _sample[dtype: DType](probabilities: NDArray[dtype], mut rng: RandomState) raises -> NDArray[dtype]:
    var result = zeros[dtype](probabilities.shape)
    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            var p = probabilities.unsafe_get(i * Int(probabilities.strides[0]) + j * Int(probabilities.strides[1]))
            result.unsafe_set(i * probabilities.shape[1] + j, SIMD[dtype, 1](Int(rng.random_sample() < Float64(p))))
    return result^


def _fixed(value: Float64, decimals: Int) -> String:
    """`'%.{decimals}f' % value`: exact binary64 to fixed decimal, ties to even.

    String.format lacks precision support on Mojo 1.0; same digit-expansion
    method as moxie.tree._export._fixed, kept local to avoid a cross-family import.
    """
    var bits = bitcast[DType.uint64](value)
    var negative = (bits >> 63) != 0
    var exponent = Int((bits >> 52) & 2047)
    var mantissa = Int(bits & 0x000fffffffffffff)
    if exponent == 2047:
        if mantissa != 0:
            return "nan"
        return "-inf" if negative else "inf"
    if exponent != 0:
        mantissa += 4503599627370496
    var power = (exponent - 1023 - 52) if exponent != 0 else -1074
    var digits = List[Int]()
    while mantissa != 0:
        digits.append(mantissa % 10)
        mantissa //= 10
    if len(digits) == 0:
        digits.append(0)
    var scale = max(0, -power)
    for _ in range(abs(power)):
        var carry = 0
        for i in range(len(digits)):
            var product = digits[i] * (2 if power >= 0 else 5) + carry
            digits[i] = product % 10
            carry = product // 10
        if carry != 0:
            digits.append(carry)
    while len(digits) <= scale:
        digits.append(0)
    if decimals < scale:
        var cut = scale - decimals
        var sticky = False
        for i in range(cut - 1):
            sticky = sticky or digits[i] != 0
        var up = digits[cut - 1] > 5 or (digits[cut - 1] == 5 and (sticky or digits[cut] % 2 != 0))
        var rounded = List[Int]()
        var carry = Int(up)
        for i in range(cut, len(digits)):
            var digit = digits[i] + carry
            rounded.append(digit % 10)
            carry = digit // 10
        if carry:
            rounded.append(carry)
        digits = rounded^
        scale = decimals
    var result = String("-" if negative else "")
    for i in range(len(digits) - 1, scale - 1, -1):
        result += String(digits[i])
    if decimals > 0:
        result += "."
        for i in range(scale - 1, -1, -1):
            result += String(digits[i])
        for _ in range(decimals - scale):
            result += "0"
    return result


@fieldwise_init
struct RBMParams(Copyable, Movable):
    var n_components: Int
    var learning_rate: Float64
    var batch_size: Int
    var n_iter: Int
    var verbose: Int
    var random_state: Optional[Int]


struct BernoulliRBM[dtype: DType = DType.float64](RBMTransformer, RBMSampling, FeatureNamesOut, InputFeatureCount, Movable):
    """Dense RBM. Integer seeds or explicitly borrowed mutable RNG contexts."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime SampleScores = NDArray[Self.dtype]
    comptime Scores = NDArray[Self.dtype]
    comptime Samples = NDArray[DType.bool]
    comptime Params = RBMParams
    var n_components: Int
    var learning_rate: Float64
    var batch_size: Int
    var n_iter: Int
    var verbose: Int
    var random_state: Optional[Int]
    var n_features_in_: Int
    var _fitted: Bool
    var _rng: RandomState
    var _has_rng: Bool
    var components_: NDArray[Self.dtype]
    var intercept_hidden_: NDArray[Self.dtype]
    var intercept_visible_: NDArray[Self.dtype]
    var h_samples_: NDArray[Self.dtype]

    def __init__(out self, n_components: Int = 256, *, learning_rate: Float64 = 0.1, batch_size: Int = 10, n_iter: Int = 10, verbose: Int = 0, random_state: Optional[Int] = None) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state.copy()
        self.n_features_in_ = 0
        self._fitted = False
        self._rng = RandomState(0)
        self._has_rng = False
        self.components_ = zeros[Self.dtype](Shape(0, 0))
        self.intercept_hidden_ = zeros[Self.dtype](Shape(0))
        self.intercept_visible_ = zeros[Self.dtype](Shape(0))
        self.h_samples_ = zeros[Self.dtype](Shape(0, 0))

    def initialize(mut self, n_components: Int, n_features: Int, batch_size: Int, mut rng: RandomState) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if n_components <= 0 or n_features <= 0 or batch_size <= 0:
            raise Error("ValueError: RBM dimensions must be positive")
        self.components_ = zeros[Self.dtype](Shape(n_components, n_features))
        self.intercept_hidden_ = zeros[Self.dtype](Shape(n_components))
        self.intercept_visible_ = zeros[Self.dtype](Shape(n_features))
        self.h_samples_ = zeros[Self.dtype](Shape(batch_size, n_components))
        self.n_features_in_ = n_features
        self._fitted = True
        for i in range(n_components * n_features):
            self.components_.unsafe_set(i, SIMD[Self.dtype, 1](rng.normal(0, 0.01)))

    def mean_hiddens(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        validate_matrix(X)
        if X.shape[1] != self.components_.shape[1]:
            raise Error("ValueError: wrong feature count")
        return _conditional(X, self.components_.T(), self.intercept_hidden_)

    def sample_visibles(self, hidden: NDArray[Self.dtype], mut rng: RandomState) raises -> NDArray[Self.dtype]:
        return _sample(_conditional(hidden, self.components_, self.intercept_visible_), rng)

    def _gibbs(self, X: NDArray[Self.dtype], mut rng: RandomState) raises -> NDArray[DType.bool]:
        var hidden = _sample(self.mean_hiddens(X), rng)
        var visible = self.sample_visibles(hidden, rng)
        var result = zeros[DType.bool](visible.shape)
        for i in range(visible.size):
            result.unsafe_set(i, visible.unsafe_get(i) != 0)
        return result^

    def free_energy(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        validate_matrix(X)
        if X.shape[1] != self.components_.shape[1]:
            raise Error("ValueError: wrong feature count")
        var hidden = X @ self.components_.T()
        for i in range(hidden.shape[0]):
            for j in range(hidden.shape[1]):
                var k = i * hidden.shape[1] + j
                hidden.unsafe_set(k, _softplus(hidden.unsafe_get(k) + self.intercept_hidden_.unsafe_get(j)))
        var result = sum_axis2d(hidden, 1)
        # Vector dot is outside the accepted NuMojo subset: use rank-2 matmul.
        var bias = self.intercept_visible_.reshape(Shape(X.shape[1], 1))
        var visible = X @ bias
        for i in range(X.shape[0]):
            result.unsafe_set(i, -result.unsafe_get(i) - visible.unsafe_get(i))
        return result^

    def _score_samples(self, X: NDArray[Self.dtype], mut rng: RandomState) raises -> NDArray[Self.dtype]:
        validate_matrix(X)
        var corrupted = X.copy()
        for i in range(X.shape[0]):
            var j = rng.randint(X.shape[1])
            var k = i * Int(corrupted.strides[0]) + j * Int(corrupted.strides[1])
            corrupted.unsafe_set(k, 1 - corrupted.unsafe_get(k))
        var original = self.free_energy(X)
        var flipped = self.free_energy(corrupted)
        for i in range(X.shape[0]):
            original.unsafe_set(i, -SIMD[Self.dtype, 1](X.shape[1]) * _softplus(original.unsafe_get(i) - flipped.unsafe_get(i)))
        return original^

    def update(mut self, X: NDArray[Self.dtype], learning_rate: Float64, mut rng: RandomState) raises:
        if learning_rate <= 0 or not isfinite(learning_rate):
            raise Error("ValueError: learning_rate must be positive")
        var positive = self.mean_hiddens(X)
        var visible = self.sample_visibles(self.h_samples_, rng)
        var negative = self.mean_hiddens(visible)
        var pos = positive.T() @ X
        var neg = negative.T() @ visible
        var hp = sum_axis2d(positive, 0)
        var hn = sum_axis2d(negative, 0)
        var vp = sum_axis2d(X, 0)
        var vn = sum_axis2d(visible, 0)
        var lr = SIMD[Self.dtype, 1](learning_rate / Float64(X.shape[0]))
        for i in range(self.components_.size):
            self.components_.unsafe_set(i, self.components_.unsafe_get(i) + lr * (pos.unsafe_get(i) - neg.unsafe_get(i)))
        for i in range(self.intercept_hidden_.size):
            self.intercept_hidden_.unsafe_set(i, self.intercept_hidden_.unsafe_get(i) + lr * (hp.unsafe_get(i) - hn.unsafe_get(i)))
        for i in range(self.intercept_visible_.size):
            self.intercept_visible_.unsafe_set(i, self.intercept_visible_.unsafe_get(i) + lr * (vp.unsafe_get(i) - vn.unsafe_get(i)))
        self.h_samples_ = _sample(negative, rng)

    def _validate_params(self) raises:
        if self.n_components <= 0 or self.batch_size <= 0 or self.n_iter < 0 or self.verbose < 0:
            raise Error("ValueError: invalid RBM integer parameter")
        if self.learning_rate <= 0 or not isfinite(self.learning_rate):
            raise Error("ValueError: learning_rate must be finite and positive")
        if self.random_state:
            _ = RandomState(self.random_state.value())

    def _check(self, X: Self.Input) raises:
        if not self._fitted:
            raise Error("NotFittedError: BernoulliRBM has not been fitted")
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: wrong feature count")

    def _seed(self) raises -> Int:
        if not self.random_state:
            raise Error("ValueError: random_state=None requires an explicit mutable RNG context")
        return self.random_state.value()

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return RBMParams(self.n_components, self.learning_rate, self.batch_size, self.n_iter, self.verbose, self.random_state.copy())

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.n_iter = params.n_iter
        self.verbose = params.verbose
        self.random_state = params.random_state.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, learning_rate=self.learning_rate, batch_size=self.batch_size, n_iter=self.n_iter, verbose=self.verbose, random_state=self.random_state.copy())

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return self.mean_hiddens(X)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        var rng = RandomState(self._seed())
        _ = self.fit(X, rng)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        mut rng: RandomState,
    ) raises -> ref[self] Self:
        self._validate_params()
        validate_matrix(X)
        self.initialize(self.n_components, X.shape[1], self.batch_size, rng)
        var begin_ns = perf_counter_ns()
        for epoch in range(self.n_iter):
            for begin in range(0, X.shape[0], self.batch_size):
                var size = min(self.batch_size, X.shape[0] - begin)
                var batch = zeros[Self.dtype](Shape(size, X.shape[1]))
                for i in range(size):
                    for j in range(X.shape[1]):
                        batch.unsafe_set(i * X.shape[1] + j, X.unsafe_get((begin + i) * Int(X.strides[0]) + j * Int(X.strides[1])))
                self.update(batch, self.learning_rate, rng)
            if self.verbose:
                # As the reference, the elapsed time is taken before scoring.
                var end_ns = perf_counter_ns()
                var scores: Self.Scores
                if self.random_state:
                    scores = self.score_samples(X)
                else:
                    scores = self._score_samples(X, rng)
                var total = Float64(0)
                for i in range(scores.size):
                    total += Float64(scores.unsafe_get(i))
                var seconds = Float64(end_ns - begin_ns) / 1e9
                print("[BernoulliRBM] Iteration ", epoch + 1, ", pseudo-likelihood = ",
                      _fixed(total / Float64(scores.size), 2), ", time = ", _fixed(seconds, 2), "s", sep="")
                begin_ns = end_ns
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, mut rng: RandomState) raises -> Self.Transformed:
        _ = self.fit(X, rng)
        return self.transform(X)

    def gibbs(mut self, X: Self.Input) raises -> Self.Samples:
        self._check(X)
        if not self._has_rng:
            self._rng = RandomState(self._seed())
            self._has_rng = True
        var rng = self._rng.copy()
        var result = self._gibbs(X, rng)
        self._rng = rng^
        return result^

    def gibbs(mut self, X: Self.Input, mut rng: RandomState) raises -> Self.Samples:
        self._check(X)
        return self._gibbs(X, rng)

    def score_samples(self, X: Self.Input) raises -> Self.Scores:
        self._check(X)
        var rng = RandomState(self._seed())
        return self._score_samples(X, rng)

    def score_samples(self, X: Self.Input, mut rng: RandomState) raises -> Self.Scores:
        self._check(X)
        return self._score_samples(X, rng)

    def partial_fit(mut self, X: Self.Input, y: Optional[Self.Input] = None) raises:
        self._validate_params()
        validate_matrix(X)
        if not self._has_rng:
            self._rng = RandomState(self._seed())
            self._has_rng = True
        # Borrowing the RNG field and mutating the whole estimator together is
        # avoided by an explicit snapshot and restoration of this owned stream.
        var rng = self._rng.copy()
        self.partial_fit(X, rng)
        self._rng = rng^

    def partial_fit(mut self, X: Self.Input, mut rng: RandomState) raises:
        self._validate_params()
        validate_matrix(X)
        comptime if Self.dtype == DType.float32:
            raise Error("NotImplementedError: Float32 partial_fit promotion is unfinished; use Float64")
        else:
            if not self._fitted:
                self.initialize(self.n_components, X.shape[1], self.batch_size, rng)
            self._check(X)
            self.update(X, self.learning_rate, rng)

    def get_n_features_in(self) raises -> Int:
        """sklearn's n_features_in_ attribute; absent (raises) before fit."""
        if not self._fitted:
            raise Error("NotFittedError: BernoulliRBM has not been fitted")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self._fitted:
            raise Error("NotFittedError: BernoulliRBM has not been fitted")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: wrong number of input feature names")
        var names = List[String]()
        for j in range(self.components_.shape[0]):
            names.append("bernoullirbm" + String(j))
        return names^
