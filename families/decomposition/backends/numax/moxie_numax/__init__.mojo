"""Opt-in Numax CPU backend for PCA with compile-time training dimensions.

The public estimator keeps NuMojo arrays; adapters make owned, row-major copies.
Requires the numax Pixi environment (Mojo 1.1 / MAX 26.6).
"""
from max.gpu.host import DeviceContext
from numojo import NDArray, Shape, zeros
from numax.core.array import Static, transpose, zeros_dyn
from numax.linalg import svd, eigh, inverse, slogdet, matmul
from moxie.decomposition._pca_backend import PCABackend


def _host[dtype: DType](X: NDArray[dtype]) raises -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=X.size)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            values.append(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
    return values^


def _array[dtype: DType](values: List[Scalar[dtype]], shape: Shape) raises -> NDArray[dtype]:
    var result = zeros[dtype](shape)
    for i in range(len(values)): result.unsafe_set(i, values[i])
    return result^


struct NumaxPCABackend[n_samples: Int, n_features: Int](PCABackend):
    """Static training shape; transform/score still accept any row count."""

    @staticmethod
    def validate_fit[dtype: DType](X: NDArray[dtype]) raises:
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        if X.ndim != 2 or X.shape[0] != Self.n_samples or X.shape[1] != Self.n_features:
            raise Error("ValueError: X does not match NumaxPCABackend's compile-time training shape")

    @staticmethod
    def full[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], NDArray[dtype]]:
        Self.validate_fit(X)
        comptime assert dtype.is_floating_point()
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        var ctx = DeviceContext(api="cpu")
        var a = Static[dtype, Self.n_samples, Self.n_features](ctx, _host(X))
        comptime r = min(Self.n_samples, Self.n_features)
        comptime if Self.n_samples >= Self.n_features:
            var factors = svd(a)
            var vt = transpose(factors.v)
            return (_array(factors.u.to_host(), Shape(Self.n_samples, r)),
                    _array(factors.s.to_host(), Shape(r)),
                    _array(vt.to_host(), Shape(r, Self.n_features)))
        else:
            # Numax accepts tall matrices: X.T = U S V.T -> X = V S U.T.
            comptime assert Self.n_features >= Self.n_samples
            var at = transpose(a)
            var factors = svd(at)
            var vt = transpose(factors.u)
            return (_array(factors.v.to_host(), Shape(Self.n_samples, r)),
                    _array(factors.s.to_host(), Shape(r)),
                    _array(vt.to_host(), Shape(r, Self.n_features)))

    @staticmethod
    def eigh[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
        comptime assert dtype.is_floating_point()
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        var ctx = DeviceContext(api="cpu")
        var a = Static[dtype, Self.n_features, Self.n_features](ctx, _host(X))
        var factors = eigh(a)
        return (_array(factors.values.to_host(), Shape(Self.n_features)),
                _array(factors.vectors.to_host(), Shape(Self.n_features, Self.n_features)))

    @staticmethod
    def inverse[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
        comptime assert dtype.is_floating_point()
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        var ctx = DeviceContext(api="cpu")
        var a = Static[dtype, Self.n_features, Self.n_features](ctx, _host(X))
        var result = inverse(a)
        return _array(result.to_host(), Shape(Self.n_features, Self.n_features))

    @staticmethod
    def slogdet[dtype: DType](X: NDArray[dtype]) raises -> Tuple[Float64, Float64]:
        comptime assert dtype.is_floating_point()
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        var ctx = DeviceContext(api="cpu")
        var a = Static[dtype, Self.n_features, Self.n_features](ctx, _host(X))
        var result = slogdet(a)
        return (Float64(result[0]), Float64(result[1]))

    @staticmethod
    def matmul[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
        comptime assert dtype.is_floating_point()
        comptime assert Self.n_samples >= 1 and Self.n_features >= 1
        var ctx = DeviceContext(api="cpu")
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[0]:
            raise Error("ValueError: matrix-product shape mismatch")
        # PCA permits a runtime n_components, including zero. Avoid sending an
        # empty matrix to MAX and preserve that zero-column sklearn contract.
        if X.shape[1] == 0 or X.shape[0] == 0 or Y.shape[1] == 0:
            return zeros[dtype](Shape(X.shape[0], Y.shape[1]))
        # MAX 26.6's dynamic GEMV crashes for e.g. (24, 5) @ (5, 1).
        # Pad singleton output dimensions to select GEMM, then crop. Padding
        # changes no reduction dimension, so it introduces no extra dot terms.
        var rows = max(2, X.shape[0])
        var cols = max(2, Y.shape[1])
        var left = _host(X)
        for i in range((rows - X.shape[0]) * X.shape[1]): left.append(0)
        var right = List[Scalar[dtype]](capacity=Y.shape[0] * cols)
        for i in range(Y.shape[0]):
            for j in range(cols):
                var value = Scalar[dtype](0)
                if j < Y.shape[1]:
                    value = Y.unsafe_get(i * Int(Y.strides[0]) + j * Int(Y.strides[1]))
                right.append(value)
        var a = zeros_dyn[dtype, 2](rows, X.shape[1], ctx=ctx)
        var b = zeros_dyn[dtype, 2](Y.shape[0], cols, ctx=ctx)
        a.copy_from_host(left)
        b.copy_from_host(right)
        var result = matmul(a, b)
        var host = result.to_host()
        var output = zeros[dtype](Shape(X.shape[0], Y.shape[1]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                output.unsafe_set(i * Y.shape[1] + j, host[i * cols + j])
        return output^


struct NumaxIncrementalPCABackend[batch_size: Int, n_features: Int, n_components: Int](PCABackend):
    """Bounded augmented SVD, including sklearn's merged final fit batch.

    Capacity is batch_size + n_components - 1 data rows, plus the retained
    basis and correction row. Padding is not counted as observed samples.
    """
    comptime Engine = NumaxPCABackend[Self.batch_size + 2 * Self.n_components, Self.n_features]

    @staticmethod
    def validate_fit[dtype: DType](X: NDArray[dtype]) raises:
        comptime assert Self.batch_size >= 1 and Self.n_components >= 1
        comptime assert Self.n_components <= Self.n_features
        if X.ndim != 2 or X.shape[1] != Self.n_features or X.shape[0] < 1 or X.shape[0] > Self.batch_size + 2 * Self.n_components:
            raise Error("ValueError: augmented batch exceeds NumaxIncrementalPCABackend capacity or feature count")

    @staticmethod
    def full[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], NDArray[dtype]]:
        Self.validate_fit(X)
        var padded = zeros[dtype](Shape(Self.batch_size + 2 * Self.n_components, Self.n_features))
        for i in range(X.shape[0]):
            for j in range(Self.n_features):
                padded.unsafe_set(i * Self.n_features + j, X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
        comptime if dtype == DType.float32 and Self.n_features == 1:
            # MAX's Float32 GEMV inside Numax gebrd crashes on tall singleton
            # columns (reproducer: 42 x 1). The Float64 Numax path is valid.
            var factors = Self.Engine.full(padded.astype[DType.float64]())
            return (factors[0].astype[dtype](), factors[1].astype[dtype](), factors[2].astype[dtype]())
        else:
            return Self.Engine.full(padded)

    @staticmethod
    def matmul[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
        return Self.Engine.matmul(X, Y)

    @staticmethod
    def eigh[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
        return Self.Engine.eigh(X)

    @staticmethod
    def inverse[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
        return Self.Engine.inverse(X)

    @staticmethod
    def slogdet[dtype: DType](X: NDArray[dtype]) raises -> Tuple[Float64, Float64]:
        return Self.Engine.slogdet(X)
