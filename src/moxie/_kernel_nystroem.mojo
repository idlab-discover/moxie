"""Dense Nystroem kernel approximation, adapted from BSD-licensed sklearn (see LICENSES).

Williams & Seeger (NIPS 2001): with a random basis B of k training rows and
basis Gram K_BB = U S V^T, the map x -> K(x, B) @ (U S^{-1/2} V^T)^T satisfies
phi(x) . phi(y) = K(x, B) K_BB^+ K(B, y). Singular values are clipped below at
1e-12 exactly as sklearn does.

Kernel evaluation reuses the completed kernel_ridge numerics engine
(`moxie._kernel_ridge.pairwise`/`pairwise_callable`); the SVD is mojo-linalg's
economy one-sided Jacobi `svd`; the products use NuMojo's rank-2 matmul.
"""
from numojo import NDArray, Shape, zeros
from std.sys import num_logical_cores
from max.algorithm import parallelize
from std.math import sqrt, isfinite
from mojo_linalg import svd, NumericalError
from moxie._random import RandomState
from moxie._kernel_features import FeatureMapState, validate_matrix, component_width, basis_width
from moxie._kernel_approximation_contracts import KernelFeatureMap
from moxie._kernel_ridge import pairwise, pairwise_callable, RowKernel


def _is_named_kernel(kernel: String) -> Bool:
    """sklearn PAIRWISE_KERNEL_FUNCTIONS keys."""
    for name in [String("additive_chi2"), "chi2", "linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"]:
        if kernel == name:
            return True
    return False


@fieldwise_init
struct NoCallableKernel(Copyable, Movable, RowKernel):
    """Placeholder kernel type for Nystroem with a named or precomputed kernel."""

    def evaluate[dtype: DType](self, X: NDArray[dtype], i: Int, Y: NDArray[dtype], j: Int) raises -> SIMD[dtype, 1]:
        raise Error("ValueError: Nystroem: no callable kernel configured")


@fieldwise_init
struct NystroemKernelParams(Copyable, Movable):
    """Typed replacement for sklearn's `kernel_params` dict of a named kernel.

    Explicit `gamma`/`coef0`/`degree` constructor values take precedence, as in
    sklearn's `_get_kernel_params`. Entries a kernel does not use are ignored
    (sklearn's `filter_params=True`).
    """

    var gamma: Optional[Float64]
    var coef0: Optional[Float64]
    var degree: Optional[Float64]


struct NystroemParams[Kernel: RowKernel & Copyable & Deinitable](Copyable, Movable):
    """Constructor configuration of `Nystroem` (no learned state)."""

    var kernel: String
    var gamma: Optional[Float64]
    var coef0: Optional[Float64]
    var degree: Optional[Float64]
    var kernel_params: Optional[NystroemKernelParams]
    var n_components: Int
    var random_state: Optional[Int]
    var n_jobs: Optional[Int]
    var kernel_function: Optional[Self.Kernel]

    def __init__(
        out self, kernel: String = "rbf", *, gamma: Optional[Float64] = None, coef0: Optional[Float64] = None,
        degree: Optional[Float64] = None, kernel_params: Optional[NystroemKernelParams] = None,
        n_components: Int = 100, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params.copy()
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kernel_function = None

    def validate(self) raises:
        """sklearn constraints: known kernel, gamma >= 0, finite coef0, degree >= 1, n_components >= 1."""
        if self.kernel != "callable" and self.kernel != "precomputed" and not _is_named_kernel(self.kernel):
            raise Error("ValueError: Nystroem: unknown kernel '" + self.kernel + "'")
        if self.kernel == "callable" and not self.kernel_function:
            raise Error("ValueError: Nystroem: kernel='callable' requires a kernel function")
        if self.gamma and (not isfinite(self.gamma.value()) or self.gamma.value() < 0):
            raise Error("ValueError: Nystroem: gamma must be a finite value >= 0, got " + String(self.gamma.value()))
        if self.coef0 and not isfinite(self.coef0.value()):
            raise Error("ValueError: Nystroem: coef0 must be finite, got " + String(self.coef0.value()))
        if self.degree and (not isfinite(self.degree.value()) or self.degree.value() < 1):
            raise Error("ValueError: Nystroem: degree must be a finite value >= 1, got " + String(self.degree.value()))
        _ = component_width(self.n_components)

    def resolved(self) raises -> NystroemKernelParams:
        """sklearn `_get_kernel_params`: explicit values override kernel_params.

        Callable and precomputed kernels reject gamma/coef0/degree. A native
        callable carries its own parameters, so kernel_params is rejected too.
        """
        if self.kernel == "callable" or self.kernel == "precomputed":
            if self.gamma or self.coef0 or self.degree:
                raise Error("ValueError: Don't pass gamma, coef0 or degree to Nystroem if using a callable or precomputed kernel")
            if self.kernel == "callable" and self.kernel_params:
                raise Error("ValueError: Nystroem: a native callable kernel stores its own parameters; kernel_params is not supported")
            return NystroemKernelParams(None, None, None)
        var result = self.kernel_params.value().copy() if self.kernel_params else NystroemKernelParams(None, None, None)
        if self.gamma:
            result.gamma = self.gamma
        if self.coef0:
            result.coef0 = self.coef0
        if self.degree:
            result.degree = self.degree
        return result^


def _evaluate_serial[
    dtype: DType, Kernel: RowKernel & Copyable & Deinitable
](params: NystroemParams[Kernel], X: NDArray[dtype], Y: NDArray[dtype], same_data: Bool) raises -> NDArray[dtype]:
    """Dense kernel matrix K(X, Y) with sklearn's per-kernel defaults."""
    var resolved = params.resolved()  # also rejects gamma/coef0/degree for callable/precomputed
    if params.kernel == "callable":
        return pairwise_callable(X, Y, params.kernel_function.value(), same_data=same_data)
    var gamma = resolved.gamma
    if params.kernel == "chi2" and not gamma:
        gamma = 1.0  # sklearn chi2_kernel(gamma=1.0); other kernels default to 1/n_features
    return pairwise(
        X, Y, params.kernel, gamma=gamma, degree=resolved.degree.or_else(3.0),
        coef0=resolved.coef0.or_else(1.0), same_data=same_data,
    )


def _workers(n_jobs: Optional[Int], rows: Int) raises -> Int:
    var jobs = n_jobs.or_else(1)
    if jobs == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if jobs < 0:
        jobs = max(1, (num_logical_cores() + jobs) + 1)
    return min(rows, jobs)


struct _KernelBlock[dtype: DType](Copyable, Movable):
    var values: Optional[NDArray[Self.dtype]]
    var error: Optional[String]

    def __init__(out self):
        self.values = None
        self.error = None


def _evaluate[
    dtype: DType, Kernel: RowKernel & Copyable & Deinitable
](params: NystroemParams[Kernel], X: NDArray[dtype], Y: NDArray[dtype], same_data: Bool) raises -> NDArray[dtype]:
    """Evaluate disjoint row blocks; join and propagate errors before publishing."""
    var workers = _workers(params.n_jobs, X.shape[0])
    if workers == 1 or params.kernel == "precomputed":
        return _evaluate_serial(params, X, Y, same_data)
    var blocks = List[NDArray[dtype]]()
    for i in range(workers):
        var start = X.shape[0] * i // workers
        var end = X.shape[0] * (i + 1) // workers
        blocks.append(X.view_with_layout(Shape(end - start, X.shape[1]), X.strides,
                                        X.offset + start * Int(X.strides[0])))
    var results = List[_KernelBlock[dtype]](length=workers, fill=_KernelBlock[dtype]())
    var slots = results.unsafe_ptr()
    def work(i: Int) {slots, blocks, params, Y}:
        try:
            slots[unsafe_offset=i].values = _evaluate_serial(params, blocks[i], Y, False)
        except error:
            slots[unsafe_offset=i].error = String(error)
    parallelize(work, workers, workers)
    var output = zeros[dtype](Shape(X.shape[0], Y.shape[0]))
    for i in range(workers):
        if results[i].error:
            raise Error(results[i].error.value())
        var start = X.shape[0] * i // workers
        for j in range(results[i].values.value().size):
            output.unsafe_set(start * Y.shape[0] + j, results[i].values.value().unsafe_get(j))
    return output^


def _normalization[dtype: DType](basis_kernel: NDArray[dtype]) raises NumericalError -> NDArray[dtype]:
    """U / sqrt(clip(S, 1e-12)) @ Vt of the (k, k) basis kernel, in the input dtype."""
    var factor = svd(basis_kernel)
    var u = factor.u()
    var s = factor.singular_values()
    var k = s.size
    var floor = Scalar[dtype](1e-12)
    for c in range(k):
        var root = sqrt(max(s.unsafe_get(c), floor))
        for r in range(u.shape[0]):
            u.unsafe_set(r * k + c, u.unsafe_get(r * k + c) / root)
    return u @ factor.vt()


struct Nystroem[dtype: DType = DType.float64, Kernel: RowKernel & Copyable & Deinitable = NoCallableKernel, output_dtype: DType = dtype](KernelFeatureMap):
    """Approximate a kernel map using a random subset of the training rows (dense input).

    Learned: `components_` `(k, p)` (copies of the basis rows),
    `component_indices_` Int64 `(k,)`, `normalization_` `(k, k)`,
    `n_features_in_`; k = min(n_components, n_samples).
    """

    comptime Params = NystroemParams[Self.Kernel]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.output_dtype]
    var params: Self.Params
    var _state: FeatureMapState
    var components_: NDArray[Self.dtype]
    var component_indices_: NDArray[DType.int64]
    var normalization_: NDArray[Self.output_dtype]
    var n_features_in_: Int
    var fit_warnings: List[String]

    def __init__(
        out self, kernel: String = "rbf", *, gamma: Optional[Float64] = None, coef0: Optional[Float64] = None,
        degree: Optional[Float64] = None, kernel_params: Optional[NystroemKernelParams] = None,
        n_components: Int = 100, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None,
    ) raises:
        """Named kernel (`rbf`, `poly`, ...) or `precomputed`."""
        self.params = Self.Params(kernel, gamma=gamma, coef0=coef0, degree=degree, kernel_params=kernel_params,
                                  n_components=n_components, random_state=random_state, n_jobs=n_jobs)
        self._state = FeatureMapState()
        self.components_ = NDArray[Self.dtype](Shape(0, 0))
        self.component_indices_ = NDArray[DType.int64](Shape(0))
        self.normalization_ = NDArray[Self.output_dtype](Shape(0, 0))
        self.n_features_in_ = 0
        self.fit_warnings = List[String]()

    def __init__(
        out self, kernel: Self.Kernel, *, n_components: Int = 100, random_state: Optional[Int] = None,
        n_jobs: Optional[Int] = None,
    ) raises:
        """Callable kernel: a typed `RowKernel` struct holding its own parameters."""
        self = Self("callable", n_components=n_components, random_state=random_state, n_jobs=n_jobs)
        self.params.kernel_function = kernel.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, kernel: Optional[String] = None, gamma: Optional[Float64] = None, coef0: Optional[Float64] = None,
        degree: Optional[Float64] = None, n_components: Optional[Int] = None, random_state: Optional[Int] = None,
        n_jobs: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Named partial update; pass a full Params record to reset a value to None."""
        if kernel:
            self.params.kernel = kernel.value()
        if gamma:
            self.params.gamma = gamma
        if coef0:
            self.params.coef0 = coef0
        if degree:
            self.params.degree = degree
        if n_components:
            self.params.n_components = n_components.value()
        if random_state:
            self.params.random_state = random_state
        if n_jobs:
            self.params.n_jobs = n_jobs
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.input_state.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Fit with a fresh `RandomState(random_state)`; None needs `fit_with_rng`."""
        if not self.params.random_state:
            raise Error("ValueError: Nystroem: random_state=None requires fit_with_rng(X, context)")
        var rng = RandomState(self.params.random_state.value())
        self.fit_with_rng(X, rng, y)
        return self

    def fit_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises:
        """Sample the basis with `rng.permutation(n_samples)[:k]`, then normalize its kernel."""
        validate_matrix(X)
        self.params.validate()
        self._check_output_dtype()
        _ = _workers(self.params.n_jobs, X.shape[0])
        var n_samples = X.shape[0]
        var warnings = List[String]()
        if self.params.n_components > n_samples:
            warnings.append(
                "n_components > n_samples. This is not possible.\nn_components was set to n_samples,"
                " which results in inefficient evaluation of the full kernel."
            )
        var k = basis_width(n_samples, self.params.n_components)
        var permutation = rng.permutation(n_samples)
        var indices = zeros[DType.int64](Shape(k))
        var p = X.shape[1]
        var basis = zeros[Self.dtype](Shape(k, p))
        var row_stride = Int(X.strides[0])
        var col_stride = Int(X.strides[1])
        for r in range(k):
            var source = Int(permutation.unsafe_get(r))
            indices.unsafe_set(r, Int64(source))
            for c in range(p):
                basis.unsafe_set(r * p + c, X.unsafe_get(source * row_stride + c * col_stride))
        var basis_kernel = self._kernel(basis, basis, True)
        var normalization: NDArray[Self.output_dtype]
        try:
            normalization = _normalization(basis_kernel)
        except e:
            raise Error("ValueError: Nystroem: SVD of the (" + String(k) + ", " + String(k) + ") basis kernel failed: " + e.kind + " in " + e.operation)
        self._state.record_fit(X, k)
        self.n_features_in_ = p
        self.components_ = basis^
        self.component_indices_ = indices^
        self.normalization_ = normalization^
        self.fit_warnings = warnings^

    def _check_output_dtype(self) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime assert Self.output_dtype == Self.dtype or Self.output_dtype == DType.float64
        if Self.dtype == DType.float32 and self.params.kernel == "laplacian" and _workers(self.params.n_jobs, Int.MAX) == 1:
            if Self.output_dtype != DType.float64:
                raise Error("ValueError: sequential Float32 laplacian requires Nystroem[output_dtype=DType.float64]")
        elif Self.output_dtype != Self.dtype:
            raise Error("ValueError: output_dtype must match input dtype except for sequential Float32 laplacian")

    def _kernel(self, X: Self.Input, Y: Self.Input, same_data: Bool) raises -> Self.Transformed:
        comptime if Self.output_dtype == Self.dtype:
            return rebind_var[Self.Transformed](_evaluate(self.params, X, Y, same_data))
        else:
            return _evaluate(self.params, X.astype[Self.output_dtype](), Y.astype[Self.output_dtype](), same_data)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """K(X, components_) @ normalization_.T, shape (n_samples, k)."""
        self._state.check_transform(X)
        self._check_output_dtype()
        var embedded = self._kernel(X, self.components_, False)
        return embedded @ self.normalization_.T()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform_with_rng(mut self, X: Self.Input, mut rng: RandomState, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit_with_rng(X, rng, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """`nystroem0 .. nystroem{k-1}` using the fitted basis size."""
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: Nystroem: call fit before get_feature_names_out")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: Nystroem: input_features length must match n_features_in_")
        var names = List[String]()
        for c in range(self._state.n_features_out_):
            names.append("nystroem" + String(c))
        return names^
