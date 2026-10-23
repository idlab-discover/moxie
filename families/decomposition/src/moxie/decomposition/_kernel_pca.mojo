"""Dense KernelPCA, following sklearn's BSD-3-Clause _kernel_pca.py.

Pairwise kernels belong to kernel_ridge.shared.numerics; eigen and randomized
SVD computations reuse the existing numerical and decomposition owners.
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, isfinite
from std.utils import Variant
from moxie.base import UnsupervisedEstimator
from moxie._kernel_ridge._kernels import pairwise, validate_kernel
from mojo_linalg import symmetric_eigh, cholesky
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import get2, set2, get1, set1, contiguous, column_means, seeded_state, feature_names
from ._svd_engine import randomized_svd

comptime KernelPower = Variant[Int, String]

@fieldwise_init
struct KernelPCAParams(Copyable, Movable):
    var n_components: Optional[Int]
    var kernel: String
    var gamma: Optional[Float64]
    var degree: Float64
    var coef0: Float64
    var kernel_params: Optional[Dict[String, Float64]]
    var alpha: Float64
    var fit_inverse_transform: Bool
    var eigen_solver: String
    var tol: Float64
    var max_iter: Optional[Int]
    var iterated_power: KernelPower
    var remove_zero_eig: Bool
    var random_state: Optional[Int]
    var copy_X: Bool
    var n_jobs: Optional[Int]

struct KernelPCA[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """Centered kernel eigenspace; optional learned kernel pre-image map."""
    comptime Params = KernelPCAParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var n_components: Optional[Int]
    var kernel: String
    var gamma: Optional[Float64]
    var degree: Float64
    var coef0: Float64
    var kernel_params: Optional[Dict[String, Float64]]
    var alpha: Float64
    var fit_inverse_transform: Bool
    var eigen_solver: String
    var tol: Float64
    var max_iter: Optional[Int]
    var iterated_power: KernelPower
    var remove_zero_eig: Bool
    var random_state: Optional[Int]
    var copy_X: Bool
    var n_jobs: Optional[Int]
    var eigenvalues_: Optional[NDArray[Self.dtype]]
    var eigenvectors_: Optional[NDArray[Self.dtype]]
    var X_fit_: Optional[NDArray[Self.dtype]]
    var X_transformed_fit_: Optional[NDArray[Self.dtype]]
    var dual_coef_: Optional[NDArray[Self.dtype]]
    var _kernel_mean: Optional[NDArray[Self.dtype]]
    var n_features_in_: Int
    var gamma_: Float64
    var _kernel_all: Float64

    def __init__(out self, n_components: Optional[Int] = None, *,
        kernel: String = "linear",
        gamma: Optional[Float64] = None,
        degree: Float64 = 3,
        coef0: Float64 = 1,
        kernel_params: Optional[Dict[String, Float64]] = None,
        alpha: Float64 = 1.0,
        fit_inverse_transform: Bool = False,
        eigen_solver: String = "auto",
        tol: Float64 = 0,
        max_iter: Optional[Int] = None,
        iterated_power: KernelPower = "auto",
        remove_zero_eig: Bool = False,
        random_state: Optional[Int] = None,
        copy_X: Bool = True,
        n_jobs: Optional[Int] = None):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components.copy()
        self.kernel = kernel.copy()
        self.gamma = gamma.copy()
        self.degree = degree.copy()
        self.coef0 = coef0.copy()
        self.kernel_params = kernel_params.copy()
        self.alpha = alpha.copy()
        self.fit_inverse_transform = fit_inverse_transform.copy()
        self.eigen_solver = eigen_solver.copy()
        self.tol = tol.copy()
        self.max_iter = max_iter.copy()
        self.iterated_power = iterated_power.copy()
        self.remove_zero_eig = remove_zero_eig.copy()
        self.random_state = random_state.copy()
        self.copy_X = copy_X.copy()
        self.n_jobs = n_jobs.copy()
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.X_fit_ = None
        self.X_transformed_fit_ = None
        self.dual_coef_ = None
        self._kernel_mean = None
        self.n_features_in_ = 0
        self.gamma_ = 0
        self._kernel_all = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components.copy(), self.kernel.copy(), self.gamma.copy(), self.degree.copy(), self.coef0.copy(), self.kernel_params.copy(), self.alpha.copy(), self.fit_inverse_transform.copy(), self.eigen_solver.copy(), self.tol.copy(), self.max_iter.copy(), self.iterated_power.copy(), self.remove_zero_eig.copy(), self.random_state.copy(), self.copy_X.copy(), self.n_jobs.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components.copy()
        self.kernel = params.kernel.copy()
        self.gamma = params.gamma.copy()
        self.degree = params.degree.copy()
        self.coef0 = params.coef0.copy()
        self.kernel_params = params.kernel_params.copy()
        self.alpha = params.alpha.copy()
        self.fit_inverse_transform = params.fit_inverse_transform.copy()
        self.eigen_solver = params.eigen_solver.copy()
        self.tol = params.tol.copy()
        self.max_iter = params.max_iter.copy()
        self.iterated_power = params.iterated_power.copy()
        self.remove_zero_eig = params.remove_zero_eig.copy()
        self.random_state = params.random_state.copy()
        self.copy_X = params.copy_X.copy()
        self.n_jobs = params.n_jobs.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *,
        n_components: Optional[Optional[Int]] = None,
        kernel: Optional[String] = None,
        gamma: Optional[Optional[Float64]] = None,
        degree: Optional[Float64] = None,
        coef0: Optional[Float64] = None,
        kernel_params: Optional[Optional[Dict[String, Float64]]] = None,
        alpha: Optional[Float64] = None,
        fit_inverse_transform: Optional[Bool] = None,
        eigen_solver: Optional[String] = None,
        tol: Optional[Float64] = None,
        max_iter: Optional[Optional[Int]] = None,
        iterated_power: Optional[KernelPower] = None,
        remove_zero_eig: Optional[Bool] = None,
        random_state: Optional[Optional[Int]] = None,
        copy_X: Optional[Bool] = None,
        n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if n_components: self.n_components = n_components.value().copy()
        if kernel: self.kernel = kernel.value().copy()
        if gamma: self.gamma = gamma.value().copy()
        if degree: self.degree = degree.value().copy()
        if coef0: self.coef0 = coef0.value().copy()
        if kernel_params: self.kernel_params = kernel_params.value().copy()
        if alpha: self.alpha = alpha.value().copy()
        if fit_inverse_transform: self.fit_inverse_transform = fit_inverse_transform.value().copy()
        if eigen_solver: self.eigen_solver = eigen_solver.value().copy()
        if tol: self.tol = tol.value().copy()
        if max_iter: self.max_iter = max_iter.value().copy()
        if iterated_power: self.iterated_power = iterated_power.value().copy()
        if remove_zero_eig: self.remove_zero_eig = remove_zero_eig.value().copy()
        if random_state: self.random_state = random_state.value().copy()
        if copy_X: self.copy_X = copy_X.value().copy()
        if n_jobs: self.n_jobs = n_jobs.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.eigenvalues_)

    def _validate(self) raises -> Int:
        if self.kernel not in [String("linear"), "poly", "rbf", "sigmoid", "cosine", "precomputed"]:
            raise Error("ValueError: unsupported KernelPCA kernel")
        if self.n_components and self.n_components.value() < 1:
            raise Error("ValueError: n_components must be positive or None")
        if self.eigen_solver not in [String("auto"), "dense", "arpack", "randomized"]:
            raise Error("ValueError: invalid eigen_solver")
        if self.gamma and not (self.gamma.value() >= 0 and isfinite(self.gamma.value())):
            raise Error("ValueError: gamma must be finite and nonnegative")
        if not (self.degree >= 0 and isfinite(self.degree) and isfinite(self.coef0)):
            raise Error("ValueError: invalid degree or coef0")
        if not (self.alpha >= 0 and isfinite(self.alpha) and self.tol >= 0 and isfinite(self.tol)):
            raise Error("ValueError: alpha and tol must be finite and nonnegative")
        if self.max_iter and self.max_iter.value() < 1:
            raise Error("ValueError: max_iter must be positive or None")
        if self.random_state and (self.random_state.value() < 0 or self.random_state.value() > 4294967295):
            raise Error("ValueError: random_state must be in [0, 2**32 - 1]")
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs must not be zero")
        if not self.copy_X:
            raise Error("NotImplementedError: KernelPCA requires copy_X=True for owned fitted snapshots")
        if self.fit_inverse_transform and self.kernel == "precomputed":
            raise Error("ValueError: inverse transform is unavailable for precomputed kernels")
        if self.iterated_power.isa[Int]():
            var count = self.iterated_power[Int]
            if count < 0: raise Error("ValueError: iterated_power must be nonnegative")
            return count
        if self.iterated_power[String] != "auto":
            raise Error("ValueError: iterated_power must be an integer or auto")
        return -1

    def _center(self, kernel: Self.Input) raises -> Self.Input:
        var result = contiguous(kernel)
        var n = kernel.shape[1]
        for i in range(kernel.shape[0]):
            var row: Float64 = 0
            for j in range(n): row += get2(kernel, i, j)
            row /= Float64(n)
            for j in range(n):
                set2(result, i, j, get2(kernel, i, j) - get1(self._kernel_mean.value(), j) - row + self._kernel_all)
        return result^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        var power = self._validate()
        validate_matrix(X)
        var n = X.shape[0]
        var k = min(n, self.n_components.value()) if self.n_components else n
        var gamma = self.gamma.value() if self.gamma else 1.0 / Float64(X.shape[1])
        var kernel = pairwise(X, X, self.kernel, gamma, self.degree, self.coef0, True)
        # KernelCenterer's matrix means expressed on accepted dense storage.
        var mean = column_means(kernel)
        var total: Float64 = 0
        for j in range(n): total += get1(mean, j)
        total /= Float64(n)
        var centered = contiguous(kernel)
        for i in range(n):
            var row: Float64 = 0
            for j in range(n): row += get2(kernel, i, j)
            row /= Float64(n)
            for j in range(n): set2(centered, i, j, get2(kernel, i, j) - get1(mean, j) - row + total)
        var values = zeros[Self.dtype](Shape(k))
        var vectors = zeros[Self.dtype](Shape(n, k))
        if self.eigen_solver == "randomized":
            var rng = seeded_state(self.random_state, "KernelPCA")
            var factors = randomized_svd(centered, k, rng, n_iter=power, flip_sign=False)
            for t in range(k):
                var dot: Float64 = 0
                for i in range(n): dot += get2(factors.U, i, t) * get2(factors.Vt, t, i)
                var sign = Float64(1 if dot > 0 else (-1 if dot < 0 else 0))
                set1(values, t, sign * get1(factors.singular_values, t))
                for i in range(n): set2(vectors, i, t, get2(factors.U, i, t))
        else:
            # Dense exact leading eigenpairs also implement the family's
            # documented dense-exact arpack adaptation (no Lanczos iterations).
            var factors = symmetric_eigh(centered)
            var all_values = factors.eigenvalues()
            var all_vectors = factors.eigenvectors()
            for t in range(k):
                set1(values, t, get1(all_values, n - 1 - t))
                for i in range(n): set2(vectors, i, t, get2(all_vectors, i, n - 1 - t))
        var largest: Float64 = get1(values, 0)
        var smallest: Float64 = largest
        for t in range(k):
            largest = max(largest, get1(values, t))
            smallest = min(smallest, get1(values, t))
        var neg_ratio: Float64 = 1e-5 if Self.dtype == DType.float64 else 5e-3
        var neg_value: Float64 = 1e-10 if Self.dtype == DType.float64 else 1e-6
        var small_ratio: Float64 = 1e-12 if Self.dtype == DType.float64 else 2e-7
        if largest < 0 or (smallest < -neg_ratio * largest and smallest < -neg_value):
            raise Error("ValueError: significant negative eigenvalues; kernel is not positive semidefinite")
        for t in range(k):
            if get1(values, t) < small_ratio * largest: set1(values, t, 0)
        # Sort after PSD cleanup; orient using the first maximal absolute entry.
        for t in range(k):
            var best = t
            for j in range(t + 1, k):
                if get1(values, j) > get1(values, best): best = j
            var temporary = get1(values, t)
            set1(values, t, get1(values, best))
            set1(values, best, temporary)
            for i in range(n):
                temporary = get2(vectors, i, t)
                set2(vectors, i, t, get2(vectors, i, best))
                set2(vectors, i, best, temporary)
            var pivot = 0
            for i in range(1, n):
                if abs(get2(vectors, i, t)) > abs(get2(vectors, pivot, t)): pivot = i
            if get2(vectors, pivot, t) < 0:
                for i in range(n): set2(vectors, i, t, -get2(vectors, i, t))
        if self.remove_zero_eig or not self.n_components:
            var positive = 0
            for t in range(k):
                if get1(values, t) > 0: positive += 1
            k = positive
        var kept_values = zeros[Self.dtype](Shape(k))
        var kept_vectors = zeros[Self.dtype](Shape(n, k))
        var transformed = zeros[Self.dtype](Shape(n, k))
        for t in range(k):
            set1(kept_values, t, get1(values, t))
            for i in range(n):
                set2(kept_vectors, i, t, get2(vectors, i, t))
                set2(transformed, i, t, get2(vectors, i, t) * sqrt(get1(values, t)))
        var dual: Optional[Self.Input] = None
        if self.fit_inverse_transform:
            var gram = pairwise(transformed, transformed, self.kernel, gamma, self.degree, self.coef0, True)
            for i in range(n): set2(gram, i, i, get2(gram, i, i) + self.alpha)
            var factor = cholesky(gram, lower=False)
            dual = factor.solve(X)
        self.eigenvalues_ = kept_values^
        self.eigenvectors_ = kept_vectors^
        self.X_fit_ = contiguous(X)
        self.X_transformed_fit_ = transformed^ if self.fit_inverse_transform else None
        self.dual_coef_ = dual^
        self._kernel_mean = mean^
        self._kernel_all = total
        self.gamma_ = gamma
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.eigenvalues_: raise Error("NotFittedError: KernelPCA is not fitted")
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_: raise Error("ValueError: wrong feature count")
        var kernel = pairwise(X, self.X_fit_.value(), self.kernel, self.gamma_, self.degree, self.coef0)
        var centered = self._center(kernel)
        var n = self.eigenvectors_.value().shape[0]
        var k = self.eigenvalues_.value().size
        var scaled = zeros[Self.dtype](Shape(n, k))
        for t in range(k):
            var value = get1(self.eigenvalues_.value(), t)
            if value > 0:
                for i in range(n): set2(scaled, i, t, get2(self.eigenvectors_.value(), i, t) / sqrt(value))
        if k == 0: return zeros[Self.dtype](Shape(X.shape[0], 0))
        return centered @ scaled

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit(X, y)
        var result = contiguous(self.eigenvectors_.value())
        for i in range(result.shape[0]):
            for t in range(result.shape[1]):
                set2(result, i, t, get2(result, i, t) * sqrt(get1(self.eigenvalues_.value(), t)))
        return result^

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        if not self.fit_inverse_transform: raise Error("NotFittedError: The fit_inverse_transform parameter was not set to True when instantiating and hence the inverse transform is not available.")
        if not self.dual_coef_: raise Error("NotFittedError: inverse map is not fitted")
        validate_matrix(X)
        if X.shape[1] != self.X_transformed_fit_.value().shape[1]: raise Error("ValueError: wrong component count")
        var kernel = pairwise(X, self.X_transformed_fit_.value(), self.kernel, self.gamma_, self.degree, self.coef0)
        return kernel @ self.dual_coef_.value()

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.eigenvalues_: raise Error("NotFittedError: KernelPCA is not fitted")
        return feature_names("kernelpca", self.eigenvalues_.value().size, input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default": raise Error("NotImplementedError: native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
