"""Dense incremental PCA using the sklearn/Ross augmented-SVD update."""
from numojo import NDArray, Shape, zeros, mean, sum
from ._base import InvertibleDecomposition, IncrementalDecomposition
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from ._contracts import validate_matrix
from ._pca_backend import PCABackend, RuntimePCABackend
from ._pca import _get, _set, _sqrt, _pca_transform, _pca_inverse_transform, _pca_get_covariance
from ._svd_engine import svd_flip


@fieldwise_init
struct IncrementalPCAParams(Copyable, Movable):
    var n_components: Optional[Int]
    var whiten: Bool
    var copy_input: Bool
    var batch_size: Optional[Int]


struct IncrementalPCA[dtype: DType = DType.float64, Backend: PCABackend = RuntimePCABackend](InvertibleDecomposition, IncrementalDecomposition, FeatureNamesOut, InputFeatureCount):
    comptime Params = IncrementalPCAParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input

    var n_components: Optional[Int]
    var whiten: Bool
    var copy: Bool
    var batch_size: Optional[Int]
    var components_: Optional[Self.Input]
    var singular_values_: Optional[Self.Input]
    var explained_variance_: Optional[Self.Input]
    var explained_variance_ratio_: Optional[Self.Input]
    var mean_: Optional[NDArray[DType.float64]]
    var var_: Optional[NDArray[DType.float64]]
    var n_samples_seen_: Int
    var n_features_in_: Int
    var n_components_: Int
    var batch_size_: Int
    var noise_variance_: Float64
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, n_components: Optional[Int] = None, *, whiten: Bool = False, copy: Bool = True, batch_size: Optional[Int] = None):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0
        self.n_components_ = 0
        self.batch_size_ = 0
        self.noise_variance_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components, self.whiten, self.copy, self.batch_size)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.whiten = params.whiten
        self.copy = params.copy_input
        self.batch_size = params.batch_size
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_components: Optional[Optional[Int]] = None, whiten: Optional[Bool] = None, copy: Optional[Bool] = None, batch_size: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if n_components: self.n_components = n_components.value()
        if whiten: self.whiten = whiten.value()
        if copy: self.copy = copy.value()
        if batch_size: self.batch_size = batch_size.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, whiten=self.whiten, copy=self.copy, batch_size=self.batch_size)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _check(self) raises:
        if not self.components_: raise Error("NotFittedError: IncrementalPCA has not been fitted")

    def _validate_params(self) raises:
        if self.n_components and self.n_components.value() < 1:
            raise Error("ValueError: n_components must be positive")
        if self.batch_size and self.batch_size.value() < 1:
            raise Error("ValueError: batch_size must be positive")

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.copy: raise Error("ValueError: copy=False requires partial_fit_inplace")
        # Only a batch is copied, never the entire fit input.
        validate_matrix(X)
        var batch = zeros[Self.dtype](Shape(X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]): _set(batch, i, j, _get(X, i, j))
        self._update(batch)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, check_input: Bool) raises -> ref[self] Self:
        # Validation remains mandatory for native arrays, even when requested off.
        _ = self.partial_fit(X, y)
        return self

    def partial_fit_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None, check_input: Bool = True) raises:
        self._update(X)

    def _update(mut self, mut X: Self.Input) raises:
        self._validate_params()
        validate_matrix(X)
        var n = X.shape[0]
        var p = X.shape[1]
        var old = self.n_samples_seen_
        var k = self.n_components.value() if self.n_components else (self.n_components_ if old else min(n, p))
        if k > p or (old == 0 and k > n):
            raise Error("ValueError: n_components exceeds features or first batch rows")
        if old and (p != self.n_features_in_ or k != self.n_components_):
            raise Error("ValueError: features and n_components must stay fixed across partial_fit")
        var rows = n + k + 1 if old else n
        var augmented = zeros[Self.dtype](Shape(rows, p))
        Self.Backend.validate_fit(augmented)
        var total = old + n
        var means = zeros[DType.float64](Shape(p))
        var variances = zeros[DType.float64](Shape(p))
        var batch_means = zeros[DType.float64](Shape(p))
        var denominator: Float64 = 0
        for j in range(p):
            var column = zeros[DType.float64](Shape(n))
            for i in range(n): column.unsafe_set(i, _get(X, i, j))
            var batch_mean = mean[returned_dtype=DType.float64](column)
            batch_means.unsafe_set(j, batch_mean)
            var correction: Float64 = 0
            for i in range(n):
                var residual = _get(X, i, j) - batch_mean
                column.unsafe_set(i, residual * residual)
                correction += residual
                _set(augmented, (k if old else 0) + i, j, residual)
            var m2 = sum(column) - correction * correction / Float64(n)
            var combined_mean = batch_mean
            if old:
                var previous = self.mean_.value().unsafe_get(j)
                var delta = previous - batch_mean
                combined_mean = (previous * Float64(old) + batch_mean * Float64(n)) / Float64(total)
                m2 += self.var_.value().unsafe_get(j) * Float64(old) + delta * delta * (Float64(old) / Float64(total)) * Float64(n)
                _set(augmented, rows - 1, j, _sqrt(Float64(old) / Float64(total) * Float64(n)) * delta)
                for c in range(k):
                    _set(augmented, c, j, Float64(self.singular_values_.value().unsafe_get(c)) * _get(self.components_.value(), c, j))
            means.unsafe_set(j, combined_mean)
            variances.unsafe_set(j, m2 / Float64(total))
            denominator += m2
        var factors = Self.Backend.full(augmented)
        var u = factors[0].copy()
        var s = factors[1].copy()
        var vt = factors[2].copy()
        svd_flip(u, vt, u_based_decision=False)
        var components = zeros[Self.dtype](Shape(k, p))
        var singular = zeros[Self.dtype](Shape(k))
        var explained = zeros[Self.dtype](Shape(k))
        var ratios = zeros[Self.dtype](Shape(k))
        for c in range(k):
            var squared = s.unsafe_get(c) * s.unsafe_get(c)
            singular.unsafe_set(c, s.unsafe_get(c))
            explained.unsafe_set(c, squared / Scalar[Self.dtype](total - 1))
            ratios.unsafe_set(c, Scalar[Self.dtype](Float64(squared) / denominator))
            for j in range(p): _set(components, c, j, _get(vt, c, j))
        var noise: Float64 = 0
        var rank = min(rows, p)
        # sklearn tests the incoming batch size, not the augmented row count.
        if k != n and k != p:
            for c in range(k, rank):
                noise += Float64(s.unsafe_get(c) * s.unsafe_get(c) / Scalar[Self.dtype](total - 1))
            noise /= Float64(rank - k)
        if not self.copy:
            var view = X.view()
            for i in range(n):
                for j in range(p): _set(view, i, j, _get(X, i, j) - batch_means.unsafe_get(j))
        self.components_ = components^
        self.singular_values_ = singular^
        self.explained_variance_ = explained^
        self.explained_variance_ratio_ = ratios^
        self.mean_ = means^
        self.var_ = variances^
        self.n_samples_seen_ = total
        self.n_features_in_ = p
        self.n_components_ = k
        self.noise_variance_ = noise

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.copy: raise Error("ValueError: copy=False requires fit_inplace")
        self._fit_batches(X)
        return self

    def _fit_batches(mut self, X: Self.Input) raises:
        self._validate_params()
        validate_matrix(X)
        var fresh = self.__sklearn_clone__()
        fresh.batch_size_ = self.batch_size.value() if self.batch_size else 5 * X.shape[1]
        var start = 0
        while start < X.shape[0]:
            var stop = min(start + fresh.batch_size_, X.shape[0])
            var minimum = self.n_components.value() if self.n_components else 0
            if X.shape[0] - stop < minimum: stop = X.shape[0]
            var batch = zeros[Self.dtype](Shape(stop - start, X.shape[1]))
            for i in range(stop - start):
                for j in range(X.shape[1]): _set(batch, i, j, _get(X, start + i, j))
            fresh._update(batch)
            start = stop
        self = fresh^

    def fit_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None) raises:
        # Fit a fresh model first; reproduce per-batch centering only on success.
        self._fit_batches(X)
        if not self.copy:
            var start = 0
            var view = X.view()
            while start < X.shape[0]:
                var stop = min(start + self.batch_size_, X.shape[0])
                var minimum = self.n_components.value() if self.n_components else 0
                if X.shape[0] - stop < minimum: stop = X.shape[0]
                for j in range(X.shape[1]):
                    var average: Float64 = 0
                    for i in range(start, stop): average += _get(X, i, j)
                    average /= Float64(stop - start)
                    for i in range(start, stop): _set(view, i, j, _get(X, i, j) - average)
                start = stop

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit_inplace(X, y)
        return self.transform(X)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_: raise Error("ValueError: feature count mismatch")
        return _pca_transform[Self.dtype, Self.Backend](X, self.components_.value(), self.mean_.value(), self.explained_variance_.value(), self.whiten)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check()
        return _pca_inverse_transform[Self.dtype, Self.Backend](X, self.components_.value(), self.mean_.value(), self.explained_variance_.value(), self.whiten)

    def get_covariance(self) raises -> Self.Input:
        self._check()
        return _pca_get_covariance[Self.dtype, Self.Backend](self.components_.value(), self.explained_variance_.value(), self.whiten, self.noise_variance_)

    def get_precision(self) raises -> Self.Input:
        self._check()
        return Self.Backend.inverse(self.get_covariance())

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` attribute, which exists only after fit."""
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: IncrementalPCA has no n_features_in_ before fit")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features length mismatch")
        var names = List[String]()
        for i in range(self.n_components_): names.append("incrementalpca" + String(i))
        return names^

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default": raise Error("NotImplementedError: native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
