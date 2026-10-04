# Adapted from scikit-learn sklearn/preprocessing/_data.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Stateless dense transforms: Normalizer, Binarizer, KernelCenterer and helpers.

See docs/api.md for the native adaptations (typed dtypes, in-place methods,
normalize_with_norms) shared by the family.
"""
from numojo import NDArray, Shape
from moxie.preprocessing._base import Preprocessor
from moxie.preprocessing._common import (
    at, put, matrix, vector, check_matrix, check_fitted, check_n_features,
    check_set_output, check_names_length, feature_names_in, prefixed_feature_names,
    check_writable,
)
from moxie.preprocessing import _dense_transforms as engine


def _norm_code(norm: String, owner: String) raises -> Int:
    if norm == "l1":
        return 1
    if norm == "l2":
        return 2
    if norm == "max":
        return 3
    raise Error("ValueError: The 'norm' parameter of " + owner + " must be a str among {'l1', 'l2', 'max'}. Got '" + norm + "' instead.")


def _check_axis(axis: Int, owner: String) raises:
    if axis != 0 and axis != 1:
        raise Error("ValueError: The 'axis' parameter of " + owner + " must be an int among {0, 1}. Got " + String(axis) + " instead.")


# ---------------------------------------------------------------------------
# Functions


def normalize[dtype: DType](X: NDArray[dtype], norm: String = "l2", *, axis: Int = 1, copy: Bool = True, return_norm: Bool = False) raises -> NDArray[dtype]:
    """Scale rows (axis=1) or features (axis=0) to unit l1, l2 or max norm.

    Returns a new array; `copy=False` cannot mutate a borrowed input (use
    `normalize_inplace`). `return_norm=True` changes the result type in
    sklearn: call `normalize_with_norms` instead.
    """
    var code = _norm_code(norm, "normalize")
    _check_axis(axis, "normalize")
    if return_norm:
        raise Error("ValueError: return_norm=True returns a tuple; call normalize_with_norms(X, norm, axis=axis)")
    check_matrix(X, allow_nan=False)
    return engine.normalize(X, axis=axis, norm=code)


def normalize_with_norms[dtype: DType](X: NDArray[dtype], norm: String = "l2", *, axis: Int = 1) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """sklearn `normalize(..., return_norm=True)`: (normalized X, guarded norms).

    Norms below 10 * eps are returned as 1, exactly as sklearn returns them.
    """
    var code = _norm_code(norm, "normalize")
    _check_axis(axis, "normalize")
    check_matrix(X, allow_nan=False)
    var norms = vector[dtype](X.shape[0] if axis == 1 else X.shape[1])
    engine.norms_into(X, norms, axis=axis, norm=code)
    return (engine.normalize(X, axis=axis, norm=code), norms^)


def normalize_inplace[dtype: DType](mut X: NDArray[dtype], norm: String = "l2", *, axis: Int = 1) raises:
    """Native copy=False: normalize a writable array in place (no allocation)."""
    var code = _norm_code(norm, "normalize")
    _check_axis(axis, "normalize")
    check_matrix(X, allow_nan=False)
    check_writable(X)
    engine.normalize_inplace(X, axis=axis, norm=code)


def binarize[dtype: DType](X: NDArray[dtype], *, threshold: Float64 = 0.0, copy: Bool = True) raises -> NDArray[dtype]:
    """Values strictly greater than `threshold` (cast to X's dtype) become 1, others 0."""
    check_matrix(X, allow_nan=False, min_samples=0, min_features=0)
    var unused = vector[dtype](0)
    return engine.transform(X, unused, unused, operation=5, lower=Scalar[dtype](threshold))


def binarize_inplace[dtype: DType](mut X: NDArray[dtype], *, threshold: Float64 = 0.0) raises:
    """Native copy=False: threshold a writable array in place."""
    check_matrix(X, allow_nan=False, min_samples=0, min_features=0)
    check_writable(X)
    var unused = vector[dtype](0)
    engine.transform_inplace(X, unused, unused, operation=5, lower=Scalar[dtype](threshold))


def add_dummy_feature[dtype: DType](X: NDArray[dtype], value: Float64 = 1.0) raises -> NDArray[DType.float64]:
    """Prepend a constant column. As in sklearn's dense hstack, output is Float64."""
    check_matrix(X, allow_nan=False)
    var n = X.shape[0]
    var p = X.shape[1]
    var result = matrix[DType.float64](n, p + 1)
    for i in range(n):
        result.unsafe_set(i * (p + 1), value)
        for j in range(p):
            result.unsafe_set(i * (p + 1) + j + 1, Float64(at(X, i, j)))
    return result^


# ---------------------------------------------------------------------------
# Normalizer


@fieldwise_init
struct NormalizerParams(Copyable, Movable):
    var norm: String
    var copy_input: Bool


struct Normalizer[dtype: DType = DType.float64](Preprocessor):
    """Normalize samples individually to unit norm. Stateless: transform works unfitted."""
    comptime Params = NormalizerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var norm: String
    var copy: Bool
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, norm: String = "l2", *, copy: Bool = True):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.norm = norm
        self.copy = copy
        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.norm, self.copy)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.norm = params.norm
        self.copy = params.copy_input
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        norm: Optional[String] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if norm:
            self.norm = norm.value()
        if copy:
            self.copy = copy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.norm, copy=self.copy)

    def __sklearn_is_fitted__(self) -> Bool:
        # Stateless (sklearn requires_fit=False).
        return True

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = _norm_code(self.norm, "Normalizer")
        check_matrix(X, allow_nan=False)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = None
        return self

    def fit_named[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=False)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def _check(self, X: Self.Input) raises -> Int:
        var code = _norm_code(self.norm, "Normalizer")
        check_matrix(X, allow_nan=False)
        if self.n_features_in_ > 0:
            check_n_features("Normalizer", self.n_features_in_, X.shape[1])
        return code

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return engine.normalize(X, axis=1, norm=self._check(X))

    def transform(self, X: Self.Input, copy: Optional[Bool]) raises -> Self.Transformed:
        """`copy` is accepted for API parity; the borrowed input is never modified."""
        return self.transform(X)

    def transform_inplace(self, mut X: Self.Input) raises:
        var code = self._check(X)
        check_writable(X)
        engine.normalize_inplace(X, axis=1, norm=code)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.n_features_in_ > 0, "Normalizer")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# Binarizer


@fieldwise_init
struct BinarizerParams(Copyable, Movable):
    var threshold: Float64
    var copy_input: Bool


struct Binarizer[dtype: DType = DType.float64](Preprocessor):
    """Threshold features: values > threshold map to 1, others to 0. Stateless."""
    comptime Params = BinarizerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var threshold: Float64
    var copy: Bool
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, threshold: Float64 = 0.0, copy: Bool = True):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.threshold = threshold
        self.copy = copy
        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.threshold, self.copy)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.threshold = params.threshold
        self.copy = params.copy_input
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        threshold: Optional[Float64] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if threshold:
            self.threshold = threshold.value()
        if copy:
            self.copy = copy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(threshold=self.threshold, copy=self.copy)

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=False)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = None
        return self

    def fit_named[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=False)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def _check(self, X: Self.Input) raises:
        check_matrix(X, allow_nan=False)
        if self.n_features_in_ > 0:
            check_n_features("Binarizer", self.n_features_in_, X.shape[1])

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return binarize(X, threshold=self.threshold)

    def transform(self, X: Self.Input, copy: Optional[Bool]) raises -> Self.Transformed:
        """`copy` is accepted for API parity; the borrowed input is never modified."""
        return self.transform(X)

    def transform_inplace(self, mut X: Self.Input) raises:
        self._check(X)
        binarize_inplace(X, threshold=self.threshold)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.n_features_in_ > 0, "Binarizer")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# KernelCenterer


@fieldwise_init
struct KernelCentererParams(Copyable, Movable):
    pass


struct KernelCenterer[dtype: DType = DType.float64](Preprocessor):
    """Center an arbitrary kernel matrix K (equivalent to centering phi(X))."""
    comptime Params = KernelCentererParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var K_fit_rows_: NDArray[Self.dtype]
    var K_fit_all_: Scalar[Self.dtype]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.K_fit_rows_ = vector[Self.dtype](0)
        self.K_fit_all_ = 0
        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params()

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        """X is the training kernel K of shape (n_samples, n_samples)."""
        check_matrix(X, allow_nan=False)
        var n = X.shape[0]
        if X.shape[1] != n:
            raise Error("ValueError: Kernel matrix must be a square matrix. Input is a " + String(X.shape[0]) + "x" + String(X.shape[1]) + " matrix.")
        var rows = vector[Self.dtype](n)
        # np.sum(axis=0): serial over rows for each column, in input precision.
        for i in range(n):
            for j in range(n):
                rows.unsafe_set(j, rows.unsafe_get(j) + at(X, i, j))
        var total = Scalar[Self.dtype](0)
        for j in range(n):
            rows.unsafe_set(j, rows.unsafe_get(j) / Scalar[Self.dtype](n))
        for j in range(n):
            total += rows.unsafe_get(j)
        self.K_fit_rows_ = rows^
        self.K_fit_all_ = total / Scalar[Self.dtype](n)
        self.n_features_in_ = n
        self.feature_names_in_ = None
        return self

    def fit_named[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=False)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def _center(self, X: Self.Input, mut out: NDArray[Self.dtype]) raises:
        check_fitted(self.__sklearn_is_fitted__(), "KernelCenterer")
        check_matrix(X, allow_nan=False)
        check_n_features("KernelCenterer", self.n_features_in_, X.shape[1])
        var n_fit = Scalar[Self.dtype](self.K_fit_rows_.shape[0])
        for i in range(X.shape[0]):
            var row = Scalar[Self.dtype](0)
            for j in range(X.shape[1]):
                row += at(X, i, j)
            var pred = row / n_fit
            for j in range(X.shape[1]):
                var v = at(X, i, j) - self.K_fit_rows_.unsafe_get(j)
                v -= pred
                put(out, i, j, v + self.K_fit_all_)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """X is K(X_test, X_fit) of shape (n_test, n_fit)."""
        var out = matrix[Self.dtype](X.shape[0], X.shape[1]) if X.ndim == 2 else matrix[Self.dtype](0, 0)
        self._center(X, out)
        return out^

    def transform(self, X: Self.Input, copy: Bool) raises -> Self.Transformed:
        """`copy` is accepted for API parity; use transform_inplace to mutate."""
        return self.transform(X)

    def transform_inplace(self, mut X: Self.Input) raises:
        check_fitted(self.__sklearn_is_fitted__(), "KernelCenterer")
        check_matrix(X, allow_nan=False)
        check_n_features("KernelCenterer", self.n_features_in_, X.shape[1])
        check_writable(X)
        var n_fit = Scalar[Self.dtype](self.K_fit_rows_.shape[0])
        for i in range(X.shape[0]):
            # The row sum is taken before the row is overwritten.
            var row = Scalar[Self.dtype](0)
            for j in range(X.shape[1]):
                row += at(X, i, j)
            var pred = row / n_fit
            for j in range(X.shape[1]):
                var v = at(X, i, j) - self.K_fit_rows_.unsafe_get(j)
                v -= pred
                put(X, i, j, v + self.K_fit_all_)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.__sklearn_is_fitted__(), "KernelCenterer")
        return prefixed_feature_names("kernelcenterer", self.n_features_in_, self.n_features_in_, self.feature_names_in_, input_features)
