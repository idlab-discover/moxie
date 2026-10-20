"""Dense sparse coding; threshold branch of sklearn's BSD-3-Clause _dict_learning.

Regression-based coding belongs to the external linear-model solver owners.
"""
from numojo import NDArray, Shape, zeros
from std.math import abs, isfinite
from moxie.base import UnsupervisedEstimator
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import get2, set2, contiguous, feature_names


def sparse_encode[dtype: DType](
    X: NDArray[dtype], dictionary: NDArray[dtype], *, gram: Optional[NDArray[dtype]] = None,
    cov: Optional[NDArray[dtype]] = None, algorithm: String = "lasso_lars",
    n_nonzero_coefs: Optional[Int] = None, alpha: Optional[Float64] = None,
    copy_cov: Bool = True, init: Optional[NDArray[dtype]] = None, max_iter: Int = 1000,
    n_jobs: Optional[Int] = None, check_input: Bool = True, verbose: Int = 0,
    positive: Bool = False,
) raises -> NDArray[dtype]:
    """Threshold covariance D @ X.T, returning dense (samples, atoms) codes."""
    if algorithm not in [String("lasso_lars"), "lasso_cd", "lars", "omp", "threshold"]:
        raise Error("ValueError: invalid sparse coding algorithm")
    if algorithm != "threshold":
        raise Error("NotImplementedError: sparse coding requires the external linear_model LARS/CD/OMP engine; use threshold")
    if alpha and not (alpha.value() >= 0 and isfinite(alpha.value())):
        raise Error("ValueError: alpha must be finite and nonnegative")
    if max_iter < 0 or verbose < 0 or (n_nonzero_coefs and n_nonzero_coefs.value() < 1):
        raise Error("ValueError: invalid sparse coding iteration/count parameter")
    if n_jobs and n_jobs.value() == 0: raise Error("ValueError: n_jobs must not be zero")
    # Memory-safe shape/finite checks remain enabled for native borrowed inputs.
    validate_matrix(X)
    validate_matrix(dictionary)
    if dictionary.shape[1] != X.shape[1]: raise Error("ValueError: dictionary and X have different feature counts")
    var covariance = zeros[dtype](Shape(0, 0))
    if cov:
        validate_matrix(cov.value())
        if cov.value().shape[0] != dictionary.shape[0] or cov.value().shape[1] != X.shape[0]:
            raise Error("ValueError: cov must have shape (atoms, samples)")
        covariance = contiguous(cov.value())
    else:
        covariance = dictionary @ X.T()
    var threshold = Scalar[dtype](alpha.value() if alpha else 1.0)
    var code = zeros[dtype](Shape(X.shape[0], dictionary.shape[0]))
    for i in range(code.shape[0]):
        for j in range(code.shape[1]):
            var value = Scalar[dtype](get2(covariance, j, i))
            var magnitude = max(abs(value) - threshold, Scalar[dtype](0))
            if value < 0: magnitude = -magnitude
            if positive: magnitude = max(magnitude, Scalar[dtype](0))
            set2(code, i, j, Float64(magnitude))
    return code^


@fieldwise_init
struct SparseCoderParams[dtype: DType](Copyable, Movable):
    var dictionary: NDArray[Self.dtype]
    var transform_algorithm: String
    var transform_n_nonzero_coefs: Optional[Int]
    var transform_alpha: Optional[Float64]
    var split_sign: Bool
    var n_jobs: Optional[Int]
    var positive_code: Bool
    var transform_max_iter: Int


struct SparseCoder[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """Stateless coding with an owned fixed dictionary; threshold mode supported."""
    comptime Params = SparseCoderParams[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var dictionary: NDArray[Self.dtype]
    var transform_algorithm: String
    var transform_n_nonzero_coefs: Optional[Int]
    var transform_alpha: Optional[Float64]
    var split_sign: Bool
    var n_jobs: Optional[Int]
    var positive_code: Bool
    var transform_max_iter: Int
    var n_components_: Int
    var n_features_in_: Int

    def __init__(out self, dictionary: NDArray[Self.dtype], *,
        transform_algorithm: String = "omp",
        transform_n_nonzero_coefs: Optional[Int] = None,
        transform_alpha: Optional[Float64] = None,
        split_sign: Bool = False,
        n_jobs: Optional[Int] = None,
        positive_code: Bool = False,
        transform_max_iter: Int = 1000):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.dictionary = dictionary.copy()
        self.transform_algorithm = transform_algorithm.copy()
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs.copy()
        self.transform_alpha = transform_alpha.copy()
        self.split_sign = split_sign.copy()
        self.n_jobs = n_jobs.copy()
        self.positive_code = positive_code.copy()
        self.transform_max_iter = transform_max_iter.copy()
        self.n_components_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.dictionary.copy(), self.transform_algorithm.copy(), self.transform_n_nonzero_coefs.copy(), self.transform_alpha.copy(), self.split_sign.copy(), self.n_jobs.copy(), self.positive_code.copy(), self.transform_max_iter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.dictionary = params.dictionary.copy()
        self.transform_algorithm = params.transform_algorithm.copy()
        self.transform_n_nonzero_coefs = params.transform_n_nonzero_coefs.copy()
        self.transform_alpha = params.transform_alpha.copy()
        self.split_sign = params.split_sign.copy()
        self.n_jobs = params.n_jobs.copy()
        self.positive_code = params.positive_code.copy()
        self.transform_max_iter = params.transform_max_iter.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *,
        dictionary: Optional[NDArray[Self.dtype]] = None,
        transform_algorithm: Optional[String] = None,
        transform_n_nonzero_coefs: Optional[Optional[Int]] = None,
        transform_alpha: Optional[Optional[Float64]] = None,
        split_sign: Optional[Bool] = None,
        n_jobs: Optional[Optional[Int]] = None,
        positive_code: Optional[Bool] = None,
        transform_max_iter: Optional[Int] = None) raises -> ref[self] Self:
        if dictionary: self.dictionary = dictionary.value().copy()
        if transform_algorithm: self.transform_algorithm = transform_algorithm.value().copy()
        if transform_n_nonzero_coefs: self.transform_n_nonzero_coefs = transform_n_nonzero_coefs.value().copy()
        if transform_alpha: self.transform_alpha = transform_alpha.value().copy()
        if split_sign: self.split_sign = split_sign.value().copy()
        if n_jobs: self.n_jobs = n_jobs.value().copy()
        if positive_code: self.positive_code = positive_code.value().copy()
        if transform_max_iter: self.transform_max_iter = transform_max_iter.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(self.dictionary)
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def _validate(self) raises:
        validate_matrix(self.dictionary)
        if self.transform_algorithm not in [String("lasso_lars"), "lasso_cd", "lars", "omp", "threshold"]:
            raise Error("ValueError: invalid transform_algorithm")
        if self.transform_alpha and not (self.transform_alpha.value() >= 0 and isfinite(self.transform_alpha.value())):
            raise Error("ValueError: invalid transform_alpha")
        if self.transform_max_iter < 0 or (self.transform_n_nonzero_coefs and self.transform_n_nonzero_coefs.value() < 1):
            raise Error("ValueError: invalid iteration/count parameter")
        if self.n_jobs and self.n_jobs.value() == 0: raise Error("ValueError: n_jobs must not be zero")
        comptime if Self.dtype == DType.float32:
            if self.split_sign:
                raise Error("NotImplementedError: Float32 split_sign requires a Float64 output; use SparseCoder[float64] for split_sign")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self._validate()
        validate_matrix(X)
        if X.shape[1] != self.dictionary.shape[1]: raise Error("ValueError: wrong feature count")
        self.n_features_in_ = X.shape[1]
        self.n_components_ = self.dictionary.shape[0]
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._validate()
        if self.n_features_in_ > 0 and X.shape[1] != self.n_features_in_: raise Error("ValueError: wrong feature count")
        var code = sparse_encode(X, self.dictionary, algorithm=self.transform_algorithm,
            alpha=self.transform_alpha, n_nonzero_coefs=self.transform_n_nonzero_coefs,
            max_iter=self.transform_max_iter, n_jobs=self.n_jobs, positive=self.positive_code)
        if not self.split_sign: return code^
        var result = zeros[Self.dtype](Shape(X.shape[0], 2 * self.dictionary.shape[0]))
        for i in range(code.shape[0]):
            for j in range(code.shape[1]):
                set2(result, i, j, max(get2(code, i, j), Float64(0)))
                set2(result, i, j + code.shape[1], max(-get2(code, i, j), Float64(0)))
        return result^

    def transform(self, X: Self.Input, y: Optional[Self.Target]) raises -> Self.Transformed:
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._validate()
        validate_matrix(X)
        var k = self.dictionary.shape[0]
        if X.shape[1] != k * (2 if self.split_sign else 1): raise Error("ValueError: wrong code component count")
        if not self.split_sign: return X @ self.dictionary
        var code = zeros[Self.dtype](Shape(X.shape[0], k))
        for i in range(X.shape[0]):
            for j in range(k): set2(code, i, j, get2(X, i, j) - get2(X, i, j + k))
        return code @ self.dictionary

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if self.n_components_ == 0: raise Error("NotFittedError: fit before requesting output names")
        return feature_names("sparsecoder", self.n_components_, input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default": raise Error("NotImplementedError: native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
