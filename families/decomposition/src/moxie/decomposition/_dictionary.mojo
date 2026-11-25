# Adapted from sklearn/decomposition/_dict_learning.py, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Batch dictionary factorization over dense NuMojo arrays."""

from numojo import NDArray, Shape, zeros
from std.math import isfinite, sqrt
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._random import RandomState
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import feature_names, get1, get2, set2, seeded_state
from ._sparse_coding import (
    reconstruct_code, sparse_encode, split_signed_code, validate_coding_options,
)
from ._svd_engine import golub_kahan_svd


comptime _DictionaryCallback[dtype: DType] = def(Int, NDArray[dtype], NDArray[dtype], Float64) thin raises -> None


@fieldwise_init
struct DictionaryLearningResult[dtype: DType](Movable):
    var code: NDArray[Self.dtype]
    var dictionary: NDArray[Self.dtype]
    var errors: List[Float64]
    var n_iter: Int

    def _take_code(deinit self) -> NDArray[Self.dtype]:
        return self.code^


def _initialize[dtype: DType](
    X: NDArray[dtype], n_components: Int,
    code_init: Optional[NDArray[dtype]], dict_init: Optional[NDArray[dtype]],
) raises -> DictionaryLearningResult[dtype]:
    var code = zeros[dtype](Shape(X.shape[0], n_components))
    var dictionary = zeros[dtype](Shape(n_components, X.shape[1]))
    if Bool(code_init) and Bool(dict_init):
        ref initial_code = code_init.value()
        ref initial_dict = dict_init.value()
        validate_matrix(initial_code)
        validate_matrix(initial_dict)
        if initial_code.shape != Shape(X.shape[0], initial_dict.shape[0]) or initial_dict.shape[1] != X.shape[1]:
            raise Error("ValueError: incompatible code_init and dict_init shapes")
        var rank = min(n_components, initial_dict.shape[0])
        for i in range(X.shape[0]):
            for k in range(rank):
                set2(code, i, k, get2(initial_code, i, k))
        for k in range(rank):
            for j in range(X.shape[1]):
                set2(dictionary, k, j, get2(initial_dict, k, j))
    else:
        var factors = golub_kahan_svd(X)
        var rank = min(n_components, factors.singular_values.shape[0])
        for i in range(X.shape[0]):
            for k in range(rank):
                set2(code, i, k, get2(factors.U, i, k))
        for k in range(rank):
            for j in range(X.shape[1]):
                set2(dictionary, k, j, get1(factors.singular_values, k) * get2(factors.Vt, k, j))
    return DictionaryLearningResult[dtype](code^, dictionary^, List[Float64](), 0)


def _resample_atom[dtype: DType](
    X: NDArray[dtype], mut dictionary: NDArray[dtype],
    mut code: NDArray[dtype], atom: Int, mut rng: RandomState,
) raises:
    var sample = rng.randint(X.shape[0])
    var features = X.shape[1]
    var mean: Float64 = 0
    for j in range(features):
        mean += get2(X, sample, j)
    mean /= Float64(features)
    var variance: Float64 = 0
    for j in range(features):
        var delta = get2(X, sample, j) - mean
        variance += delta * delta
    var deviation = sqrt(variance / Float64(features))
    var noise_scale = 0.01 * (deviation if deviation != 0 else 1)
    for j in range(features):
        set2(dictionary, atom, j, get2(X, sample, j) + rng.normal(scale=noise_scale))
    for i in range(X.shape[0]):
        set2(code, i, atom, 0)


def _update_dictionary[dtype: DType](
    X: NDArray[dtype], mut dictionary: NDArray[dtype],
    mut code: NDArray[dtype], mut rng: RandomState,
    positive: Bool, verbose: Int,
) raises:
    var transposed = code.T()
    var gram = transposed @ code
    var cross = transposed @ X
    var n_unused = 0
    for k in range(dictionary.shape[0]):
        var weight = get2(gram, k, k)
        if weight > 1e-6:
            for j in range(dictionary.shape[1]):
                # Gauss-Seidel uses the newest values of atoms already updated.
                var product: Float64 = 0
                for atom in range(dictionary.shape[0]):
                    product += get2(gram, k, atom) * get2(dictionary, atom, j)
                set2(dictionary, k, j, get2(dictionary, k, j) + (get2(cross, k, j) - product) / weight)
        else:
            _resample_atom(X, dictionary, code, k, rng)
            n_unused += 1

        var squared_norm: Float64 = 0
        for j in range(dictionary.shape[1]):
            var value = get2(dictionary, k, j)
            if positive and value < 0:
                value = 0
                set2(dictionary, k, j, value)
            squared_norm += value * value
        var norm = sqrt(squared_norm)
        if norm > 1:
            for j in range(dictionary.shape[1]):
                set2(dictionary, k, j, get2(dictionary, k, j) / norm)
    if verbose > 0 and n_unused > 0:
        print(n_unused, "unused atoms resampled.")


def _objective[dtype: DType](
    X: NDArray[dtype], code: NDArray[dtype], dictionary: NDArray[dtype], alpha: Float64,
) raises -> Float64:
    var reconstruction = code @ dictionary
    var squared_error: Float64 = 0
    var l1: Float64 = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var residual = get2(X, i, j) - get2(reconstruction, i, j)
            squared_error += residual * residual
        for k in range(code.shape[1]):
            l1 += abs(get2(code, i, k))
    return 0.5 * squared_error + alpha * l1


def dict_learning[dtype: DType](
    X: NDArray[dtype], n_components: Optional[Int], *, alpha: Float64,
    max_iter: Int = 100, tol: Float64 = 1e-8, method: String = "lars",
    n_jobs: Optional[Int] = None, dict_init: Optional[NDArray[dtype]] = None,
    code_init: Optional[NDArray[dtype]] = None,
    callback: Optional[_DictionaryCallback[dtype]] = None,
    verbose: Int = 0, random_state: Optional[Int] = None, positive_dict: Bool = False,
    positive_code: Bool = False, method_max_iter: Int = 1000,
) raises -> DictionaryLearningResult[dtype]:
    """Alternately fit sparse codes and unit-ball dictionary atoms."""
    validate_matrix(X)
    var count = n_components.or_else(X.shape[1])
    if count < 1 or max_iter < 0 or method_max_iter < 0 or verbose < 0:
        raise Error("ValueError: invalid dictionary component or iteration count")
    if not (alpha >= 0 and isfinite(alpha) and tol >= 0 and isfinite(tol)):
        raise Error("ValueError: alpha and tol must be finite and nonnegative")
    if method != "lars" and method != "cd":
        raise Error("ValueError: dictionary learning method must be 'lars' or 'cd'")
    if positive_code and method == "lars":
        raise Error("ValueError: Positive constraint not supported for 'lars' coding method.")
    if n_jobs.or_else(1) == 0:
        raise Error("ValueError: n_jobs must not be zero")
    var rng = seeded_state(random_state, "dict_learning")
    var result = _initialize(X, count, code_init, dict_init)
    var algorithm = "lasso_lars" if method == "lars" else "lasso_cd"
    for iteration in range(max_iter):
        # Zero CD sweeps preserve the current code while still updating the atoms.
        if method != "cd" or method_max_iter > 0:
            var initial: Optional[NDArray[dtype]] = None
            if method == "cd":
                ref code = result.code
                initial = code.view_with_layout(code.shape, code.strides, code.offset)
            result.code = sparse_encode(X, result.dictionary, algorithm=algorithm, alpha=alpha,
                init=initial, n_jobs=n_jobs, max_iter=method_max_iter, positive=positive_code)
        _update_dictionary(X, result.dictionary, result.code, rng, positive_dict, verbose)
        var cost = _objective(X, result.code, result.dictionary, alpha)
        result.errors.append(cost)
        if verbose > 0:
            print("Dictionary learning iteration", iteration, "objective", cost)
        if iteration > 0 and result.errors[iteration - 1] - cost < tol * cost:
            break
        if iteration % 5 == 0 and callback:
            callback.value()(iteration, result.code, result.dictionary, cost)
    result.n_iter = len(result.errors)
    return result^


@fieldwise_init
struct DictionaryLearningParams[dtype: DType](Copyable, Movable):
    var n_components: Optional[Int]
    var alpha: Float64
    var max_iter: Int
    var tol: Float64
    var fit_algorithm: String
    var transform_algorithm: String
    var transform_n_nonzero_coefs: Optional[Int]
    var transform_alpha: Optional[Float64]
    var n_jobs: Optional[Int]
    var code_init: Optional[NDArray[Self.dtype]]
    var dict_init: Optional[NDArray[Self.dtype]]
    var callback: Optional[_DictionaryCallback[Self.dtype]]
    var verbose: Int
    var split_sign: Bool
    var random_state: Optional[Int]
    var positive_code: Bool
    var positive_dict: Bool
    var transform_max_iter: Int


struct DictionaryLearning[dtype: DType = DType.float64](
    InvertibleDecomposition, FeatureNamesOut, InputFeatureCount,
):
    """Learn a dense dictionary and encode data as sparse combinations of its atoms."""
    comptime Params = DictionaryLearningParams[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: Optional[Int]
    var alpha: Float64
    var max_iter: Int
    var tol: Float64
    var fit_algorithm: String
    var transform_algorithm: String
    var transform_n_nonzero_coefs: Optional[Int]
    var transform_alpha: Optional[Float64]
    var n_jobs: Optional[Int]
    var code_init: Optional[NDArray[Self.dtype]]
    var dict_init: Optional[NDArray[Self.dtype]]
    var callback: Optional[_DictionaryCallback[Self.dtype]]
    var verbose: Int
    var split_sign: Bool
    var random_state: Optional[Int]
    var positive_code: Bool
    var positive_dict: Bool
    var transform_max_iter: Int

    var components_: NDArray[Self.dtype]
    var error_: List[Float64]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self, n_components: Optional[Int] = None, *,
        alpha: Float64 = 1,
        max_iter: Int = 1000,
        tol: Float64 = 1e-8,
        fit_algorithm: String = "lars",
        transform_algorithm: String = "omp",
        transform_n_nonzero_coefs: Optional[Int] = None,
        transform_alpha: Optional[Float64] = None,
        n_jobs: Optional[Int] = None,
        code_init: Optional[NDArray[Self.dtype]] = None,
        dict_init: Optional[NDArray[Self.dtype]] = None,
        callback: Optional[_DictionaryCallback[Self.dtype]] = None,
        verbose: Int = 0,
        split_sign: Bool = False,
        random_state: Optional[Int] = None,
        positive_code: Bool = False,
        positive_dict: Bool = False,
        transform_max_iter: Int = 1000,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components.copy()
        self.alpha = alpha.copy()
        self.max_iter = max_iter.copy()
        self.tol = tol.copy()
        self.fit_algorithm = fit_algorithm.copy()
        self.transform_algorithm = transform_algorithm.copy()
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs.copy()
        self.transform_alpha = transform_alpha.copy()
        self.n_jobs = n_jobs.copy()
        self.code_init = code_init.copy()
        self.dict_init = dict_init.copy()
        self.callback = callback.copy()
        self.verbose = verbose.copy()
        self.split_sign = split_sign.copy()
        self.random_state = random_state.copy()
        self.positive_code = positive_code.copy()
        self.positive_dict = positive_dict.copy()
        self.transform_max_iter = transform_max_iter.copy()
        self.components_ = zeros[Self.dtype](Shape(0, 0))
        self.error_ = []
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.n_components.copy(),
            self.alpha.copy(),
            self.max_iter.copy(),
            self.tol.copy(),
            self.fit_algorithm.copy(),
            self.transform_algorithm.copy(),
            self.transform_n_nonzero_coefs.copy(),
            self.transform_alpha.copy(),
            self.n_jobs.copy(),
            self.code_init.copy(),
            self.dict_init.copy(),
            self.callback.copy(),
            self.verbose.copy(),
            self.split_sign.copy(),
            self.random_state.copy(),
            self.positive_code.copy(),
            self.positive_dict.copy(),
            self.transform_max_iter.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_components = params.n_components.copy()
        self.alpha = params.alpha.copy()
        self.max_iter = params.max_iter.copy()
        self.tol = params.tol.copy()
        self.fit_algorithm = params.fit_algorithm.copy()
        self.transform_algorithm = params.transform_algorithm.copy()
        self.transform_n_nonzero_coefs = params.transform_n_nonzero_coefs.copy()
        self.transform_alpha = params.transform_alpha.copy()
        self.n_jobs = params.n_jobs.copy()
        self.code_init = params.code_init.copy()
        self.dict_init = params.dict_init.copy()
        self.callback = params.callback.copy()
        self.verbose = params.verbose.copy()
        self.split_sign = params.split_sign.copy()
        self.random_state = params.random_state.copy()
        self.positive_code = params.positive_code.copy()
        self.positive_dict = params.positive_dict.copy()
        self.transform_max_iter = params.transform_max_iter.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        n_components: Optional[Optional[Int]] = None,
        alpha: Optional[Float64] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        fit_algorithm: Optional[String] = None,
        transform_algorithm: Optional[String] = None,
        transform_n_nonzero_coefs: Optional[Optional[Int]] = None,
        transform_alpha: Optional[Optional[Float64]] = None,
        n_jobs: Optional[Optional[Int]] = None,
        code_init: Optional[Optional[NDArray[Self.dtype]]] = None,
        dict_init: Optional[Optional[NDArray[Self.dtype]]] = None,
        callback: Optional[Optional[_DictionaryCallback[Self.dtype]]] = None,
        verbose: Optional[Int] = None,
        split_sign: Optional[Bool] = None,
        random_state: Optional[Optional[Int]] = None,
        positive_code: Optional[Bool] = None,
        positive_dict: Optional[Bool] = None,
        transform_max_iter: Optional[Int] = None,
    ) raises -> ref[self] Self:
        if n_components:
            self.n_components = n_components.value().copy()
        if alpha:
            self.alpha = alpha.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        if tol:
            self.tol = tol.value().copy()
        if fit_algorithm:
            self.fit_algorithm = fit_algorithm.value().copy()
        if transform_algorithm:
            self.transform_algorithm = transform_algorithm.value().copy()
        if transform_n_nonzero_coefs:
            self.transform_n_nonzero_coefs = transform_n_nonzero_coefs.value().copy()
        if transform_alpha:
            self.transform_alpha = transform_alpha.value().copy()
        if n_jobs:
            self.n_jobs = n_jobs.value().copy()
        if code_init:
            self.code_init = code_init.value().copy()
        if dict_init:
            self.dict_init = dict_init.value().copy()
        if callback:
            self.callback = callback.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if split_sign:
            self.split_sign = split_sign.value().copy()
        if random_state:
            self.random_state = random_state.value().copy()
        if positive_code:
            self.positive_code = positive_code.value().copy()
        if positive_dict:
            self.positive_dict = positive_dict.value().copy()
        if transform_max_iter:
            self.transform_max_iter = transform_max_iter.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            self.n_components, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
            fit_algorithm=self.fit_algorithm, transform_algorithm=self.transform_algorithm,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
            transform_alpha=self.transform_alpha, n_jobs=self.n_jobs, code_init=self.code_init,
            dict_init=self.dict_init, callback=self.callback, verbose=self.verbose,
            split_sign=self.split_sign, random_state=self.random_state,
            positive_code=self.positive_code, positive_dict=self.positive_dict,
            transform_max_iter=self.transform_max_iter,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self.fit_transform(X, y)
        return self

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        """Fit the dictionary and return the training codes from the solver."""
        validate_coding_options(
            self.transform_algorithm, self.transform_alpha, self.transform_n_nonzero_coefs,
            self.transform_max_iter, self.n_jobs,
        )
        var result = dict_learning(
            X, self.n_components, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
            method=self.fit_algorithm, n_jobs=self.n_jobs, dict_init=self.dict_init,
            code_init=self.code_init, callback=self.callback, verbose=self.verbose,
            random_state=self.random_state, positive_dict=self.positive_dict,
            positive_code=self.positive_code, method_max_iter=self.transform_max_iter,
        )
        swap(self.components_, result.dictionary)
        swap(self.error_, result.errors)
        self.n_iter_ = result.n_iter
        self.n_features_in_ = X.shape[1]
        return result^._take_code()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Recode input rows with the fitted atoms and transform parameters."""
        _ = self.get_n_features_in()
        var code = sparse_encode(
            X, self.components_, algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=self.transform_alpha.or_else(self.alpha),
            max_iter=self.transform_max_iter, n_jobs=self.n_jobs, positive=self.positive_code,
        )
        return split_signed_code(code) if self.split_sign else code^

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _ = self.get_n_features_in()
        return reconstruct_code(X, self.components_, self.split_sign)

    def get_n_features_in(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: DictionaryLearning must be fitted first")
        return self.n_features_in_

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None,
    ) raises -> List[String]:
        var n_features_in = self.get_n_features_in()
        return feature_names(
            "dictionarylearning", self.components_.shape[0], input_features, n_features_in,
        )

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
