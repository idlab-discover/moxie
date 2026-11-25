# Adapted from sklearn/decomposition/_dict_learning.py, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Batch dictionary factorization over dense NuMojo arrays."""

from numojo import NDArray, Shape, zeros
from std.math import isfinite, sqrt
from moxie._random import RandomState
from ._contracts import validate_matrix
from ._matrix import get1, get2, set2, seeded_state
from ._sparse_coding import sparse_encode
from ._svd_engine import golub_kahan_svd


@fieldwise_init
struct DictionaryLearningResult[dtype: DType](Movable):
    var code: NDArray[Self.dtype]
    var dictionary: NDArray[Self.dtype]
    var errors: List[Float64]
    var n_iter: Int


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
    callback: Optional[def(Int, NDArray[dtype], NDArray[dtype], Float64) thin raises -> None] = None,
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
