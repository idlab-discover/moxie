from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.decomposition import DictionaryLearningResult, dict_learning


def fail_callback[dtype: DType](
    iteration: Int, code: NDArray[dtype], dictionary: NDArray[dtype], cost: Float64,
) raises:
    assert_equal(iteration, 0)
    assert_equal(code.shape, Shape(3, 2))
    assert_equal(dictionary.shape, Shape(2, 2))
    assert_true(cost > 0)
    raise Error("stop from dictionary callback")


def contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(3, 2))
    x.unsafe_set(0, 1)
    x.unsafe_set(3, 2)
    var initial_code = zeros[dtype](Shape(3, 2))
    var initial_dict = zeros[dtype](Shape(2, 2))
    initial_dict.unsafe_set(0, 1)
    initial_dict.unsafe_set(3, 1)
    var code_view = initial_code.view()
    var dict_view = initial_dict.view()
    var initialized: DictionaryLearningResult[dtype] = dict_learning(
        x, 2, alpha=0.3, max_iter=0, code_init=code_view^, dict_init=dict_view^,
    )
    assert_equal(initialized.n_iter, 0)
    assert_equal(len(initialized.errors), 0)
    initialized.code.unsafe_set(0, 3)
    initialized.dictionary.unsafe_set(0, 4)
    assert_equal(Float64(initial_code.unsafe_get(0)), 0)
    assert_equal(Float64(initial_dict.unsafe_get(0)), 1)

    var reversed_columns = x.view_with_layout(x.shape, NDArrayStrides(2, -1), 1)
    var strided = dict_learning(reversed_columns, 2, alpha=2, max_iter=3, random_state=12)
    var contiguous = dict_learning(reversed_columns.T().T(), 2, alpha=2, max_iter=3, random_state=12)
    assert_equal(strided.n_iter, contiguous.n_iter)
    for i in range(strided.dictionary.size):
        assert_equal(Float64(strided.dictionary.unsafe_get(i)), Float64(contiguous.dictionary.unsafe_get(i)))
    assert_equal(Float64(x.unsafe_get(0)), 1)
    assert_equal(Float64(x.unsafe_get(3)), 2)

    with assert_raises(contains="stop from dictionary callback"):
        _ = dict_learning(x, 2, alpha=0.3, callback=fail_callback[dtype])
    with assert_raises(contains="Positive constraint"):
        _ = dict_learning(x, 2, alpha=0.3, positive_code=True)
    with assert_raises(contains="incompatible code_init and dict_init shapes"):
        _ = dict_learning(x, 2, alpha=0.3, code_init=initial_dict.copy(), dict_init=initial_dict.copy())
    with assert_raises(contains="invalid dictionary component or iteration count"):
        _ = dict_learning(x, 0, alpha=0.3)
    with assert_raises(contains="alpha and tol must be finite and nonnegative"):
        _ = dict_learning(x, 2, alpha=-1)
    with assert_raises(contains="dictionary learning method must be"):
        _ = dict_learning(x, 2, alpha=1, method="omp")
    with assert_raises(contains="n_jobs must not be zero"):
        _ = dict_learning(x, 2, alpha=1, n_jobs=0)
    with assert_raises(contains="random_state must be in"):
        _ = dict_learning(x, 2, alpha=1, random_state=-1)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS: batch dictionary learning Float32 and Float64 contracts")
