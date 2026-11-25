from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.decomposition import (
    DictionaryLearning, DictionaryLearningParams, DictionaryLearningResult, dict_learning,
)


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


comptime _Callback[dtype: DType] = def(Int, NDArray[dtype], NDArray[dtype], Float64) thin raises -> None


def estimator_contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(3, 2))
    x.unsafe_set(0, 1)
    x.unsafe_set(3, -2)
    var initial_code = zeros[dtype](Shape(3, 2))
    initial_code.unsafe_set(0, 7)
    var initial_dict = zeros[dtype](Shape(2, 2))
    initial_dict.unsafe_set(0, 1)
    initial_dict.unsafe_set(3, 1)
    var model = DictionaryLearning[dtype](
        2, alpha=0.25, max_iter=0, code_init=initial_code.view(),
        dict_init=initial_dict.view(), random_state=12,
    )
    assert_equal(model.get_params().transform_algorithm, "omp")
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(x)
    with assert_raises(contains="NotFittedError"):
        _ = model.inverse_transform(initial_code)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()

    # Both initializer configuration and fitted state own independent arrays.
    initial_code.unsafe_set(0, 9)
    initial_dict.unsafe_set(0, 11)
    var training: NDArray[dtype] = model.fit_transform(x, initial_code.copy())
    assert_equal(training.shape, Shape(3, 2))
    assert_equal(Float64(training.unsafe_get(0)), 7)
    assert_equal(Float64(model.components_.unsafe_get(0)), 1)
    assert_equal(model.get_n_features_in(), 2)
    assert_equal(model.n_iter_, 0)
    assert_equal(len(model.error_), 0)
    var omp = model.transform(x)
    assert_equal(Float64(omp.unsafe_get(0)), 1)
    assert_equal(Float64(omp.unsafe_get(3)), -2)

    var params: DictionaryLearningParams[dtype] = model.get_params()
    params.dict_init.value().unsafe_set(0, 31)
    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    assert_equal(Float64(fresh.fit_transform(x).unsafe_get(0)), 7)
    assert_equal(Float64(fresh.components_.unsafe_get(0)), 1)
    fresh.components_.unsafe_set(0, 45)
    assert_equal(Float64(model.components_.unsafe_get(0)), 1)

    _ = model.set_params(transform_algorithm="threshold", split_sign=True)
    var split: NDArray[dtype] = model.transform(x)
    assert_equal(split.shape, Shape(3, 4))
    assert_equal(Float64(split.unsafe_get(0)), 0.75)
    assert_equal(Float64(split.unsafe_get(7)), 1.75)
    var inverse: NDArray[dtype] = model.inverse_transform(split)
    assert_equal(Float64(inverse.unsafe_get(0)), 0.75)
    assert_equal(Float64(inverse.unsafe_get(3)), -1.75)
    assert_equal(model.get_feature_names_out(), [String("dictionarylearning0"), "dictionarylearning1"])
    with assert_raises(contains="wrong code component count"):
        _ = model.inverse_transform(training)
    with assert_raises(contains="different feature counts"):
        _ = model.transform(split)
    with assert_raises(contains="input_features should have length"):
        var names: List[String] = ["x"]
        _ = model.get_feature_names_out(names^)

    _ = model.set_params(max_iter=2, callback=Optional[_Callback[dtype]](fail_callback[dtype]))
    with assert_raises(contains="stop from dictionary callback"):
        _ = model.fit(x)
    assert_equal(model.get_n_features_in(), 2)
    assert_equal(model.n_iter_, 0)
    assert_equal(Float64(model.components_.unsafe_get(0)), 1)

    params = model.get_params()
    params.callback = None
    params.code_init = None
    params.dict_init = None
    params.n_components = None
    params.fit_algorithm = "cd"
    _ = model.set_params(params)
    var wide = zeros[dtype](Shape(3, 4))
    _ = model.fit(wide)
    assert_equal(model.get_n_features_in(), 4)
    assert_equal(model.components_.shape, Shape(4, 4))
    assert_equal(model.n_iter_, 2)
    assert_equal(len(model.error_), 2)
    _ = model.set_params(transform_algorithm="invalid")
    with assert_raises(contains="invalid transform_algorithm"):
        _ = model.fit(x)
    assert_equal(model.get_n_features_in(), 4)
    _ = model.set_params(transform_algorithm="threshold")
    var vector = zeros[dtype](Shape(4))
    with assert_raises(contains="expected a dense rank-2 matrix"):
        _ = model.fit(vector)
    with assert_raises(contains="expected a dense rank-2 matrix"):
        _ = model.transform(vector)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    estimator_contracts[DType.float32]()
    estimator_contracts[DType.float64]()
    print("PASS: batch dictionary function and estimator Float32 and Float64 contracts")
