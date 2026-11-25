"""Dense coding, parameter and layout contracts."""
from std.testing import assert_equal, assert_false, assert_raises
from std.utils.numerics import inf
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import SparseCoder, sparse_encode


def check[dtype: DType]() raises:
    var d = zeros[dtype](Shape(2, 3))
    d.unsafe_set(0, 1)
    d.unsafe_set(4, 1)
    var x = zeros[dtype](Shape(2, 3))
    x.unsafe_set(0, 3)
    x.unsafe_set(1, -2)
    var model = SparseCoder[dtype](d, transform_algorithm="threshold")
    var z = model.transform(x)  # Stateless transform before fit is intentional.
    assert_equal(z.unsafe_get(0), 2)
    assert_equal(z.unsafe_get(1), -1)
    assert_equal(model.inverse_transform(z).unsafe_get(0), 2)
    model.fit(x)
    assert_equal(model.n_components_, 2)
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.get_feature_names_out()[0], "sparsecoder0")
    d.unsafe_set(0, 99)
    assert_equal(model.dictionary.unsafe_get(0), 1)
    var c = clone(model)
    assert_false(c.__sklearn_is_fitted__())
    assert_equal(c.transform(x).unsafe_get(0), 2)
    var params = c.get_params()
    params.transform_alpha = 0.0
    c.set_params(params)
    assert_equal(c.fit_transform(x).unsafe_get(0), 3)
    c.set_params(positive_code=True)
    assert_equal(c.transform(x).unsafe_get(1), 0)
    var reversed = x.view_with_layout(Shape(2, 3), NDArrayStrides(-3, 1), 3)
    assert_equal(c.transform(reversed).unsafe_get(2), 3)
    with assert_raises(contains="NotImplementedError: sparse coding requires"):
        _ = sparse_encode(x, d)  # default lasso_lars unavailable
    with assert_raises(contains="wrong feature count"):
        _ = model.transform(zeros[dtype](Shape(1, 4)))
    with assert_raises(contains="wrong code component count"):
        _ = model.inverse_transform(zeros[dtype](Shape(1, 3)))
    with assert_raises(contains="alpha must be finite and nonnegative"):
        _ = sparse_encode(x, d, algorithm="threshold", alpha=-1.0)
    with assert_raises(contains="cov must have shape"):
        _ = sparse_encode(x, d, algorithm="threshold", cov=zeros[dtype](Shape(1, 1)))
    with assert_raises(contains="insufficient samples"):
        _ = sparse_encode(zeros[dtype](Shape(0, 3)), d, algorithm="threshold")
    x.unsafe_set(0, inf[dtype]())
    with assert_raises(contains="input must be finite"):
        _ = model.transform(x)
    x.unsafe_set(0, 3)
    comptime if dtype == DType.float32:
        c.set_params(split_sign=True)
        with assert_raises(contains="Float32 split_sign requires a Float64 output"):
            _ = c.transform(x)


def lasso_cd[dtype: DType]() raises:
    var dictionary = zeros[dtype](Shape(2, 3))
    dictionary.unsafe_set(0, 1)
    dictionary.unsafe_set(4, 1)
    var x = zeros[dtype](Shape(2, 3))
    x.unsafe_set(0, 3)
    x.unsafe_set(1, -2)
    x.unsafe_set(3, -4)
    x.unsafe_set(4, 5)
    var reversed = x.view_with_layout(Shape(2, 3), NDArrayStrides(-3, 1), 3)
    var gram_buffer = zeros[dtype](Shape(3, 3))
    gram_buffer.unsafe_set(4, 1)
    gram_buffer.unsafe_set(8, 1)
    var gram = gram_buffer.view_with_layout(Shape(2, 2), NDArrayStrides(3, 1), 4)
    var init_buffer = zeros[dtype](Shape(2, 2))
    init_buffer.unsafe_set(2, 2)
    var init = init_buffer.view_with_layout(Shape(2, 2), NDArrayStrides(-2, 1), 2)
    var code = sparse_encode(reversed, dictionary, gram=gram.view(), init=init.view(), algorithm="lasso_cd")
    assert_equal(code.shape, Shape(2, 2))
    assert_equal(code.unsafe_get(0), -3)
    assert_equal(code.unsafe_get(1), 4)
    assert_equal(code.unsafe_get(2), 2)
    assert_equal(code.unsafe_get(3), -1)
    assert_equal(init_buffer.unsafe_get(2), 2)
    assert_equal(gram_buffer.unsafe_get(8), 1)

    var model = SparseCoder[dtype](dictionary, transform_algorithm="lasso_cd")
    assert_equal(model.transform(x).unsafe_get(1), -1)
    var serial = sparse_encode(x, dictionary, algorithm="lasso_cd", n_jobs=2)
    assert_equal(serial.unsafe_get(1), -1)
    _ = model.set_params(positive_code=True).fit(x)
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.transform(x).unsafe_get(1), 0)
    assert_equal(clone(model).transform(x).unsafe_get(3), 4)

    var singleton = x.view_with_layout(Shape(1, 3), NDArrayStrides(3, 1), 0)
    var vector = zeros[dtype](Shape(2))
    var single_code = sparse_encode(singleton, dictionary, init=vector.view(), algorithm="lasso_cd")
    assert_equal(single_code.shape, Shape(1, 2))
    assert_equal(single_code.unsafe_get(0), 2)
    with assert_raises(contains="init must have shape"):
        _ = sparse_encode(x, dictionary, init=vector.view(), algorithm="lasso_cd")
    init_buffer.unsafe_set(0, inf[dtype]())
    with assert_raises(contains="init must be finite"):
        _ = sparse_encode(x, dictionary, init=init.view(), algorithm="lasso_cd")
    with assert_raises(contains="gram must have shape"):
        _ = sparse_encode(x, dictionary, gram=vector.view(), algorithm="lasso_cd")
    with assert_raises(contains="lasso_cd requires max_iter >= 1"):
        _ = sparse_encode(x, dictionary, max_iter=0, algorithm="lasso_cd")
    x.unsafe_set(0, inf[dtype]())
    with assert_raises(contains="input must be finite"):
        _ = sparse_encode(x, dictionary, check_input=False, algorithm="lasso_cd")


def omp[dtype: DType]() raises:
    var dictionary = zeros[dtype](Shape(2, 3))
    dictionary.unsafe_set(0, 2)
    dictionary.unsafe_set(4, 1)
    var x = zeros[dtype](Shape(2, 3))
    x.unsafe_set(0, 3)
    x.unsafe_set(1, -4)
    x.unsafe_set(4, -5)
    var model = SparseCoder[dtype](dictionary)
    var default_code = model.transform(x)
    assert_equal(default_code.shape, Shape(2, 2))
    assert_equal(default_code.unsafe_get(0), 1.5)
    assert_equal(default_code.unsafe_get(1), 0)
    assert_equal(default_code.unsafe_get(3), -5)
    _ = model.set_params(transform_n_nonzero_coefs=Optional(2)).fit(x)
    assert_equal(clone(model).transform(x).unsafe_get(1), -4)

    var gram_storage = zeros[dtype](Shape(3, 3))
    gram_storage.unsafe_set(4, 4)
    gram_storage.unsafe_set(8, 1)
    var gram = gram_storage.view_with_layout(Shape(2, 2), NDArrayStrides(3, 1), 4)
    var cov_storage = zeros[dtype](Shape(2, 2))
    cov_storage.unsafe_set(0, -8)
    cov_storage.unsafe_set(1, 12)
    cov_storage.unsafe_set(2, 7)
    var cov = cov_storage.view_with_layout(Shape(2, 2), NDArrayStrides(2, -1), 1)
    var code = sparse_encode(x, dictionary, algorithm="omp", n_nonzero_coefs=2,
        gram=gram.view(), cov=cov.view(), copy_cov=False, check_input=False)
    assert_equal(code.unsafe_get(0), 3)
    assert_equal(code.unsafe_get(1), 0)
    assert_equal(code.unsafe_get(2), -2)
    assert_equal(code.unsafe_get(3), 7)
    assert_equal(cov_storage.unsafe_get(0), -8)
    assert_equal(cov_storage.unsafe_get(1), 12)
    assert_equal(gram_storage.unsafe_get(4), 4)

    with assert_raises(contains="Positive constraint not supported for 'omp'"):
        _ = sparse_encode(x, dictionary, algorithm="omp", positive=True)
    with assert_raises(contains="cannot exceed the number of atoms"):
        _ = sparse_encode(x, dictionary, algorithm="omp", n_nonzero_coefs=3)
    with assert_raises(contains="gram must have shape"):
        _ = sparse_encode(x, dictionary, algorithm="omp", gram=x.view())
    with assert_raises(contains="cov must have shape"):
        _ = sparse_encode(x, dictionary, algorithm="omp", cov=x.view())
    cov_storage.unsafe_set(0, inf[dtype]())
    with assert_raises(contains="OMP: input cov contains NaN or infinity"):
        _ = sparse_encode(x, dictionary, algorithm="omp", cov=cov.view())
    var reversed = x.view_with_layout(Shape(2, 3), NDArrayStrides(-3, 1), 3)
    assert_equal(model.transform(reversed).unsafe_get(1), -5)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    lasso_cd[DType.float32]()
    lasso_cd[DType.float64]()
    omp[DType.float32]()
    omp[DType.float64]()
    print("PASS: threshold, Lasso CD and OMP contracts Float32/Float64")
