"""Threshold coding contract checks; unsupported regression solvers raise."""
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
    with assert_raises(): _ = sparse_encode(x, d)  # default lasso_lars unavailable
    var default = SparseCoder[dtype](d)
    default.fit(x)
    with assert_raises(): _ = default.transform(x)  # default omp unavailable
    with assert_raises(): _ = model.transform(zeros[dtype](Shape(1, 4)))
    with assert_raises(): _ = model.inverse_transform(zeros[dtype](Shape(1, 3)))
    with assert_raises(): _ = sparse_encode(x, d, algorithm="threshold", alpha=-1.0)
    with assert_raises(): _ = sparse_encode(x, d, algorithm="threshold", cov=zeros[dtype](Shape(1, 1)))
    with assert_raises(): _ = sparse_encode(zeros[dtype](Shape(0, 3)), d, algorithm="threshold")
    x.unsafe_set(0, inf[dtype]())
    with assert_raises(): _ = model.transform(x)
    x.unsafe_set(0, 3)
    comptime if dtype == DType.float32:
        c.set_params(split_sign=True)
        with assert_raises(): _ = c.transform(x)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: threshold sparse coding contracts Float32/Float64")
