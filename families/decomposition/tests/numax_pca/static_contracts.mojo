"""Existing lifecycle contracts plus static-shape and inference-row checks."""
from std.testing import assert_equal, assert_raises
from numojo import Shape, zeros
from moxie.decomposition import PCA
from moxie_numax import NumaxPCABackend
from contracts import check


def static_shape[dtype: DType]() raises:
    var model = PCA[dtype, NumaxPCABackend[6, 3]](2)
    var X = zeros[dtype](Shape(6, 3))
    for i in range(6):
        for j in range(3): X.unsafe_set(i * 3 + j, Scalar[dtype]((i + j) * (i + j) + i % (j + 1)))
    model.fit(X)
    var other_rows = zeros[dtype](Shape(2, 3))
    assert_equal(model.transform(other_rows).shape[0], 2)
    assert_equal(model.score_samples(other_rows).size, 2)
    with assert_raises(): model.fit(other_rows)
    assert_equal(model.n_samples_, 6)
    # Check deep-copy adapters don't retain aliases to the caller's matrix.
    var before = model.singular_values_.value().unsafe_get(0)
    X.unsafe_set(0, 100)
    assert_equal(model.singular_values_.value().unsafe_get(0), before)


def main() raises:
    check[DType.float32, NumaxPCABackend[6, 3]]()
    check[DType.float64, NumaxPCABackend[6, 3]]()
    static_shape[DType.float32]()
    static_shape[DType.float64]()
    print("PASS: Numax PCA contracts in Float32 and Float64")
