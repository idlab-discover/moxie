"""Dtype changes across refits, owned strided inputs, and public result arms."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone, SupervisedEstimator
from moxie.kernel_ridge import KernelRidge, ScalarOrVector
from moxie._kernel_ridge._validation import dense_cast


def through_contract[
    E: SupervisedEstimator
](mut model: E, X: E.Input, y: E.Target) raises:
    _ = model.fit(X, y)


def main() raises:
    var X = nm.arange[DType.float32](6).reshape(nm.Shape(3, 2))
    var y = nm.arange[DType.float64](6).reshape(nm.Shape(3, 2))
    var model = KernelRidge[DType.float32]()
    _ = model.fit(X, y)
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float64]]())
    assert_true(model.X_fit_.isa[nm.NDArray[DType.float32]]())
    var original = model.dual_coef_[nm.NDArray[DType.float64]].copy()
    var duplicate = model.copy()
    duplicate.dual_coef_[nm.NDArray[DType.float64]].unsafe_set(0, 100)
    assert_equal(
        model.dual_coef_[nm.NDArray[DType.float64]].unsafe_get(0),
        original.unsafe_get(0),
    )
    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.predict(X.astype[DType.int64]())
    _ = model.set_params(alpha=ScalarOrVector(List[Float64]([1.0, 2.0])))
    _ = model.fit(X, y)
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float32]]())
    assert_true(model.predict(X).isa[nm.NDArray[DType.float32]]())
    assert_true(
        model.predict(X.astype[DType.float64]()).isa[
            nm.NDArray[DType.float64]
        ]()
    )
    var before = model.dual_coef_[nm.NDArray[DType.float32]].copy()
    var bad = y.copy()
    bad.unsafe_set(0, Float64("nan"))
    with assert_raises():
        _ = model.fit(X.astype[DType.float64](), bad)
    assert_true(model.X_fit_.isa[nm.NDArray[DType.float32]]())
    assert_equal(
        model.dual_coef_[nm.NDArray[DType.float32]].unsafe_get(0),
        before.unsafe_get(0),
    )
    _ = model.set_params(kernel="laplacian")
    assert_true(model.predict(X).isa[nm.NDArray[DType.float64]]())
    _ = model.fit(X, y.astype[DType.float32]())
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float64]]())
    _ = model.set_params(kernel="linear", alpha=ScalarOrVector(1.0))
    var y32 = y.astype[DType.float32]()
    through_contract(model, X, y32)
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float32]]())
    _ = model.fit(X, y32, sample_weight=1.0)
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float32]]())
    _ = model.fit(X, y32, sample_weight=2.0)
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float64]]())
    _ = model.fit(
        X, y32, sample_weight=ScalarOrVector(List[Float64]([1.0, 1.0, 1.0]))
    )
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float64]]())
    # Conversion must traverse logical elements of negative/offset strides.
    var reverse = X.view_with_layout(
        nm.Shape(3, 2), nm.NDArrayStrides(-2, 1), 4
    )
    var reverse_y = y.view_with_layout(
        nm.Shape(3, 2), nm.NDArrayStrides(-2, 1), 4
    )
    var expected_X = nm.zeros[DType.float64](nm.Shape(3, 2))
    for i in range(3):
        for j in range(2):
            expected_X.unsafe_set(i * 2 + j, Float64((2 - i) * 2 + j))
    var converted = dense_cast[DType.float64](reverse)
    for i in range(6):
        assert_equal(converted.unsafe_get(i), expected_X.unsafe_get(i))
    _ = model.fit(reverse, reverse_y)
    var direct = KernelRidge()
    _ = direct.fit(expected_X, expected_X)
    var actual = model.predict(expected_X)
    var expected = direct.predict(expected_X)
    for i in range(6):
        assert_true(
            abs(
                actual[nm.NDArray[DType.float64]].unsafe_get(i)
                - expected[nm.NDArray[DType.float64]].unsafe_get(i)
            )
            < 1e-10
        )
    # Retain exact integer snapshots, even when floating computation rounds them.
    var integer_X = nm.ones[DType.int64](nm.Shape(3, 2))
    integer_X.unsafe_set(0, 9007199254740993)
    _ = model.set_params(kernel="rbf")
    _ = model.fit(integer_X, y32)
    integer_X.unsafe_set(0, 0)
    assert_equal(
        model.X_fit_[nm.NDArray[DType.int64]].unsafe_get(0),
        Int64(9007199254740993),
    )
    assert_true(model.dual_coef_.isa[nm.NDArray[DType.float64]]())
    assert_true(model.predict(X).isa[nm.NDArray[DType.float64]]())
    print("Dense dtype contracts passed")
