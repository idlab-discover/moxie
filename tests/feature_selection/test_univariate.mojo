"""Exercise the public family namespace across variance and univariate stages."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.feature_selection import (
    VarianceThreshold, SelectKBest, BuiltinScorer, SelectionParameter,
    chi2, f_classif, f_oneway, f_regression, r_regression,
)

def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(6, 3))
    var y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(3 * i, 7)
        X.unsafe_set(3 * i + 1, SIMD[dtype, 1](i))
        X.unsafe_set(3 * i + 2, SIMD[dtype, 1](i % 2))
        y.unsafe_set(i, SIMD[dtype, 1](i))
    var variance = VarianceThreshold[dtype]()
    variance.fit_named(X, None, [String("constant"), String("trend"), String("alternating")])
    var varying = variance.transform(X)
    var selector = SelectKBest(BuiltinScorer[dtype]("r_regression"), k=SelectionParameter(1))
    selector.fit_named(varying, y.copy(), variance.get_feature_names_out())
    assert_equal(selector.get_feature_names_out()[0], "trend")
    assert_equal(selector.get_support(True)[List[Int]][0], 0)
    var selected: nm.NDArray[dtype] = selector.transform(varying)
    var restored = variance.inverse_transform(selector.inverse_transform(selected))
    for i in range(6):
        assert_equal(selected.unsafe_get(i), SIMD[dtype, 1](i))
        assert_equal(restored.unsafe_get(3 * i), 0)
        assert_equal(restored.unsafe_get(3 * i + 1), SIMD[dtype, 1](i))
        assert_equal(restored.unsafe_get(3 * i + 2), 0)
    selected.unsafe_set(0, 99)
    assert_equal(varying.unsafe_get(0), 0)
    assert_equal(X.unsafe_get(1), 0)
    var fresh = clone(selector)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(varying)
    # The function exports must resolve to implementations too.
    assert_equal(r_regression(varying, y).shape[0], 2)
    assert_equal(f_regression(varying, y).scores.shape[0], 2)
    var labels = nm.zeros[dtype](nm.Shape(6))
    for i in range(3, 6):
        labels.unsafe_set(i, 1)
    assert_equal(chi2(varying, labels).scores.shape[0], 2)
    assert_equal(f_classif(varying, labels).scores.shape[0], 2)
    var groups = List[nm.NDArray[dtype]]()
    groups.append(nm.zeros[dtype](nm.Shape(3, 2)))
    groups.append(nm.ones[dtype](nm.Shape(3, 2)))
    assert_equal(f_oneway(groups).scores.shape[0], 2)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("public variance/univariate composition passed")
