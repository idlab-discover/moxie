"""The normal family namespace must resolve concrete, usable estimators."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie import clone
from moxie.impute import SimpleImputer, MissingIndicator, KNNImputer, IterativeImputer


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(2, 2))
    X.unsafe_set(0, Float64("nan"))
    X.unsafe_set(2, 4)
    var imputer = SimpleImputer(add_indicator=True)
    var filled = imputer.fit_transform(X)
    assert_equal(filled.shape[1], 3)
    assert_equal(filled.unsafe_get(0), 4)
    assert_equal(filled.unsafe_get(2), 1)
    var fresh = clone(imputer)
    assert_false(fresh.__sklearn_is_fitted__())
    var indicator = MissingIndicator()
    var mask = indicator.fit_transform(X)
    assert_equal(mask.shape[1], 1)
    assert_equal(mask.unsafe_get(0), True)
    assert_equal(mask.unsafe_get(1), False)
    var knn = KNNImputer()
    assert_equal(knn.fit_transform(X).unsafe_get(0), 4)
    with assert_raises():
        _ = IterativeImputer()
    print("PASS: concrete public estimators and explicit deferred estimators")
