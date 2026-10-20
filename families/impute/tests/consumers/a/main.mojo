from moxie.impute import StringMissingIndicator
from std.testing import assert_equal, assert_false, assert_raises
import numojo as nm
from moxie import ProbeA, ProbeB
from moxie.impute import SimpleImputer, IntegerSimpleImputer, StringSimpleImputer, StringMatrix, MissingIndicator, KNNImputer
from moxie.impute._plumbing import MissingValue

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(2, 2))
    x.unsafe_set(0, -1)
    var imputer = SimpleImputer(missing_values=MissingValue[DType.float64](-1))
    assert_equal(imputer.fit_transform(x).unsafe_get(0), 0)
    var knn = KNNImputer(missing_values=MissingValue[DType.float64](-1))
    assert_equal(knn.fit_transform(x).unsafe_get(0), 0)
    var indicator = MissingIndicator(missing_values=MissingValue[DType.float64](-1))
    assert_equal(indicator.fit_transform(x).unsafe_get(0), True)
    var integers = nm.zeros[DType.int64](nm.Shape(2, 1))
    integers.unsafe_set(0, 3)
    var integer_model = IntegerSimpleImputer[DType.int64](strategy="most_frequent", missing_values=MissingValue[DType.int64](0), add_indicator=True)
    var integer_result = integer_model.fit_transform(integers)
    assert_equal(integer_result[nm.NDArray[DType.int64]].unsafe_get(2), 3)
    assert_equal(integer_result[nm.NDArray[DType.int64]].unsafe_get(3), 1)
    var text_cells: List[Optional[String]] = ["blue", None]
    var strings = StringMatrix(text_cells, 2, 1)
    var text_indicator = StringMissingIndicator()
    assert_equal(text_indicator.fit_transform(strings).unsafe_get(1), True)
    var text_model = StringSimpleImputer(strategy="most_frequent", add_indicator=True)
    var text_result = text_model.fit_transform(strings)
    assert_equal(text_result.values.get(1, 0).value(), "blue")
    assert_equal(text_result.indicators.unsafe_get(1), True)
    var a = ProbeA()
    var b = ProbeB()
    with assert_raises():
        _ = a.transform(x)
    a.fit(x)
    b.fit(x)
    var ma = a.transform(x)
    var mb = b.transform(x)
    assert_equal(ma.shape[1], 1)
    assert_equal(ma.unsafe_get(0), mb.unsafe_get(0))
    assert_equal(ma.unsafe_get(1), mb.unsafe_get(1))
    var clone = a.__sklearn_clone__()
    assert_false(clone.layout.state.is_fitted())
    var shared = x.view()
    shared.unsafe_set(3, -1)
    assert_equal(x.unsafe_get(3), -1)
    a.fit(shared)
    assert_equal(len(a.layout.indicator_features), 2)
    assert_equal(len(b.layout.indicator_features), 1)
    var norms = b.norms(x)
    assert_equal(norms.unsafe_get(0), 1)
    print("PASS: two composed consumers share family and numerical storage")
