from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow, FeatureEntry, FeatureKey, FeatureValue


def main() raises:
    var row = FeatureRow()
    row.append(FeatureEntry(FeatureKey("a"), FeatureValue(Int64(2))))
    row.append(FeatureEntry(FeatureKey("b"), FeatureValue(Int64(3))))
    row.append(FeatureEntry(FeatureKey("c"), FeatureValue(Int64(5))))
    var rows = FeatureBatch()
    rows.append(row^)
    var model = DictVectorizer(sparse=False)
    _ = model.fit(rows)

    var backing = zeros[DType.float32](Shape(1,6))
    backing.unsafe_set(5,1.25)
    backing.unsafe_set(1,-2.5)
    var view = backing.view_with_layout(Shape(1,3), NDArrayStrides(strides=List[Int]([6,-2])),5)
    var inverse: List[Dict[FeatureKey,Float32]] = model.inverse_transform(view)
    assert_equal(len(inverse[0]),2)
    assert_equal(inverse[0][FeatureKey("a")],1.25)
    assert_equal(inverse[0][FeatureKey("c")],-2.5)
    backing.unsafe_set(5,9)
    assert_equal(inverse[0][FeatureKey("a")],1.25)
    var wide = zeros[DType.uint64](Shape(1,3))
    wide.unsafe_set(1,UInt64(18446744073709551615))
    var exact: List[Dict[FeatureKey,UInt64]] = model.inverse_transform(wide)
    assert_equal(exact[0][FeatureKey("b")],UInt64(18446744073709551615))
    var small = DictVectorizer[DType.int8](sparse=False)
    _ = small.fit(rows)
    var uncast = small.inverse_transform(wide)
    assert_equal(uncast[0][FeatureKey("b")],UInt64(18446744073709551615))
    backing.unsafe_set(5,Float32(Float64("nan")))
    with assert_raises():
        _ = model.inverse_transform(view)

    var mask = zeros[DType.float64](Shape(6))
    mask.unsafe_set(5,Float64("nan"))
    mask.unsafe_set(1,-2)
    var mask_view = mask.view_with_layout(Shape(3),NDArrayStrides(strides=List[Int]([-2])),5)
    var selected = model.restrict(mask_view).transform(rows)
    assert_equal(selected.shape[1],2)
    assert_equal(selected.unsafe_get(0),2)
    assert_equal(selected.unsafe_get(1),5)
    assert_equal(model.get_feature_names_out()[1],"c")
    var index_backing = zeros[DType.int16](Shape(4))
    index_backing.unsafe_set(3,-1)
    index_backing.unsafe_set(1,0)
    var indices = index_backing.view_with_layout(Shape(2),NDArrayStrides(strides=List[Int]([-2])),3)
    var reordered = model.fit(rows).restrict(indices,indices=True).transform(rows)
    assert_equal(reordered.unsafe_get(0),5)
    assert_equal(reordered.unsafe_get(1),2)
    _ = model.fit(rows)
    var invalid = zeros[DType.uint64](Shape(1))
    invalid.unsafe_set(0,UInt64(18446744073709551615))
    with assert_raises():
        _ = model.restrict(invalid,indices=True)
    assert_equal(model.get_feature_names_out()[0],"a")
    assert_equal(model.vocabulary_.size(),3)
    with assert_raises():
        _ = model.restrict(mask_view,indices=True)
    var matrix_mask = zeros[DType.bool](Shape(1,3))
    with assert_raises():
        _ = model.restrict(matrix_mask)
    var fresh = DictVectorizer(sparse=False)
    with assert_raises():
        _ = fresh.restrict(indices,indices=True)
    var bool_mask = zeros[DType.bool](Shape(3))
    bool_mask.unsafe_set(1,True)
    with assert_raises():
        _ = model.restrict(bool_mask,indices=True)
    var one = model.restrict(bool_mask).transform(rows)
    assert_equal(one.unsafe_get(0),3)
    var empty = zeros[DType.int64](Shape(0))
    var none = model.restrict(empty,indices=True).transform(rows)
    assert_equal(none.shape[1],0)
    var list_selected = model.fit(rows).restrict(List[Bool]([True,False,True])).restrict(List[Int]([1]),indices=True).transform(rows)
    assert_equal(list_selected.unsafe_get(0),5)
    print("PASS: mixed inverse dtypes, array restriction and fluent selection")
