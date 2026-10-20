from std.testing import assert_equal, assert_true, assert_raises
from numojo import zeros, Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow, FeatureEntry, FeatureKey, FeatureValue


def main() raises:
    var row = FeatureRow()
    row.append(FeatureEntry(FeatureKey("z"), FeatureValue(Float64(2))))
    row.append(FeatureEntry(FeatureKey("a"), FeatureValue(Float64(3))))
    var rows = FeatureBatch()
    rows.append(row.copy())
    var v = DictVectorizer(sparse=False)
    with assert_raises():
        _ = v.transform(rows)
    with assert_raises():
        _ = v.get_feature_names_out()
    with assert_raises():
        v.restrict(List[Int]())
    var x = zeros[DType.float64](Shape(1,2))
    with assert_raises():
        _ = v.inverse_transform(x)
    var sparse = DictVectorizer()
    with assert_raises():
        sparse.fit(rows)
    with assert_raises():
        _ = sparse.fit_transform(rows)
    v.fit(rows, Optional[Int](9))
    assert_equal(v.get_feature_names_out(List[String]())[0], "a")
    var names = v.get_feature_names_out()
    names[0] = "changed"
    assert_equal(v.get_feature_names_out()[0], "a")
    var params = v.get_params(False)
    params.separator = "!"
    assert_equal(v.separator, "=")
    with assert_raises():
        _ = v.transform(FeatureBatch())
    with assert_raises():
        _ = v.fit_transform(FeatureBatch())
    assert_equal(v.vocabulary_.size(), 2)
    var single = v.transform(row)
    assert_equal(single.unsafe_get(0),3)
    var single_fit = v.fit_transform(row)
    assert_equal(single_fit.unsafe_get(1),2)
    # Actual negative-stride/offset NDArray view, borrowed by inverse.
    var base = zeros[DType.float64](Shape(2,4))
    base.unsafe_set(3,5)
    base.unsafe_set(1,7)
    base.unsafe_set(7,11)
    base.unsafe_set(5,13)
    var view = base.view_with_layout(Shape(2,2), NDArrayStrides(strides=List[Int]([4,-2])), 3)
    var inverse = v.inverse_transform(view)
    assert_equal(inverse[0][FeatureKey("a")],5)
    assert_equal(inverse[0][FeatureKey("z")],7)
    assert_equal(inverse[1][FeatureKey("z")],13)
    var empty = zeros[DType.float64](Shape(1,0))
    with assert_raises():
        _ = v.inverse_transform(empty)
    var vector = zeros[DType.float64](Shape(2))
    with assert_raises():
        _ = v.inverse_transform(vector)
    x.unsafe_set(0,Float64("nan"))
    with assert_raises():
        _ = v.inverse_transform(x)
    v.restrict(List[Bool]([False,True]))
    assert_equal(v.get_feature_names_out()[0],"z")
    v.fit(rows)
    v.restrict(List[Int]([-1,0]), indices=True)
    assert_equal(v.get_feature_names_out()[0],"z")
    v.fit(rows)
    v.restrict(List[Int]([0,0]), indices=True)
    assert_equal(v.vocabulary_.column(FeatureKey("a")),1)
    with assert_raises():
        _ = v.transform(rows)
    v.fit(rows)
    with assert_raises():
        v.restrict(List[Int]([5]), indices=True)
    v.set_params(sparse=True)
    with assert_raises():
        _ = v.transform(rows)
    v.set_params(sparse=False, separator="", sort=False)
    assert_equal(v.separator, "")
    var snapshot = v.copy()
    v.restrict(List[Int]([0,1]))
    assert_equal(v.vocabulary_.size(),1)
    assert_equal(snapshot.vocabulary_.size(),2)
    assert_equal(v.get_feature_names_out()[0],"z")
    v.fit(FeatureBatch())
    assert_true(v.__sklearn_is_fitted__())
    var unknown = v.transform(row)
    assert_equal(unknown.shape[1],0)
    print("PASS: lifecycle, negative capabilities and strided inverse")
