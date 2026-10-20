from std.testing import assert_equal, assert_raises
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureKey, FeatureEntry, FeatureValue, FeatureRow, FeatureBatch
from moxie.base import clone


def main() raises:
    var row = FeatureRow()
    row.append(FeatureEntry(FeatureKey("version"), FeatureValue(List[String]( ["1", "2", "1"] ))))
    row.append(FeatureEntry(FeatureKey("ham"), FeatureValue(Float64(2))))
    var batch = FeatureBatch()
    batch.append(row.copy())
    var v = DictVectorizer(sparse=False)
    var x = v.fit_transform(batch)
    assert_equal(x.shape[0],1)
    assert_equal(x.shape[1],3)
    assert_equal(x.unsafe_get(0),2)
    assert_equal(x.unsafe_get(1),2)
    var inv = v.inverse_transform(x)
    assert_equal(inv[0][FeatureKey("ham")],2)
    var cloned = clone(v)
    with assert_raises():
        _ = cloned.transform(batch)
    var support: List[Int] = [2,0]
    v.restrict(support, indices=True)
    assert_equal(v.get_feature_names_out()[0],"version=2")
    var params = v.get_params()
    var chained = v.set_params(params).set_params(sort=False).fit(batch).transform(batch)
    assert_equal(chained.shape[1],3)
    assert_equal(chained.unsafe_get(0),2)
    assert_equal(v.get_feature_names_out()[0],"version=1")
    print("PASS: dense dictionary contracts")
