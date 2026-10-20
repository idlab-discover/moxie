from moxie.feature_extraction._probe_a import produce
from moxie.feature_extraction._probe_b import consume
from moxie._random import RandomState
from std.testing import assert_equal
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow


def exchange[dtype: DType]() raises:
    var x = produce[dtype]()
    var snapshot = x.copy()
    consume(x)
    assert_equal(x.unsafe_get(1), 9)
    assert_equal(snapshot.unsafe_get(1), 4)


def main() raises:
    exchange[DType.float32]()
    exchange[DType.float64]()
    var v = DictVectorizer(sparse=False)
    var rows = FeatureBatch()
    rows.append(FeatureRow())
    var output = v.fit_transform(rows)
    assert_equal(output.shape[1], 0)
    var rng = RandomState(42)
    assert_equal(rng.randint(10), 6)
    print("PASS: composed producer/consumer shares native storage")
