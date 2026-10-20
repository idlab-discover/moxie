"""Shared dense validation and Bernoulli List compatibility."""
import numojo as nm
from max.algorithm import sum as max_sum
from moxie._dense import sum_axis2d
from std.testing import assert_equal, assert_true
from moxie._naive_bayes_dense import prepare_batch, int_classes, FeatureAlpha
from moxie._naive_bayes_common import _alpha
from moxie.naive_bayes import BernoulliNB

def main() raises:
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    var y: List[Int] = [7, -1]
    var classes: List[Int] = [-1, 7]
    var labeled = prepare_batch(X, y, List[Int](), classes.copy())
    var batch = labeled.batch.copy()
    assert_equal(int_classes(labeled.domain)[0], -1)
    assert_equal(batch.n_features, 1)
    assert_equal(batch.weights[0], 1.0)
    var alpha = FeatureAlpha(1.0, None)
    alpha.validate(1)
    assert_equal(_alpha(0.0, False), 1e-10)
    var legacy_X: List[List[Float64]] = [[1.0], [0.0]]
    var model = BernoulliNB()
    _ = model.fit(legacy_X, y)
    var probs = model.predict_proba(legacy_X)
    assert_equal(len(probs), 2)
    assert_equal(len(probs[0]), 2)
    assert_true(abs(probs[0][0] + probs[0][1] - 1.0) < 1e-7)
    print("PASS: shared validation and canonical Bernoulli List predictions")
