"""Consumer fixture, not an estimator implementation."""
from moxie._naive_bayes_dense import DenseMatrix, validate_matrix
from moxie._naive_bayes_common import _alpha
from moxie._dense import sum_axis2d
from moxie.naive_bayes import BernoulliNB
from std.testing import assert_equal

def consume(mut X: DenseMatrix) raises -> Float64:
    _ = validate_matrix(X)
    X.unsafe_set(0, 5)
    var counts = sum_axis2d(X, 0)
    return counts.item(0)

def check_model() raises:
    var X: List[List[Float64]] = [[2.0, 0.0], [0.0, 2.0]]
    var y: List[Int] = [8, -1]
    var model = BernoulliNB()
    _ = model.fit(X, y)
    assert_equal(model.predict(X)[1], -1)
    assert_equal(model.classes_[0], -1)
    assert_equal(_alpha(0.0, False), 1e-10)
