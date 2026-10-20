"""Producer fixture, not an estimator implementation."""
import numojo as nm
from moxie._naive_bayes_dense import DenseMatrix, validate_matrix
from moxie._naive_bayes_common import _alpha
from moxie.naive_bayes import MultinomialNB
from std.testing import assert_equal

def data() raises -> DenseMatrix:
    return nm.ones[DType.float64](nm.Shape(2, 3))

def check_model() raises:
    var X: List[List[Float64]] = [[2.0, 0.0], [0.0, 2.0]]
    var y: List[Int] = [8, -1]
    var model = MultinomialNB()
    _ = model.fit(X, y)
    assert_equal(model.predict(X)[0], 8)
    assert_equal(model.classes_[0], -1)
    assert_equal(_alpha(0.0, False), 1e-10)
    var dense = data()
    assert_equal(validate_matrix(dense), 3)
