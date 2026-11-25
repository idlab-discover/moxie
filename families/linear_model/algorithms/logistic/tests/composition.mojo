"""The public dense classifier as the final step of a standardization pipeline."""

from moxie.base import clone
from moxie.linear_model import LogisticRegression
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import Shape, zeros
from std.testing import assert_almost_equal, assert_equal, assert_false


def check_pipeline(solver: String) raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = zeros[DType.int64](Shape(8))
    for i in range(8):
        X.unsafe_set(2 * i, Float64(i))
        X.unsafe_set(2 * i + 1, Float64((i * 3) % 5))
        y.unsafe_set(i, Int64(i % 3))

    var model = make_pipeline(StandardScaler(), LogisticRegression(solver=solver.copy(), tol=1e-9))
    var probability = model.fit(X, y).predict_proba(X)
    var scores = model.decision_function(X)
    var predictions = model.predict(X)
    assert_equal(model.score(X, y), 0.5)
    assert_equal(probability.shape[0], 8)
    assert_equal(probability.shape[1], 3)

    # Pinned LogisticRegression fitted by sklearn's StandardScaler pipeline.
    var expected = [0.3129830836953335, 0.24086251574162415, 0.44615440056304234]
    var expected_scores = [-0.03086436385503641, -0.2927872106256095, 0.32365157448064547]
    var expected_labels = [2, 0, 0, 0, 1, 2, 1, 1]
    for i in range(8):
        assert_equal(Int(predictions.unsafe_get(i)), expected_labels[i])
    for k in range(3):
        assert_almost_equal(probability.unsafe_get(k), expected[k], atol=2e-7)
        assert_almost_equal(scores.unsafe_get(k), expected_scores[k], atol=2e-7)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.C = 2.0
    var changed = fresh.set_params(params).fit(X, y).predict_proba(X)
    assert_equal(fresh.estimator.C, 2.0)
    assert_equal(model.estimator.C, 1.0)
    assert_false(abs(changed.unsafe_get(0) - probability.unsafe_get(0)) < 1e-5)
    assert_equal(fresh.estimator.solver, solver)


def main() raises:
    check_pipeline("lbfgs")
    check_pipeline("newton-cholesky")
    print("PASS: dense LogisticRegression with both solvers through StandardScaler pipeline")
