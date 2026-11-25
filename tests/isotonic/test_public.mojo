"""Public exports and fluent contracts, also compiled outside the repository."""
from numojo import NDArrayStrides, Shape, array, zeros
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import RegressorMixin, WeightedSupervisedEstimator, clone, get_tags
from moxie.isotonic import IsotonicRegression, isotonic_regression, check_increasing


def generic[E: RegressorMixin & WeightedSupervisedEstimator](
    mut model: E, params: E.Params, X: E.Input, y: E.Target, weights: E.SampleWeight,
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y, sample_weight=weights).predict(X)


def exercise[dtype: DType]() raises:
    var x: List[Scalar[dtype]] = [0, 1, 2]
    var y: List[Scalar[dtype]] = [3, 1, 2]
    var weights: List[Scalar[dtype]] = [1, 1, 1]
    var model = IsotonicRegression[dtype]()
    var tags = get_tags(model)
    assert_true(tags.input_tags.one_d_array)
    assert_false(tags.input_tags.two_d_array)
    assert_false(tags._skip_test)
    tags.input_tags.one_d_array = False
    tags._skip_test = True
    var expected: List[Scalar[dtype]] = [2, 2, 2]
    var fitted: List[Scalar[dtype]] = isotonic_regression(y)
    assert_equal(fitted, expected)
    assert_true(check_increasing(x, x))
    var params = model.get_params()
    params.out_of_bounds = "clip"
    var predictions = generic(
        model, params, array(x, [3, 1]), array(y, [3]), array(weights, [3]),
    )
    assert_equal(predictions.ndim, 1)
    assert_equal(predictions.tolist(), expected)
    assert_equal(model.set_params(out_of_bounds=String("clip")).fit(x, y).predict(x), expected)
    var columns: List[List[Scalar[dtype]]] = [[0], [1], [2]]
    assert_equal(model.fit(columns, y, sample_weight=weights.copy()).predict(columns), expected)
    var copied = model.copy()
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_true(get_tags(model).input_tags.one_d_array)
    assert_false(get_tags(model)._skip_test)
    assert_true(get_tags(copied).input_tags.one_d_array)
    assert_true(get_tags(fresh).input_tags.one_d_array)
    with assert_raises():
        _ = model.fit(x, List[Scalar[dtype]]())
    assert_equal(model.predict(x), expected)
    _ = model.fit(x, x)
    assert_equal(copied.predict(x), expected)
    assert_equal(model.score(x, x, sample_weight=weights.copy()), 1.0)
    var retained: List[Scalar[dtype]] = copied.X_thresholds_.copy()
    var endpoints: List[Scalar[dtype]] = [0, 2]
    assert_equal(retained, endpoints)


def exercise_views[dtype: DType]() raises:
    var buffer: List[Scalar[dtype]] = [99, 0, 99, 1, 99, 2]
    var backing = array(buffer, [6])
    var X = backing.view_with_layout(Shape(3, 1), NDArrayStrides(-2, 1), 5)
    var y = backing.view_with_layout(Shape(3), NDArrayStrides(2), 1)
    var weight_data: List[Scalar[dtype]] = [-1, 1, -1, 2, -1, 1]
    var weight_buffer = array(weight_data, [6])
    var weights = weight_buffer.view_with_layout(Shape(3), NDArrayStrides(-2), 5)
    var expected: List[Scalar[dtype]] = [1, 1, 1]
    var model = IsotonicRegression[dtype]()
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    with assert_raises(contains="requires y"):
        _ = model.fit_transform(X)
    assert_equal(model.fit_transform(X, y, sample_weight=weights).tolist(), expected)
    assert_equal(model.transform(X).tolist(), expected)
    assert_equal(model.f_(X).tolist(), expected)
    assert_equal(model.fit_transform(X, y).tolist(), expected)
    var index_values: List[Scalar[dtype]] = [2, 1, 0]
    var target = array(index_values, [3])
    _ = model.fit(X, target)
    assert_equal(model.score(X, target, sample_weight=weights), 1.0)
    with assert_raises(contains="1 feature"):
        _ = model.fit(zeros[dtype](Shape(3, 2)), target)
    with assert_raises(contains="y should be a 1d array"):
        _ = model.fit(X, array(index_values, [3, 1]))
    with assert_raises(contains="sample_weight should be a 1d array"):
        _ = model.fit(X, target, sample_weight=array(index_values, [3, 1]))
    with assert_raises(contains="at least one sample"):
        _ = model.predict(zeros[dtype](Shape(0)))
    assert_equal(backing.tolist(), buffer)
    assert_equal(weight_buffer.tolist(), weight_data)
    assert_equal(model.predict(X).tolist(), index_values)
    backing.unsafe_set(1, 99)
    assert_equal(model.predict(target).tolist(), index_values)


def exercise_metadata() raises:
    var model = IsotonicRegression()
    var X: List[Float64] = [0, 1, 2]
    var y: List[Float64] = [3, 1, 2]
    var weights: List[Float64] = [0, 1, 1]
    assert_true(model.get_metadata_routing().fit.value().isa[NoneType]())
    assert_true(model.get_metadata_routing().score.value().isa[NoneType]())
    _ = model.set_fit_request(sample_weight=False).set_score_request(sample_weight="score_weights")
    _ = model.set_fit_request().fit(X, y, sample_weight=weights)
    assert_equal(model.X_min_, 1.0)
    var query: List[Float64] = [1, 2]
    var target: List[Float64] = [1, 3]
    var score_weights: List[Float64] = [1, 0]
    assert_equal(model.score(query, target, sample_weight=score_weights.copy()), 1.0)
    var copied = model.copy()
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    var snapshot = model.get_metadata_routing()
    assert_false(snapshot.fit.value()[Bool])
    with assert_raises(contains="metadata alias must be an ASCII identifier"):
        _ = model.set_score_request(sample_weight="invalid-weight")
    assert_equal(model.get_metadata_routing().score.requested_name().or_else(""), "score_weights")
    var metadata = Dict[String, List[Float64]]()
    metadata["score_weights"] = score_weights.copy()
    var routed = snapshot.score.route(metadata)
    metadata["score_weights"][0] = 0.0
    assert_equal(routed^.or_else([]), score_weights)
    snapshot.score.set(False)
    _ = model.set_fit_request(sample_weight=None).set_score_request(sample_weight=True)
    assert_true(model.get_metadata_routing().fit.value().isa[NoneType]())
    assert_equal(copied.get_metadata_routing().score.requested_name().or_else(""), "score_weights")
    assert_equal(cloned.get_metadata_routing().score.requested_name().or_else(""), "score_weights")


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    exercise_views[DType.float32]()
    exercise_views[DType.float64]()
    exercise_metadata()
    print("PASS: isotonic public exports, fluent contracts, metadata requests and dtype preservation")
