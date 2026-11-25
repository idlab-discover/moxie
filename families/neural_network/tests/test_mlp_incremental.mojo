"""Dense public MLP incremental state and trait contracts."""

from numojo import Shape, arange, zeros
from std.testing import assert_equal, assert_false, assert_raises
from moxie._random import RandomState
from moxie.base import clone
from moxie.neural_network import MLPClassifier, MLPClassifierAPI, MLPRegressor, MLPRegressorAPI
from test_mlp import equal


def classify[Model: MLPClassifierAPI](
    mut model: Model, X: Model.Input, y: Model.Target, classes: Optional[Model.Classes] = None,
) raises -> Model.Prediction:
    return model.partial_fit(X, y, classes=classes).predict(X)


def regress[Model: MLPRegressorAPI](
    mut model: Model, X: Model.Input, y: Model.Target,
) raises -> Model.Prediction:
    return model.partial_fit(X, y).predict(X)


def classifier() raises:
    var X = arange[DType.float32](8).reshape(Shape(4, 2))
    var labels = zeros[DType.int32](Shape(4))
    var domain = zeros[DType.int32](Shape(4))
    domain.unsafe_set(0, 5)
    domain.unsafe_set(1, -7)
    domain.unsafe_set(3, 5)
    var model = MLPClassifier[DType.float32, DType.int32](
        [3], random_state=17, batch_size=10, max_iter=3,
    )
    with assert_raises(contains="first partial_fit"):
        _ = model.partial_fit(X, labels)
    with assert_raises(contains="nonempty vector"):
        _ = model.partial_fit(X, labels, classes=domain.reshape(Shape(4, 1)))
    assert_false(model.__sklearn_is_fitted__())

    var prediction = classify(model, X, labels, Optional(domain.copy()))
    equal(prediction, model.predict(X))
    assert_equal(model.classes_().size, 3)
    assert_equal(model.classes_().unsafe_get(0), Int32(-7))
    assert_equal(model.classes_().unsafe_get(2), Int32(5))
    assert_equal(model.n_iter_(), 1)
    assert_equal(model.t_(), 4)
    assert_equal(len(model.fit_warnings), 1)
    assert_equal(model.fit_warnings[0].category, "UserWarning")
    var untouched = model.copy()
    var fresh = clone(model)
    with assert_raises(contains="first partial_fit"):
        _ = fresh.partial_fit(X, labels)

    var unseen = labels.copy()
    unseen.unsafe_set(0, 6)
    with assert_raises(contains="unseen label"):
        _ = model.partial_fit(X, unseen)
    with assert_raises(contains="classes differ"):
        _ = model.partial_fit(X, labels, classes=labels.copy())
    with assert_raises(contains="input width"):
        _ = model.partial_fit(zeros[DType.float32](Shape(4, 1)), labels)
    assert_equal(model.t_(), 4)
    assert_equal(len(model.fit_warnings), 1)
    equal(model.predict_proba(X), untouched.predict_proba(X))

    labels.unsafe_set(0, 5)
    _ = model.partial_fit(X, labels, classes=domain.copy())
    _ = untouched.partial_fit(X, labels)
    equal(model.predict_proba(X), untouched.predict_proba(X))
    assert_equal(model.t_(), 8)
    var curve = model.loss_curve_()
    assert_equal(len(curve.value()), 2)

    _ = model.fit(X, labels)
    assert_equal(model.classes_().size, 2)
    assert_equal(model.t_(), 12)
    var ordinary_curve = model.loss_curve_()
    assert_equal(len(ordinary_curve.value()), 3)

    var rng = RandomState(17)
    var explicit = MLPClassifier[DType.float32, DType.int32]([3])
    var weights = arange[DType.float32](4)
    _ = explicit.partial_fit(X, labels, rng, classes=domain.copy(), sample_weight=weights)
    assert_equal(explicit.n_outputs_(), 3)

    var warm = MLPClassifier[DType.float32, DType.int32]([3], random_state=17, warm_start=True)
    with assert_raises(contains="same classes"):
        _ = warm.partial_fit(X, labels, classes=domain.copy())

    var lbfgs = MLPClassifier[DType.float32, DType.int32](solver="lbfgs", random_state=17)
    with assert_raises(contains="requires sgd or adam"):
        _ = lbfgs.partial_fit(X, labels, classes=domain.copy())


def regressor() raises:
    var X = arange[DType.float64](8).reshape(Shape(4, 2))
    var y = arange[DType.float64](4)
    var model = MLPRegressor[DType.float64](
        hidden_layer_sizes=[3], random_state=17, batch_size=2, shuffle=False,
    )
    equal(regress(model, X, y), model.predict(X))
    var control = model.copy()
    var weights = zeros[DType.float64](Shape(4))
    weights.unsafe_set(0, 1.0)
    weights.unsafe_set(1, 1.0)
    # A zero-mass second minibatch fails after the first optimizer update.
    with assert_raises(contains="nonzero sum"):
        _ = model.partial_fit(X, y, sample_weight=weights)
    assert_equal(model.t_(), 4)
    equal(model.predict(X), control.predict(X))
    _ = model.partial_fit(X, y)
    _ = control.partial_fit(X, y)
    equal(model.predict(X), control.predict(X))
    assert_equal(model.t_(), 8)
    assert_equal(model.n_iter_(), 1)
    assert_equal(len(model.fit_warnings), 0)

    with assert_raises(contains="output shape"):
        _ = model.partial_fit(X, zeros[DType.float64](Shape(4, 2)))
    with assert_raises(contains="hidden layer sizes"):
        var changed = model.get_params()
        changed.hidden_layer_sizes = [4]
        _ = model.set_params(changed).partial_fit(X, y)

    var stopping = MLPRegressor[DType.float64](early_stopping=True, random_state=17)
    with assert_raises(contains="does not support early_stopping"):
        _ = stopping.partial_fit(X, y)
    var unseeded = MLPRegressor[DType.float64]()
    with assert_raises(contains="explicit mutable RNG context"):
        _ = unseeded.partial_fit(X, y)
    var lbfgs = MLPRegressor[DType.float64](solver="lbfgs", random_state=17)
    with assert_raises(contains="requires sgd or adam"):
        _ = lbfgs.partial_fit(X, y)


def main() raises:
    classifier()
    regressor()
    print("PASS dense MLP incremental traits, class domains, warnings, and atomic update")
