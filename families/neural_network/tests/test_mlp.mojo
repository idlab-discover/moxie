"""Dense public MLP configuration, state and input contracts."""

from numojo import NDArray, NDArrayStrides, Shape, arange, ones, zeros
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie._random import RandomState
from moxie.base import clone
from moxie.neural_network import MLPClassifier, MLPRegressor


def equal[dtype: DType](left: NDArray[dtype], right: NDArray[dtype]) raises:
    assert_equal(left.shape, right.shape)
    for i in range(left.size):
        assert_equal(left.unsafe_get(i), right.unsafe_get(i))


def classifier() raises:
    var X = arange[DType.float64](12).reshape(Shape(6, 2))
    var y = arange[DType.int32](6)
    var model = MLPClassifier[DType.float64, DType.int32]([3], random_state=17, max_iter=3)
    var defaults = MLPClassifier().get_params()
    assert_equal(defaults.hidden_layer_sizes[0], 100)
    assert_equal(defaults.solver, "adam")
    assert_false(defaults.random_state)
    assert_false(defaults.batch_size)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.coefs_()
    with assert_raises(contains="NotFittedError"):
        _ = model.classes_()
    with assert_raises(contains="explicit mutable RNG context"):
        var unseeded = MLPClassifier()
        _ = unseeded.fit(X, y.astype[DType.int64]())

    var expected = model.fit(X, y).predict_proba(X)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.get_n_features_in(), 2)
    assert_equal(model.n_layers_(), 3)
    assert_equal(model.n_outputs_(), 6)
    assert_equal(model.out_activation_(), "softmax")
    var classes = model.classes_()
    assert_equal(classes.unsafe_get(5), Int32(5))
    classes.unsafe_set(5, Int32(99))
    assert_equal(model.classes_().unsafe_get(5), Int32(5))
    _ = model.fit(X, y)
    equal(expected, model.predict_proba(X))

    var copied = model.copy()
    var coefs = model.coefs_()
    coefs[0].unsafe_set(0, 12345.0)
    equal(expected, model.predict_proba(X))
    var changed = clone(model)
    assert_false(changed.__sklearn_is_fitted__())
    assert_equal(len(changed.fit_warnings), 0)
    var params = changed.get_params()
    params.hidden_layer_sizes = []
    params.max_iter = 1
    params.batch_size = 100
    _ = changed.set_params(params).fit(X, y)
    assert_equal(len(changed.fit_warnings), 2)
    assert_equal(changed.fit_warnings[0].category, "UserWarning")
    assert_equal(changed.fit_warnings[1].category, "ConvergenceWarning")
    assert_equal(changed.n_layers_(), 2)
    assert_equal(model.n_layers_(), 3)

    var warm = MLPClassifier[DType.float64, DType.int32](
        [3], warm_start=True, max_iter=1, shuffle=False,
    )
    var rng = RandomState(17)
    _ = warm.fit(X, y, rng)
    var probe = rng.copy()
    var warm_prediction = warm.predict_proba(X)
    with assert_raises(contains="same classes"):
        _ = warm.fit(X, zeros[DType.int32](Shape(6)), rng)
    equal(warm_prediction, warm.predict_proba(X))
    equal(warm.classes_(), y)
    assert_equal(len(warm.fit_warnings), 1)
    _ = warm.fit(X, y, rng)
    assert_equal(warm.t_(), 12)
    assert_equal(rng.next_uint32(), probe.next_uint32())
    var fresh = clone(warm)
    assert_true(fresh.get_params().warm_start)
    assert_false(fresh.__sklearn_is_fitted__())

    with assert_raises(contains="label"):
        _ = model.fit(X, zeros[DType.int32](Shape(6, 2)))
    with assert_raises(contains="one label per sample"):
        _ = model.fit(X, zeros[DType.int32](Shape(5)))
    with assert_raises(contains="input width"):
        _ = model.predict(zeros[DType.float64](Shape(6, 3)))
    with assert_raises(contains="sample weights"):
        _ = model.fit(X, y, sample_weight=zeros[DType.float64](Shape(6)))
    with assert_raises(contains="finite matrix"):
        var invalid = X.copy()
        invalid.unsafe_set(0, nan[DType.float64]())
        _ = model.fit(invalid, y)
    equal(expected, model.predict_proba(X))
    equal(expected, copied.predict_proba(X))
    assert_equal(len(model.fit_warnings), 1)


def regressor[dtype: DType]() raises:
    # The reversed strided view and its compact copy give identical seeded fits.
    var source = arange[dtype](32).reshape(Shape(8, 4))
    var X = source.view_with_layout(Shape(6, 2), NDArrayStrides(-4, 2), 24)
    var dense = X.contiguous()
    var target_source = arange[dtype](12)
    var y = target_source.view_with_layout(Shape(6), NDArrayStrides(2), 1)
    var target = y.contiguous()
    var weights_source = ones[dtype](Shape(12))
    var weights = weights_source.view_with_layout(Shape(6), NDArrayStrides(-2), 10)
    var first = MLPRegressor[dtype](solver="lbfgs", hidden_layer_sizes=[2], max_iter=2)
    var second = MLPRegressor[dtype](solver="lbfgs", hidden_layer_sizes=[2], max_iter=2, random_state=23)
    var rng = RandomState(23)
    _ = first.fit(X, y, rng, sample_weight=weights)
    _ = second.fit(dense, target, sample_weight=weights.contiguous())
    equal(first.predict(X), second.predict(dense))
    equal(first.coefs_[DType.float64]()[0], second.coefs_[DType.float64]()[0])
    assert_equal(first.parameter_dtype(), DType.float64)
    assert_false(first.loss_curve_())
    assert_false(first.best_loss_())
    assert_equal(first.t_(), 0)
    assert_equal(first.predict(X).ndim, 1)
    assert_equal(first.score(X, y), second.score(dense, target))

    comptime if dtype == DType.float32:
        with assert_raises(contains="requested dtype"):
            _ = first.coefs_()
    with assert_raises(contains="same output width"):
        _ = first.score(X, zeros[dtype](Shape(6, 2)))
    with assert_raises(contains="Poisson targets must be nonnegative"):
        var poisson = MLPRegressor[dtype](loss="poisson", random_state=0)
        _ = poisson.fit(X, target - 2)
    with assert_raises(contains="early stopping"):
        var stopping = MLPRegressor[dtype](early_stopping=True, random_state=0)
        _ = stopping.fit(X, target)
    var warm = MLPRegressor[dtype](warm_start=True, random_state=0, max_iter=1)
    _ = warm.fit(X, target)
    var warm_copy = warm.copy()
    with assert_raises(contains="input width"):
        _ = warm.fit(zeros[dtype](Shape(6, 1)), target)
    assert_equal(warm.t_(), 6)
    _ = warm.fit(X, target)
    assert_equal(warm.t_(), 12)
    assert_equal(warm_copy.t_(), 6)
    assert_equal(warm.n_iter_(), 1)

    var warm_params = warm.get_params()
    warm_params.solver = "lbfgs"
    _ = warm.set_params(warm_params).fit(X, target)
    assert_equal(warm.parameter_dtype(), DType.float64)
    assert_equal(warm.t_(), 12)
    var history = warm.loss_curve_()
    assert_equal(len(history.value()), 2)
    warm_params.solver = "sgd"
    _ = warm.set_params(warm_params).fit(X, target)
    assert_equal(warm.parameter_dtype(), dtype)
    assert_equal(warm.t_(), 18)
    history = warm.loss_curve_()
    assert_equal(len(history.value()), 3)

    var lbfgs_first = clone(warm)
    warm_params.solver = "lbfgs"
    _ = lbfgs_first.set_params(warm_params).fit(X, target)
    warm_params.solver = "adam"
    _ = lbfgs_first.set_params(warm_params).fit(X, target)
    assert_equal(lbfgs_first.parameter_dtype(), dtype)
    assert_equal(lbfgs_first.t_(), 6)

    var constant = MLPRegressor[dtype](
        hidden_layer_sizes=[], solver="sgd", max_iter=1, random_state=0,
    )
    _ = constant.fit(dense, target)
    assert_equal(constant.score(dense, ones[dtype](Shape(6))), 0.0)


def main() raises:
    classifier()
    regressor[DType.float32]()
    regressor[DType.float64]()
    print("PASS dense public MLP validation, cloning, atomic refit, RNG and strided input")
