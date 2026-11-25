"""Inspect the owned children of supervised and unsupervised pipelines."""
from moxie.base import clone
from moxie.dummy import DummyRegressor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def main() raises:
    var X = zeros[DType.float64](Shape(2, 1))
    X.itemset(1, 2)
    var y = zeros[DType.float64](Shape(2))
    y.itemset(1, 8)
    var pipe = make_pipeline(StandardScaler(), DummyRegressor())
    var named = pipe.named_steps()
    var keys = named.keys()
    assert_equal(len(named), 2)
    assert_equal(len(keys), 2)
    assert_equal(keys[0], "standardscaler")
    assert_equal(keys[1], "dummyregressor")
    assert_true("dummyregressor" in named)
    assert_false("missing" in named)
    assert_false(named.get[DummyRegressor[]]("dummyregressor").__sklearn_is_fitted__())
    with assert_raises(contains="KeyError"):
        _ = named.get[DummyRegressor[]]("missing")
    with assert_raises(contains="TypeError"):
        _ = named.get[StandardScaler[]]("dummyregressor")
    assert_equal(len(pipe), 2)
    assert_false(pipe.step[0]().__sklearn_is_fitted__())
    assert_false(pipe.step[-1]().__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = pipe.slice[stop=-1]().transform(X)
    _ = pipe.fit(X, y)
    assert_true(Pointer(to=pipe.step[0]()) == Pointer(to=pipe.step[-2]()))
    assert_true(Pointer(to=pipe.step[1]()) == Pointer(to=pipe.step[-1]()))
    assert_true(Pointer(to=pipe.step[0]()) == Pointer(to=pipe.transformer))
    assert_true(Pointer(to=pipe.step[1]()) == Pointer(to=pipe.estimator))

    var name = String("dummy") + "regressor"
    ref last = pipe.named_step[DummyRegressor[]](name)
    assert_true(Pointer(to=last) == Pointer(to=pipe.step[-1]()))
    assert_true(Pointer(to=last) == Pointer(to=named.get[DummyRegressor[]](name)))
    print("named_steps", keys[0], keys[1], named.get[DummyRegressor[]](name).__sklearn_is_fitted__())
    var transformed = pipe.step[0]().transform(X)
    var prediction = last.predict(transformed)
    print("supervised", len(pipe), transformed.item(0), prediction.item(0))

    var prefix = pipe.slice[stop=-1]()
    var tail = pipe.slice[-1, 99]()
    var full = pipe.slice[-99, 99]()
    var empty = pipe.slice[2, 0]()
    var names = prefix.get_feature_names_out()
    assert_equal(len(names), 1)
    print("slice_feature", names[0])
    assert_true(Pointer(to=prefix.step[-1]()) == Pointer(to=pipe.step[0]()))
    assert_true(Pointer(to=tail.named_step[DummyRegressor[]](name)) == Pointer(to=last))
    assert_true(Pointer(to=full.estimator()) == Pointer(to=pipe))
    assert_true(empty.__sklearn_is_fitted__())
    with assert_raises(contains="KeyError"):
        _ = prefix.named_step[DummyRegressor[]](name)
    with assert_raises(contains="TypeError"):
        _ = tail.named_step[StandardScaler[]](name)
    print("slices", len(prefix), len(tail), len(full), len(empty),
          prefix.transform(X).item(0), tail.predict(transformed).item(0), full.predict(X).item(0))

    var fresh = clone(pipe)
    var fresh_named = fresh.named_steps()
    assert_false(fresh.named_step[DummyRegressor[]](name).__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator_name = "regression"
    _ = fresh.set_params(params)
    assert_false("dummyregressor" in fresh_named)
    assert_true("regression" in fresh_named)
    var renamed_keys = fresh_named.keys()
    print("renamed_steps", renamed_keys[0], renamed_keys[1])
    try:
        _ = fresh.named_step[DummyRegressor[]](name)
        raise Error("expected missing name")
    except error:
        assert_equal(String(error), "KeyError: no pipeline step named 'dummyregressor'")
    try:
        _ = fresh.named_step[StandardScaler[]]("regression")
        raise Error("expected incorrect type")
    except error:
        assert_equal(String(error), "TypeError: pipeline step 'regression' has a different type")
    assert_true(
        Pointer(to=fresh.named_step[DummyRegressor[]]("regression"))
        == Pointer(to=fresh.step[1]())
    )

    fresh.estimator_name = fresh.transformer_name
    assert_equal(len(fresh_named), 1)
    var duplicate_keys = fresh_named.keys()
    assert_equal(len(duplicate_keys), 1)
    assert_true(
        Pointer(to=fresh_named.get[DummyRegressor[]](duplicate_keys[0]))
        == Pointer(to=fresh.step[1]())
    )
    print("duplicate_steps", duplicate_keys[0])

    var nested = make_pipeline(StandardScaler(), pipe^)
    assert_equal(len(nested), 2)
    print("nested", len(nested), len(nested.step[-1]()))

    var scalers = make_pipeline(StandardScaler(), StandardScaler())
    var named_scalers = scalers.named_steps()
    _ = scalers.fit(X)
    assert_true(
        Pointer(to=scalers.named_step[StandardScaler[]]("standardscaler-1"))
        == Pointer(to=scalers.step[0]())
    )
    ref final_scaler = scalers.named_step[StandardScaler[]]("standardscaler-2")
    assert_true(
        Pointer(to=named_scalers.get[StandardScaler[]]("standardscaler-1"))
        == Pointer(to=scalers.step[0]())
    )
    assert_true(
        Pointer(to=named_scalers.get[StandardScaler[]]("standardscaler-2"))
        == Pointer(to=final_scaler)
    )
    assert_true(Pointer(to=final_scaler) == Pointer(to=scalers.step[-1]()))
    var both = final_scaler.transform(scalers.step[0]().transform(X))
    print("transformers", len(scalers), both.item(0), scalers.slice[]().transform(X).item(0))

    var live_prefix = scalers.slice[stop=1]()
    var shifted = X.copy()
    shifted.itemset(0, 4)
    shifted.itemset(1, 6)
    _ = scalers.fit(shifted)
    print("refit_slice", live_prefix.transform(X).item(0))
