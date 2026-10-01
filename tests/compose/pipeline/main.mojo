"""ColumnTransformer composed with the real DummyRegressor merged into Moxie."""
from moxie.base import clone
from moxie.compose import ColumnTransformer, ColumnEntry, NoTransformer
from moxie.dummy import DummyRegressor
from moxie.pipeline import Pipeline
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from parity_io import ResultLine


def check[dtype: DType, multi: Bool]() raises:
    var columns = ColumnTransformer(
        ColumnEntry[NoTransformer[dtype], dtype]("selected", "passthrough", [2, 0, 2]),
        remainder="passthrough", n_jobs=2)
    var pipe = Pipeline(columns^, DummyRegressor[dtype](),
                        transformer_name="columns", estimator_name="dummy")
    var X = zeros[dtype](Shape(4, 3))
    var Z = zeros[dtype](Shape(2, 3))
    var y: NDArray[dtype]
    comptime if multi:
        y = zeros[dtype](Shape(4, 2))
    else:
        y = zeros[dtype](Shape(4))
    assert_equal(y.size, 8 if multi else 4)
    for i in range(4):
        for j in range(3):
            X.unsafe_set(i * 3 + j, Scalar[dtype](i * 10 + j))
        comptime if multi:
            y.unsafe_set(i * 2, Scalar[dtype](i * 2))
            y.unsafe_set(i * 2 + 1, Scalar[dtype](10 + i * 2))
        else:
            y.unsafe_set(i, Scalar[dtype](i * 2))
    for i in range(2):
        for j in range(3):
            Z.unsafe_set(i * 3 + j, Scalar[dtype](100 + i * 10 + j))
    with assert_raises():
        _ = pipe.predict(Z)
    pipe.fit(X, y)
    assert_true(pipe.__sklearn_is_fitted__())
    assert_true(pipe.transformer.__sklearn_is_fitted__())
    assert_equal(pipe.estimator.n_features_in_.value(), 4)
    var transformed = pipe.transformer.transform(Z)
    var order: List[Int] = [2, 0, 2, 1]
    for i in range(2):
        for j in range(4):
            assert_equal(transformed.unsafe_get(i * 4 + j), Z.unsafe_get(i * 3 + order[j]))
    assert_equal(pipe.transformer.get_feature_names_out()[3], "remainder__x1")
    assert_equal(pipe.transformer.output_indices_.value()["selected"].stop, 3)
    var output = ResultLine(String(Int(dtype == DType.float32), "-", Int(multi)))
    output.add("transformed", transformed)
    output.add("mean", pipe.predict(Z))
    # The pipeline fits its stored components; clone recursively drops learned state.
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.predict(Z)
    var params = fresh.get_params()
    params.estimator.strategy = "median"
    fresh.set_params(params)
    assert_equal(pipe.get_params().estimator.strategy, "mean")
    y.unsafe_set(0, 100)
    fresh.fit(X, y)
    output.add("median", fresh.predict(Z))
    output.add("original", pipe.predict(Z))
    pipe.fit(X, y)
    output.add("refit", pipe.predict(Z))
    # Dummy accepts any feature count, so the prefix must enforce its own schema.
    var wrong = zeros[dtype](Shape(2, 2))
    with assert_raises():
        _ = pipe.predict(wrong)
    with assert_raises():
        pipe.fit(wrong, y)
    assert_true(pipe.__sklearn_is_fitted__())
    output.add("after_failure", pipe.predict(Z))
    # A parameter update changes the prefix only on refit.
    var updated = pipe.get_params()
    updated.transformer.remainder = "drop"
    pipe.set_params(updated)
    assert_equal(pipe.transformer.transform(Z).shape[1], 4)
    pipe.fit(X, y)
    assert_equal(pipe.estimator.n_features_in_.value(), 3)
    output.add("drop_remainder", pipe.transformer.transform(Z))
    output.add("updated", pipe.predict(Z))
    assert_equal(fresh.transformer.transform(Z).shape[1], 4)
    output.emit()


def main() raises:
    check[DType.float64, False]()
    check[DType.float32, False]()
    check[DType.float64, True]()
    check[DType.float32, True]()
