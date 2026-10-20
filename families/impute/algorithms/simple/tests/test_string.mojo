import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie import clone
from moxie.impute import StringMatrix, StringSimpleImputer


def data() raises -> StringMatrix:
    var cells: List[Optional[String]] = ["b", None, None, "a", "z", None, None, "y", None]
    return StringMatrix(cells, 3, 3)


def storage() raises:
    var cells: List[Optional[String]] = ["a", None]
    var X = StringMatrix(cells, 1, 2)
    cells[0] = String("changed")
    assert_equal(X.get(0, 0).value(), "a")
    var copied = X.copy()
    copied.set(0, 0, String("other"))
    assert_equal(X.get(0, 0).value(), "a")
    var exported = X.values()
    exported[0] = None
    assert_equal(X.get(0, 0).value(), "a")
    with assert_raises():
        _ = X.get(-1, 0)
    with assert_raises():
        X.set(0, 2, None)
    with assert_raises():
        _ = StringMatrix(cells, 2, 2)
    with assert_raises():
        _ = StringMatrix(cells, -1, 2)
    with assert_raises():
        _ = StringMatrix(cells, 9223372036854775807, 2)
    var strings: List[String] = ["", "é", "🐈"]
    var complete = StringMatrix.from_strings(strings, 1, 3)
    assert_equal(complete.get(0, 0).value(), "")
    assert_equal(complete.get(0, 2).value(), "🐈")


def strategies() raises:
    var X = data()
    var model = StringSimpleImputer(strategy="most_frequent", add_indicator=True, copy=False)
    with assert_raises():
        _ = model.transform(X)
    with assert_raises():
        _ = model.get_feature_names_out()
    var output = model.fit_transform(X)
    assert_equal(output.shape[0], 3)
    assert_equal(output.shape[1], 5)
    assert_equal(output.values.columns(), 2)
    assert_equal(output.values.get(2, 0).value(), "a")
    assert_equal(output.values.get(0, 1).value(), "y")
    assert_equal(output.indicators.shape[1], 3)
    assert_true(output.indicators.unsafe_get(2))
    assert_false(output.indicators.unsafe_get(0))
    assert_equal(model.statistics_[0].value(), "a")
    assert_false(Bool(model.statistics_[2]))
    assert_equal(len(model.indicator_.value().features_), 3)
    with assert_raises():
        _ = model.inverse_transform(output)
    output.values.set(0, 0, String("changed"))
    assert_equal(X.get(0, 0).value(), "b")
    X.set(1, 0, String("changed"))
    assert_equal(model.statistics_[0].value(), "a")
    var copied = model.copy()
    copied.statistics_[0] = String("copied")
    assert_equal(model.statistics_[0].value(), "a")
    var unfitted = clone(model)
    assert_false(unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.get_params().strategy, "most_frequent")
    _ = model.set_params(strategy="constant", fill_value=String(""), keep_empty_features=True)
    assert_false(model.__sklearn_is_fitted__())
    var filled = model.fit_transform(X)
    assert_equal(filled.shape[1], 6)
    assert_equal(filled.values.get(0, 1).value(), "")
    assert_equal(filled.values.get(0, 2).value(), "")
    _ = model.set_params(fill_value=None, add_indicator=False)
    var defaults = model.fit_transform(X)
    assert_equal(defaults.values.get(0, 1).value(), "missing_value")
    assert_equal(defaults.indicators.shape[1], 0)
    with assert_raises():
        _ = model.inverse_transform(defaults)
    var named: List[String] = ["color", "size", "empty"]
    model.fit_named(X, named)
    named[0] = "changed"
    assert_equal(model.get_feature_names_out()[0], "color")
    with assert_raises():
        _ = model.get_feature_names_out(named.copy())
    _ = model.fit(X)
    assert_equal(model.get_feature_names_out()[0], "x0")
    var wrong = StringMatrix([String("x")], 1, 1)
    with assert_raises():
        _ = model.transform(wrong)
    with assert_raises():
        model.fit_named(X, [String("bad")])
    assert_equal(model.n_features_in_, 3)


def sentinels_and_empty() raises:
    var X = StringMatrix([String("NA"), String("a"), String("b"), String("a")], 2, 2)
    var model = StringSimpleImputer(strategy="most_frequent", missing_values=String("NA"), add_indicator=True)
    var output = model.fit_transform(X)
    assert_equal(output.values.get(0, 0).value(), "b")
    assert_equal(output.indicators.shape[1], 1)
    var T = StringMatrix([String("NA"), String("NA")], 1, 2)
    var fresh_missing = model.transform(T)
    assert_equal(fresh_missing.values.get(0, 1).value(), "a")
    assert_equal(fresh_missing.indicators.shape[1], 1)
    with assert_raises():
        _ = model.fit(data())
    assert_equal(model.statistics_[0].value(), "b")
    _ = model.set_params(missing_values=None, fill_value=String("fill"), strategy="constant")
    assert_false(model.__sklearn_is_fitted__())
    assert_false(Bool(model.get_params().missing_values))
    _ = model.fit_transform(data())
    var empty_cells: List[Optional[String]] = [None, None]
    var empty = StringMatrix(empty_cells, 2, 1)
    _ = model.set_params(strategy="most_frequent")
    var dropped = model.fit_transform(empty)
    assert_equal(dropped.values.rows(), 2)
    assert_equal(dropped.values.columns(), 0)
    assert_equal(dropped.indicators.shape[1], 1)
    _ = model.set_params(keep_empty_features=True)
    _ = model.fit(X)
    with assert_raises():
        _ = model.fit(empty)
    assert_equal(model.n_features_in_, 2)
    assert_true(model.__sklearn_is_fitted__())
    var zero_rows = StringMatrix(List[Optional[String]](), 0, 2)
    var zero_cols = StringMatrix(List[Optional[String]](), 2, 0)
    with assert_raises():
        _ = model.fit(zero_rows)
    with assert_raises():
        _ = model.fit(zero_cols)
    for strategy in [String("mean"), String("median"), String("callable"), String("invalid")]:
        var invalid = StringSimpleImputer(strategy=strategy)
        with assert_raises():
            _ = invalid.fit(X)


def main() raises:
    storage()
    strategies()
    sentinels_and_empty()
    print("PASS: StringSimpleImputer storage, strategies, missingness, lifecycle and limits")
