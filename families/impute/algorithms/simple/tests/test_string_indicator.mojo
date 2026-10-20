from std.testing import assert_equal, assert_false, assert_raises
from moxie import clone
from moxie.base import UnsupervisedEstimator, TransformerMixin
from moxie.impute import StringMatrix, StringMissingIndicator


def generic_fit[T: UnsupervisedEstimator & TransformerMixin](mut model: T, X: T.Input) raises -> T.Transformed where conforms_to(T.Target, Deinitable):
    return model.fit(X).transform(X)


def main() raises:
    var cells: List[Optional[String]] = ["a", None, "b", "c"]
    var X = StringMatrix(cells, 2, 2)
    var model = StringMissingIndicator()
    with assert_raises():
        _ = model.transform(X)
    with assert_raises():
        _ = model.get_feature_names_out()
    var result = generic_fit(model, X)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 1)
    assert_equal(result.unsafe_get(0), True)
    assert_equal(result.unsafe_get(1), False)
    assert_equal(model.features_, [1])
    var names: List[String] = ["first", "second"]
    model.fit_named(X, names)
    names[1] = "changed"
    assert_equal(model.get_feature_names_out(), [String("missingindicator_second")])
    with assert_raises():
        _ = model.get_feature_names_out(names.copy())
    var copied = clone(model)
    assert_false(copied.__sklearn_is_fitted__())
    var T = X.copy()
    T.set(0, 0, None)
    with assert_raises():
        _ = model.transform(T)
    _ = model.set_params(error_on_new=False).fit(X)
    assert_equal(model.transform(T).shape[1], 1)
    _ = model.set_params(features=String("all"), sparse=False).fit(X)
    assert_equal(model.transform(T).unsafe_get(0), True)
    assert_equal(model.features_, [0, 1])
    _ = model.set_params(missing_values=String("a"), sparse=String("auto"))
    var complete = StringMatrix.from_strings([String("a"), String("b")], 1, 2)
    assert_equal(model.fit_transform(complete).unsafe_get(0), True)
    with assert_raises():
        _ = model.fit(X)
    assert_equal(model.transform(complete).unsafe_get(0), True)
    _ = model.set_params(missing_values=None).fit(X)
    assert_equal(model.transform(X).unsafe_get(0), False)
    assert_equal(model.get_feature_names_out(), [String("missingindicator_x0"), String("missingindicator_x1")])
    var params = model.get_params()
    params.features = "missing-only"
    _ = model.set_params(params).fit(complete)
    assert_equal(model.transform(complete).shape[1], 0)
    _ = model.fit(X)
    assert_equal(model.features_, [1])
    result.unsafe_set(0, False)
    assert_equal(model.transform(X).unsafe_get(0), True)
    var narrow = StringMatrix.from_strings([String("a")], 1, 1)
    with assert_raises():
        _ = model.transform(narrow)
    var empty = StringMatrix(List[Optional[String]](), 0, 2)
    with assert_raises():
        _ = model.fit(empty)
    _ = model.set_params(sparse=True)
    with assert_raises():
        _ = model.fit(X)
    _ = model.set_params(sparse=String("invalid"))
    with assert_raises():
        _ = model.fit(X)
    _ = model.set_params(sparse=False, features=String("invalid"))
    with assert_raises():
        _ = model.fit(X)
    print("PASS: string indicator contracts and generic estimator consumer")
