"""Lifecycle contracts for OneHotEncoder and OrdinalEncoder."""
from numojo import NDArray, Shape, Item, zeros
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie._categorical import categorical_column, CategoricalColumn, StringCategories, NumericCategories
from moxie.preprocessing import InvertiblePreprocessor, OneHotEncoder, OrdinalEncoder
from moxie.preprocessing._vocabulary import StringKind, StringNaNKind, NumericKind


def generic_inverse[P: InvertiblePreprocessor](model: P, Z: P.Transformed) raises -> P.Input:
    # Compile-time trait conformance plus generic dispatch.
    return model.inverse_transform(Z)


def strings(values: List[Optional[String]]) raises -> CategoricalColumn[StringCategories]:
    return categorical_column(values)


def table() raises -> List[CategoricalColumn[StringCategories]]:
    var color: List[Optional[String]] = [Optional[String]("red"), Optional[String]("blue"), None, Optional[String]("red")]
    var size: List[Optional[String]] = [Optional[String]("s"), Optional[String]("m"), Optional[String]("m"), Optional[String]("l")]
    var result: List[CategoricalColumn[StringCategories]] = [strings(color), strings(size)]
    return result^


def test_one_hot() raises:
    var X = table()
    var model = OneHotEncoder(sparse_output=False)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(X)
    var Z = model.fit_transform(X)
    # color: blue, red, None (missing last); size: l, m, s.
    assert_equal(Z.shape[1], 6)
    assert_equal(model.n_features_in_, 2)
    assert_false(Bool(model.categories_[0][2]))
    var names = model.get_feature_names_out()
    assert_equal(names[2], "x0_None")
    assert_equal(names[5], "x1_s")
    var back = generic_inverse(model, Z)
    assert_equal(back[0].get(0).value(), "red")
    assert_false(Bool(back[0].get(2)))
    # Columns with their own dictionaries are translated by value.
    var other: List[Optional[String]] = [Optional[String]("violet"), Optional[String]("red")]
    var sizes: List[Optional[String]] = [Optional[String]("m"), Optional[String]("xl")]
    var unseen: List[CategoricalColumn[StringCategories]] = [strings(other), strings(sizes)]
    with assert_raises(contains="unknown categories ['violet']"):
        _ = model.transform(unseen)
    _ = model.set_params(handle_unknown="ignore")
    _ = model.fit(X)
    var ignored = model.transform(unseen)
    var row_total = Float64(0)
    for j in range(3):
        row_total += ignored.item(0, j)
    assert_equal(row_total, 0)
    # Refit replaces learned state; clone copies configuration only.
    _ = model.set_params(drop="first")
    _ = model.fit(X)
    assert_equal(model.transform(X).shape[1], 4)
    assert_equal(model.drop_idx_.value()[1].value(), 0)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.drop.value(), "first")
    var named: List[String] = ["color", "size"]
    _ = model.fit_named(X, named)
    assert_equal(model.get_feature_names_out()[0], "color_red")
    _ = model.fit(X)
    assert_false(Bool(model.feature_names_in_))
    # Sparse output stays deferred, including sklearn's default.
    var default = OneHotEncoder()
    with assert_raises(contains="sparse output"):
        _ = default.fit(X)
    with assert_raises(contains="features"):
        var narrow: List[CategoricalColumn[StringCategories]] = [X[0].copy()]
        _ = model.transform(narrow)


def test_infrequent_and_explicit() raises:
    var values: List[Optional[String]] = [Optional[String]("a"), Optional[String]("a"), Optional[String]("a"), Optional[String]("b"), Optional[String]("c")]
    var X: List[CategoricalColumn[StringCategories]] = [strings(values)]
    var model = OneHotEncoder(sparse_output=False, min_frequency=2.0, handle_unknown="infrequent_if_exist")
    var Z = model.fit_transform(X)
    assert_equal(Z.shape[1], 2)
    assert_equal(model.infrequent_categories_.value()[0].value()[1].value(), "c")
    assert_equal(model.get_feature_names_out()[1], "x0_infrequent_sklearn")
    assert_equal(generic_inverse(model, Z)[0].get(4).value(), "infrequent_sklearn")
    # Explicit categories keep caller order; None marks the missing category.
    var cats: List[List[Optional[String]]] = [[Optional[String]("c"), Optional[String]("b"), Optional[String]("a"), None]]
    var ordered = OrdinalEncoder(categories=cats^)
    var codes = ordered.fit_transform(X)
    assert_equal(codes.item(0, 0), 2)
    assert_equal(codes.item(4, 0), 0)
    var bad: List[List[Optional[String]]] = [[None, Optional[String]("a")]]
    var invalid = OrdinalEncoder(categories=bad^)
    with assert_raises(contains="Nan should be the last"):
        _ = invalid.fit(X)


def test_ordinal_missing() raises:
    var cells: List[Optional[String]] = [Optional[String]("x"), None, Optional[String]("y")]
    var X: List[CategoricalColumn[StringCategories]] = [strings(cells)]
    # StringKind: missing is sklearn's None, an ordinary final category.
    var plain = OrdinalEncoder()
    assert_equal(plain.fit_transform(X).item(1, 0), 2)
    # StringNaNKind: missing is sklearn's NaN and uses encoded_missing_value.
    var nan_style = OrdinalEncoder[StringNaNKind]()
    assert_true(isnan(nan_style.fit_transform(X).item(1, 0)))
    var coded = OrdinalEncoder[StringNaNKind, DType.int64](encoded_missing_value=-1.0)
    var Z = coded.fit_transform(X)
    assert_equal(Z.item(1, 0), -1)
    assert_false(Bool(generic_inverse(coded, Z)[0].get(1)))
    var integer = OrdinalEncoder[StringNaNKind, DType.int64]()
    with assert_raises(contains="There are missing values"):
        _ = integer.fit(X)
    var unknown = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-5.0)
    _ = unknown.fit(X)
    var other: List[Optional[String]] = [Optional[String]("z")]
    var unseen: List[CategoricalColumn[StringCategories]] = [strings(other)]
    var encoded = unknown.transform(unseen)
    assert_equal(encoded.item(0, 0), -5)
    assert_false(Bool(unknown.inverse_transform(encoded)[0].get(0)))
    assert_equal(clone(unknown).unknown_value.value(), -5)
    assert_equal(unknown.get_feature_names_out()[0], "x0")


def test_numeric() raises:
    var values = zeros[DType.float64](Shape(4))
    values.unsafe_set(0, 2.5)
    values.unsafe_set(1, Float64("nan"))
    values.unsafe_set(2, -1.0)
    values.unsafe_set(3, 2.5)
    var X: List[CategoricalColumn[NumericCategories[DType.float64]]] = [categorical_column(values)]
    var model = OneHotEncoder[NumericKind[DType.float64]](sparse_output=False)
    var Z = model.fit_transform(X)
    var names = model.get_feature_names_out()
    assert_equal(names[0], "x0_-1.0")
    assert_equal(names[2], "x0_nan")
    assert_equal(Z.item(1, 2), 1)
    var counts = zeros[DType.int64](Shape(3))
    counts.unsafe_set(0, 7)
    counts.unsafe_set(1, 3)
    counts.unsafe_set(2, 7)
    var I: List[CategoricalColumn[NumericCategories[DType.int64]]] = [categorical_column(counts)]
    var ordinal = OrdinalEncoder[NumericKind[DType.int64], DType.int64]()
    var codes = ordinal.fit_transform(I)
    assert_equal(codes.item(0, 0), 1)
    assert_equal(codes.item(1, 0), 0)
    assert_equal(ordinal.inverse_transform(codes)[0].get(2).value(), 7)


def main() raises:
    test_one_hot()
    test_infrequent_and_explicit()
    test_ordinal_missing()
    test_numeric()
    print("PASS encoder lifecycle contracts")
