import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie import clone
from moxie._categorical import CategoricalColumn, CategoryDomain, NumericCategories, StringCategories
from moxie.compose import ColumnTable
from moxie.impute import MissingIndicator
from moxie.impute._plumbing import MissingValue


def emit(X: nm.NDArray[DType.bool]) raises:
    print(X.shape[0], X.shape[1])
    for i in range(X.size):
        print(Int(X.unsafe_get(i)))


def codes(values: List[Int64]) raises -> nm.NDArray[DType.int64]:
    var result = nm.zeros[DType.int64](nm.Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def check_indicator[Domain: CategoryDomain](
    X: List[CategoricalColumn[Domain]], Z: List[CategoricalColumn[Domain]],
    names: List[String],
) raises:
    var strict = MissingIndicator(sparse=False)
    with assert_raises(contains="NotFittedError"):
        _ = strict.transform(Z)
    strict.fit_named(X, names)
    with assert_raises(contains="new missing feature"):
        _ = strict.transform(Z)
    assert_false(clone(strict).__sklearn_is_fitted__())

    var options: List[String] = ["missing-only", "all"]
    for features in options:
        var model = MissingIndicator(features=features, error_on_new=False)
        model.fit_named(X, names)
        print(len(model.features_))
        for feature in model.features_:
            print(feature)
        for name in model.get_feature_names_out():
            print(name)
        var replay = model.transform(Z)
        emit(replay)

        var refit = model.fit_transform(Z)
        assert_equal(refit.shape[1], 2)
        assert_false(Bool(model.feature_names_in_))
        var invalid: List[CategoricalColumn[Domain]] = [X[0].copy(), Z[1].copy()]
        with assert_raises(contains="matching rows"):
            _ = model.fit(invalid)
        assert_equal(model.transform(Z).shape[1], 2)

    var finite_marker = MissingIndicator(missing_values=MissingValue[DType.float64](0))
    with assert_raises(contains="column missing tag"):
        _ = finite_marker.fit(X)
    var sparse = MissingIndicator(sparse=True)
    with assert_raises(contains="sparse output is deferred"):
        _ = sparse.fit(X)


def main() raises:
    var colors = StringCategories(["red", "blue", "unused"])
    var unseen_colors = StringCategories(["blue", "green", "red"])
    var integers = NumericCategories[DType.int64](codes([9007199254740992, 9007199254740993]))
    var unseen_integers = NumericCategories[DType.int64](codes([4, 9007199254740993]))
    var train = ColumnTable(nm.zeros[DType.float64](nm.Shape(3, 1)), ["weight"])
    train.add_categorical("color", CategoricalColumn(colors, codes([0, -1, 1])), position=0)
    train.add_categorical("color2", CategoricalColumn(colors, codes([1, 0, 1])))
    train.add_categorical("count", CategoricalColumn(integers, codes([0, -1, 1])))
    train.add_categorical("count2", CategoricalColumn(integers, codes([1, 0, 1])))
    var query = ColumnTable(nm.zeros[DType.float64](nm.Shape(2, 1)), ["weight"])
    query.add_categorical("count2", CategoricalColumn(unseen_integers, codes([1, -1])))
    query.add_categorical("color2", CategoricalColumn(unseen_colors, codes([0, -1])))
    query.add_categorical("count", CategoricalColumn(unseen_integers, codes([0, -1])))
    query.add_categorical("color", CategoricalColumn(unseen_colors, codes([1, -1])))
    check_indicator(train.categorical_columns([0, 2]), query.categorical_columns([4, 2]),
                    ["color", "color2"])
    check_indicator(train.categorical_columns[NumericCategories[DType.int64]]([3, 4]),
                    query.categorical_columns[NumericCategories[DType.int64]]([3, 1]),
                    ["count", "count2"])
