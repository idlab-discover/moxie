import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie import clone
from moxie._categorical import CategoricalColumn, NumericCategories
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.impute import CategoricalNumericImputer, MissingIndicator
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OrdinalEncoder
from moxie.preprocessing._vocabulary import NumericKind


def vector[dtype: DType](values: List[Scalar[dtype]]) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](nm.Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def emit[dtype: DType](X: List[CategoricalColumn[NumericCategories[dtype]]], rows: Int) raises:
    print(rows, len(X))
    for i in range(rows):
        for column in X:
            print(column.get(i).value())


def compare[dtype: DType](lo: Scalar[dtype], hi: Scalar[dtype], extra: Scalar[dtype]) raises:
    comptime Model = CategoricalNumericImputer[dtype]
    var train = ColumnTable(nm.zeros[DType.float64](nm.Shape(4, 1)), ["measurement"])
    var values = NumericCategories[dtype](vector[dtype]([lo, hi, extra]))
    train.add_categorical("count", Model.Column(values, vector[DType.int64]([1, -1, 0, -1])))
    train.add_categorical("empty", Model.Column(values, vector[DType.int64]([-1, -1, -1, -1])))
    train.add_categorical("complete", Model.Column(values, vector[DType.int64]([1, 1, 0, 1])))
    var query = ColumnTable(nm.zeros[DType.float64](nm.Shape(2, 1)), ["measurement"])
    var other = NumericCategories[dtype](vector[dtype]([hi, extra]))
    query.add_categorical("complete", Model.Column(other, vector[DType.int64]([1, -1])))
    query.add_categorical("count", Model.Column(other, vector[DType.int64]([0, -1])))
    query.add_categorical("empty", Model.Column(other, vector[DType.int64]([0, -1])))
    var X = train.categorical_columns[Model.Domain]([1, 2, 3])
    var Z = query.categorical_columns[Model.Domain]([2, 3, 1])
    var names: List[String] = ["count", "empty", "complete"]

    for strategy in ["most_frequent", "constant"]:
        for keep in [False, True]:
            var model = Model(strategy=strategy, fill_value=extra, keep_empty_features=keep)
            var fresh = clone(model)
            with assert_raises(contains="NotFittedError"):
                _ = fresh.transform(Z)
            var fitted = model.fit_transform(X)
            assert_equal(fitted[0].domain().size(), 3)
            for statistic in model.statistics_:
                if statistic:
                    print(statistic.value())
                else:
                    print("missing")
            var out_names = model.get_feature_names_out(names.copy())
            print(len(out_names))
            for name in out_names:
                print(name)
            emit[dtype](fitted, 4)
            var replay = model.transform(Z)
            emit[dtype](replay, 2)
            assert_equal(replay[0].domain().size(), 3 if strategy == "most_frequent" else 2)
            assert_false(Bool(X[0].get(1)))
            assert_false(Bool(Z[0].get(1)))
            assert_false(clone(model).__sklearn_is_fitted__())
            with assert_raises(contains="feature count differs"):
                _ = model.transform([Z[0].copy()])
            with assert_raises(contains="matching rows"):
                _ = model.fit([X[0].copy(), Z[1].copy(), X[2].copy()])
            assert_equal(model.transform(Z)[0].get(1).value(), replay[0].get(1).value())
            var snapshot = model.copy()
            _ = model.fit(Z)
            if strategy == "most_frequent":
                assert_equal(model.transform(Z)[0].get(1).value(), hi)
            assert_equal(snapshot.transform(Z)[0].get(1).value(), replay[0].get(1).value())
            _ = model.set_params(strategy="constant", keep_empty_features=True)
            assert_false(model.__sklearn_is_fitted__())
            assert_equal(model.fit_transform(X)[1].get(1).value(), extra)
            var params = model.get_params()
            params.fill_value = None
            _ = model.set_params(params)
            assert_equal(model.fit_transform(X)[1].get(1).value(), Scalar[dtype](0))

    var invalid = Model(strategy="mean")
    with assert_raises(contains="most_frequent or constant"):
        _ = invalid.fit(X)
    var empty = Model()
    assert_equal(len(empty.fit_transform([X[1].copy()])), 0)
    with assert_raises(contains="samples and features"):
        _ = empty.fit([])


def check_boolean() raises:
    comptime Model = CategoricalNumericImputer[DType.bool]
    var domain = Model.Domain(vector[DType.bool]([False, True]))
    var column = Model.Column(domain, vector[DType.int64]([0, 1, -1]))
    var absent = Model.Column(domain, vector[DType.int64]([-1, -1, -1]))
    var model = Model(keep_empty_features=True)
    var X: Model.Input = [column^, absent^]
    var result = model.fit_transform(X)
    assert_equal(result[0].get(2).value(), False)
    assert_equal(result[1].get(2).value(), False)
    _ = model.set_params(strategy="constant", fill_value=Scalar[DType.bool](True))
    var constant = model.fit_transform(X)
    assert_equal(constant[0].get(0).value(), False)
    assert_equal(constant[0].get(2).value(), True)
    assert_equal(constant[1].get(2).value(), True)
    for output in [result^, constant^]:
        print(3, len(output))
        for i in range(3):
            for column in output:
                print(Int(column.get(i).value()))


def check_composition() raises:
    comptime Model = CategoricalNumericImputer[DType.int64]
    var domain = Model.Domain(vector[DType.int64]([9007199254740992, 9007199254740993]))
    var X = ColumnTable(nm.zeros[DType.float64](nm.Shape(3, 1)), ["measurement"])
    X.add_categorical("count", Model.Column(domain, vector[DType.int64]([1, -1, 0])), position=0)
    var Z = ColumnTable(nm.zeros[DType.float64](nm.Shape(2, 1)), ["measurement"])
    Z.add_categorical("count", Model.Column(domain, vector[DType.int64]([-1, 1])))
    var stage = make_pipeline(Model(), OrdinalEncoder[NumericKind[DType.int64]]())
    assert_equal(stage.transformer_name, "simpleimputer")
    var names: List[String] = ["count"]
    var model = ColumnTransformer(ColumnEntry("category", stage^, names^))
    var fitted = model.fit_transform(X)
    var held_out = model.transform(Z)
    print(fitted.shape[0], fitted.shape[1])
    for i in range(fitted.size):
        print(fitted.unsafe_get(i))
    print(held_out.shape[0], held_out.shape[1])
    for i in range(held_out.size):
        print(held_out.unsafe_get(i))
    for name in model.get_feature_names_out():
        print(name)
    var indicator = MissingIndicator()
    var missing = indicator.fit_transform(X.categorical_columns[Model.Domain]([0]))
    assert_equal(missing.shape[1], 1)
    assert_equal(missing.unsafe_get(1), True)


def main() raises:
    compare[DType.int64](9007199254740992, 9007199254740993, 9007199254740994)
    compare[DType.float32](-1.5, 0.5, 9.25)
    compare[DType.float64](-1.5, 0.5, 9.25)
    check_boolean()
    var invalid_fill = CategoricalNumericImputer[DType.float64](
        strategy="constant", fill_value=Float64("nan"),
    )
    with assert_raises(contains="must be finite"):
        _ = invalid_fill.fit([])
    check_composition()
