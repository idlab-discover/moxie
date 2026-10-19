import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._categorical import categorical_column, StringCategories, CategoricalColumn
from moxie.tree import TreeTable, TreeColumn, StringColumn, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from moxie._random import RandomState

def main() raises:
    var strings: List[Optional[String]] = [String("red"), String("blue"), String("red"), String("blue"), None, String("")]
    var numeric = nm.zeros[DType.float32](nm.Shape(6))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i % 2))
    var col = categorical_column(strings)
    var table = TreeTable([TreeColumn(numeric.copy()), TreeColumn(col.copy())])
    numeric.unsafe_set(0, 99)
    col.set(0, String("blue"))
    assert_equal(table._columns[0][nm.NDArray[DType.float32]].unsafe_get(0), Float32(0))
    assert_equal(table._columns[1][StringColumn].get(0).value(), "red")
    var model = DecisionTreeClassifier(random_state=0, categorical_features=Optional[List[Int]]([1]))
    _ = model.fit(table, y)
    assert_equal(model.score(table, y), Float64(1))
    _ = model.predict_proba(table)
    _ = model.predict_log_proba(table)
    _ = model.cost_complexity_pruning_path(table, y)
    var clone = model.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    var copy = model.__sklearn_clone__()
    _ = copy.fit(table, y)
    var query_strings: List[Optional[String]] = [String("unknown"), None]
    var query = TreeTable([TreeColumn(nm.zeros[DType.float32](nm.Shape(2))), TreeColumn(categorical_column(query_strings))])
    var leaves = model.apply(query)
    assert_equal(leaves.unsafe_get(0), leaves.unsafe_get(1))
    var matrix = nm.zeros[DType.float32](nm.Shape(6, 2))
    with assert_raises():
        _ = model.predict(matrix)
    with assert_raises():
        var bad = TreeTable([TreeColumn(numeric.copy()), TreeColumn(nm.zeros[DType.float32](nm.Shape(2)))])
    with assert_raises():
        var bad = TreeTable([TreeColumn(matrix.copy())])
    var wrong = TreeTable([TreeColumn(numeric.copy()), TreeColumn(numeric.copy())])
    with assert_raises():
        _ = model.predict(wrong)
    var unselected = DecisionTreeClassifier()
    with assert_raises():
        _ = unselected.fit(table, y)
    with assert_raises():
        _ = model.fit(table, nm.zeros[DType.float64](nm.Shape(1)))
    assert_equal(model.score(table, y), Float64(1))
    var rng = RandomState(0)
    _ = clone.fit_with_random_state(table, y, rng)
    assert_equal(clone.score(table, y), Float64(1))
    var weights = nm.full[DType.float64](nm.Shape(6), 1)
    var reg = DecisionTreeRegressor(random_state=0, categorical_features=Optional[List[Int]]([1]))
    _ = reg.fit(table, y, sample_weight=weights)
    assert_equal(reg.score(table, y), Float64(1))
    var extra = ExtraTreeClassifier(random_state=0, categorical_features=Optional[List[Int]]([1]))
    _ = extra.fit(table, y)
    _ = extra.predict(table)
    var extra_reg = ExtraTreeRegressor(random_state=0, categorical_features=Optional[List[Int]]([1]))
    _ = extra_reg.fit(table, y)
    _ = extra_reg.predict(table)
    var params = model.get_params()
    params.categorical_features = None
    _ = model.set_params(params)
    _ = model.fit(matrix, y)
    _ = model.predict(matrix)
    assert_equal(copy.score(table, y), Float64(1))
    var fixed_domain = StringCategories(["unused", "red", "blue", ""])
    var explicit = CategoricalColumn[StringCategories].from_values(fixed_domain, strings)
    var reordered = TreeTable([TreeColumn(nm.zeros[DType.float32](nm.Shape(6))), TreeColumn(explicit^)])
    _ = clone.fit(reordered, y)
    assert_equal(clone.tree_.n_categories[1], 3)
    assert_equal(clone.score(table, y), Float64(1))
    var missing: List[Optional[String]] = [None, None, None, None, None, None]
    var all_missing = TreeTable([TreeColumn(nm.zeros[DType.float32](nm.Shape(6))), TreeColumn(categorical_column(missing))])
    _ = clone.fit(all_missing, y)
    assert_equal(clone.tree_.n_categories[1], 0)
    _ = clone.predict(query)
    var zeros = TreeTable([TreeColumn(nm.zeros[DType.float32](nm.Shape(6))), TreeColumn(nm.zeros[DType.float32](nm.Shape(6)))])
    _ = model.fit(zeros, y)
    _ = model.predict(matrix)
    var constrained = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([1]), monotonic_cst=Optional[List[Int]]([0, 1]))
    with assert_raises():
        _ = constrained.fit(reordered, y)
    print("PASS mixed-table schema, snapshots, unknowns, lifecycle and estimator methods")
