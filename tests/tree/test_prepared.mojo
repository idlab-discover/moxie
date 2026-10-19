"""Prepared-input tree engine: encode once, build many with per-tree weights and seeds."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._categorical import categorical_column
from moxie._random import RandomState
from moxie.tree import TreeTable, TreeColumn, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from moxie.tree._classes import ClassWeightMap
from moxie.tree._records import TreeData, Features, Values

def assert_same_tree(a: TreeData, b: TreeData) raises:
    assert_equal(a.node_count, b.node_count)
    assert_equal(a.max_depth, b.max_depth)
    for i in range(a.node_count):
        ref x = a.nodes[i]
        ref z = b.nodes[i]
        assert_equal(x.left_child, z.left_child)
        assert_equal(x.right_child, z.right_child)
        assert_equal(x.feature, z.feature)
        assert_true(x.threshold == z.threshold or (x.threshold != x.threshold and z.threshold != z.threshold))
        assert_true(x.left_cat_bitset == z.left_cat_bitset)
        assert_equal(x.impurity, z.impurity)
        assert_equal(x.n_node_samples, z.n_node_samples)
        assert_equal(x.weighted_n_node_samples, z.weighted_n_node_samples)
        assert_equal(x.missing_go_to_left, z.missing_go_to_left)
        assert_equal(x.split_kind, z.split_kind)
    var stride = a.n_outputs * a.max_n_classes
    for i in range(a.node_count * stride):
        assert_equal(a.value.unsafe_get(i), b.value.unsafe_get(i))

def assert_same_values[dtype: DType](a: nm.NDArray[dtype], b: nm.NDArray[dtype]) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))

def fixture(mut X: Features, mut y: Values, mut weights: Values, mut counts: Values) raises:
    var rng = RandomState(91)
    var n = X.shape[0]
    for i in range(n):
        for j in range(X.shape[1]):
            var value = Float32(rng.randint(0, 9)) / 4
            if j == 1 and i % 11 == 3:
                value = Float32(Float64("nan"))
            if j == 2:
                value = Float32(10 * rng.randint(0, 4))
            X.unsafe_set(i * X.shape[1] + j, value)
        y.unsafe_set(i, Float64(rng.randint(0, 3)))
        weights.unsafe_set(i, Float64(rng.randint(1, 4)) / 2)
        counts.unsafe_set(i, Float64(rng.randint(0, 3)))

def main() raises:
    var n = 40
    var X = nm.zeros[DType.float32](nm.Shape(n, 3))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var weights = nm.zeros[DType.float64](nm.Shape(n))
    var counts = nm.zeros[DType.float64](nm.Shape(n))
    fixture(X, y, weights, counts)
    var X_before = X.copy()

    # fit == prepare_fit + fit_prepared, including class and caller weights.
    var balanced = DecisionTreeClassifier(random_state=3, max_features="sqrt", class_weight=String("balanced"))
    _ = balanced.fit(X, y, sample_weight=weights)
    var template = DecisionTreeClassifier(random_state=3, max_features="sqrt", class_weight=String("balanced"))
    var prepared = template.prepare_fit(X, y, sample_weight=weights)
    assert_equal(prepared.n_samples(), n)
    var member = template.__sklearn_clone__()
    _ = member.fit_prepared(prepared)
    assert_same_tree(balanced.tree_, member.tree_)
    assert_same_values(balanced.feature_importances_, member.feature_importances_)
    assert_equal(member.max_features_, 1)
    assert_equal(member.n_classes_[0], 3)

    # A per-build weight replaces caller weights; class factors still multiply it.
    var expanded = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        expanded.unsafe_set(i, counts.unsafe_get(i) * prepared.class_weight.unsafe_get(i))
    var unweighted = DecisionTreeClassifier(random_state=3, max_features="sqrt")
    _ = unweighted.fit(X, y, sample_weight=expanded)
    var counted = template.__sklearn_clone__()
    _ = counted.fit_prepared(prepared, sample_weight=counts)
    assert_same_tree(unweighted.tree_, counted.tree_)

    # One prepared input serves many differently seeded members without mutation.
    var forest_template = ExtraTreeRegressor(max_depth=4)
    var shared = forest_template.prepare_fit(X, y)
    var prepared_X = shared.X.copy()
    for seed in range(4):
        var rng = RandomState(1000 + seed)
        var tree = forest_template.__sklearn_clone__()
        _ = tree.fit_prepared_with_random_state(shared, rng, sample_weight=counts)
        var direct = ExtraTreeRegressor(max_depth=4, random_state=1000 + seed)
        _ = direct.fit(X, y, sample_weight=counts)
        assert_same_tree(direct.tree_, tree.tree_)
        assert_same_values(direct.predict(X), tree.predict_encoded(shared.X))
        assert_same_values(direct.predict(X), tree.predict(X))
    for i in range(prepared_X.size):
        var a = prepared_X.unsafe_get(i)
        var b = shared.X.unsafe_get(i)
        assert_true(a == b or (a != a and b != b))
    for i in range(X.size):
        var a = X.unsafe_get(i)
        var b = X_before.unsafe_get(i)
        assert_true(a == b or (a != a and b != b))

    # Categorical input is encoded once; members share the encoder for prediction.
    var regression_y = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        regression_y.unsafe_set(i, Float64(i % 5) + Float64(X.unsafe_get(i * 3 + 2)) / 10)
    var categorical = DecisionTreeRegressor(max_features=2, categorical_features=Optional[List[Int]]([2]))
    var encoded_input = categorical.prepare_fit(X, regression_y)
    assert_equal(Int(encoded_input.n_categories.unsafe_get(2)), 4)
    var query = X.copy()
    query.unsafe_set(2, 999)
    var encoded_query = encoded_input.encoder.encode(query, False, False)
    assert_true(encoded_query.unsafe_get(2) != encoded_query.unsafe_get(2))
    for seed in range(3):
        var tree = categorical.__sklearn_clone__()
        var rng = RandomState(seed)
        _ = tree.fit_prepared_with_random_state(encoded_input, rng, sample_weight=counts)
        var direct = DecisionTreeRegressor(max_features=2, categorical_features=Optional[List[Int]]([2]), random_state=seed)
        _ = direct.fit(X, regression_y, sample_weight=counts)
        assert_same_tree(direct.tree_, tree.tree_)
        assert_same_values(direct.apply(query), tree.apply_encoded(encoded_query))
        assert_same_values(direct.predict(query), tree.predict_encoded(encoded_query))
        assert_true(tree.is_categorical_.value()[2])

    # String table input: prepared encoder and member encoder agree.
    var strings: List[Optional[String]] = []
    var numeric = nm.zeros[DType.float32](nm.Shape(n))
    var names: List[String] = ["a", "b", "c"]
    for i in range(n):
        strings.append(Optional[String](names[i % 3].copy()) if i % 7 != 0 else None)
        numeric.unsafe_set(i, Float32(i % 4))
    var table = TreeTable([TreeColumn(numeric^), TreeColumn(categorical_column(strings))], feature_names=Optional[List[String]](["n", "s"]))
    var binary = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        binary.unsafe_set(i, Float64(i % 3 == 1))
    var table_model = ExtraTreeClassifier(categorical_features=Optional[List[Int]]([1]), random_state=5)
    var table_input = table_model.prepare_fit(table, binary)
    var table_member = table_model.__sklearn_clone__()
    _ = table_member.fit_prepared(table_input)
    var table_direct = table_model.__sklearn_clone__()
    _ = table_direct.fit(table, binary)
    assert_same_tree(table_direct.tree_, table_member.tree_)
    var from_input = table_input.encoder.encode(table, False, False)
    var from_member = table_member.input_encoder().encode(table, False, False)
    assert_equal(from_input.size, from_member.size)
    for i in range(from_input.size):
        var a = from_input.unsafe_get(i)
        var b = from_member.unsafe_get(i)
        assert_true(a == b or (a != a and b != b))
    var proba = table_member.predict_proba_encoded(from_input)
    var direct_proba = table_direct.predict_proba(table)
    assert_same_values(proba[0], direct_proba[0])
    assert_equal(table_member.feature_names_in_.value()[1], "s")
    with assert_raises(contains="TreeTable"):
        _ = table_member.predict_encoded(table_input.encoder.encode(X, False, False))

    # Misuse is rejected without disturbing fitted state.
    var mismatched = DecisionTreeClassifier(random_state=3, max_features="sqrt", criterion="entropy", class_weight=String("balanced"))
    with assert_raises(contains="prepared input"):
        _ = mismatched.fit_prepared(prepared)
    var weight_map = ClassWeightMap()
    weight_map[0.0] = 2.0
    weight_map[1.0] = 1.0
    weight_map[2.0] = 1.0
    var other_weights = DecisionTreeClassifier(random_state=3, max_features="sqrt", class_weight=weight_map^)
    with assert_raises(contains="prepared input"):
        _ = other_weights.fit_prepared(prepared)
    var before = counted.tree_.node_count
    with assert_raises(contains="shape"):
        _ = counted.fit_prepared(prepared, sample_weight=nm.zeros[DType.float64](nm.Shape(n - 1)))
    with assert_raises(contains="nonzero"):
        _ = counted.fit_prepared(prepared, sample_weight=nm.zeros[DType.float64](nm.Shape(n)))
    var bad = counts.copy()
    bad.unsafe_set(0, Float64("inf"))
    with assert_raises(contains="nonfinite"):
        _ = counted.fit_prepared(prepared, sample_weight=bad)
    var deep = template.__sklearn_clone__()
    var params = deep.get_params()
    params.max_depth = 0
    _ = deep.set_params(params)
    with assert_raises(contains="max_depth"):
        _ = deep.fit_prepared(prepared)
    assert_equal(counted.tree_.node_count, before)
    var unfitted = DecisionTreeClassifier()
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.predict_encoded(X)
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.input_encoder()
    with assert_raises(contains="feature count"):
        _ = counted.apply_encoded(nm.zeros[DType.float32](nm.Shape(2, 2)))

    # Build-only parameters may differ between members of one prepared input.
    var shallow = template.__sklearn_clone__()
    var shallow_params = shallow.get_params()
    shallow_params.max_depth = 2
    shallow_params.min_samples_leaf = 3
    _ = shallow.set_params(shallow_params)
    _ = shallow.fit_prepared(prepared, sample_weight=counts)
    var shallow_direct = DecisionTreeClassifier(random_state=3, max_features="sqrt", max_depth=2, min_samples_leaf=3)
    _ = shallow_direct.fit(X, y, sample_weight=expanded)
    assert_same_tree(shallow_direct.tree_, shallow.tree_)
    print("PASS prepared tree inputs: fit equivalence, shared encoding, per-build weights/seeds, misuse")
