import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from balsa import encode, decode_editable, validate, ValidationOptions, Operator, TaskType, NodeType
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor, export_treelite

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 1))
    var y = nm.zeros[DType.float64](nm.Shape(6, 1))
    var y2 = nm.zeros[DType.float64](nm.Shape(6, 2))
    var labels = nm.zeros[DType.float64](nm.Shape(6, 1))
    var targets: List[Float64] = [1, 1, 1, 5, 5, 6]
    for i in range(6):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, targets[i])
        y2.unsafe_set(i * 2, targets[i])
        y2.unsafe_set(i * 2 + 1, -targets[i])
        labels.unsafe_set(i, Float64(i // 2))

    var regressor = DecisionTreeRegressor()
    regressor.params.max_depth = 1
    with assert_raises(contains="NotFittedError"):
        _ = export_treelite(regressor)
    _ = regressor.fit(X, y)
    var model = export_treelite(regressor)
    assert_equal(model.task_type, TaskType.REGRESSOR)
    assert_equal(model.average_tree_output, 1)
    assert_equal(model.postprocessor_name(), "identity")
    assert_equal(model.target_id[0], 0)
    ref tree = model.trees[0]
    assert_equal(Int(tree.num_nodes), 3)
    assert_equal(tree.node_type[0], NodeType.NUMERICAL)
    assert_equal(tree.cmp[0], Operator.LE)
    assert_equal(tree.threshold[0], 2.5)
    assert_equal(tree.leaf_value[2], regressor.tree_.value.unsafe_get(2))
    # Equal child sizes make sklearn send NaN right; the Treelite reference
    # always routes missing values left, and the export follows it.
    assert_equal(regressor.tree_.nodes[0].missing_go_to_left, 0)
    assert_equal(tree.default_left[0], 1)
    assert_equal(tree.data_count[0], 6)
    assert_equal(tree.gain_present[1], 0)
    var copy = decode_editable[DType.float64](encode(model, options=ValidationOptions(enabled=True)))
    validate(copy)
    assert_equal(copy.trees[0].threshold[0], 2.5)

    var single_leaf = DecisionTreeRegressor()
    single_leaf.params.min_samples_split = 7
    _ = single_leaf.fit(X, y)
    assert_equal(len(export_treelite(single_leaf).trees[0].gain), 0)

    var multi = ExtraTreeRegressor(random_state=0)
    _ = multi.fit(X, y2)
    var multi_model = export_treelite(multi)
    assert_equal(multi_model.num_target, 2)
    assert_equal(multi_model.target_id[0], -1)
    assert_equal(multi_model.leaf_vector_shape[0], 2)

    var classifier = DecisionTreeClassifier()
    classifier.params.max_depth = 1
    _ = classifier.fit(X, labels)
    var clf_model = export_treelite(classifier)
    assert_equal(clf_model.task_type, TaskType.MULTI_CLF)
    assert_equal(clf_model.postprocessor_name(), "identity_multiclass")
    assert_equal(clf_model.class_id[0], -1)
    assert_equal(clf_model.num_class[0], 3)
    assert_equal(len(clf_model.trees[0].leaf_vector), 6)
    assert_equal(clf_model.trees[0].leaf_vector_begin[0], clf_model.trees[0].leaf_vector_end[0])

    var single_class = DecisionTreeClassifier()
    _ = single_class.fit(X, nm.zeros[DType.float64](nm.Shape(6, 1)))
    with assert_raises(contains="at least 2 classes"):
        _ = export_treelite(single_class)
    print("PASS Treelite export metadata, NaN routing contract, round trip, single-leaf and error cases")
