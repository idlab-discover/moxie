from std.testing import TestSuite, assert_equal, assert_raises

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.grower import grow_tree
from moxie.ensemble._hist_gradient_boosting.splitting import _SplitOptions
from numojo import NDArray, NDArrayStrides, Shape


def test_grow_from_strided_mapper_output() raises:
    var X = NDArray[DType.float64](Shape(6, 1))
    var values: List[Float64] = [0, 2, 1, 0, 2, 1]
    for i in range(len(values)):
        X.unsafe_set(i, values[i])
    var mapper = _BinMapper(n_bins=4)
    var binned = mapper.fit(X).transform(X)
    var reverse = binned.view_with_layout(
        Shape(6, 1), NDArrayStrides(-1, binned.strides[1]), 5
    )
    # Reversed inputs are [1, 2, 0, 1, 2, 0].
    var gradients: List[Float32] = [0, -4, 4, 0, -4, 4]
    var hessians = List[Float32]()
    var monotonic: List[Int] = [1]
    var grown = grow_tree(
        reverse,
        4,
        Span(gradients),
        Span(hessians),
        mapper.n_bins_non_missing_,
        mapper.is_categorical_,
        monotonic,
        options=_SplitOptions(min_samples_leaf=1),
        shrinkage=0.25,
    )
    assert_equal(len(grown.tree.nodes), 5)
    assert_equal(len(grown.leaves), 3)
    assert_equal(grown.partition, [2, 5, 0, 3, 1, 4])
    var training = grown.tree.predict(reverse)
    var forward = grown.tree.predict(binned)
    for i in range(len(values)):
        assert_equal(
            training.unsafe_get(i), forward.unsafe_get(len(values) - 1 - i)
        )
    for id in grown.leaves:
        ref leaf = grown.tree.nodes[id]
        for position in range(leaf.start, leaf.stop):
            assert_equal(
                training.unsafe_get(grown.partition[position]), leaf.value
            )
    with assert_raises(contains="feature count"):
        _ = grown.tree.predict(NDArray[DType.uint8](Shape(1, 2)))
    var first_leaf = grown.leaves[0]
    var first_row = grown.partition[grown.tree.nodes[first_leaf].start]
    grown.tree.nodes[first_leaf].value = 17.0
    var tree = grown^.take_tree()
    assert_equal(tree.predict(reverse).unsafe_get(first_row), 17.0)


def test_interaction_feature_ties_use_ascending_ids() raises:
    var X = NDArray[DType.uint8](Shape(4, 17))
    for row in range(4):
        for feature in range(17):
            X.unsafe_set(row * 17 + feature, UInt8(row % 2))
    var gradients: List[Float32] = [1, -1, 1, -1]
    var hessians = List[Float32]()
    var counts = List[UInt32](length=17, fill=2)
    var categorical = List[Bool](length=17, fill=False)
    var monotonic = List[Int](length=17, fill=0)
    var groups: List[List[Int]] = [[16, 1]]
    var grown = grow_tree(
        X,
        3,
        Span(gradients),
        Span(hessians),
        counts,
        categorical,
        monotonic,
        interaction_groups=groups,
        options=_SplitOptions(min_samples_leaf=1),
    )
    assert_equal(grown.tree.nodes[0].split.value().feature, 1)


def test_invalid_configuration_and_stump() raises:
    var X = NDArray[DType.uint8](Shape(2, 1))
    X.unsafe_set(0, 0)
    X.unsafe_set(1, 1)
    var gradients: List[Float32] = [-1, 1]
    var hessians = List[Float32]()
    var counts: List[UInt32] = [2]
    var categorical: List[Bool] = [False]
    var monotonic: List[Int] = [0]
    var out_of_range: List[List[Int]] = [[1]]
    with assert_raises(contains="interaction feature index"):
        _ = grow_tree(
            X,
            3,
            Span(gradients),
            Span(hessians),
            counts,
            categorical,
            monotonic,
            interaction_groups=out_of_range,
        )
    with assert_raises(contains="invalid split options"):
        _ = grow_tree(
            X,
            3,
            Span(gradients),
            Span(hessians),
            counts,
            categorical,
            monotonic,
            max_leaf_nodes=1,
            options=_SplitOptions(min_samples_leaf=0),
        )
    var grown = grow_tree(
        X,
        3,
        Span(gradients),
        Span(hessians),
        counts,
        categorical,
        monotonic,
        max_leaf_nodes=1,
        options=_SplitOptions(min_samples_leaf=1),
    )
    assert_equal(grown.leaves, [0])
    assert_equal(grown.partition, [0, 1])
    assert_equal(grown.tree.nodes[0].value, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
