from std.math import inf, nan
from std.testing import TestSuite, assert_equal, assert_raises

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.grower import grow_tree
from moxie.ensemble._hist_gradient_boosting.predictor import make_raw_tree
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


def test_raw_numeric_prediction_snapshots_updated_leaf_values() raises:
    var X = NDArray[DType.float64](Shape(4, 1))
    for i in range(4):
        X.unsafe_set(i, Float64(i))
    var mapper = _BinMapper(n_bins=5)
    var binned = mapper.fit(X).transform(X)
    var gradients: List[Float32] = [3, 3, -3, -3]
    var hessians = List[Float32]()
    var grown = grow_tree(
        binned,
        5,
        Span(gradients),
        Span(hessians),
        mapper.n_bins_non_missing_,
        mapper.is_categorical_,
        [0],
        options=_SplitOptions(min_samples_leaf=1),
    )
    var left = grown.tree.nodes[0].left
    grown.tree.nodes[left].value = -17.0
    var raw = make_raw_tree(grown.tree, mapper)
    var expected = grown.tree.predict(binned)
    grown.tree.nodes[left].value = -9.0
    var domains = List[Optional[List[Float64]]]()
    var codes: List[Float64] = [0, 1, 2, 3]
    domains.append(codes^)
    _ = mapper.fit(X, known_categories=domains^)

    var reverse = X.view_with_layout(Shape(4, 1), NDArrayStrides(-1, 1), 3)
    var actual = raw.predict(reverse)
    for i in range(4):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(3 - i))
    assert_equal(actual.unsafe_get(3), -17.0)
    with assert_raises(contains="feature count"):
        _ = raw.predict(NDArray[DType.float64](Shape(1, 2)))
    with assert_raises(contains="NotFittedError"):
        _ = make_raw_tree(grown.tree, _BinMapper(n_bins=5))
    with assert_raises(contains="numeric split"):
        _ = make_raw_tree(grown.tree, mapper)


def test_float32_raw_categorical_prediction() raises:
    var X = NDArray[DType.float32](Shape(72, 1))
    var categories: List[Float32] = [2, 254, nan[DType.float32]()]
    var gradients = List[Float32](capacity=72)
    for row in range(72):
        X.unsafe_set(row, categories[row // 24])
        gradients.append(Float32((row // 24 - 1) * 3))
    var domains = List[Optional[List[Float64]]]()
    var codes: List[Float64] = [2, 254]
    domains.append(codes^)
    var mapper = _BinMapper(n_bins=4)
    var binned = mapper.fit(X, known_categories=domains^).transform(X)
    var hessians = List[Float32]()
    var grown = grow_tree(
        binned,
        4,
        Span(gradients),
        Span(hessians),
        mapper.n_bins_non_missing_,
        mapper.is_categorical_,
        [0],
        options=_SplitOptions(min_samples_leaf=1),
    )
    var raw = make_raw_tree(grown.tree, mapper)
    var expected = grown.tree.predict(binned)
    var query_values: List[Float32] = [
        2, 254, nan[DType.float32](), 3, 2.5, -1, 255, inf[DType.float32](),
    ]
    var query = NDArray[DType.float32](Shape(len(query_values), 1))
    for i in range(len(query_values)):
        query.unsafe_set(i, query_values[i])
    var predictions = raw.predict(query)
    assert_equal(len(grown.leaves), 3)
    for i in range(len(query_values)):
        assert_equal(predictions.unsafe_get(i), expected.unsafe_get(min(i, 2) * 24))


def test_single_category_can_split_from_missing() raises:
    var X = NDArray[DType.float64](Shape(48, 1))
    var gradients = List[Float32](capacity=48)
    for row in range(48):
        X.unsafe_set(row, 42.0 if row < 24 else nan[DType.float64]())
        gradients.append(Float32(-3 if row < 24 else 3))
    var domains = List[Optional[List[Float64]]]()
    var codes: List[Float64] = [42]
    domains.append(codes^)
    var mapper = _BinMapper(n_bins=4)
    var binned = mapper.fit(X, known_categories=domains^).transform(X)
    var hessians = List[Float32]()
    var grown = grow_tree(
        binned,
        4,
        Span(gradients),
        Span(hessians),
        mapper.n_bins_non_missing_,
        mapper.is_categorical_,
        [0],
        options=_SplitOptions(min_samples_leaf=1),
    )
    var query = NDArray[DType.float64](Shape(3, 1))
    query.unsafe_set(0, 42.0)
    query.unsafe_set(1, nan[DType.float64]())
    query.unsafe_set(2, 7.0)
    var predictions = make_raw_tree(grown.tree, mapper).predict(query)
    assert_equal(predictions.unsafe_get(0), 3.0)
    assert_equal(predictions.unsafe_get(1), -3.0)
    assert_equal(predictions.unsafe_get(2), -3.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
