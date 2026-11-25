# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Best-first growth and binned prediction for one boosting tree."""

from std.math import isfinite

from numojo import NDArray, Shape

from .._sampling import numpy_sum
from .histogram import _HistogramBin, _Histograms, compute_histograms
from .splitting import (
    _Split,
    _SplitOptions,
    _ValueBounds,
    find_node_split,
    partition_rows,
)


@fieldwise_init
struct _TreeNode(Copyable, Movable):
    var depth: Int
    var start: Int
    var stop: Int
    var value: Float64
    var left: Int
    var right: Int
    var split: Optional[_Split]

    def __init__(out self, depth: Int, start: Int, stop: Int, value: Float64):
        self = Self(depth, start, stop, value, -1, -1, None)

    def is_leaf(self) -> Bool:
        return not self.split


@fieldwise_init
struct _BinnedTree(Copyable, Movable):
    var nodes: List[_TreeNode]
    var n_features: Int
    var missing_bin: UInt8

    def predict(self, X_binned: NDArray[DType.uint8]) raises -> NDArray[DType.float64]:
        """Predict from the training mapper's byte encoding."""
        if X_binned.ndim != 2 or X_binned.shape[1] != self.n_features:
            raise Error(
                "ValueError: binned prediction feature count does not match tree"
            )
        var result = NDArray[DType.float64](Shape(X_binned.shape[0]))
        for row in range(X_binned.shape[0]):
            var index = 0
            while True:
                ref node = self.nodes[index]
                if node.split:
                    ref split = node.split.value()
                    var code = X_binned.unsafe_get(
                        row * X_binned.strides[0] + split.feature * X_binned.strides[1]
                    )
                    if code > self.missing_bin:
                        raise Error("ValueError: prediction bin exceeds tree bin range")
                    index = (
                        node.left
                        if split.goes_left(code, self.missing_bin)
                        else node.right
                    )
                else:
                    result.unsafe_set(row, node.value)
                    break
        return result^


@fieldwise_init
struct _GrownTree(Movable):
    """Own the predictor and training rows needed for loss-specific leaf updates."""

    var tree: _BinnedTree
    var partition: List[Int]
    var leaves: List[Int]

    def take_tree(deinit self) -> _BinnedTree:
        return self.tree^


@fieldwise_init
struct _InteractionPath(Copyable, Movable):
    var groups: List[Int]
    var features: List[Int]


def _allowed_features(
    n_features: Int, groups: List[List[Int]], active_groups: List[Int]
) -> List[Int]:
    var allowed = List[Bool](length=n_features, fill=not groups)
    for group in active_groups:
        for feature in groups[group]:
            allowed[feature] = True
    var features = List[Int]()
    for feature in range(n_features):
        if allowed[feature]:
            features.append(feature)
    return features^


def _child_path(
    path: _InteractionPath,
    split_feature: Int,
    n_features: Int,
    groups: List[List[Int]],
) -> _InteractionPath:
    if not groups:
        return path.copy()
    var active_groups = List[Int]()
    for index in path.groups:
        if split_feature in groups[index]:
            active_groups.append(index)
    var features = _allowed_features(n_features, groups, active_groups)
    return _InteractionPath(active_groups^, features^)


@fieldwise_init
struct _PendingSplit(Movable):
    var split: _Split
    var histograms: _Histograms
    var path: _InteractionPath
    var bounds: _ValueBounds

    def take_histograms(deinit self) -> _Histograms:
        return self.histograms^


@fieldwise_init
struct _Priority(ImplicitlyCopyable, Movable):
    var node: Int
    var gain: Float64


def _sift_up(mut queue: List[_Priority], position: Int):
    var entry = queue[position]
    var hole = position
    while hole > 0:
        var parent = (hole - 1) // 2
        if queue[parent].gain >= entry.gain:
            break
        queue[hole] = queue[parent]
        hole = parent
    queue[hole] = entry


def _pop_best(mut queue: List[_Priority]) -> Int:
    var node = queue[0].node
    var last = queue.pop()
    if queue:
        # Match heapq's right-child tie rule, then sift the replacement up.
        var hole = 0
        var child = 1
        while child < len(queue):
            var right = child + 1
            if right < len(queue) and queue[right].gain >= queue[child].gain:
                child = right
            queue[hole] = queue[child]
            hole = child
            child = 2 * hole + 1
        queue[hole] = last
        _sift_up(queue, hole)
    return node


@fieldwise_init
struct _Growth(Movable):
    var nodes: List[_TreeNode]
    var queue: List[_Priority]
    var pending: List[Optional[_PendingSplit]]
    var options: _SplitOptions
    var has_missing: List[Bool]

    def take_nodes(deinit self) -> List[_TreeNode]:
        return self.nodes^

    def add_node(mut self, var node: _TreeNode):
        self.nodes.append(node^)
        self.pending.append(None)

    def consider(
        mut self,
        index: Int,
        total: _HistogramBin,
        bounds: _ValueBounds,
        var histograms: _Histograms,
        path: _InteractionPath,
        counts: List[UInt32],
        categorical: List[Bool],
        monotonic: List[Int],
    ) raises:
        var split = find_node_split(
            histograms,
            total,
            self.nodes[index].value,
            counts,
            self.has_missing,
            categorical,
            monotonic,
            allowed_features=Span(path.features),
            bounds=bounds,
            options=self.options,
        )
        if split:
            var winner = split.take()
            self.queue.append(_Priority(index, winner.gain))
            _sift_up(self.queue, len(self.queue) - 1)
            self.pending[index] = _PendingSplit(
                winner^, histograms^, path.copy(), bounds
            )


def _child_bounds(
    split: _Split, parent: _ValueBounds, monotonic: Int
) -> Tuple[_ValueBounds, _ValueBounds]:
    if monotonic == 0:
        return (parent, parent)
    var middle = (split.value_left + split.value_right) / 2.0
    var lower = _ValueBounds(parent.lower, middle)
    var upper = _ValueBounds(middle, parent.upper)
    return (lower, upper) if monotonic > 0 else (upper, lower)


def _root_totals(
    bins: List[_HistogramBin], n_rows: Int, unit_hessian: Bool
) raises -> _HistogramBin:
    var sums = NDArray[DType.float64](Shape(len(bins)))
    for i in range(len(bins)):
        sums.unsafe_set(i, bins[i].sum_gradients)
    var gradient = numpy_sum(sums)
    var hessian = Float64(n_rows)
    if not unit_hessian:
        for i in range(len(bins)):
            sums.unsafe_set(i, bins[i].sum_hessians)
        hessian = numpy_sum(sums)
    return _HistogramBin(gradient, hessian, n_rows)


def grow_tree[gradient_origin: ImmOrigin, hessian_origin: ImmOrigin, //](
    X_binned: NDArray[DType.uint8],
    n_bins: Int,
    gradients: Span[Float32, gradient_origin],
    hessians: Span[Float32, hessian_origin],
    counts: List[UInt32],
    categorical: List[Bool],
    monotonic: List[Int],
    *,
    interaction_groups: List[List[Int]] = [],
    max_leaf_nodes: Optional[Int] = None,
    max_depth: Optional[Int] = None,
    options: _SplitOptions = _SplitOptions(),
    shrinkage: Float64 = 1.0,
) raises -> _GrownTree:
    """Grow one tree using all features available on each interaction path."""
    if X_binned.ndim != 2 or X_binned.shape[0] == 0 or X_binned.shape[1] == 0:
        raise Error("ValueError: grower requires a nonempty rank-2 binned matrix")
    var n_rows = X_binned.shape[0]
    var n_features = X_binned.shape[1]
    if n_bins < 2 or n_bins > 256 or len(counts) != n_features or (
        len(categorical) != n_features or len(monotonic) != n_features
    ):
        raise Error("ValueError: invalid bin count or feature metadata")
    var depth_limit = max_depth.or_else(n_rows)
    var leaf_limit = max_leaf_nodes.or_else(n_rows)
    if depth_limit < 1 or leaf_limit < 1 or not isfinite(shrinkage) or shrinkage < 0:
        raise Error("ValueError: invalid tree depth, leaf limit, or shrinkage")
    options.validate()
    for feature in range(n_features):
        if Int(counts[feature]) >= n_bins or monotonic[feature] < -1 or (
            monotonic[feature] > 1 or (categorical[feature] and monotonic[feature] != 0)
        ):
            raise Error("ValueError: invalid feature bin count or monotonic constraint")
    var root_groups = List[Int](capacity=len(interaction_groups))
    for index in range(len(interaction_groups)):
        for feature in interaction_groups[index]:
            if feature < 0 or feature >= n_features:
                raise Error("ValueError: interaction feature index is out of range")
        root_groups.append(index)
    var root_features = _allowed_features(n_features, interaction_groups, root_groups)
    if not root_features:
        raise Error("ValueError: interaction groups must allow at least one feature")
    var root_path = _InteractionPath(root_groups^, root_features^)
    var missing_bin = UInt8(n_bins - 1)

    var partition = List[Int](capacity=n_rows)
    for row in range(n_rows):
        partition.append(row)
    var scratch = List[Int](length=n_rows, fill=0)
    var histograms = compute_histograms(
        X_binned,
        n_bins,
        Span(partition),
        gradients,
        hessians,
        allowed_features=root_path.features.copy(),
    )
    var has_missing = List[Bool](length=n_features, fill=False)
    for feature in root_path.features:
        has_missing[feature] = histograms.feature(feature)[Int(missing_bin)].count > 0
    var total = _root_totals(
        histograms.feature(root_path.features[0]), n_rows, not hessians
    )
    var growth = _Growth(
        List[_TreeNode](),
        List[_Priority](),
        List[Optional[_PendingSplit]](),
        options,
        has_missing^,
    )
    growth.add_node(_TreeNode(0, 0, n_rows, 0.0))
    if leaf_limit > 1 and n_rows // 2 >= options.min_samples_leaf and (
        total.sum_hessians >= options.min_hessian_to_split
    ):
        growth.consider(
            0, total, _ValueBounds(), histograms^, root_path, counts, categorical,
            monotonic,
        )

    var n_leaves = 1
    while growth.queue and n_leaves < leaf_limit:
        var index = _pop_best(growth.queue)
        var branch = growth.pending[index].take()
        var start = growth.nodes[index].start
        var stop = growth.nodes[index].stop
        var depth = growth.nodes[index].depth + 1
        var middle = start + partition_rows(
            X_binned,
            branch.split,
            missing_bin,
            Span(partition)[start:stop],
            Span(scratch),
        )
        if not growth.has_missing[branch.split.feature]:
            branch.split.missing_go_to_left = (
                branch.split.left.count > branch.split.right.count
            )
        var left = len(growth.nodes)
        var right = left + 1
        growth.add_node(_TreeNode(depth, start, middle, branch.split.value_left))
        growth.add_node(_TreeNode(depth, middle, stop, branch.split.value_right))
        growth.nodes[index].left = left
        growth.nodes[index].right = right
        growth.nodes[index].split = branch.split.copy()
        n_leaves += 1
        if n_leaves == leaf_limit or depth == depth_limit:
            continue

        var search_left = branch.split.left.count // 2 >= options.min_samples_leaf
        var search_right = branch.split.right.count // 2 >= options.min_samples_leaf
        if not (search_left or search_right):
            continue
        var path = _child_path(
            branch.path, branch.split.feature, n_features, interaction_groups
        )
        var left_bounds: _ValueBounds
        var right_bounds: _ValueBounds
        left_bounds, right_bounds = _child_bounds(
            branch.split, branch.bounds, monotonic[branch.split.feature]
        )
        var left_is_smaller = middle - start < stop - middle
        var small_start = start if left_is_smaller else middle
        var small_stop = middle if left_is_smaller else stop
        var smaller = compute_histograms(
            X_binned,
            n_bins,
            Span(partition)[small_start:small_stop],
            gradients,
            hessians,
            allowed_features=path.features.copy(),
        )
        var left_total = branch.split.left
        var right_total = branch.split.right
        var larger = branch^.take_histograms().subtract_sibling(smaller)
        var left_hist: _Histograms
        var right_hist: _Histograms
        if left_is_smaller:
            left_hist = smaller^
            right_hist = larger^
        else:
            left_hist = larger^
            right_hist = smaller^
        if search_left:
            growth.consider(
                left, left_total, left_bounds, left_hist^, path, counts, categorical,
                monotonic,
            )
        if search_right:
            growth.consider(
                right, right_total, right_bounds, right_hist^, path, counts,
                categorical, monotonic,
            )

    var leaves = List[Int](capacity=n_leaves)
    for index in range(len(growth.nodes)):
        if growth.nodes[index].is_leaf():
            growth.nodes[index].value *= shrinkage
            leaves.append(index)
    var tree = _BinnedTree(growth^.take_nodes(), n_features, missing_bin)
    return _GrownTree(tree^, partition^, leaves^)
