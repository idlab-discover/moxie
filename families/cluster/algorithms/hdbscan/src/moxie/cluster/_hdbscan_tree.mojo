"""Condensed-tree extraction and flat HDBSCAN clustering."""
# Adapted from scikit-learn (BSD-3-Clause). See LICENSES/scikit-learn.txt.

import numojo as nm
from std.math import isfinite, isnan, max, min
from moxie.cluster._contracts import Labels, MergeTree


@fieldwise_init
struct CondensedEdge(ImplicitlyCopyable):
    var parent: Int
    var child: Int
    var lambda_value: Float64
    var size: Int


@fieldwise_init
struct CondensedTree(Movable):
    var n_samples: Int
    var n_clusters: Int
    var edges: List[CondensedEdge]


@fieldwise_init
struct _PendingNode(ImplicitlyCopyable):
    var linkage_node: Int
    var cluster: Int


def _emit_pruned(
    tree: MergeTree,
    node: Int,
    parent: Int,
    lambda_value: Float64,
    mut queue: List[Int],
    mut edges: List[CondensedEdge],
) raises:
    queue.clear()
    queue.append(node)
    var cursor = 0
    while cursor < len(queue):
        var child = queue[cursor]
        cursor += 1
        if child < tree.n_leaves:
            edges.append(CondensedEdge(parent, child, lambda_value, 1))
        else:
            var row = child - tree.n_leaves
            queue.append(Int(tree.children.item(row, 0)))
            queue.append(Int(tree.children.item(row, 1)))


def condense_tree(tree: MergeTree, min_cluster_size: Int) raises -> CondensedTree:
    """Prune small branches and emit edges in sklearn's breadth-first order."""
    if min_cluster_size < 2:
        raise Error("ValueError: min_cluster_size must be at least 2")
    tree.validate()
    var n = tree.n_leaves
    if (
        n < 2
        or tree.children.shape[0] != n - 1
        or tree.n_connected_components != 1
    ):
        raise Error("ValueError: expected a complete HDBSCAN linkage tree")
    if not tree.distances:
        raise Error("ValueError: HDBSCAN linkage tree requires merge distances")
    ref heights = tree.distances.value()

    var sizes = List[Int](capacity=2 * n - 1)
    for _ in range(n):
        sizes.append(1)
    for row in range(n - 1):
        var distance = heights[nm.Item(row)]
        if isnan(distance):
            raise Error("ValueError: merge distances must not contain NaN")
        if row > 0 and distance < heights[nm.Item(row - 1)]:
            raise Error("ValueError: merge distances must be nondecreasing")
        var left = Int(tree.children.item(row, 0))
        var right = Int(tree.children.item(row, 1))
        sizes.append(sizes[left] + sizes[right])

    var next_cluster = n + 1
    var pending = List[_PendingNode]()
    pending.append(_PendingNode(2 * n - 2, n))
    var pruned = List[Int]()
    var edges = List[CondensedEdge](capacity=n)
    var cursor = 0
    while cursor < len(pending):
        var node = pending[cursor]
        cursor += 1
        var row = node.linkage_node - n
        var distance = heights[nm.Item(row)]
        var lambda_value = 1.0 / distance if distance > 0 else Float64("inf")
        var left = Int(tree.children.item(row, 0))
        var right = Int(tree.children.item(row, 1))
        var left_large = sizes[left] >= min_cluster_size
        var right_large = sizes[right] >= min_cluster_size
        var split = left_large and right_large
        for side in range(2):
            var child = left if side == 0 else right
            if sizes[child] < min_cluster_size:
                _emit_pruned(tree, child, node.cluster, lambda_value, pruned, edges)
            elif split:
                edges.append(
                    CondensedEdge(node.cluster, next_cluster, lambda_value, sizes[child])
                )
                pending.append(_PendingNode(child, next_cluster))
                next_cluster += 1
            else:
                pending.append(_PendingNode(child, node.cluster))
    return CondensedTree(n, next_cluster - n, edges^)


struct _Cluster(ImplicitlyCopyable):
    var parent: Int
    var size: Int
    var birth: Float64
    var death: Float64
    var stability: Float64
    var child_stability: Float64
    var has_children: Bool
    var selected: Bool

    def __init__(out self):
        self.parent = 0
        self.size = 0
        self.birth = 0
        self.death = 0
        self.stability = 0
        self.child_stability = 0
        self.has_children = False
        self.selected = False


def _cluster_stats(tree: CondensedTree) -> List[_Cluster]:
    var clusters = List[_Cluster](capacity=tree.n_clusters)
    for _ in range(tree.n_clusters):
        clusters.append(_Cluster())
    var previous_parent = -1
    for edge in tree.edges:
        var parent = edge.parent - tree.n_samples
        # sklearn's probability normalization uses the last contiguous run
        # of edges for a parent, which can recur after a branch is pruned.
        if parent != previous_parent:
            clusters[parent].death = edge.lambda_value
            previous_parent = parent
        else:
            clusters[parent].death = max(clusters[parent].death, edge.lambda_value)
        clusters[parent].stability += (
            edge.lambda_value - clusters[parent].birth
        ) * Float64(edge.size)
        if edge.size > 1:
            var child = edge.child - tree.n_samples
            clusters[parent].has_children = True
            clusters[child].parent = parent
            clusters[child].birth = edge.lambda_value
            clusters[child].size = edge.size
            if parent == 0:
                clusters[0].size += edge.size
    return clusters^


def _select_eom(mut clusters: List[_Cluster], first: Int, max_size: Int):
    for node in range(len(clusters) - 1, first - 1, -1):
        var cluster = clusters[node]
        var score = cluster.stability
        if cluster.child_stability > score or cluster.size > max_size:
            score = cluster.child_stability
        else:
            clusters[node].selected = True
        if node > 0:
            clusters[cluster.parent].child_stability += score

    var covered = List[Bool](capacity=len(clusters))
    covered.append(clusters[0].selected)
    for node in range(1, len(clusters)):
        var ancestor_selected = covered[clusters[node].parent]
        if ancestor_selected:
            clusters[node].selected = False
        covered.append(ancestor_selected or clusters[node].selected)


def _merge_epsilon(
    mut clusters: List[_Cluster], epsilon: Float64, allow_root: Bool
):
    # Compute where an upward walk from each child would stop. Condensed
    # cluster IDs are topological, so each parent's answer is already known.
    var upward = List[Int](capacity=len(clusters))
    var selected = List[Bool](capacity=len(clusters))
    upward.append(0)
    selected.append(clusters[0].selected)
    for _ in range(1, len(clusters)):
        selected.append(False)
    for node in range(1, len(clusters)):
        var parent = clusters[node].parent
        var target: Int
        if parent == 0:
            target = 0 if allow_root else node
        elif 1.0 / clusters[parent].birth > epsilon:
            target = parent
        else:
            target = upward[parent]
        upward.append(target)
        if clusters[node].selected:
            if 1.0 / clusters[node].birth < epsilon:
                selected[target] = True
            else:
                selected[node] = True
    for node in range(len(clusters)):
        clusters[node].selected = selected[node]


@fieldwise_init
struct FlatClustering(Movable):
    var labels: Labels
    var probabilities: nm.NDArray[DType.float64]


def _label_points(
    tree: CondensedTree, clusters: List[_Cluster], allow_root: Bool, epsilon: Float64
) raises -> FlatClustering:
    var assigned = List[Int](capacity=len(clusters))
    var label_map = List[Int](capacity=len(clusters))
    var label_count = 0
    for node in range(len(clusters)):
        if clusters[node].selected:
            assigned.append(node)
            label_map.append(label_count)
            label_count += 1
        else:
            assigned.append(0 if node == 0 else assigned[clusters[node].parent])
            label_map.append(-1)

    var labels = nm.zeros[DType.int64](nm.Shape(tree.n_samples))
    var probabilities = nm.zeros[DType.float64](nm.Shape(tree.n_samples))
    var root_threshold = clusters[0].death
    if epsilon != 0:
        root_threshold = 1.0 / epsilon
    for edge in tree.edges:
        if edge.size > 1:
            continue
        var cluster = assigned[edge.parent - tree.n_samples]
        if cluster == 0:
            if (
                not allow_root or label_count != 1 or edge.lambda_value < root_threshold
            ):
                labels.unsafe_set(edge.child, -1)
                continue
            if label_map[0] == -1:
                raise Error(
                    "KeyError: HDBSCAN root is absent from the cluster label map"
                )
        labels.unsafe_set(edge.child, Int64(label_map[cluster]))
        var death = clusters[cluster].death
        var probability = 1.0 if death == 0 or not isfinite(edge.lambda_value) else (
            min(edge.lambda_value, death) / death
        )
        probabilities.unsafe_set(edge.child, probability)
    return FlatClustering(labels^, probabilities^)


def tree_to_labels(
    tree: MergeTree,
    min_cluster_size: Int = 10,
    *,
    cluster_selection_method: String = "eom",
    allow_single_cluster: Bool = False,
    cluster_selection_epsilon: Float64 = 0,
    max_cluster_size: Optional[Int] = None,
) raises -> FlatClustering:
    """Extract exact cluster IDs and membership strengths from dense linkage."""
    if cluster_selection_method != "eom" and cluster_selection_method != "leaf":
        raise Error(
            "ValueError: cluster_selection_method must be 'eom' or 'leaf'"
        )
    if not isfinite(cluster_selection_epsilon) or cluster_selection_epsilon < 0:
        raise Error(
            "ValueError: cluster_selection_epsilon must be finite and nonnegative"
        )
    if max_cluster_size and max_cluster_size.value() < 1:
        raise Error("ValueError: max_cluster_size must be at least 1")

    var condensed = condense_tree(tree, min_cluster_size)
    var clusters = _cluster_stats(condensed)
    if cluster_selection_method == "eom":
        _select_eom(
            clusters,
            0 if allow_single_cluster else 1,
            max_cluster_size.value() if max_cluster_size else tree.n_leaves + 1,
        )
    else:
        # sklearn's leaf search starts with actual cluster-to-cluster edges.
        for node in range(1, len(clusters)):
            clusters[node].selected = not clusters[node].has_children
    if cluster_selection_epsilon != 0 and len(clusters) > 1:
        _merge_epsilon(clusters, cluster_selection_epsilon, allow_single_cluster)
    return _label_points(
        condensed, clusters, allow_single_cluster, cluster_selection_epsilon
    )
