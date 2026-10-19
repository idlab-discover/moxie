# scikit-learn developers, BSD-3-Clause; see family LICENSES/scikit-learn.txt.
"""Depth-first construction and weakest-link cost-complexity pruning."""
import numojo as nm
from std.math import isfinite
from moxie._random import RandomState
from moxie.tree._records import (TreeData, TrainingData, SplitWorkspace, SplitRecord,
    ParentInfo, BuilderConfig, Node, CategoryBits, PruningPath)
from moxie.tree._splitter import DenseSplitter
from moxie.tree._engine_api import TreeBuilder
from moxie.tree._tree import Tree, new_tree, copy_tree, validate_tree

@fieldwise_init
struct Branch(Copyable, Movable):
    var start: Int
    var end: Int
    var parent: Int
    var left: Bool
    var depth: Int
    var info: ParentInfo


@fieldwise_init
struct Candidate(Copyable, Movable):
    var branch: Branch
    var node_id: Int
    var split: SplitRecord


def heap_push(mut heap: List[Candidate], var value: Candidate):
    heap.append(value.copy())
    var hole = len(heap) - 1
    while hole > 0:
        var parent = (hole - 1) // 2
        if heap[parent].split.improvement >= value.split.improvement:
            break
        heap[hole] = heap[parent].copy()
        hole = parent
    heap[hole] = value^


def heap_pop(mut heap: List[Candidate]) -> Candidate:
    var result = heap[0].copy()
    var saved = heap.pop()
    if len(heap) == 0:
        return result^
    var hole = 0
    var child = 2
    while child < len(heap):
        # std::pop_heap selects the right child for equal priorities.
        if heap[child].split.improvement < heap[child - 1].split.improvement:
            child -= 1
        heap[hole] = heap[child].copy()
        hole = child
        child = 2 * (hole + 1)
    if child == len(heap):
        heap[hole] = heap[child - 1].copy()
        hole = child - 1
    while hole > 0:
        var parent = (hole - 1) // 2
        if heap[parent].split.improvement >= saved.split.improvement:
            break
        heap[hole] = heap[parent].copy()
        hole = parent
    heap[hole] = saved^
    return result^

struct DepthFirstTreeBuilder[kind: Int](TreeBuilder):
    var splitter: DenseSplitter[Self.kind]
    var config: BuilderConfig
    def __init__(out self, var splitter: DenseSplitter[Self.kind], config: BuilderConfig) raises:
        if config.min_samples_split < 2 or config.min_samples_leaf < 1 or config.max_depth < 0 or not isfinite(config.min_weight_leaf) or not config.min_impurity_decrease >= 0 or (config.max_leaf_nodes != -1 and config.max_leaf_nodes < 2):
            raise Error("ValueError: invalid BuilderConfig")
        self.splitter = splitter^
        self.config = config.copy()

    def prune(self, tree: TreeData, ccp_alpha: Float64) raises -> TreeData:
        return prune(tree, ccp_alpha)

    def cost_complexity_pruning_path(self, tree: TreeData) raises -> PruningPath:
        return cost_complexity_pruning_path(tree)

    def build(mut self, mut tree: TreeData, data: TrainingData, mut rng: RandomState) raises:
        if tree.node_count != 0:
            raise Error("ValueError: build requires fresh tree storage")
        if self.config.max_leaf_nodes != -1:
            self._best_first(tree, data, rng)
            return
        var n = data.X.shape[0]
        var p = data.X.shape[1]
        var work = SplitWorkspace(nm.zeros[DType.int64](nm.Shape(n)), nm.zeros[DType.int64](nm.Shape(p)),
            nm.zeros[DType.int64](nm.Shape(p)), nm.zeros[DType.float32](nm.Shape(n)), 0, n, 0)
        self.splitter.initialize(data, work, rng)
        var stack = List[Branch]()
        stack.append(Branch(0, work.end, -2, False, 0, ParentInfo(-Float64("inf"), Float64("inf"), Float64("inf"), 0)))
        var storage = Tree()
        var constrained = False
        for j in range(p):
            constrained = constrained or data.monotonic_cst.unsafe_get(j * Int(data.monotonic_cst.strides[0])) != 0
        while len(stack) > 0:
            var task = stack.pop()
            var weight = self.splitter.node_reset(data, work, task.start, task.end)
            var impurity = task.info.impurity
            if task.parent == -2:
                impurity = self.splitter.node_impurity()
            task.info.impurity = impurity
            var value = nm.zeros[DType.float64](nm.Shape(tree.n_outputs, tree.max_n_classes))
            self.splitter.node_value(value)
            if constrained:
                self.splitter.clip_node_value(value, task.info.lower_bound, task.info.upper_bound)
            var leaf = task.depth >= self.config.max_depth or task.end - task.start < self.config.min_samples_split or task.end - task.start < 2 * self.config.min_samples_leaf or weight < 2 * self.config.min_weight_leaf or impurity <= 2.220446049250313e-16
            var split = SplitRecord(-2, task.end, -2, CategoryBits(0), -1, 0, 0, 0, task.info.lower_bound, task.info.upper_bound, 0)
            if not leaf:
                split = self.splitter.node_split(data, work, task.info)
                leaf = split.pos >= task.end or split.improvement + 2.220446049250313e-16 < self.config.min_impurity_decrease
            var node = Node(-1, -1, split.feature, split.threshold, split.left_cat_bitset, impurity,
                task.end - task.start, weight, split.missing_go_to_left, Int8(-1) if leaf else split.split_kind)
            var node_id = storage.add_node(tree, task.parent, task.left, node)
            tree.max_depth = max(tree.max_depth, task.depth)
            for j in range(value.size):
                tree.value.unsafe_set(node_id * value.size + j, value.unsafe_get(j))
            if not leaf:
                var left_info = task.info.copy()
                var right_info = task.info.copy()
                left_info.impurity = split.impurity_left
                right_info.impurity = split.impurity_right
                var sign = data.monotonic_cst.unsafe_get(split.feature * Int(data.monotonic_cst.strides[0]))
                if sign > 0:
                    left_info.upper_bound = split.upper_bound
                    right_info.lower_bound = split.lower_bound
                elif sign < 0:
                    left_info.lower_bound = split.lower_bound
                    right_info.upper_bound = split.upper_bound
                stack.append(Branch(split.pos, task.end, node_id, False, task.depth + 1, right_info^))
                stack.append(Branch(task.start, split.pos, node_id, True, task.depth + 1, left_info^))
        storage.resize(tree, tree.node_count)
        validate_tree(tree)


    def _candidate(mut self, mut tree: TreeData, data: TrainingData, mut work: SplitWorkspace, var branch: Branch, constrained: Bool, mut frontier: List[Candidate]) raises:
        var weight = self.splitter.node_reset(data, work, branch.start, branch.end)
        if branch.parent == -2:
            branch.info.impurity = self.splitter.node_impurity()
        # Best-first does not retain ancestor constant-feature prefixes.
        branch.info.n_constant_features = 0
        var value = nm.zeros[DType.float64](nm.Shape(tree.n_outputs, tree.max_n_classes))
        self.splitter.node_value(value)
        if constrained:
            self.splitter.clip_node_value(value, branch.info.lower_bound, branch.info.upper_bound)
        var leaf = branch.depth >= self.config.max_depth or branch.end - branch.start < self.config.min_samples_split or branch.end - branch.start < 2 * self.config.min_samples_leaf or weight < 2 * self.config.min_weight_leaf or branch.info.impurity <= 2.220446049250313e-16
        var split = SplitRecord(-2, branch.end, -2, CategoryBits(0), -1, 0, 0, 0, branch.info.lower_bound, branch.info.upper_bound, 0)
        if not leaf:
            split = self.splitter.node_split(data, work, branch.info)
            leaf = split.pos >= branch.end or split.improvement + 2.220446049250313e-16 < self.config.min_impurity_decrease
        var storage = Tree()
        var id = storage.add_node(tree, branch.parent, branch.left, Node(-1, -1, split.feature, split.threshold, split.left_cat_bitset,
            branch.info.impurity, branch.end - branch.start, weight, split.missing_go_to_left, Int8(-1) if leaf else split.split_kind))
        tree.max_depth = max(tree.max_depth, branch.depth)
        for j in range(value.size):
            tree.value.unsafe_set(id * value.size + j, value.unsafe_get(j))
        if not leaf:
            heap_push(frontier, Candidate(branch^, id, split^))

    def _best_first(mut self, mut tree: TreeData, data: TrainingData, mut rng: RandomState) raises:
        var n = data.X.shape[0]
        var p = data.X.shape[1]
        var work = SplitWorkspace(nm.zeros[DType.int64](nm.Shape(n)), nm.zeros[DType.int64](nm.Shape(p)), nm.zeros[DType.int64](nm.Shape(p)), nm.zeros[DType.float32](nm.Shape(n)), 0, n, 0)
        self.splitter.initialize(data, work, rng)
        var constrained = False
        for j in range(p):
            constrained = constrained or data.monotonic_cst.unsafe_get(j * Int(data.monotonic_cst.strides[0])) != 0
        var frontier = List[Candidate]()
        self._candidate(tree, data, work, Branch(0, work.end, -2, False, 0, ParentInfo(-Float64("inf"), Float64("inf"), Float64("inf"), 0)), constrained, frontier)
        var leaves = 1
        while len(frontier) > 0:
            var candidate = heap_pop(frontier)
            var id = candidate.node_id
            if leaves >= self.config.max_leaf_nodes:
                tree.nodes[id].split_kind = -1
                tree.nodes[id].feature = -2
                tree.nodes[id].threshold = -2
                continue
            leaves += 1
            var left = candidate.branch.info.copy()
            var right = candidate.branch.info.copy()
            left.impurity = candidate.split.impurity_left
            right.impurity = candidate.split.impurity_right
            var sign = data.monotonic_cst.unsafe_get(candidate.split.feature * Int(data.monotonic_cst.strides[0]))
            if sign > 0:
                left.upper_bound = candidate.split.upper_bound
                right.lower_bound = candidate.split.lower_bound
            elif sign < 0:
                left.lower_bound = candidate.split.lower_bound
                right.upper_bound = candidate.split.upper_bound
            self._candidate(tree, data, work, Branch(candidate.branch.start, candidate.split.pos, id, True, candidate.branch.depth + 1, left^), constrained, frontier)
            self._candidate(tree, data, work, Branch(candidate.split.pos, candidate.branch.end, id, False, candidate.branch.depth + 1, right^), constrained, frontier)
        var storage = Tree()
        storage.resize(tree, tree.node_count)
        validate_tree(tree)


@fieldwise_init
struct PruningResult(Movable):
    var leaves: nm.NDArray[DType.uint8]
    var path: PruningPath

def pruning_sequence(tree: TreeData, limit: Float64, stop: Bool) raises -> PruningResult:
    validate_tree(tree)
    var active = nm.full[DType.uint8](nm.Shape(tree.node_count), 1)
    var leaf = nm.zeros[DType.uint8](nm.Shape(tree.node_count))
    var risk = nm.zeros[DType.float64](nm.Shape(tree.node_count))
    var counts = nm.zeros[DType.int64](nm.Shape(tree.node_count))
    var alphas = nm.zeros[DType.float64](nm.Shape(tree.node_count))
    var impurities = nm.zeros[DType.float64](nm.Shape(tree.node_count))
    var used = 0
    var alpha = Float64(0)
    var root_weight = tree.nodes[0].weighted_n_node_samples
    for i in range(tree.node_count):
        leaf.unsafe_set(i, UInt8(tree.nodes[i].left_child == -1))
    while True:
        for reverse in range(tree.node_count):
            var i = tree.node_count - 1 - reverse
            if active.unsafe_get(i) == 0:
                continue
            if leaf.unsafe_get(i) != 0:
                risk.unsafe_set(i, tree.nodes[i].impurity * tree.nodes[i].weighted_n_node_samples / root_weight)
                counts.unsafe_set(i, 1)
            else:
                var left = tree.nodes[i].left_child
                var right = tree.nodes[i].right_child
                risk.unsafe_set(i, risk.unsafe_get(left) + risk.unsafe_get(right))
                counts.unsafe_set(i, counts.unsafe_get(left) + counts.unsafe_get(right))
        alphas.unsafe_set(used, alpha)
        impurities.unsafe_set(used, risk.unsafe_get(0))
        used += 1
        var chosen = -1
        var smallest = Float64("inf")
        for i in range(tree.node_count):
            if active.unsafe_get(i) != 0 and leaf.unsafe_get(i) == 0:
                var a = (tree.nodes[i].impurity * tree.nodes[i].weighted_n_node_samples / root_weight - risk.unsafe_get(i)) / Float64(counts.unsafe_get(i) - 1)
                if a < smallest:
                    smallest = a
                    chosen = i
        if chosen == -1 or (stop and smallest > limit):
            break
        alpha = smallest
        leaf.unsafe_set(chosen, 1)
        var stack = List[Int]()
        stack.append(tree.nodes[chosen].left_child)
        stack.append(tree.nodes[chosen].right_child)
        while len(stack) > 0:
            var i = stack.pop()
            active.unsafe_set(i, 0)
            if leaf.unsafe_get(i) == 0:
                stack.append(tree.nodes[i].left_child)
                stack.append(tree.nodes[i].right_child)
    var a = nm.zeros[DType.float64](nm.Shape(used))
    var r = nm.zeros[DType.float64](nm.Shape(used))
    for i in range(used):
        a.unsafe_set(i, alphas.unsafe_get(i))
        r.unsafe_set(i, impurities.unsafe_get(i))
    return PruningResult(leaf^, PruningPath(a^, r^))

def cost_complexity_pruning_path(tree: TreeData) raises -> PruningPath:
    var result = pruning_sequence(tree, 0, False)
    return PruningPath(result.path.ccp_alphas.copy(), result.path.impurities.copy())

def prune(tree: TreeData, ccp_alpha: Float64) raises -> TreeData:
    if ccp_alpha < 0 or ccp_alpha != ccp_alpha:
        raise Error("ValueError: ccp_alpha must be nonnegative")
    if ccp_alpha == 0:
        return copy_tree(tree)
    var result = pruning_sequence(tree, ccp_alpha, True)
    var leaves = result.leaves.copy()
    var output = new_tree(tree.n_features, tree.n_classes, tree.n_categories)
    var storage = Tree()
    var pending = List[Branch]()
    pending.append(Branch(0, 0, -2, False, 0, ParentInfo(0, 0, 0, 0)))
    var width = tree.n_outputs * tree.max_n_classes
    while len(pending) > 0:
        var t = pending.pop()
        var node = tree.nodes[t.start].copy()
        var left = node.left_child
        var right = node.right_child
        if leaves.unsafe_get(t.start) != 0:
            node.split_kind = -1
        var id = storage.add_node(output, t.parent, t.left, node)
        output.max_depth = max(output.max_depth, t.depth)
        for j in range(width):
            output.value.unsafe_set(id * width + j, tree.value.unsafe_get(t.start * width + j))
        if node.split_kind != -1:
            pending.append(Branch(right, 0, id, False, t.depth + 1, ParentInfo(0, 0, 0, 0)))
            pending.append(Branch(left, 0, id, True, t.depth + 1, ParentInfo(0, 0, 0, 0)))
    storage.resize(output, output.node_count)
    return output^
