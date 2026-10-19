# scikit-learn developers, BSD-3-Clause; see family LICENSES/scikit-learn.txt.
"""Best and randomized numeric/categorical split search; exact feature draw protocol."""
import numojo as nm
from std.math import isfinite
from moxie._random import RandomState
from moxie.tree._criterion import WeightedCriterion
from moxie.tree._records import (TrainingData, SplitWorkspace, SplitRecord, ParentInfo,
    CriterionConfig, SplitterConfig, CategoryBits, Values, validate_training_layout)
from moxie.tree._utils import TreeRandom
from moxie.tree._partitioner import sort_feature, partition, swap, find_range, DensePartitioner
from moxie.tree._engine_api import Splitter

struct DenseSplitter[kind: Int](Splitter):
    var criterion: WeightedCriterion[Self.kind]
    var config: SplitterConfig
    var random: Bool
    var partitioner: DensePartitioner
    var rng: TreeRandom
    var total_weight: Float64

    def __init__(out self, criterion: CriterionConfig, config: SplitterConfig, random: Bool) raises:
        if config.max_features < 1 or config.min_samples_leaf < 1 or not isfinite(config.min_weight_leaf):
            raise Error("ValueError: invalid SplitterConfig")
        self.criterion = WeightedCriterion[Self.kind](criterion)
        self.config = config.copy()
        self.random = random
        self.rng = TreeRandom()
        self.partitioner = DensePartitioner()
        self.total_weight = 0

    def initialize(mut self, data: TrainingData, mut work: SplitWorkspace, mut rng: RandomState) raises:
        validate_training_layout(data)
        if work.samples.ndim != 1 or work.samples.size < data.X.shape[0] or work.samples.strides[0] != 1 or work.feature_values.ndim != 1 or work.feature_values.size < data.X.shape[0] or work.feature_values.strides[0] != 1:
            raise Error("ValueError: invalid sample workspace")
        if work.features.ndim != 1 or work.constant_features.ndim != 1 or work.features.size != data.X.shape[1] or work.constant_features.size != data.X.shape[1] or work.features.strides[0] != 1 or work.constant_features.strides[0] != 1:
            raise Error("ValueError: invalid feature workspace")
        self.rng.initialize(rng)
        self.total_weight = 0
        var n = 0
        for i in range(data.X.shape[0]):
            var w = Float64(1)
            if data.sample_weight:
                w = data.sample_weight.value().unsafe_get(i * Int(data.sample_weight.value().strides[0]))
            self.total_weight += w
            if w != 0:
                work.samples.unsafe_set(n, Int64(i))
                n += 1
        work.start = 0
        work.end = n
        for j in range(data.X.shape[1]):
            work.features.unsafe_set(j, Int64(j))

    def node_reset(mut self, data: TrainingData, mut work: SplitWorkspace, start: Int, end: Int) raises -> Float64:
        work.start = start
        work.end = end
        self.criterion.initialize(data, work.samples, start, end, self.total_weight)
        return self.criterion.weighted_n_node_samples

    def node_value(self, mut dest: Values) raises:
        self.criterion.node_value(dest)
    def clip_node_value(self, mut dest: Values, lower: Float64, upper: Float64) raises:
        self.criterion.clip_node_value(dest, lower, upper)
    def node_impurity(self) raises -> Float64:
        return self.criterion.node_impurity()

    def _valid(self) raises -> Bool:
        if self.criterion.weighted_n_left <= 0 or self.criterion.weighted_n_right <= 0:
            return False
        if self.criterion.weighted_n_left < self.config.min_weight_leaf or self.criterion.weighted_n_right < self.config.min_weight_leaf:
            return False
        # The pinned splitter checks child totals, not per-class signed counts.
        # Preserve that observable behavior, including signed class proportions.
        return True

    def node_split(mut self, data: TrainingData, mut work: SplitWorkspace, mut parent: ParentInfo) raises -> SplitRecord:
        var best = SplitRecord(-2, work.end, -2, CategoryBits(0), -1, 0, 0, 0, parent.lower_bound, parent.upper_bound, 0)
        var best_proxy = -Float64("inf")
        var remaining = work.features.size
        var found = 0
        var drawn = 0
        var known = parent.n_constant_features
        var total = known
        var visited = 0
        var last_feature = -1
        while remaining > total and (visited < self.config.max_features or visited <= found + drawn):
            visited += 1
            var j = self.rng.randint(drawn, remaining - found)
            if j < known:
                swap(work.features, drawn, j)
                drawn += 1
                continue
            j += found
            var f = Int(work.features.unsafe_get(j))
            last_feature = f
            var categorical = data.n_categories.unsafe_get(f * Int(data.n_categories.strides[0])) > 0
            var bits = CategoryBits(0)
            var low = Float32(0)
            var high = Float32(0)
            var constant: Bool
            if self.random:
                var limits = self.partitioner.find_min_max(data, work, f)
                low = limits.minimum
                high = limits.maximum
                constant = work.n_missing == work.end - work.start or (work.n_missing == 0 and high <= low + Float32(1e-7))
            else:
                constant = self.partitioner.sort_samples_and_feature_values(data, work, f)
            if constant:
                swap(work.features, j, total)
                found += 1
                total += 1
                continue
            remaining -= 1
            swap(work.features, remaining, j)
            var end_finite = work.end - work.n_missing
            var searches = 2 if work.n_missing > 0 and not self.random else 1
            var saved = work.samples.copy()
            var threshold = Float64(0)
            var random_left = False
            if self.random:
                if work.n_missing > 0:
                    random_left = self.rng.randint(0, 2) != 0
                if not categorical:
                    threshold = self.rng.uniform(Float64(low), Float64(high))
                if threshold == Float64(high):
                    threshold = Float64(low)
            for search in range(searches):
                var missing_left = random_left if self.random else search == 1
                if search == 1:
                    for q in range(work.n_missing):
                        work.samples.unsafe_set(work.start + q, saved.unsafe_get(end_finite + q))
                    for q in range(end_finite - work.start):
                        work.samples.unsafe_set(work.start + work.n_missing + q, saved.unsafe_get(work.start + q))
                if not self.random:
                    self.criterion.reset()
                var p = work.start
                while p < end_finite:
                    if self.random:
                        var candidate = SplitRecord(f, work.end, threshold, CategoryBits(0), 0, 0, 0, 0, parent.lower_bound, parent.upper_bound, UInt8(missing_left))
                        if categorical:
                            candidate.split_kind = 2
                            for _ in range(100):
                                bits[0] = UInt32(self.rng.randint(1, 2147483647))
                                candidate.left_cat_bitset = bits
                                p = self.partitioner.partition_samples(data, work, candidate)
                                if p > work.start and p < work.end:
                                    break
                        else:
                            p = self.partitioner.partition_samples(data, work, candidate)
                        # As in the reference, the criterion views the partitioned order.
                        self.criterion.reset()
                    else:
                        while p + 1 < end_finite and (work.feature_values.unsafe_get(p + 1) == work.feature_values.unsafe_get(p) if categorical else work.feature_values.unsafe_get(p + 1) <= work.feature_values.unsafe_get(p) + Float32(1e-7)):
                            p += 1
                        var previous = p
                        p += 1
                        if p == end_finite:
                            threshold = Float64("inf")
                        else:
                            threshold = Float64(work.feature_values.unsafe_get(previous)) / 2 + Float64(work.feature_values.unsafe_get(p)) / 2
                            if threshold == Float64(work.feature_values.unsafe_get(p)):
                                threshold = Float64(work.feature_values.unsafe_get(previous))
                    var position = p + (work.n_missing if missing_left and not self.random else 0)
                    if position - work.start >= self.config.min_samples_leaf and work.end - position >= self.config.min_samples_leaf:
                        self.criterion.update(position)
                        if self._valid():
                            var sign = data.monotonic_cst.unsafe_get(f * Int(data.monotonic_cst.strides[0]))
                            var valid = True
                            if sign != 0:
                                valid = self.criterion.check_monotonicity(sign, parent.lower_bound, parent.upper_bound)
                            if valid:
                                var proxy = self.criterion.proxy_impurity_improvement()
                                if proxy > best_proxy:
                                    best_proxy = proxy
                                    var direction = missing_left
                                    if work.n_missing == 0:
                                        direction = position - work.start > work.end - position
                                    best = SplitRecord(f, position, threshold, bits, Int8(2 if self.random else 1) if categorical else Int8(0), 0, 0, 0, parent.lower_bound, parent.upper_bound, UInt8(direction))
                                    if categorical and not self.random:
                                        self.partitioner.cat_position_to_split_bitset(work, position, missing_left, best)
                    if self.random:
                        break
        if best.pos < work.end:
            if not self.random or best.feature != last_feature:
                best.pos = self.partitioner.partition_samples(data, work, best)
            self.criterion.reset()
            self.criterion.update(best.pos)
            var children = self.criterion.children_impurity()
            best.impurity_left = children.left
            best.impurity_right = children.right
            best.improvement = self.criterion.impurity_improvement(parent.impurity, children.left, children.right)
            if data.monotonic_cst.unsafe_get(best.feature * Int(data.monotonic_cst.strides[0])) != 0:
                var middle = self.criterion.middle_value()
                best.lower_bound = middle
                best.upper_bound = middle
        for q in range(known):
            work.features.unsafe_set(q, work.constant_features.unsafe_get(q))
        for q in range(known, total):
            work.constant_features.unsafe_set(q, work.features.unsafe_get(q))
        parent.n_constant_features = total
        return best^
