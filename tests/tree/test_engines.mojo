"""Compile-time engine extension points, with no numerical implementations."""
from moxie._random import RandomState
from moxie.tree._records import (
    Node, TreeData, TrainingData, SplitWorkspace, SplitRecord, ParentInfo, Values,
    Features, Indices, ChildImpurities, FeatureRange, SplitPosition, PruningPath,
)

from moxie.tree._engine_api import Criterion, Partitioner, Splitter, TreeStorage, TreeBuilder
from std.testing import assert_raises

struct CriterionProbe(Criterion):
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def initialize(mut self, data: TrainingData, samples: Indices,
                   start: Int, end: Int, weighted_n_samples: Float64) raises:
        raise Error("Scaffold probe: engine not implemented")
    def reset(mut self) raises:
        raise Error("Scaffold probe: engine not implemented")
    def reverse_reset(mut self) raises:
        raise Error("Scaffold probe: engine not implemented")
    def update(mut self, new_pos: Int) raises:
        raise Error("Scaffold probe: engine not implemented")
    def node_impurity(self) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")
    def children_impurity(self) raises -> ChildImpurities:
        raise Error("Scaffold probe: engine not implemented")
    def node_value(self, mut dest: Values) raises:
        raise Error("Scaffold probe: engine not implemented")
    def clip_node_value(self, mut dest: Values, lower: Float64, upper: Float64) raises:
        raise Error("Scaffold probe: engine not implemented")
    def middle_value(self) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")
    def impurity_improvement(self, parent: Float64, left: Float64, right: Float64) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")
    def proxy_impurity_improvement(self) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")
    def check_monotonicity(self, constraint: Int8, lower: Float64, upper: Float64) raises -> Bool:
        raise Error("Scaffold probe: engine not implemented")

struct PartitionerProbe(Partitioner):
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def init_node_split(mut self, mut work: SplitWorkspace, start: Int, end: Int) raises:
        raise Error("Scaffold probe: engine not implemented")
    def sort_samples_and_feature_values(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises -> Bool:
        raise Error("Scaffold probe: engine not implemented")
    def shift_missing_to_the_left(mut self, mut work: SplitWorkspace) raises:
        raise Error("Scaffold probe: engine not implemented")
    def find_min_max(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises -> FeatureRange:
        raise Error("Scaffold probe: engine not implemented")
    def next_p(mut self, work: SplitWorkspace, position: SplitPosition, missing_left: Bool) raises -> SplitPosition:
        raise Error("Scaffold probe: engine not implemented")
    def partition_samples(mut self, data: TrainingData, mut work: SplitWorkspace, split: SplitRecord) raises -> Int:
        raise Error("Scaffold probe: engine not implemented")
    def partition_samples_final(mut self, data: TrainingData, mut work: SplitWorkspace, split: SplitRecord) raises:
        raise Error("Scaffold probe: engine not implemented")
    def cat_position_to_split_bitset(self, work: SplitWorkspace, position: Int, missing_left: Bool, mut split: SplitRecord) raises:
        raise Error("Scaffold probe: engine not implemented")
    def sort_categories(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises:
        raise Error("Scaffold probe: engine not implemented")

struct SplitterProbe(Splitter):
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def initialize(mut self, data: TrainingData, mut work: SplitWorkspace, mut rng: RandomState) raises:
        raise Error("Scaffold probe: engine not implemented")
    def node_reset(mut self, data: TrainingData, mut work: SplitWorkspace, start: Int, end: Int) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")
    def node_split(mut self, data: TrainingData, mut work: SplitWorkspace, mut parent: ParentInfo) raises -> SplitRecord:
        raise Error("Scaffold probe: engine not implemented")
    def node_value(self, mut dest: Values) raises:
        raise Error("Scaffold probe: engine not implemented")
    def clip_node_value(self, mut dest: Values, lower: Float64, upper: Float64) raises:
        raise Error("Scaffold probe: engine not implemented")
    def node_impurity(self) raises -> Float64:
        raise Error("Scaffold probe: engine not implemented")

struct TreeStorageProbe(TreeStorage):
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def resize(mut self, mut tree: TreeData, capacity: Int) raises:
        raise Error("Scaffold probe: engine not implemented")
    def add_node(mut self, mut tree: TreeData, parent: Int, is_left: Bool,
                 node: Node) raises -> Int:
        raise Error("Scaffold probe: engine not implemented")
    def apply(self, tree: TreeData, X: Features) raises -> Indices:
        raise Error("Scaffold probe: engine not implemented")
    def predict(self, tree: TreeData, X: Features) raises -> Values:
        raise Error("Scaffold probe: engine not implemented")
    def compute_node_depths(self, tree: TreeData) raises -> Indices:
        raise Error("Scaffold probe: engine not implemented")
    def compute_feature_importances(self, tree: TreeData, normalize: Bool = True) raises -> Values:
        raise Error("Scaffold probe: engine not implemented")

struct TreeBuilderProbe(TreeBuilder):
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def build(mut self, mut tree: TreeData, data: TrainingData, mut rng: RandomState) raises:
        raise Error("Scaffold probe: engine not implemented")
    def prune(self, tree: TreeData, ccp_alpha: Float64) raises -> TreeData:
        raise Error("Scaffold probe: engine not implemented")
    def cost_complexity_pruning_path(self, tree: TreeData) raises -> PruningPath:
        raise Error("Scaffold probe: engine not implemented")

def main() raises:
    var probe_Criterion = CriterionProbe()
    var probe_Partitioner = PartitionerProbe()
    var probe_Splitter = SplitterProbe()
    var probe_TreeStorage = TreeStorageProbe()
    var probe_TreeBuilder = TreeBuilderProbe()
    with assert_raises():
        _ = probe_Criterion.node_impurity()
    print("Engine signature probes pass (raising test doubles only)")
