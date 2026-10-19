# Interface field/signature adaptation from scikit-learn (BSD-3-Clause).
# Copyright the scikit-learn developers; see LICENSES/scikit-learn.txt.
"""Shared tree records. Algorithms live in their designated child owners."""
from numojo import NDArray, Shape
from std.utils import Variant
from moxie._categorical import CategoricalColumn, StringCategories

comptime TREE_LEAF = -1
comptime TREE_UNDEFINED = -2
comptime SPLIT_LEAF = -1
comptime SPLIT_NUMERIC = 0
comptime SPLIT_CATEGORICAL_BITSET = 1
comptime SPLIT_CATEGORICAL_HASH = 2
comptime CategoryBits = SIMD[DType.uint32, 8]
comptime Features = NDArray[DType.float32]
comptime Values = NDArray[DType.float64]
comptime Indices = NDArray[DType.int64]
comptime Mask = NDArray[DType.uint8]
comptime Constraints = NDArray[DType.int8]

@fieldwise_init
struct Node(Copyable, Movable):
    var left_child: Int
    var right_child: Int
    var feature: Int
    var threshold: Float64
    var left_cat_bitset: CategoryBits
    var impurity: Float64
    var n_node_samples: Int
    var weighted_n_node_samples: Float64
    var missing_go_to_left: UInt8
    var split_kind: Int8

@fieldwise_init
struct ParentInfo(Copyable, Movable):
    var lower_bound: Float64
    var upper_bound: Float64
    var impurity: Float64
    var n_constant_features: Int

@fieldwise_init
struct SplitRecord(Copyable, Movable):
    var feature: Int
    var pos: Int
    var threshold: Float64
    var left_cat_bitset: CategoryBits
    var split_kind: Int8
    var improvement: Float64
    var impurity_left: Float64
    var impurity_right: Float64
    var lower_bound: Float64
    var upper_bound: Float64
    var missing_go_to_left: UInt8

@fieldwise_init
struct TreeData(Movable):
    """One owned representation; storage engine owns allocation and traversal."""
    var n_features: Int
    var n_classes: List[Int]
    var n_categories: List[Int]
    var n_outputs: Int
    var max_n_classes: Int
    var max_depth: Int
    var node_count: Int
    var capacity: Int
    var nodes: List[Node]
    var value: Values

@fieldwise_init
struct TrainingData(Movable):
    """Explicit owned handles; callers borrow this record throughout a build."""
    var X: Features
    var y: Values
    var sample_weight: Optional[Values]
    var missing_values_in_feature_mask: Mask
    var n_categories: Indices
    var monotonic_cst: Constraints

@fieldwise_init
struct SplitWorkspace(Movable):
    var samples: Indices
    var features: Indices
    var constant_features: Indices
    var feature_values: Features
    var start: Int
    var end: Int
    var n_missing: Int

@fieldwise_init
struct CriterionConfig(Movable):
    var n_outputs: Int
    var n_classes: List[Int]
    var n_samples: Int

@fieldwise_init
struct SplitterConfig(Copyable, Movable):
    var max_features: Int
    var min_samples_leaf: Int
    var min_weight_leaf: Float64

@fieldwise_init
struct BuilderConfig(Copyable, Movable):
    var min_samples_split: Int
    var min_samples_leaf: Int
    var min_weight_leaf: Float64
    var max_depth: Int
    var min_impurity_decrease: Float64
    var max_leaf_nodes: Int

@fieldwise_init
struct ChildImpurities(Copyable, Movable):
    var left: Float64
    var right: Float64

@fieldwise_init
struct FeatureRange(Copyable, Movable):
    var minimum: Float32
    var maximum: Float32

@fieldwise_init
struct SplitPosition(Copyable, Movable):
    var previous: Int
    var current: Int

@fieldwise_init
struct PruningPath(Movable):
    var ccp_alphas: Values
    var impurities: Values


def validate_range(start: Int, pos: Int, end: Int, size: Int) raises:
    if start < 0 or start > pos or pos > end or end > size:
        raise Error("ValueError: expected 0 <= start <= pos <= end <= size")


def validate_parent(parent: ParentInfo, n_features: Int) raises:
    if not parent.lower_bound <= parent.upper_bound:
        raise Error("ValueError: invalid monotonic bounds")
    if parent.n_constant_features < 0 or parent.n_constant_features > n_features:
        raise Error("ValueError: invalid constant feature count")


def value_offset(node: Int, output: Int, category: Int,
                 capacity: Int, n_outputs: Int, max_n_classes: Int) raises -> Int:
    if capacity < 0 or n_outputs <= 0 or max_n_classes <= 0:
        raise Error("ValueError: invalid value dimensions")
    # Overflow is checked before performing any multiplication.
    comptime LIMIT = 9223372036854775807
    if n_outputs > LIMIT // max_n_classes:
        raise Error("ValueError: value stride overflow")
    var stride = n_outputs * max_n_classes
    if capacity > LIMIT // stride:
        raise Error("ValueError: value capacity overflow")
    if node < 0 or node >= capacity or output < 0 or output >= n_outputs or category < 0 or category >= max_n_classes:
        raise Error("ValueError: value index out of bounds")
    return node * stride + output * max_n_classes + category


def snapshot_values(values: Values) raises -> Values:
    """Independent fitted snapshot, never an implicit shared view."""
    return values.copy()


def validate_split(split: SplitRecord, start: Int, end: Int, n_features: Int) raises:
    if start < 0 or end < start or n_features <= 0:
        raise Error("ValueError: invalid split dimensions")
    if not split.lower_bound <= split.upper_bound or split.missing_go_to_left > 1:
        raise Error("ValueError: invalid split bounds or missing direction")
    if split.split_kind == SPLIT_LEAF:
        if split.pos < end:
            raise Error("ValueError: leaf split position must be at or beyond end")
        return
    if split.split_kind < SPLIT_NUMERIC or split.split_kind > SPLIT_CATEGORICAL_HASH:
        raise Error("ValueError: unknown split kind")
    if split.pos <= start or split.pos >= end or split.feature < 0 or split.feature >= n_features:
        raise Error("ValueError: invalid nonempty split or feature")
    if split.split_kind == SPLIT_NUMERIC and split.threshold != split.threshold:
        raise Error("ValueError: numeric threshold cannot be NaN")


def validate_training_layout(data: TrainingData) raises:
    """Shape/contiguity only; finite values and encoded labels are engine checks."""
    if data.X.ndim != 2 or data.y.ndim != 2:
        raise Error("ValueError: X and y must be rank two")
    var n = data.X.shape[0]
    var p = data.X.shape[1]
    if n <= 0 or p <= 0 or data.y.shape[0] != n or data.y.shape[1] <= 0:
        raise Error("ValueError: invalid training dimensions")
    if data.y.strides[1] != 1:
        raise Error("ValueError: y output axis must be contiguous")
    if data.missing_values_in_feature_mask.ndim != 1 or data.n_categories.ndim != 1 or data.monotonic_cst.ndim != 1:
        raise Error("ValueError: feature metadata must be vectors")
    if data.missing_values_in_feature_mask.size != p or data.n_categories.size != p or data.monotonic_cst.size != p:
        raise Error("ValueError: feature metadata length mismatch")
    if data.sample_weight:
        if data.sample_weight.value().ndim != 1 or data.sample_weight.value().size != n:
            raise Error("ValueError: sample_weight must match rows")


comptime StringColumn = CategoricalColumn[StringCategories]
comptime TreeColumn = Variant[Features, StringColumn]

struct TreeTable(Copyable, Movable):
    """Owned ordered Float32/string columns for dense tree estimators.

    Numeric columns are rank-one arrays; string columns use shared categorical
    storage. Constructors snapshot columns, including their dictionaries.
    """
    var _columns: List[TreeColumn]
    var shape: Shape

    var feature_names: Optional[List[String]]
    var categorical_mask: List[Bool]

    def __init__(out self, columns: List[TreeColumn], *, feature_names: Optional[List[String]] = None, categorical_mask: Optional[List[Bool]] = None) raises:
        if len(columns) == 0:
            raise Error("ValueError: TreeTable requires at least one column")
        var rows = -1
        for column in columns:
            var count: Int
            if column.isa[Features]():
                if column[Features].ndim != 1:
                    raise Error("ValueError: numeric table columns must be rank one")
                count = column[Features].size
            else:
                count = column[StringColumn].rows()
            if rows >= 0 and count != rows:
                raise Error("ValueError: table columns must have equal lengths")
            rows = count
        if feature_names and len(feature_names.value()) != len(columns):
            raise Error("ValueError: feature names must match table columns")
        if categorical_mask and len(categorical_mask.value()) != len(columns):
            raise Error("ValueError: categorical dtype mask must match table columns")
        self.feature_names = feature_names.copy()
        self.categorical_mask = List[Bool]()
        for j in range(len(columns)):
            self.categorical_mask.append(categorical_mask.value()[j] if categorical_mask else columns[j].isa[StringColumn]())
        self._columns = columns.copy()
        self.shape = Shape(rows, len(columns))
