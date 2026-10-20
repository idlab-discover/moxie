# Contract adaptation from scikit-learn; see LICENSES/scikit-learn.txt.
"""Family-owned dense layouts; no learning engines."""
from numojo import NDArray
from moxie.tree._records import Features, Values, Indices, Mask, CategoryBits

comptime Bins = NDArray[DType.uint8]
comptime Gradients = NDArray[DType.float32]
comptime Counts = NDArray[DType.uint32]

@fieldwise_init
struct SamplingPlan(Copyable, Movable):
    var n_samples: Int
    var n_features: Int
    var max_samples: Int
    var max_features: Int
    var bootstrap_samples: Bool
    var bootstrap_features: Bool

@fieldwise_init
struct SampleSelection(Movable):
    var rows: Indices
    var features: Indices
    var multiplicities: Indices

@fieldwise_init
struct ClassAlignment(Movable):
    var columns: Indices
    var n_classes: Int

@fieldwise_init
struct OOBAccumulator(Movable):
    var sums: Values
    var counts: Indices

@fieldwise_init
struct BinMetadata(Movable):
    var thresholds: List[Values]
    var n_bins_non_missing: Indices
    var is_categorical: Mask
    var known_categories: List[Values]
    var missing_bin: UInt8

@fieldwise_init
struct Histogram(Movable):
    var gradients: Values
    var hessians: Values
    var counts: Counts

@fieldwise_init
struct HistogramSplit(Copyable, Movable):
    var feature: Int
    var bin_threshold: UInt8
    var missing_left: Bool
    var gain: Float64
    var is_categorical: Bool
    var left_categories: CategoryBits
    var n_left: Int
    var n_right: Int
    var gradient_left: Float64
    var gradient_right: Float64
    var hessian_left: Float64
    var hessian_right: Float64

@fieldwise_init
struct HistogramNode(Copyable, Movable):
    var value: Float64
    var count: UInt32
    var feature_idx: Int
    var num_threshold: Float64
    var missing_go_to_left: UInt8
    var left: UInt32
    var right: UInt32
    var gain: Float64
    var depth: UInt32
    var is_leaf: UInt8
    var bin_threshold: UInt8
    var is_categorical: UInt8
    var bitset_idx: UInt32

@fieldwise_init
struct HistogramPredictor(Movable):
    var nodes: List[HistogramNode]
    var raw_left_categories: List[CategoryBits]
    var binned_left_categories: List[CategoryBits]

@fieldwise_init
struct GrowerConfig(Copyable, Movable):
    var max_leaf_nodes: Int
    var max_depth: Int
    var min_samples_leaf: Int
    var l2_regularization: Float64
    var min_hessian_to_split: Float64
    var min_gain_to_split: Float64
    var shrinkage: Float64
    var feature_fraction: Float64

@fieldwise_init
struct NamedEstimator[E: Movable & Deinitable](Movable):
    """E may be a child-owned closed native variant adapter; None means drop."""
    var name: String
    var estimator: Optional[Self.E]
