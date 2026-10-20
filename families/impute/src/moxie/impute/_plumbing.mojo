"""Dense missingness and output-layout contracts; no imputation engine."""
import numojo as nm
from std.math import isfinite
from moxie.utils.validation import FittedState


struct MissingValue[dtype: DType](Copyable, Movable):
    """Default is NaN; an explicit numeric scalar selects equality masking."""
    var is_nan: Bool
    var value: SIMD[Self.dtype, 1]

    def __init__(out self):
        self.is_nan = True
        self.value = 0

    def __init__(out self, value: SIMD[Self.dtype, 1]) raises:
        self.is_nan = Bool(value != value)
        if not self.is_nan and not isfinite(value):
            raise Error("ValueError: missing_values must be finite or NaN")
        self.value = value


def validate_matrix[dtype: DType](x: nm.NDArray[dtype]) raises:
    if x.ndim != 2:
        raise Error("ValueError: expected a dense rank-2 matrix")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise Error("ValueError: input must have samples and features")


def missing_mask[dtype: DType](
    x: nm.NDArray[dtype], missing: MissingValue[dtype]
) raises -> nm.NDArray[DType.bool]:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    validate_matrix(x)
    var result = nm.zeros[DType.bool](nm.Shape(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            var nan = Bool(value != value)
            if (nan and not missing.is_nan) or (not nan and not isfinite(value)):
                raise Error("ValueError: non-missing values must be finite")
            var absent = nan if missing.is_nan else Bool(value == missing.value)
            result.unsafe_set(i * x.shape[1] + j, absent)
    return result^


def validate_indices(indices: List[Int], n_features: Int) raises:
    var previous = -1
    for index in indices:
        if index <= previous or index >= n_features:
            raise Error("ValueError: feature indices must be ascending, unique and in range")
        previous = index


def select_features[dtype: DType](
    x: nm.NDArray[dtype], indices: List[Int]
) raises -> nm.NDArray[dtype]:
    """Copy a sorted feature subset, including a zero-column result."""
    validate_matrix(x)
    validate_indices(indices, x.shape[1])
    var result = nm.zeros[dtype](nm.Shape(x.shape[0], len(indices)))
    for i in range(x.shape[0]):
        for j in range(len(indices)):
            result.unsafe_set(i * len(indices) + j, x.unsafe_get(
                i * Int(x.strides[0]) + indices[j] * Int(x.strides[1])))
    return result^


struct FeatureLayout(Copyable, Movable):
    """Fitted bookkeeping over ORIGINAL feature indices, independently owned."""
    var state: FittedState
    var retained_features: List[Int]
    var indicator_features: List[Int]
    var empty_features: List[Int]
    var error_on_new: Bool
    var all_features: Bool

    def __init__(out self):
        self.state = FittedState()
        self.retained_features = List[Int]()
        self.indicator_features = List[Int]()
        self.empty_features = List[Int]()
        self.error_on_new = True
        self.all_features = False

    def fit(
        mut self, mask: nm.NDArray[DType.bool], *,
        keep_empty_features: Bool = False, features: String = "missing-only",
        error_on_new: Bool = True,
    ) raises:
        validate_matrix(mask)
        if features != "all" and features != "missing-only":
            raise Error("ValueError: features must be all or missing-only")
        var next_layout = Self()
        next_layout.error_on_new = error_on_new
        next_layout.all_features = features == "all"
        for j in range(mask.shape[1]):
            var count = 0
            for i in range(mask.shape[0]):
                count += Int(mask.unsafe_get(i * Int(mask.strides[0]) + j * Int(mask.strides[1])))
            if count == mask.shape[0]:
                next_layout.empty_features.append(j)
            if keep_empty_features or count < mask.shape[0]:
                next_layout.retained_features.append(j)
            if next_layout.all_features or count > 0:
                next_layout.indicator_features.append(j)
        next_layout.state.record_fit(mask.shape[0], mask.shape[1])
        self = next_layout^

    def set_retained(mut self, indices: List[Int]) raises:
        """Simple's custom statistic may discard additional invalid columns."""
        if not self.state.is_fitted():
            raise Error("NotFittedError: fit layout before setting retained features")
        validate_indices(indices, self.state.n_features_in_.value())
        self.retained_features = indices.copy()

    def check_mask(self, mask: nm.NDArray[DType.bool]) raises:
        validate_matrix(mask)
        self.state.check_features(mask.shape[1])
        if self.error_on_new and not self.all_features:
            var next_feature = 0
            for j in range(mask.shape[1]):
                if next_feature < len(self.indicator_features):
                    if self.indicator_features[next_feature] == j:
                        next_feature += 1
                        continue
                for i in range(mask.shape[0]):
                    if mask.unsafe_get(i * Int(mask.strides[0]) + j * Int(mask.strides[1])):
                        raise Error("ValueError: new missing feature at transform")

    def indicator(self, mask: nm.NDArray[DType.bool]) raises -> nm.NDArray[DType.bool]:
        self.check_mask(mask)
        return select_features(mask, self.indicator_features)


def append_indicator[dtype: DType](
    filled: nm.NDArray[dtype], original_mask: nm.NDArray[DType.bool],
    layout: FeatureLayout,
) raises -> nm.NDArray[dtype]:
    """Filled already contains retained columns; append original mask columns."""
    comptime assert dtype == DType.float32 or dtype == DType.float64 or dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64
    layout.check_mask(original_mask)
    if filled.ndim != 2:
        raise Error("ValueError: expected a rank-2 filled matrix")
    if filled.shape[0] != original_mask.shape[0] or filled.shape[1] != len(layout.retained_features):
        raise Error("ValueError: filled matrix does not match retained layout")
    var width = filled.shape[1] + len(layout.indicator_features)
    var result = nm.zeros[dtype](nm.Shape(filled.shape[0], width))
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            result.unsafe_set(i * width + j, filled.unsafe_get(
                i * Int(filled.strides[0]) + j * Int(filled.strides[1])))
        for j in range(len(layout.indicator_features)):
            var absent = original_mask.unsafe_get(i * Int(original_mask.strides[0]) +
                layout.indicator_features[j] * Int(original_mask.strides[1]))
            result.unsafe_set(i * width + filled.shape[1] + j, SIMD[dtype, 1](Int(absent)))
    return result^


struct ImputationStep[Model: Copyable & Deinitable](Copyable, Movable):
    """Owned fitted model for one iterative step, never an estimator clone."""
    var feat_idx: Int
    var neighbor_feat_idx: List[Int]
    var estimator: Self.Model

    def __init__(out self, feat_idx: Int, neighbor_feat_idx: List[Int],
                 var estimator: Self.Model, n_features: Int) raises:
        if feat_idx < 0 or feat_idx >= n_features:
            raise Error("ValueError: target feature out of range")
        # Neighbor ordering is significant and need not be sorted.
        for i in range(len(neighbor_feat_idx)):
            var index = neighbor_feat_idx[i]
            if index < 0 or index >= n_features or index == feat_idx:
                raise Error("ValueError: invalid predictor feature")
            for j in range(i):
                if neighbor_feat_idx[j] == index:
                    raise Error("ValueError: duplicate predictor feature")
        self.feat_idx = feat_idx
        self.neighbor_feat_idx = neighbor_feat_idx.copy()
        self.estimator = estimator^
