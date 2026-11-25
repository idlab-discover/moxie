# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Compile a fitted binned tree for traversal on raw feature values."""

from std.math import inf, isnan

from numojo import NDArray, Shape

from .binning import _BinMapper, _CategoryBits
from .grower import _BinnedTree
from .splitting import _contains_category


@fieldwise_init
struct _RawSplit(Copyable, Movable):
    var feature: Int
    var left: Int
    var right: Int
    var missing_go_to_left: Bool
    var threshold: Float64
    # For a categorical split, store the categories routed away from missing.
    var opposite_categories: Optional[_CategoryBits]

    def child(self, value: Float64) -> Int:
        var goes_left = self.missing_go_to_left
        if self.opposite_categories:
            if value >= 0 and value <= 254:
                var code = UInt8(value)
                if Float64(code) == value and _contains_category(
                    self.opposite_categories.value(), code
                ):
                    goes_left = not goes_left
        elif not isnan(value):
            goes_left = value <= self.threshold
        return self.left if goes_left else self.right


@fieldwise_init
struct _RawNode(Copyable, Movable):
    var value: Float64
    var split: Optional[_RawSplit]


@fieldwise_init
struct _RawTree(Copyable, Movable):
    var nodes: List[_RawNode]
    var n_features: Int

    def predict[dtype: DType](
        self, X: NDArray[dtype]
    ) raises -> NDArray[DType.float64]:
        """Predict from numerical features and encoded category IDs."""
        comptime assert dtype in (DType.float32, DType.float64), "prediction requires Float32 or Float64"
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise Error("ValueError: raw prediction feature count does not match tree")

        var result = NDArray[DType.float64](Shape(X.shape[0]))
        for row in range(X.shape[0]):
            var index = 0
            while self.nodes[index].split:
                ref split = self.nodes[index].split.value()
                var offset = row * X.strides[0] + split.feature * X.strides[1]
                index = split.child(Float64(X.unsafe_get(offset)))
            result.unsafe_set(row, self.nodes[index].value)
        return result^


def make_raw_tree(tree: _BinnedTree, mapper: _BinMapper) raises -> _RawTree:
    """Snapshot a tree and the split metadata from its original fitted mapper."""
    mapper._require_fitted()
    if (
        len(mapper.bin_thresholds_) != tree.n_features
        or mapper.missing_values_bin_idx_ != tree.missing_bin
    ):
        raise Error("ValueError: fitted mapper dimensions do not match tree bins")

    var nodes = List[_RawNode](capacity=len(tree.nodes))
    for node in tree.nodes:
        if not node.split:
            nodes.append(_RawNode(node.value, None))
            continue

        ref split = node.split.value()
        ref thresholds = mapper.bin_thresholds_[split.feature]
        var threshold = inf[DType.float64]()
        var opposite: Optional[_CategoryBits] = None
        if split.left_categories:
            if not mapper.is_categorical_[split.feature]:
                raise Error("ValueError: mapper marks a categorical split as numeric")
            var bits = _CategoryBits(fill=0)
            ref left_categories = split.left_categories.value()
            for i in range(len(thresholds)):
                if _contains_category(left_categories, UInt8(i)) != split.missing_go_to_left:
                    var code = Int(thresholds[i])
                    bits[code // 32] |= UInt32(1) << UInt32(code % 32)
            opposite = bits^
        else:
            var bin_index = Int(split.bin_idx)
            if mapper.is_categorical_[split.feature] or bin_index > len(thresholds):
                raise Error("ValueError: mapper bins do not match numeric split")
            # The final nonmissing bin separates all observed numbers from NaNs.
            if bin_index < len(thresholds):
                threshold = thresholds[bin_index]
        nodes.append(
            _RawNode(
                node.value,
                _RawSplit(
                    split.feature,
                    node.left,
                    node.right,
                    split.missing_go_to_left,
                    threshold,
                    opposite^,
                ),
            )
        )
    return _RawTree(nodes^, tree.n_features)
