# scikit-learn developers, BSD-3-Clause; see families/ensemble/LICENSES/scikit-learn.txt.
"""Accumulate per-node statistics over dense byte bins."""

from numojo import NDArray


@fieldwise_init
struct _HistogramBin(ImplicitlyCopyable, Movable):
    var sum_gradients: Float64
    var sum_hessians: Float64
    var count: Int


struct _Histograms(Movable):
    """One row of bins per available feature, indexed by original feature ID."""

    var _columns: List[List[_HistogramBin]]
    var _n_bins: Int
    var _constant_hessian: Bool

    def __init__(
        out self, n_features: Int, n_bins: Int, constant_hessian: Bool
    ):
        self._columns = List[List[_HistogramBin]](capacity=n_features)
        for _ in range(n_features):
            self._columns.append(List[_HistogramBin]())
        self._n_bins = n_bins
        self._constant_hessian = constant_hessian

    def feature[origin: ImmOrigin](
        ref[origin] self, index: Int
    ) raises -> ref[self._columns[0]] List[_HistogramBin]:
        """Borrow a feature's bins, or an empty list if it was excluded."""
        if index < 0 or index >= len(self._columns):
            raise Error("ValueError: histogram feature index is out of range")
        return self._columns[index]

    def subtract_sibling(var self, sibling: Self) raises -> Self:
        """Consume the parent and reuse its bin storage for the other child."""
        if (
            len(self._columns) != len(sibling._columns)
            or self._n_bins != sibling._n_bins
            or self._constant_hessian != sibling._constant_hessian
        ):
            raise Error("ValueError: incompatible histogram layouts")
        for feature in range(len(self._columns)):
            if not sibling._columns[feature]:
                self._columns[feature] = List[_HistogramBin]()
                continue
            if not self._columns[feature]:
                raise Error("ValueError: sibling feature is absent from parent")
            for code in range(self._n_bins):
                ref parent_bin = self._columns[feature][code]
                var sibling_bin = sibling._columns[feature][code]
                if sibling_bin.count > parent_bin.count:
                    raise Error("ValueError: sibling bin count exceeds parent")
                parent_bin.count -= sibling_bin.count
                parent_bin.sum_gradients -= sibling_bin.sum_gradients
                parent_bin.sum_hessians -= sibling_bin.sum_hessians
        return self^


def compute_histograms[
    rows_origin: ImmOrigin, gradients_origin: ImmOrigin, hessians_origin: ImmOrigin, //
](
    X_binned: NDArray[DType.uint8],
    n_bins: Int,
    sample_indices: Span[Int, rows_origin],
    gradients: Span[Float32, gradients_origin],
    hessians: Span[Float32, hessians_origin],
    *,
    allowed_features: Optional[List[Int]] = None,
) raises -> _Histograms:
    """Build selected feature histograms in sample partition order.

    An empty hessians span selects the constant-Hessian path. Its bin sums
    remain zero, and the splitter can derive each Hessian from the bin count
    and the loss's constant. Each input row index addresses the full arrays.
    """
    if X_binned.ndim != 2 or X_binned.shape[1] == 0:
        raise Error("ValueError: X_binned must be a rank-2 matrix with features")
    if n_bins < 1 or n_bins > 256:
        raise Error("ValueError: n_bins must be in [1, 256]")
    var n_rows = X_binned.shape[0]
    var n_features = X_binned.shape[1]
    var constant_hessian = len(hessians) == 0
    if len(gradients) != n_rows or (
        not constant_hessian and len(hessians) != n_rows
    ):
        raise Error("ValueError: gradient and Hessian lengths must match X_binned")
    for row in sample_indices:
        if row < 0 or row >= n_rows:
            raise Error("ValueError: histogram sample index is out of range")

    var result = _Histograms(n_features, n_bins, constant_hessian)
    var empty_bin = _HistogramBin(0.0, 0.0, 0)
    if allowed_features:
        for feature in allowed_features.value():
            if feature < 0 or feature >= n_features:
                raise Error("ValueError: histogram feature index is out of range")
            if result._columns[feature]:
                raise Error("ValueError: duplicate histogram feature")
            result._columns[feature] = List[_HistogramBin](length=n_bins, fill=empty_bin)
    else:
        for feature in range(n_features):
            result._columns[feature] = List[_HistogramBin](length=n_bins, fill=empty_bin)

    var row_stride = X_binned.strides[0]
    var feature_stride = X_binned.strides[1]
    for feature in range(n_features):
        ref bins = result._columns[feature]
        if not bins:
            continue
        var offset = feature * feature_stride
        for row in sample_indices:
            var code = Int(X_binned.unsafe_get(offset + row * row_stride))
            if code >= n_bins:
                raise Error("ValueError: binned value exceeds histogram bin range")
            ref bin = bins[code]
            bin.count += 1
            bin.sum_gradients += Float64(gradients[row])
            if not constant_hessian:
                bin.sum_hessians += Float64(hessians[row])
    return result^
