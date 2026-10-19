# SPDX-License-Identifier: BSD-3-Clause
from std.math import ceil, cbrt
from std.utils import Variant
import numojo as nm
from ._calibration_algorithms import Array, vector, order


def _curve_float64(
    y_true: Array,
    y_prob: Array,
    *,
    pos_label: Optional[Float64] = None,
    n_bins: Variant[Int, String] = 5,
    strategy: String = "uniform",
) raises -> Tuple[Array, Array]:
    vector(y_true)
    vector(y_prob)
    if y_true.size != y_prob.size:
        raise Error("ValueError: inconsistent sample count")
    var positive = Float64(1)
    if pos_label:
        positive = pos_label.value()
    var labels = order(y_true)
    var count = 1
    for i in range(y_true.size):
        if i > 0 and y_true.item(labels[i]) != y_true.item(labels[i - 1]):
            count += 1
        var label = y_true.item(i)
        if not pos_label and label != -1 and label != 0 and label != 1:
            raise Error("ValueError: pos_label is required for these labels")
        if y_prob.item(i) < 0 or y_prob.item(i) > 1:
            raise Error("ValueError: y_prob must be in [0, 1]")
    if count > 2 or (
        not pos_label
        and y_true.item(labels[0]) == -1
        and y_true.item(labels[y_true.size - 1]) == 0
    ):
        raise Error(
            "ValueError: only binary classification is supported; specify"
            " pos_label"
        )
    var bins: Int
    if n_bins.isa[Int]():
        bins = n_bins[Int]
    else:
        if n_bins[String] != "cube_root":
            raise Error(
                "ValueError: n_bins must be a positive integer or cube_root"
            )
        bins = Int(ceil(cbrt(Float64(y_true.size))))
    if bins < 1:
        raise Error("ValueError: n_bins must be positive")
    if strategy != "uniform" and strategy != "quantile":
        raise Error("ValueError: strategy must be uniform or quantile")
    var indices = order(y_prob)
    var edges = nm.zeros[DType.float64](nm.Shape(bins + 1))
    for j in range(bins + 1):
        var fraction = Float64(j) / Float64(bins)
        if strategy == "uniform":
            edges[nm.Item(j)] = fraction
        else:
            var position = fraction * Float64(y_prob.size - 1)
            var lo = Int(position)
            var hi = min(lo + 1, y_prob.size - 1)
            var lower = y_prob.item(indices[lo])
            edges[nm.Item(j)] = lower + (position - Float64(lo)) * (
                y_prob.item(indices[hi]) - lower
            )
    var counts = nm.zeros[DType.float64](nm.Shape(bins))
    var sums = nm.zeros[DType.float64](nm.Shape(bins))
    var positives = nm.zeros[DType.float64](nm.Shape(bins))
    for i in range(y_true.size):
        var lo = 0
        var hi = bins - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if y_prob.item(i) <= edges.item(mid + 1):
                hi = mid
            else:
                lo = mid + 1
        counts[nm.Item(lo)] = counts.item(lo) + (1)
        sums[nm.Item(lo)] = sums.item(lo) + (y_prob.item(i))
        if y_true.item(i) == positive:
            positives[nm.Item(lo)] = positives.item(lo) + (1)
    var nonempty = 0
    for j in range(bins):
        if counts.item(j) > 0:
            nonempty += 1
    var prob_true = nm.zeros[DType.float64](nm.Shape(nonempty))
    var prob_pred = nm.zeros[DType.float64](nm.Shape(nonempty))
    var k = 0
    for j in range(bins):
        if counts.item(j) > 0:
            prob_true[nm.Item(k)] = positives.item(j) / counts.item(j)
            prob_pred[nm.Item(k)] = sums.item(j) / counts.item(j)
            k += 1
    return (prob_true^, prob_pred^)


def calibration_curve[
    Y: DType, P: DType
](
    y_true: nm.NDArray[Y],
    y_prob: nm.NDArray[P],
    *,
    pos_label: Optional[Float64] = None,
    n_bins: Variant[Int, String] = 5,
    strategy: String = "uniform",
) raises -> Tuple[Array, Array]:
    comptime assert Y == DType.float32 or Y == DType.float64
    comptime assert P == DType.float32 or P == DType.float64
    comptime if Y == DType.float64 and P == DType.float64:
        return _curve_float64(
            rebind[Array](y_true),
            rebind[Array](y_prob),
            pos_label=pos_label,
            n_bins=n_bins,
            strategy=strategy,
        )
    else:
        var y = y_true.astype[DType.float64]()
        var p = y_prob.astype[DType.float64]()
        return _curve_float64(
            y, p, pos_label=pos_label, n_bins=n_bins, strategy=strategy
        )


def calibration_curve[
    P: DType
](
    y_true: List[String],
    y_prob: nm.NDArray[P],
    *,
    pos_label: Optional[String] = None,
    n_bins: Variant[Int, String] = 5,
    strategy: String = "uniform",
) raises -> Tuple[Array, Array]:
    comptime assert P == DType.float32 or P == DType.float64
    if not pos_label:
        raise Error("ValueError: pos_label must be specified for string labels")
    var classes = List[String]()
    var labels = nm.zeros[DType.float64](nm.Shape(len(y_true)))
    for i in range(len(y_true)):
        var seen = False
        for label in classes:
            seen = seen or y_true[i] == label
        if not seen:
            classes.append(y_true[i].copy())
            if len(classes) > 2:
                raise Error(
                    "ValueError: only binary classification is supported"
                )
        labels[nm.Item(i)] = 1 if y_true[i] == pos_label.value() else 0
    var probabilities = y_prob.astype[DType.float64]()
    return _curve_float64(
        labels, probabilities, n_bins=n_bins, strategy=strategy
    )
