from std.memory import bitcast
from std.sys import argv

from moxie.ensemble._hist_gradient_boosting.histogram import (
    _HistogramBin,
    compute_histograms,
)
from moxie.ensemble._hist_gradient_boosting.splitting import (
    _SplitOptions,
    _ValueBounds,
    compute_node_value,
    find_node_split,
    partition_rows,
)
from numojo import NDArray, Shape


def as_float64(bits: Int) -> Float64:
    return bitcast[DType.float64](Int64(bits))


def run_case(fields: List[Int]) raises:
    var n_rows = fields[0]
    var n_features = fields[1]
    var n_bins = fields[2]
    var n_selected = fields[3]
    var n_node = fields[4]
    var constant = fields[5] == 1
    var options = _SplitOptions(
        l2_regularization=as_float64(fields[6]),
        min_hessian_to_split=as_float64(fields[7]),
        min_samples_leaf=fields[8],
        min_gain_to_split=as_float64(fields[9]),
    )
    var bounds = _ValueBounds(as_float64(fields[10]), as_float64(fields[11]))
    var total = _HistogramBin(
        as_float64(fields[12]), as_float64(fields[13]), n_node
    )

    var offset = 14
    var selected = List[Int](capacity=n_selected)
    for _ in range(n_selected):
        selected.append(fields[offset])
        offset += 1
    var rows = List[Int](capacity=n_node)
    for _ in range(n_node):
        rows.append(fields[offset])
        offset += 1

    var counts = List[UInt32](capacity=n_features)
    var missing = List[Bool](capacity=n_features)
    var categorical = List[Bool](capacity=n_features)
    var monotonic = List[Int](capacity=n_features)
    for _ in range(n_features):
        counts.append(UInt32(fields[offset]))
        missing.append(fields[offset + 1] == 1)
        categorical.append(fields[offset + 2] == 1)
        monotonic.append(fields[offset + 3])
        offset += 4

    var X = NDArray[DType.uint8](Shape(n_rows, n_features), order="F")
    for i in range(X.size):
        X.unsafe_set(i, UInt8(fields[offset]))
        offset += 1
    var gradients = List[Float32](capacity=n_rows)
    var hessians = List[Float32](capacity=0 if constant else n_rows)
    for _ in range(n_rows):
        gradients.append(bitcast[DType.float32](Int32(fields[offset])))
        offset += 1
    if not constant:
        for _ in range(n_rows):
            hessians.append(bitcast[DType.float32](Int32(fields[offset])))
            offset += 1

    var histograms = compute_histograms(
        X,
        n_bins,
        Span(rows),
        Span(gradients),
        Span(hessians),
        allowed_features=selected.copy(),
    )
    var value = compute_node_value(total, bounds, options.l2_regularization)
    print(bitcast[DType.int64](value))
    var result = find_node_split(
        histograms,
        total,
        value,
        counts,
        missing,
        categorical,
        monotonic,
        allowed_features=Span(selected),
        bounds=bounds,
        options=options,
    )
    if result:
        ref split = result.value()
        print(
            split.feature,
            split.bin_idx,
            Int(split.missing_go_to_left),
            split.left.count,
            split.right.count,
            Int(Bool(split.left_categories)),
            bitcast[DType.int64](split.gain),
            bitcast[DType.int64](split.left.sum_gradients),
            bitcast[DType.int64](split.left.sum_hessians),
            bitcast[DType.int64](split.right.sum_gradients),
            bitcast[DType.int64](split.right.sum_hessians),
            bitcast[DType.int64](split.value_left),
            bitcast[DType.int64](split.value_right),
        )
        if split.left_categories:
            ref bits = split.left_categories.value()
            for word in bits:
                print(word)
        var scratch = List[Int](length=n_node, fill=0)
        _ = partition_rows(
            X, split, UInt8(n_bins - 1), Span(rows), Span(scratch)
        )
        for row in rows:
            print(row)
    else:
        print("no split")


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = List[Int]()
        for field in line.split(" "):
            fields.append(atol(String(field)))
        run_case(fields)
