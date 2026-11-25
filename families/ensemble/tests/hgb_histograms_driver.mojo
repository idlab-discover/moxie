from std.memory import bitcast
from std.sys import argv

from moxie.ensemble._hist_gradient_boosting.histogram import _Histograms, compute_histograms
from numojo import NDArray, Shape


def emit(histograms: _Histograms, n_features: Int) raises:
    for feature in range(n_features):
        for bin in histograms.feature(feature):
            print(
                bin.count,
                bitcast[DType.int64](bin.sum_gradients),
                bitcast[DType.int64](bin.sum_hessians),
            )


def run_case(fields: List[Int]) raises:
    var n_rows = fields[0]
    var n_features = fields[1]
    var n_bins = fields[2]
    var n_selected = fields[3]
    var n_sibling = fields[4]
    var n_parent = fields[5]
    var constant_hessian = fields[6] == 1
    var offset = 7
    var selected = List[Int](capacity=n_selected)
    for _ in range(n_selected):
        selected.append(fields[offset])
        offset += 1
    var sibling_rows = List[Int](capacity=n_sibling)
    for _ in range(n_sibling):
        sibling_rows.append(fields[offset])
        offset += 1
    var parent_rows = List[Int](capacity=n_parent)
    for _ in range(n_parent):
        parent_rows.append(fields[offset])
        offset += 1
    var X = NDArray[DType.uint8](Shape(n_rows, n_features), order="F")
    for i in range(X.size):
        X.unsafe_set(i, UInt8(fields[offset]))
        offset += 1
    var gradients = List[Float32](capacity=n_rows)
    var hessians = List[Float32](capacity=0 if constant_hessian else n_rows)
    for _ in range(n_rows):
        gradients.append(bitcast[DType.float32](Int32(fields[offset])))
        offset += 1
    if not constant_hessian:
        for _ in range(n_rows):
            hessians.append(bitcast[DType.float32](Int32(fields[offset])))
            offset += 1

    var parent = compute_histograms(
        X, n_bins, Span(parent_rows), Span(gradients), Span(hessians)
    )
    var sibling = compute_histograms(
        X, n_bins, Span(sibling_rows), Span(gradients), Span(hessians),
        allowed_features=selected^,
    )
    emit(parent, n_features)
    emit(sibling, n_features)
    var other = parent^.subtract_sibling(sibling)
    emit(other, n_features)


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = List[Int]()
        for field in line.split(" "):
            fields.append(atol(String(field)))
        run_case(fields)
