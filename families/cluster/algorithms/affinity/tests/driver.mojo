"""Native executable for differential validation; decimal CLI data only."""
import numojo as nm
from std.sys import argv
from moxie.cluster import AffinityPropagation, affinity_propagation
from moxie.cluster._affinity_propagation import Preference
from moxie._random import RandomState


def run[dtype: DType]() raises:
    var args = argv()
    var rows = Int(args[2])
    var cols = Int(args[3])
    var mode = args[4]
    var seed = Int(args[5])
    var damping = Float64(args[6])
    var iterations = Int(args[7])
    var window = Int(args[8])
    var pref_count = Int(args[9])
    var copy_input = Int(args[10]) != 0
    var layout = Int(args[11])
    var input = nm.zeros[dtype](nm.Shape(rows, cols))
    for i in range(rows * cols):
        input.unsafe_set(i, Float64(args[12 + i]).cast[dtype]())
    var pref = Preference()
    if pref_count == 1:
        pref = Preference(Float64(args[12 + rows * cols]))
    elif pref_count > 1:
        var values = nm.zeros[DType.float64](nm.Shape(pref_count))
        for i in range(pref_count):
            values.unsafe_set(i, Float64(args[12 + rows * cols + i]))
        pref = Preference(values)
    var x = input.view()
    if layout == 1:
        x = input.view_with_layout(nm.Shape(rows, cols), nm.NDArrayStrides(-cols, -1), rows * cols - 1)
    var model = AffinityPropagation[dtype](damping=damping, max_iter=iterations, convergence_iter=window, copy=copy_input, preference=pref^, affinity=mode, random_state=seed)
    model.fit_inplace(x)
    print("META", model.n_iter_, model.cluster_centers_indices_.size, model.converged_, model.equal_similarities_)
    for i in range(model.labels_.size): print("LABEL", model.labels_.item(i))
    for i in range(model.cluster_centers_indices_.size): print("CENTER", model.cluster_centers_indices_.item(i))
    for i in range(rows):
        for j in range(rows): print("S", model.affinity_matrix_.item(i, j))
    for i in range(rows):
        for j in range(cols): print("INPUT", x.item(i, j))
    if mode == "euclidean":
        var predicted = model.predict(x)
        for i in range(rows): print("PREDICT", predicted.item(i))
        for i in range(model.cluster_centers_.value().shape[0]):
            for j in range(cols): print("VALUE", model.cluster_centers_.value().item(i, j))


def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
