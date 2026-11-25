from std.memory import bitcast
from std.sys import argv

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.grower import grow_tree
from moxie.ensemble._hist_gradient_boosting.losses import _RegressionLoss
from moxie.ensemble._hist_gradient_boosting.predictor import make_raw_tree
from moxie.ensemble._hist_gradient_boosting.splitting import _SplitOptions
from numojo import NDArray, Shape


comptime Vector = NDArray[DType.float64]


def read_values[origin: ImmOrigin](
    fields: List[StringSpan[origin]], mut offset: Int, mut values: Vector
) raises:
    for i in range(values.size):
        values.unsafe_set(i, bitcast[DType.float64](Int64(atol(String(fields[offset])))))
        offset += 1


def run_case[origin: ImmOrigin](fields: List[StringSpan[origin]]) raises:
    var loss = _RegressionLoss(
        String(fields[0]), bitcast[DType.float64](Int64(atol(String(fields[1]))))
    )
    var n_rows = atol(String(fields[2]))
    var n_features = atol(String(fields[3]))
    var n_query = atol(String(fields[4]))
    var n_stages = atol(String(fields[5]))
    var weighted = atol(String(fields[6])) != 0
    var offset = 7
    var x = Vector(Shape(n_rows, n_features))
    var query = Vector(Shape(n_query, n_features))
    var y = Vector(Shape(n_rows))
    var supplied_raw = Vector(Shape(n_rows))
    read_values(fields, offset, x)
    read_values(fields, offset, query)
    read_values(fields, offset, y)
    read_values(fields, offset, supplied_raw)
    var weights: Optional[Vector] = None
    if weighted:
        var w = Vector(Shape(n_rows))
        read_values(fields, offset, w)
        weights = w^

    var baseline = loss.fit_intercept(y, weights)
    var gradients = List[Float32]()
    var hessians = List[Float32]()
    loss.gradient_hessian(y, supplied_raw, weights, gradients, hessians)
    var score = loss.mean_loss(y, supplied_raw, weights)
    print(bitcast[DType.int64](baseline), bitcast[DType.int64](score))
    for i in range(n_rows):
        var hessian = hessians[i] if hessians else Float32(1)
        print(bitcast[DType.int32](gradients[i]), bitcast[DType.int32](hessian))

    if n_stages == 0:
        return
    comptime n_bins = 17
    comptime shrinkage = 0.125
    var mapper = _BinMapper(n_bins=n_bins, random_state=0)
    _ = mapper.fit(x, sample_weight=weights)
    var binned = mapper.transform(x)
    var monotonic = List[Int](length=n_features, fill=0)
    var raw = Vector(Shape(n_rows))
    var query_raw = Vector(Shape(n_query))
    for i in range(n_rows):
        raw.unsafe_set(i, baseline)
    for i in range(n_query):
        query_raw.unsafe_set(i, baseline)
    for _ in range(n_stages):
        loss.gradient_hessian(y, raw, weights, gradients, hessians)
        var grown = grow_tree(
            binned,
            n_bins,
            Span(gradients),
            Span(hessians),
            mapper.n_bins_non_missing_,
            mapper.is_categorical_,
            monotonic,
            max_leaf_nodes=5,
            max_depth=3,
            options=_SplitOptions(min_samples_leaf=3, l2_regularization=0.25),
            shrinkage=shrinkage,
        )
        loss.update_leaves(grown, y, raw, weights)
        for leaf in grown.leaves:
            ref node = grown.tree.nodes[leaf]
            for index in range(node.start, node.stop):
                var row = grown.partition[index]
                raw.unsafe_set(row, raw.unsafe_get(row) + node.value)
        var prediction = make_raw_tree(grown.tree, mapper).predict(query)
        print(bitcast[DType.int64](loss.mean_loss(y, raw, weights)))
        for i in range(n_query):
            var updated = query_raw.unsafe_get(i) + prediction.unsafe_get(i)
            query_raw.unsafe_set(i, updated)
            print(bitcast[DType.int64](updated), bitcast[DType.int64](loss.inverse_link(updated)))


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() != 0:
            run_case(line.split(" "))
