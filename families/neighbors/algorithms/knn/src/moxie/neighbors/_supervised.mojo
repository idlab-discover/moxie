"""Shared neighbor target aggregation; numerical payloads use NuMojo storage."""
import numojo as nm
from std.math import isfinite, floor
from moxie.neighbors._distance import at
from moxie.neighbors._query_engine import NeighborWeights, BuiltinWeights
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._unsupervised import NearestNeighbors


def target_at[dtype: DType](y: nm.NDArray[dtype], i: Int, j: Int) raises -> Float64:
    if y.ndim == 1: return Float64(y.unsafe_get(i * Int(y.strides[0])))
    return at(y, i, j)


def target_snapshot[dtype: DType](y: nm.NDArray[dtype], n: Int, classifier: Bool) raises -> nm.NDArray[dtype]:
    if y.ndim != 1 and y.ndim != 2: raise Error("ValueError: y must have one or two dimensions")
    if y.shape[0] != n or n == 0: raise Error("ValueError: target row mismatch")
    var outputs = 1 if y.ndim == 1 else y.shape[1]
    if outputs == 0: raise Error("ValueError: empty target")
    for i in range(n):
        for j in range(outputs):
            var v = target_at(y, i, j)
            if not isfinite(v): raise Error("ValueError: nonfinite target")
            if classifier and floor(v) != v: raise Error("ValueError: continuous classification target")
    return y.copy()


def class_arrays[dtype: DType](y: nm.NDArray[dtype]) raises -> List[nm.NDArray[dtype]]:
    var result = List[nm.NDArray[dtype]]()
    var outputs = 1 if y.ndim == 1 else y.shape[1]
    for col in range(outputs):
        var tmp = nm.zeros[dtype](nm.Shape(y.shape[0]))
        var count = 0
        for row in range(y.shape[0]):
            var v = target_at(y, row, col)
            var pos = 0
            while pos < count and Float64(tmp.unsafe_get(pos)) < v: pos += 1
            if pos < count and Float64(tmp.unsafe_get(pos)) == v: continue
            for k in range(count, pos, -1): tmp.unsafe_set(k, tmp.unsafe_get(k-1))
            tmp.unsafe_set(pos, v.cast[dtype]())
            count += 1
        var labels = nm.zeros[dtype](nm.Shape(count))
        for k in range(count): labels.unsafe_set(k, tmp.unsafe_get(k))
        result.append(labels^)
    return result^

@fieldwise_init
struct WeightedNeighbors(Movable):
    var offsets: List[Int]
    var indices: List[Int]
    var weights: nm.NDArray[DType.float64]


def weighted_neighbors[dtype: DType, M: DenseMetric & Deinitable, W: NeighborWeights](model: NearestNeighbors[dtype, M],
        X: Optional[nm.NDArray[dtype]], radius_mode: Bool, weighting: W) raises -> WeightedNeighbors:
    var offsets = List[Int]()
    var indices: List[Int]
    var distances = nm.zeros[DType.float64](nm.Shape(0))
    if radius_mode:
        var r = model.radius_neighbors(X)
        offsets = r.offsets.copy()
        indices = r.indices.copy()
        distances = nm.zeros[DType.float64](nm.Shape(len(indices)))
        for j in range(len(indices)): distances.unsafe_set(j, Float64(r.distances.value().unsafe_get(j)))
    else:
        var k = model.kneighbors(X)
        for i in range(k.n_queries + 1): offsets.append(i*k.k)
        indices = k.indices.copy()
        distances = nm.zeros[DType.float64](nm.Shape(len(indices)))
        for j in range(len(indices)): distances.unsafe_set(j, Float64(k.distances.value().unsafe_get(j)))
    var weights = weighting.apply(distances, offsets)
    if weights.ndim != 1 or weights.shape[0] != len(indices): raise Error("ValueError: callable weight shape mismatch")
    # Deliberate native adaptation: finite nonnegative voting/interpolation weights.
    for j in range(len(indices)):
        var w = weights.unsafe_get(j * Int(weights.strides[0]))
        if not isfinite(w) or w < 0: raise Error("ValueError: callable weights must be finite and nonnegative")
    return WeightedNeighbors(offsets^, indices^, weights^)


def regression[dtype: DType](y: nm.NDArray[dtype], rows: WeightedNeighbors) raises -> nm.NDArray[dtype]:
    var n = len(rows.offsets)-1
    var cols = 1 if y.ndim == 1 else y.shape[1]
    var output = nm.zeros[dtype](nm.Shape(n) if y.ndim == 1 else nm.Shape(n, cols))
    for i in range(n):
        var denominator = Float64(0)
        for k in range(rows.offsets[i], rows.offsets[i+1]):
            denominator += rows.weights.unsafe_get(k * Int(rows.weights.strides[0]))
        for col in range(cols):
            var total = Float64(0)
            for k in range(rows.offsets[i], rows.offsets[i+1]):
                total += rows.weights.unsafe_get(k * Int(rows.weights.strides[0])) * target_at(y, rows.indices[k], col)
            output.unsafe_set(i*cols+col, (total/denominator if denominator != 0 else Float64("nan")).cast[dtype]())
    return output^


def probabilities[dtype: DType](y: nm.NDArray[dtype], classes: List[nm.NDArray[dtype]],
        rows: WeightedNeighbors, outliers: Optional[nm.NDArray[dtype]] = None) raises -> List[nm.NDArray[DType.float64]]:
    var result = List[nm.NDArray[DType.float64]]()
    for col in range(len(classes)):
        var nc = classes[col].shape[0]
        var output = nm.zeros[DType.float64](nm.Shape(len(rows.offsets)-1, nc))
        for i in range(len(rows.offsets)-1):
            var denom = Float64(0)
            for k in range(rows.offsets[i], rows.offsets[i+1]):
                var w = rows.weights.unsafe_get(k * Int(rows.weights.strides[0]))
                var label = target_at(y, rows.indices[k], col)
                for c in range(nc):
                    if label == Float64(classes[col].unsafe_get(c)):
                        output.unsafe_set(i*nc+c, output.unsafe_get(i*nc+c)+w)
                denom += w
            if rows.offsets[i] == rows.offsets[i+1]:
                if not outliers: raise Error("ValueError: no neighbors found; specify outlier_label")
                for c in range(nc):
                    if outliers.value().unsafe_get(col) == classes[col].unsafe_get(c): output.unsafe_set(i*nc+c, 1)
            elif denom == 0: raise Error("ValueError: all neighbors have zero weights")
            else:
                for c in range(nc): output.unsafe_set(i*nc+c, output.unsafe_get(i*nc+c)/denom)
        result.append(output^)
    return result^


def classified[dtype: DType](classes: List[nm.NDArray[dtype]], probs: List[nm.NDArray[DType.float64]],
        outputs_2d: Bool, outliers: Optional[nm.NDArray[dtype]] = None) raises -> nm.NDArray[dtype]:
    var n = probs[0].shape[0]
    var cols = len(classes)
    var output = nm.zeros[dtype](nm.Shape(n, cols) if outputs_2d else nm.Shape(n))
    for col in range(cols):
        for i in range(n):
            var best = 0
            var total = Float64(0)
            for c in range(classes[col].shape[0]):
                var p = at(probs[col], i, c)
                total += p
                if p > at(probs[col], i, best): best = c
            var label = classes[col].unsafe_get(best)
            if total == 0 and outliers: label = outliers.value().unsafe_get(col)
            output.unsafe_set(i*cols+col, label)
    return output^


def prediction_score[dtype: DType](pred: nm.NDArray[dtype], y: nm.NDArray[dtype], classifier: Bool,
        sample_weight: Optional[nm.NDArray[dtype]] = None) raises -> Float64:
    var checked = target_snapshot(y, pred.shape[0], classifier)
    var cols = 1 if y.ndim == 1 else y.shape[1]
    var pc = 1 if pred.ndim == 1 else pred.shape[1]
    if pc != cols: raise Error("ValueError: output count mismatch")
    for i in range(pred.shape[0]):
        for j in range(pc):
            if not isfinite(target_at(pred,i,j)): raise Error("ValueError: nonfinite prediction in score")
    var n = y.shape[0]
    if sample_weight:
        if sample_weight.value().ndim != 1 or sample_weight.value().shape[0] != n: raise Error("ValueError: invalid sample_weight shape")
    var weights = nm.ones[DType.float64](nm.Shape(n))
    var sumw = Float64(0)
    for i in range(n):
        var w = Float64(1)
        if sample_weight: w = Float64(sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0])))
        if not isfinite(w) or w < 0: raise Error("ValueError: invalid sample weight")
        weights.unsafe_set(i,w)
        sumw += w
    if sumw == 0: raise Error("ValueError: weights sum to zero")
    if classifier:
        if cols > 1:
            for which in range(2):
                var first = target_at(y,0,0) if which == 0 else target_at(pred,0,0)
                var second = first
                for i in range(n):
                    for j in range(cols):
                        var value = target_at(y,i,j) if which == 0 else target_at(pred,i,j)
                        if value != first:
                            if second == first: second = value
                            elif value != second: raise Error("ValueError: multiclass-multioutput scoring is not supported by sklearn")
        var correct = Float64(0)
        for i in range(n):
            var matches = True
            for j in range(cols):
                if target_at(pred,i,j) != target_at(y,i,j): matches = False
            if matches: correct += weights.unsafe_get(i)
        return correct/sumw
    if n < 2: return Float64("nan")
    var score = Float64(0)
    for j in range(cols):
        var mean = Float64(0)
        for i in range(n): mean += weights.unsafe_get(i)*target_at(y,i,j)/sumw
        var residual = Float64(0)
        var variance = Float64(0)
        for i in range(n):
            var delta = target_at(pred,i,j)-target_at(y,i,j)
            var centered = target_at(y,i,j)-mean
            residual += weights.unsafe_get(i)*delta*delta
            variance += weights.unsafe_get(i)*centered*centered
        score += Float64(1) if residual == 0 else (Float64(0) if variance == 0 else 1-residual/variance)
    return score/Float64(cols)
