"""Dense univariate statistics. Formulas follow pinned sklearn (BSD-3-Clause)."""
import numojo as nm
from std.math import sqrt, isnan, isinf, isfinite
from moxie._labels import StringTarget, encode_column
from moxie._dense import sum_axis2d, row_squared_norms
from moxie.feature_selection._contracts import ScoreResult
from moxie.feature_selection._selection import SelectionState
from moxie.feature_selection._univariate_probability import f_survival, chi_survival

def _validate_x[dt: DType](x: nm.NDArray[dt]) raises:
    comptime assert dt == DType.float32 or dt == DType.float64
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise Error("ValueError: expected nonempty rank-2 X")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))):
                raise Error("ValueError: X contains NaN or infinity")

def _validate_y[dt: DType](y: nm.NDArray[dt], n: Int, multilabel: Bool = False) raises:
    comptime assert dt == DType.float32 or dt == DType.float64 or dt == DType.int64
    if y.ndim != 1 and not (y.ndim == 2 and (multilabel or y.shape[1] == 1)):
        raise Error("ValueError: expected rank-1 y or supported indicator matrix")
    if y.shape[0] != n:
        raise Error("ValueError: inconsistent sample counts")
    var columns = y.shape[1] if y.ndim == 2 else 1
    if columns == 0:
        raise Error("ValueError: empty target")
    for i in range(n):
        for j in range(columns):
            var offset = i * Int(y.strides[0])
            if y.ndim == 2:
                offset += j * Int(y.strides[1])
            if not isfinite(y.unsafe_get(offset)):
                raise Error("ValueError: y contains NaN or infinity")

def _score_matrix[dt: DType](X: nm.NDArray[dt]) raises -> nm.NDArray[DType.float64]:
    # The adopted astype reads contiguous memory; gather views before casting.
    if X.is_c_contiguous():
        return X.astype[DType.float64]()
    var selection = SelectionState()
    var mask = List[Bool]()
    for _ in range(X.shape[1]):
        mask.append(True)
    selection.commit(mask)
    return selection.transform(X).astype[DType.float64]()

def r_regression[dt: DType, target_dt: DType = dt](X: nm.NDArray[dt], y: nm.NDArray[target_dt], *, center: Bool = True, force_finite: Bool = True) raises -> nm.NDArray[DType.float64]:
    _validate_x(X)
    _validate_y(y, X.shape[0])
    var x64 = _score_matrix(X)
    var target = nm.zeros[DType.float64](nm.Shape(1, X.shape[0]))
    for i in range(X.shape[0]):
        target.unsafe_set(i, Float64(y.unsafe_get(i * Int(y.strides[0]))))
    if center:
        var average = nm.sum(target) / Float64(X.shape[0])
        for i in range(X.shape[0]):
            target.unsafe_set(i, target.unsafe_get(i) - average)
    var transposed = x64.T()
    var squared_norms = row_squared_norms(transposed)
    var sums = sum_axis2d(x64, 0)
    var target_norm = row_squared_norms(target)
    var products = target @ x64
    var result = nm.zeros[DType.float64](nm.Shape(X.shape[1]))
    for j in range(X.shape[1]):
        var norm = squared_norms.unsafe_get(j)
        if center:
            var mean = sums.unsafe_get(j) / Float64(X.shape[0])
            norm -= Float64(X.shape[0]) * mean * mean
        var value = products.unsafe_get(j) / sqrt(norm) / sqrt(target_norm.unsafe_get(0))
        if force_finite and isnan(value):
            value = 0
        result.unsafe_set(j, value)
    return result^

def f_regression[dt: DType, target_dt: DType = dt](X: nm.NDArray[dt], y: nm.NDArray[target_dt], *, center: Bool = True, force_finite: Bool = True) raises -> ScoreResult[DType.float64]:
    var scores = r_regression(X, y, center=center, force_finite=force_finite)
    var pvalues = nm.zeros[DType.float64](nm.Shape(X.shape[1]))
    var degrees = Float64(X.shape[0] - (2 if center else 1))
    for j in range(X.shape[1]):
        var r2 = scores.unsafe_get(j) * scores.unsafe_get(j)
        var f = r2 / (1 - r2) * degrees
        var p = f_survival(f, 1, degrees)
        if force_finite:
            if isinf(f):
                f = 1.7976931348623157e308
            elif isnan(f):
                f = 0
                p = 1
        scores.unsafe_set(j, f)
        pvalues.unsafe_set(j, p)
    return ScoreResult[DType.float64](scores^, pvalues^)

def f_oneway[dt: DType](groups: List[nm.NDArray[dt]]) raises -> ScoreResult[DType.float64]:
    if len(groups) == 0:
        raise Error("ValueError: at least one group required")
    if groups[0].ndim != 1 and groups[0].ndim != 2:
        raise Error("ValueError: groups must have rank one or two")
    var p = groups[0].shape[1] if groups[0].ndim == 2 else 1
    var total = nm.zeros[dt](nm.Shape(p))
    var squares = nm.zeros[dt](nm.Shape(p))
    var between = nm.zeros[DType.float64](nm.Shape(p))
    var n = 0
    for k in range(len(groups)):
        if groups[k].ndim != 1 and groups[k].ndim != 2:
            raise Error("ValueError: groups must have rank one or two")
        var columns = groups[k].shape[1] if groups[k].ndim == 2 else 1
        if columns != p:
            raise Error("ValueError: group feature counts differ")
        var samples = groups[k].shape[0]
        var matrix = nm.zeros[dt](nm.Shape(samples, p))
        for i in range(samples):
            for j in range(p):
                var offset = i * Int(groups[k].strides[0])
                if groups[k].ndim == 2:
                    offset += j * Int(groups[k].strides[1])
                matrix.unsafe_set(i * p + j, groups[k].unsafe_get(offset))
        var group_sum = sum_axis2d(matrix, 0)
        var squared = matrix * matrix
        var group_squares = sum_axis2d(squared, 0)
        n += samples
        for j in range(p):
            var s = group_sum.unsafe_get(j)
            total.unsafe_set(j, total.unsafe_get(j) + s)
            squares.unsafe_set(j, squares.unsafe_get(j) + group_squares.unsafe_get(j))
            between.unsafe_set(j, between.unsafe_get(j) + Float64(s * s) / Float64(samples))
    var scores = nm.zeros[DType.float64](nm.Shape(p))
    var pvalues = nm.zeros[DType.float64](nm.Shape(p))
    for j in range(p):
        var correction = total.unsafe_get(j) * total.unsafe_get(j) / SIMD[dt, 1](n)
        var ss_total = Float64(squares.unsafe_get(j) - correction)
        var ss_between = between.unsafe_get(j) - Float64(correction)
        var f = (ss_between / Float64(len(groups) - 1)) / ((ss_total - ss_between) / Float64(n - len(groups)))
        scores.unsafe_set(j, f)
        pvalues.unsafe_set(j, f_survival(f, Float64(len(groups) - 1), Float64(n - len(groups))))
    return ScoreResult[DType.float64](scores^, pvalues^)

def _class_codes[dt: DType](y: nm.NDArray[dt], n: Int, classification: Bool = True) raises -> nm.NDArray[DType.float64]:
    _validate_y(y, n)
    var encoded = encode_column(y, classification=classification)
    # Only compact class IDs cross the floating-point boundary, never labels.
    return encoded.index.inverse.astype[DType.float64]()

def _class_codes(y: StringTarget, n: Int, classification: Bool = True) raises -> nm.NDArray[DType.float64]:
    if y.rows() != n or y.columns() != 1 or n == 0:
        raise Error("ValueError: expected one target per sample")
    var encoded = encode_column(y, classification=classification)
    return encoded.index.inverse.astype[DType.float64]()

def f_classif[dt: DType, target_dt: DType = dt](X: nm.NDArray[dt], y: nm.NDArray[target_dt]) raises -> ScoreResult[DType.float64]:
    _validate_x(X)
    _validate_y(y, X.shape[0])
    var encoded = encode_column(y, classification=False)
    var groups = List[nm.NDArray[dt]]()
    for k in range(encoded.classes.shape[0]):
        var count = Int(encoded.index.counts.unsafe_get(k))
        var group = nm.zeros[dt](nm.Shape(count, X.shape[1]))
        var row = 0
        for i in range(X.shape[0]):
            if encoded.index.inverse.unsafe_get(i) == Int64(k):
                for j in range(X.shape[1]):
                    group.unsafe_set(row * X.shape[1] + j, X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
                row += 1
        groups.append(group^)
    return f_oneway(groups)

def chi2[dt: DType, target_dt: DType = dt](X: nm.NDArray[dt], y: nm.NDArray[target_dt]) raises -> ScoreResult[DType.float64]:
    _validate_x(X)
    _validate_y(y, X.shape[0], True)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])) < 0:
                raise Error("ValueError: X must be non-negative")
    var multilabel = y.ndim == 2 and y.shape[1] > 1
    var encoded = encode_column(y)
    ref labels = encoded.classes
    var classes = y.shape[1] if multilabel else max(2, labels.shape[0])
    var indicators = nm.zeros[DType.float64](nm.Shape(X.shape[0], classes))
    for i in range(X.shape[0]):
        for k in range(classes):
            var member: Bool
            if multilabel:
                var value = y.unsafe_get(i * Int(y.strides[0]) + k * Int(y.strides[1]))
                if value != 0 and value != 1:
                    raise Error("ValueError: multilabel targets must be binary indicators")
                member = value == 1
            elif labels.shape[0] == 1:
                member = k == 0
            else:
                member = encoded.index.inverse.unsafe_get(i) == Int64(k)
            indicators.unsafe_set(i * classes + k, Float64(member))
    var x64 = _score_matrix(X)
    var observed = indicators.T() @ x64
    var feature_sum = sum_axis2d(x64, 0)
    var counts = sum_axis2d(indicators, 0)
    var residuals = nm.zeros[DType.float64](nm.Shape(classes, X.shape[1]))
    for k in range(classes):
        for j in range(X.shape[1]):
            var expected = counts.unsafe_get(k) / Float64(X.shape[0]) * feature_sum.unsafe_get(j)
            var difference = observed.unsafe_get(k * X.shape[1] + j) - expected
            residuals.unsafe_set(k * X.shape[1] + j, difference * difference / expected)
    var scores = sum_axis2d(residuals, 0)
    var pvalues = nm.zeros[DType.float64](nm.Shape(X.shape[1]))
    for j in range(X.shape[1]):
        pvalues.unsafe_set(j, chi_survival(scores.unsafe_get(j), Float64(classes - 1)))
    return ScoreResult[DType.float64](scores^, pvalues^)

def f_classif[dt: DType](X: nm.NDArray[dt], y: StringTarget) raises -> ScoreResult[DType.float64]:
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0], False)
    return f_classif(X, codes)

def chi2[dt: DType](X: nm.NDArray[dt], y: StringTarget) raises -> ScoreResult[DType.float64]:
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0])
    return chi2(X, codes)
