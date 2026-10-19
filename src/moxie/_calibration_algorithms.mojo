# SPDX-License-Identifier: BSD-3-Clause
"""Calibration-specific fitting on native Float64 NuMojo arrays."""
import numojo as nm
from .isotonic import IsotonicRegression
from std.math import exp, log, log1p, isfinite

comptime Array = nm.NDArray[DType.float64]


def vector(x: Array) raises:
    if x.ndim != 1 or x.size == 0:
        raise Error("ValueError: expected a nonempty one-dimensional array")
    for i in range(x.size):
        if not isfinite(x.item(i)):
            raise Error("ValueError: input must be finite")


def matrix(x: Array) raises:
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise Error("ValueError: expected a nonempty matrix")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.item(i, j)):
                raise Error("ValueError: input must be finite")


def weights(n: Int, sample_weight: Optional[Array]) raises -> Array:
    var w = nm.ones[DType.float64](nm.Shape(n))
    if sample_weight:
        vector(sample_weight.value())
        if sample_weight.value().size != n:
            raise Error("ValueError: inconsistent sample weights")
        w = sample_weight.value().copy()
    var total = Float64(0)
    for i in range(n):
        if w.item(i) < 0:
            raise Error("ValueError: negative sample weights are unsupported")
        total += w.item(i)
    if total <= 0 or not isfinite(total):
        raise Error("ValueError: sample weights must have positive finite sum")
    return w^


def order(x: Array) raises -> List[Int]:
    var indices = List[Int]()
    for i in range(x.size):
        indices.append(i)
    var scratch = indices.copy()
    var width = 1
    while width < x.size:
        var start = 0
        while start < x.size:
            var middle = min(start + width, x.size)
            var end = min(start + 2 * width, x.size)
            var left = start
            var right = middle
            for k in range(start, end):
                var take_left = right == end
                if left < middle and right < end:
                    take_left = x.item(indices[left]) <= x.item(indices[right])
                if take_left:
                    scratch[k] = indices[left]
                    left += 1
                else:
                    scratch[k] = indices[right]
                    right += 1
            start = end
        indices = scratch.copy()
        width *= 2
    return indices^


def expit(x: Float64) -> Float64:
    if x >= 0:
        return 1 / (1 + exp(-x))
    var e = exp(x)
    return e / (1 + e)


def sigmoid_loss(
    x: Array, targets: Array, w: Array, a: Float64, b: Float64
) raises -> Float64:
    var loss = Float64(0)
    for i in range(x.size):
        var z = a * x.item(i) + b
        var t = targets.item(i)
        loss += w.item(i) * (max(z, 0.0) - (1 - t) * z + log1p(exp(-abs(z))))
    return loss


struct _SigmoidCalibration(Copyable, Movable):
    var a_: Float64
    var b_: Float64
    var fitted: Bool

    def __init__(out self):
        self.a_ = 0
        self.b_ = 0
        self.fitted = False

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        vector(X)
        vector(y)
        if X.size != y.size:
            raise Error("ValueError: inconsistent sample count")
        var w = weights(X.size, sample_weight)
        var prior0 = Float64(0)
        var prior1 = Float64(0)
        var scale = Float64(0)
        for i in range(X.size):
            scale = max(scale, abs(X.item(i)))
            if y.item(i) > 0:
                prior1 += w.item(i)
            else:
                prior0 += w.item(i)
        if scale < 30:
            scale = 1
        var x = nm.zeros[DType.float64](nm.Shape(X.size))
        var targets = nm.zeros[DType.float64](nm.Shape(X.size))
        for i in range(X.size):
            x[nm.Item(i)] = X.item(i) / scale
            targets[nm.Item(i)] = (prior1 + 1) / (prior1 + 2) if y.item(
                i
            ) > 0 else 1 / (prior0 + 2)
        var a = Float64(0)
        var b = log((prior0 + 1) / (prior1 + 1))
        var loss = sigmoid_loss(x, targets, w, a, b)
        var converged = False
        for _ in range(200):
            var h00 = Float64(1e-12)
            var h01 = Float64(0)
            var h11 = Float64(1e-12)
            var g0 = Float64(0)
            var g1 = Float64(0)
            for i in range(x.size):
                var f = x.item(i)
                var p = expit(-(a * f + b))
                var q = expit(a * f + b)
                var curvature = w.item(i) * p * q
                var gradient = w.item(i) * (targets.item(i) - p)
                h00 += curvature * f * f
                h01 += curvature * f
                h11 += curvature
                g0 += gradient * f
                g1 += gradient
            if max(abs(g0), abs(g1)) < 1e-8:
                converged = True
                break
            var det = h00 * h11 - h01 * h01
            if det <= 0 or not isfinite(det):
                raise Error(
                    "RuntimeError: sigmoid Hessian is numerically singular"
                )
            var da = -(h11 * g0 - h01 * g1) / det
            var db = -(-h01 * g0 + h00 * g1) / det
            var descent = g0 * da + g1 * db
            var step = Float64(1)
            var accepted = False
            for _ in range(50):
                var candidate = sigmoid_loss(
                    x, targets, w, a + step * da, b + step * db
                )
                if candidate <= loss + 1e-4 * step * descent:
                    a += step * da
                    b += step * db
                    if abs(loss - candidate) <= 1e-14 * max(1.0, abs(loss)):
                        converged = True
                    loss = candidate
                    accepted = True
                    break
                step *= 0.5
            if converged:
                break
            if not accepted:
                raise Error("RuntimeError: sigmoid line search failed")
        if not converged or not isfinite(a) or not isfinite(b):
            raise Error("RuntimeError: sigmoid fitting did not converge")
        self.a_ = a / scale
        self.b_ = b
        self.fitted = True

    def predict(self, X: Array) raises -> Array:
        if not self.fitted:
            raise Error("NotFittedError: sigmoid calibration is unfitted")
        vector(X)
        var out = nm.zeros[DType.float64](nm.Shape(X.size))
        for i in range(X.size):
            out[nm.Item(i)] = expit(-(self.a_ * X.item(i) + self.b_))
        return out^


def _list(x: Array) raises -> List[Float64]:
    var values = List[Float64]()
    for i in range(x.size):
        values.append(x.item(i))
    return values^


def _array(x: List[Float64]) raises -> Array:
    var values = nm.zeros[DType.float64](nm.Shape(len(x)))
    for i in range(len(x)):
        values[nm.Item(i)] = x[i]
    return values^


struct _IsotonicCalibration(Copyable, Movable):
    var X_thresholds_: Array
    var y_thresholds_: Array
    var _model: IsotonicRegression[DType.float64]

    def __init__(out self) raises:
        self.X_thresholds_ = nm.zeros[DType.float64](nm.Shape(0))
        self.y_thresholds_ = nm.zeros[DType.float64](nm.Shape(0))
        self._model = IsotonicRegression[DType.float64](out_of_bounds="clip")

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        vector(X)
        vector(y)
        if X.size != y.size:
            raise Error("ValueError: inconsistent sample count")
        var w = weights(X.size, sample_weight)
        var model = IsotonicRegression[DType.float64](out_of_bounds="clip")
        _ = model.fit(_list(X), _list(y), sample_weight=_list(w))
        var xs = _array(model.X_thresholds_)
        var ys = _array(model.y_thresholds_)
        for i in range(ys.size):
            if not isfinite(ys.item(i)):
                raise Error("RuntimeError: isotonic weighted sums overflowed")
        self.X_thresholds_ = xs^
        self.y_thresholds_ = ys^
        self._model = model^

    def predict(self, X: Array) raises -> Array:
        vector(X)
        return _array(self._model.predict(_list(X)))


def logits(X: Array) raises -> Array:
    if X.ndim == 1:
        vector(X)
    else:
        matrix(X)
    var n = X.shape[0]
    var binary = X.ndim == 1 or X.shape[1] == 1
    var c = 2 if binary else X.shape[1]
    var out = nm.zeros[DType.float64](nm.Shape(n, c))
    var probabilities = not binary
    if not binary:
        for i in range(n):
            var total = Float64(0)
            for j in range(c):
                var value = X.item(i, j)
                total += value
                probabilities = probabilities and value >= 0 and value <= 1
            probabilities = probabilities and abs(total - 1) <= 1.001e-5
    for i in range(n):
        for j in range(c):
            if binary:
                var value = X.item(i) if X.ndim == 1 else X.item(i, 0)
                out[nm.Item(i, j)] = -value if j == 0 else value
            else:
                var value = X.item(i, j)
                out[nm.Item(i, j)] = log(
                    value + 1e-12
                ) if probabilities else value
    return out^


def temperature_loss(
    x: Array, y: Array, w: Array, log_beta: Float64
) raises -> Float64:
    var beta = exp(log_beta)
    var result = Float64(0)
    for i in range(x.shape[0]):
        var peak = x.item(i, 0)
        for j in range(1, x.shape[1]):
            peak = max(peak, x.item(i, j))
        var total = Float64(0)
        for j in range(x.shape[1]):
            total += exp(beta * (x.item(i, j) - peak))
        result += w.item(i) * (
            log(total) + beta * (peak - x.item(i, Int(y.item(i))))
        )
    return result


struct _TemperatureScaling(Copyable, Movable):
    var beta_: Float64
    var n_classes_: Int

    def __init__(out self):
        self.beta_ = 1
        self.n_classes_ = 0

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        var x = logits(X)
        vector(y)
        if x.shape[0] != y.size:
            raise Error("ValueError: inconsistent sample count")
        for i in range(y.size):
            var label = y.item(i)
            if (
                label < 0
                or label >= Float64(x.shape[1])
                or label != Float64(Int(label))
            ):
                raise Error(
                    "ValueError: temperature targets must be encoded class"
                    " indices"
                )
        var w = weights(y.size, sample_weight)
        var lo = Float64(-10)
        var hi = Float64(10)
        var ratio = Float64(0.6180339887498949)
        var left = hi - ratio * (hi - lo)
        var right = lo + ratio * (hi - lo)
        var fl = temperature_loss(x, y, w, left)
        var fr = temperature_loss(x, y, w, right)
        for _ in range(100):
            if fl < fr:
                hi = right
                right = left
                fr = fl
                left = hi - ratio * (hi - lo)
                fl = temperature_loss(x, y, w, left)
            else:
                lo = left
                left = right
                fl = fr
                right = lo + ratio * (hi - lo)
                fr = temperature_loss(x, y, w, right)
            if hi - lo < 1e-10:
                break
        if not isfinite(min(fl, fr)):
            raise Error("RuntimeError: temperature objective overflowed")
        self.beta_ = exp((lo + hi) / 2)
        self.n_classes_ = x.shape[1]

    def predict(self, X: Array) raises -> Array:
        if self.n_classes_ == 0:
            raise Error("NotFittedError: temperature calibration is unfitted")
        var x = logits(X)
        if x.shape[1] != self.n_classes_:
            raise Error("ValueError: inconsistent class count")
        var out = nm.zeros[DType.float64](nm.Shape(x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            var peak = x.item(i, 0)
            for j in range(1, x.shape[1]):
                peak = max(peak, x.item(i, j))
            var total = Float64(0)
            for j in range(x.shape[1]):
                var value = exp(self.beta_ * (x.item(i, j) - peak))
                out[nm.Item(i, j)] = value
                total += value
            for j in range(x.shape[1]):
                out[nm.Item(i, j)] = out.item(i, j) / (total)
        return out^
