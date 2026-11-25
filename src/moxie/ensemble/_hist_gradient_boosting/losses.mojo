# scikit-learn developers, BSD-3-Clause. See families/ensemble/LICENSES/scikit-learn.txt.
# NumPy interpolation and reduction conventions. See families/ensemble/LICENSES/NumPy.txt.
"""Regression and classification loss operations for histogram gradient boosting."""

from std.builtin.sort import sort
from std.math import floor, isfinite, isnan, nan
from std.sys import llvm_intrinsic

from numojo import NDArray, Shape

from .._aggregation import _get
from .._sampling import _at, check_sample_weight, numpy_sum
from .binning import _Observation
from .grower import _GrownTree

comptime _Vector = NDArray[DType.float64]


def _exp(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


def _log(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


def _weighted_mean(values: _Vector, weights: Optional[_Vector]) raises -> Float64:
    if not weights:
        return numpy_sum(values) / Float64(values.size)
    ref weight = weights.value()
    var total = numpy_sum(weight)
    if total == 0:
        raise Error("ValueError: sample weights sum to zero")
    var products = _Vector(Shape(values.size))
    for i in range(values.size):
        products.unsafe_set(i, _at(values, i) * _at(weight, i))
    return numpy_sum(products) / total


def _ordered_location(
    mut values: List[_Observation], fraction: Float64, median: Bool, weighted: Bool
) -> Float64:
    if not values:
        return nan[DType.float64]()
    sort(Span(values))
    var size = len(values)
    if weighted:
        var total = Float64(0)
        for observation in values:
            total += observation.weight
        if total == 0:
            return nan[DType.float64]()
        var rank = fraction * total
        var cumulative = Float64(0)
        for observation in values:
            cumulative += observation.weight
            # An underflowed rank still skips leading zero weights.
            if cumulative >= rank and cumulative > 0:
                return observation.value
        return values[size - 1].value

    if median:
        return (values[(size - 1) // 2].value + values[size // 2].value) / 2.0 if (
            size % 2 == 0
        ) else values[size // 2].value

    var position = Float64(size - 1) * fraction
    var lower = Int(floor(position))
    var a = values[lower].value
    var b = values[min(lower + 1, size - 1)].value
    var blend = position - Float64(lower)
    var delta = b - a
    return a + delta * blend if blend < 0.5 else b - delta * (1.0 - blend)


struct _RegressionLoss(Copyable, Movable):
    var _name: String
    var _quantile: Float64

    def __init__(out self, loss: String, quantile: Optional[Float64] = None) raises:
        if loss == "quantile":
            var level = quantile.or_else(0.0)
            if not (level > 0 and level < 1):
                raise Error("ValueError: quantile must be strictly between 0 and 1")
        elif loss not in ("squared_error", "absolute_error", "poisson", "gamma"):
            raise Error("ValueError: unknown histogram boosting regression loss")
        self._name = loss
        self._quantile = quantile.or_else(0.5)

    def _uses_log_link(self) -> Bool:
        return self._name == "poisson" or self._name == "gamma"

    def needs_leaf_update(self) -> Bool:
        return self._name == "absolute_error" or self._name == "quantile"

    def _fraction(self) -> Float64:
        if self._name == "absolute_error":
            return 0.5
        # sklearn passes this round trip through a percentile-valued API.
        return (100.0 * self._quantile) / 100.0

    def _check_vectors(
        self, y: _Vector, raw: _Vector, weights: Optional[_Vector]
    ) raises:
        if y.ndim != 1 or raw.ndim != 1 or raw.size != y.size or y.size == 0:
            raise Error("ValueError: loss expects nonempty, matching target and raw vectors")
        if weights and (weights.value().ndim != 1 or weights.value().size != y.size):
            raise Error("ValueError: sample_weight must have one entry per target")

    def fit_intercept(self, y: _Vector, weights: Optional[_Vector] = None) raises -> Float64:
        """Validate the training target and return the initial raw prediction."""
        if y.ndim != 1 or y.size == 0:
            raise Error("ValueError: loss expects a nonempty target vector")
        if weights:
            check_sample_weight(weights.value(), y.size)
        var has_positive = False
        for i in range(y.size):
            var target = _at(y, i)
            if not isfinite(target):
                raise Error("ValueError: y must be finite")
            if self._name == "poisson" and target < 0:
                raise Error("ValueError: Poisson loss requires nonnegative targets")
            if self._name == "gamma" and target <= 0:
                raise Error("ValueError: Gamma loss requires strictly positive targets")
            has_positive = has_positive or target > 0
            if weights and _at(weights.value(), i) < 0:
                raise Error("ValueError: sample_weight must be nonnegative")
        if self._name == "poisson" and not has_positive:
            raise Error("ValueError: Poisson loss requires a positive target sum")

        if self.needs_leaf_update():
            var values = List[_Observation](capacity=y.size)
            for i in range(y.size):
                var weight = _at(weights.value(), i) if weights else 1.0
                values.append(_Observation(_at(y, i), weight))
            return _ordered_location(
                values, self._fraction(), self._name == "absolute_error", Bool(weights)
            )
        var mean = _weighted_mean(y, weights)
        if self._uses_log_link():
            # sklearn clips the prediction to ten Float64 epsilons.
            comptime lower_bound = Float64(10.0 * 2.220446049250313e-16)
            return _log(lower_bound if mean < lower_bound else mean)
        return mean

    def inverse_link(self, raw_prediction: Float64) -> Float64:
        return _exp(raw_prediction) if self._uses_log_link() else raw_prediction

    def gradient_hessian(
        self,
        y: _Vector,
        raw: _Vector,
        weights: Optional[_Vector],
        mut gradients: List[Float32],
        mut hessians: List[Float32],
    ) raises:
        """Refill reusable buffers. An empty Hessian buffer means unit Hessians."""
        self._check_vectors(y, raw, weights)
        var unit_hessian = not weights and not self._uses_log_link()
        gradients.clear()
        hessians.clear()
        gradients.reserve(y.size)
        if not unit_hessian:
            hessians.reserve(y.size)
        for i in range(y.size):
            var target = _at(y, i)
            var prediction = _at(raw, i)
            var gradient: Float64
            var hessian = Float64(1.0)
            if self._name == "poisson":
                hessian = _exp(prediction)
                gradient = hessian - target
            elif self._name == "gamma":
                hessian = target * _exp(-prediction)
                gradient = 1.0 - hessian
            elif self._name == "absolute_error":
                gradient = 1.0 if prediction > target else -1.0
            elif self._name == "quantile":
                gradient = -self._quantile if target >= prediction else 1.0 - self._quantile
            else:
                gradient = prediction - target
            if weights:
                var weight = _at(weights.value(), i)
                gradient *= weight
                hessian *= weight
            gradients.append(Float32(gradient))
            if not unit_hessian:
                hessians.append(Float32(hessian))

    def mean_loss(
        self, y: _Vector, raw: _Vector, weights: Optional[_Vector] = None
    ) raises -> Float64:
        """Evaluate the objective used by loss-based early stopping."""
        self._check_vectors(y, raw, weights)
        var values = _Vector(Shape(y.size))
        for i in range(y.size):
            var target = _at(y, i)
            var prediction = _at(raw, i)
            var difference = prediction - target
            var loss: Float64
            if self._name == "poisson":
                loss = _exp(prediction) - target * prediction
            elif self._name == "gamma":
                loss = prediction + target * _exp(-prediction)
            elif self._name == "absolute_error":
                loss = abs(difference)
            elif self._name == "quantile":
                loss = self._quantile * (target - prediction) if target >= prediction else (
                    (1.0 - self._quantile) * difference
                )
            else:
                loss = 0.5 * difference * difference
            values.unsafe_set(i, loss)
        return _weighted_mean(values, weights)

    def update_leaves(
        self,
        mut grown: _GrownTree,
        y: _Vector,
        raw: _Vector,
        weights: Optional[_Vector],
    ) raises:
        """Replace nonsmooth-loss Newton values with shrunk residual quantiles."""
        if not self.needs_leaf_update():
            return
        self._check_vectors(y, raw, weights)
        if len(grown.partition) != y.size:
            raise Error("ValueError: loss vectors must match the grown tree's training rows")
        var values = List[_Observation]()
        for leaf in grown.leaves:
            ref node = grown.tree.nodes[leaf]
            values.clear()
            values.reserve(node.stop - node.start)
            for position in range(node.start, node.stop):
                var row = grown.partition[position]
                var weight = _at(weights.value(), row) if weights else 1.0
                var residual = _at(y, row) - _at(raw, row)
                if isnan(residual):
                    if not weights:
                        values.clear()
                        break
                    continue
                values.append(_Observation(residual, weight))
            var update = _ordered_location(
                values, self._fraction(), self._name == "absolute_error", Bool(weights)
            )
            grown.tree.nodes[leaf].value = grown.shrinkage * update


def _log1pexp(x: Float64) -> Float64:
    if x <= -37:
        return _exp(x)
    if x <= -2:
        var tail = _exp(x)
        var shifted = 1 + tail
        # Compensate for rounding in 1 + tail before taking the log.
        return tail if shifted == 1 else _log(shifted) * (tail / (shifted - 1))
    if x <= 18:
        return _log(1 + _exp(x))
    if x <= 33.3:
        return x + _exp(-x)
    return x


struct _ClassificationLoss(Copyable, Movable):
    """Binary and multinomial log loss over ordered, encoded class indices."""

    var _n_classes: Int

    def __init__(out self, n_classes: Int) raises:
        if n_classes < 1:
            raise Error("ValueError: classification loss requires at least one class")
        self._n_classes = n_classes

    def raw_columns(self) -> Int:
        return 1 if self._n_classes <= 2 else self._n_classes

    def _check_raw(self, raw: _Vector) raises:
        if raw.ndim != 2 or raw.shape[1] != self.raw_columns():
            raise Error("ValueError: raw predictions must have one column per tree in a stage")

    def _check_targets(self, y: _Vector, weights: Optional[_Vector]) raises:
        if y.ndim != 1 or y.size == 0:
            raise Error("ValueError: expected a nonempty vector of encoded class indices")
        if weights:
            ref weight = weights.value()
            if weight.ndim != 1 or weight.size != y.size:
                raise Error("ValueError: sample_weight must have one entry per target")
        for i in range(y.size):
            var code = _at(y, i)
            if not (code >= 0 and code < Float64(self._n_classes) and code == floor(code)):
                raise Error("ValueError: invalid encoded class index")

    def _check_inputs(self, y: _Vector, raw: _Vector, weights: Optional[_Vector]) raises:
        self._check_raw(raw)
        self._check_targets(y, weights)
        if raw.shape[0] != y.size:
            raise Error("ValueError: raw predictions must have one row per target")

    def fit_intercept(self, y: _Vector, weights: Optional[_Vector] = None) raises -> _Vector:
        """Return one initial raw value per tree, in stage column order."""
        self._check_targets(y, weights)
        if weights:
            ref weight = weights.value()
            check_sample_weight(weight, y.size)
            for i in range(y.size):
                if _at(weight, i) < 0:
                    raise Error("ValueError: sample_weight must be nonnegative")

        comptime eps = Float64(2.220446049250313e-16)
        var baseline = _Vector(Shape(self.raw_columns()))
        if self.raw_columns() == 1:
            var prior = _weighted_mean(y, weights)
            if not isnan(prior):
                prior = prior.clamp(10 * eps, 1 - 10 * eps)
            baseline.unsafe_set(0, _log(prior / (1 - prior)))
            return baseline^

        var total = numpy_sum(weights.value()) if weights else Float64(y.size)
        if total == 0:
            raise Error("ValueError: sample weights sum to zero")
        var class_weights = _Vector(Shape(y.size))
        for k in range(self._n_classes):
            for i in range(y.size):
                var value = 0.0
                if _at(y, i) == Float64(k):
                    value = _at(weights.value(), i) if weights else 1.0
                class_weights.unsafe_set(i, value)
            var prior = numpy_sum(class_weights) / total
            baseline.unsafe_set(k, prior if isnan(prior) else prior.clamp(eps, 1 - eps))
        var logs = _Vector(Shape(self._n_classes))
        for k in range(self._n_classes):
            logs.unsafe_set(k, _log(baseline.unsafe_get(k)))
        var geometric_mean = _exp(numpy_sum(logs) / Float64(self._n_classes))
        for k in range(self._n_classes):
            baseline.unsafe_set(k, _log(baseline.unsafe_get(k) / geometric_mean))
        return baseline^

    def _shifted_exponentials(
        self, raw: _Vector, row: Int, mut exps: List[Float64]
    ) raises -> Tuple[Float64, Float64]:
        var peak = _get(raw, row, 0)
        for k in range(1, self._n_classes):
            var value = _get(raw, row, k)
            if peak < value:
                peak = value
        var total = Float64(0)
        for k in range(self._n_classes):
            exps[k] = _exp(_get(raw, row, k) - peak)
            total += exps[k]
        return (peak, total)

    def gradient_hessian(
        self,
        y: _Vector,
        raw: _Vector,
        weights: Optional[_Vector],
        mut gradients: List[Float32],
        mut hessians: List[Float32],
    ) raises:
        """Refill column-major derivatives so each tree can borrow a contiguous span."""
        self._check_inputs(y, raw, weights)
        gradients.resize(raw.size, fill=0.0)
        hessians.resize(raw.size, fill=0.0)
        var binary = self.raw_columns() == 1
        var exps = List[Float64](length=0 if binary else self._n_classes, fill=0.0)
        for i in range(y.size):
            var target = _at(y, i)
            var weight = _at(weights.value(), i) if weights else 1.0
            if binary:
                var prediction = _get(raw, i, 0)
                var gradient: Float64
                var hessian: Float64
                if prediction > -37:
                    var tail = _exp(-prediction)
                    var denom = 1 + tail
                    gradient = ((1 - target) - target * tail) / denom
                    hessian = tail / (denom * denom)
                else:
                    hessian = _exp(prediction)
                    gradient = hessian - target
                gradients[i] = Float32(weight * gradient)
                hessians[i] = Float32(weight * hessian)
            else:
                var _, total = self._shifted_exponentials(raw, i, exps)
                for k in range(self._n_classes):
                    var probability = exps[k] / total
                    var offset = k * y.size + i
                    gradients[offset] = Float32((probability - Float64(target == Float64(k))) * weight)
                    hessians[offset] = Float32((probability * (1 - probability)) * weight)

    def mean_loss(
        self, y: _Vector, raw: _Vector, weights: Optional[_Vector] = None
    ) raises -> Float64:
        self._check_inputs(y, raw, weights)
        var values = _Vector(Shape(y.size))
        var binary = self.raw_columns() == 1
        var exps = List[Float64](length=0 if binary else self._n_classes, fill=0.0)
        for i in range(y.size):
            var target = _at(y, i)
            if binary:
                var prediction = _get(raw, i, 0)
                values.unsafe_set(i, _log1pexp(prediction) - target * prediction)
            else:
                var peak, total = self._shifted_exponentials(raw, i, exps)
                values.unsafe_set(i, (_log(total) + peak) - _get(raw, i, Int(target)))
        return _weighted_mean(values, weights)

    def predict_proba(self, raw: _Vector) raises -> _Vector:
        self._check_raw(raw)
        var binary = self.raw_columns() == 1
        var n_columns = 2 if binary else self._n_classes
        var probabilities = _Vector(Shape(raw.shape[0], n_columns))
        var exps = List[Float64](length=0 if binary else n_columns, fill=0.0)
        for i in range(raw.shape[0]):
            if binary:
                var probability = 1 / (1 + _exp(-_get(raw, i, 0)))
                probabilities.unsafe_set(i * n_columns, 1 - probability)
                probabilities.unsafe_set(i * n_columns + 1, probability)
            else:
                var _, total = self._shifted_exponentials(raw, i, exps)
                for k in range(n_columns):
                    probabilities.unsafe_set(i * n_columns + k, exps[k] / total)
        return probabilities^
