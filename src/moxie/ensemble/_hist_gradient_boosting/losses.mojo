# scikit-learn developers, BSD-3-Clause. See families/ensemble/LICENSES/scikit-learn.txt.
# NumPy interpolation and reduction conventions. See families/ensemble/LICENSES/NumPy.txt.
"""Single-output regression loss operations for histogram gradient boosting."""

from std.builtin.sort import sort
from std.math import floor, isfinite, isnan, nan
from std.sys import llvm_intrinsic

from numojo import NDArray, Shape

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
