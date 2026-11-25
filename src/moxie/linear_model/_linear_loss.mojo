# Adapted from scikit-learn sklearn/linear_model/_linear_loss.py and
# sklearn/_loss/_loss.pyx.tp; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense weighted linear-model objective and analytic gradient."""

from std.math import abs, exp, log1p, max
from mojo_optimize import Objective
from numojo import NDArray, Shape, zeros


def logistic_probability(score: Float64) -> Float64:
    if score < 0:
        var tail = exp(score)
        return tail / (1 + tail)
    return 1 / (1 + exp(-score))


struct LogisticLoss[source_dtype: DType](Objective):
    """Half-binomial or half-multinomial mean loss with feature-only L2."""

    comptime dtype = DType.float64

    var _X: NDArray[Self.source_dtype]
    var _targets: List[Int]
    var _weights: NDArray[DType.float64]
    var _scores: List[Float64]
    var _probabilities: List[Float64]
    var _active: List[Int]
    var hessian: NDArray[DType.float64]
    var hessian_warning: Bool
    var _l2: Float64
    var _fit_intercept: Bool

    def __init__(
        out self,
        var X: NDArray[Self.source_dtype],
        var targets: List[Int],
        var normalized_weights: NDArray[DType.float64],
        outputs: Int,
        l2: Float64,
        fit_intercept: Bool,
        newton: Bool,
    ) raises:
        var active = List[Int]()
        if newton:
            var features = X.shape[1]
            var columns = features + Int(fit_intercept)
            for j in range(columns):
                # Fix the final class at zero for unpenalized columns.
                var count = outputs
                if outputs > 1 and (l2 == 0 or j == features):
                    count -= 1
                for k in range(count):
                    active.append(j * outputs + k)
        var order = len(active)
        if order > 0 and order > Int.MAX // 8 // order:
            raise Error("ValueError: LogisticRegression Hessian size overflows")
        self.hessian = zeros[DType.float64](Shape(order, order))
        self.hessian_warning = False
        self._active = active^
        self._probabilities = List[Float64](
            length=outputs if newton else 0, fill=0
        )
        self._X = X^
        self._targets = targets^
        self._weights = normalized_weights^
        self._scores = List[Float64](length=outputs, fill=0)
        self._l2 = l2
        self._fit_intercept = fit_intercept

    def evaluate(
        mut self,
        weights: NDArray[DType.float64],
        mut gradient: NDArray[DType.float64],
    ) raises -> Float64:
        return self._evaluate[False](weights, gradient)

    def active_parameters(self) -> List[Int]:
        return self._active.copy()

    def gradient_hessian(
        mut self,
        weights: NDArray[DType.float64],
        mut gradient: NDArray[DType.float64],
    ) raises -> Float64:
        return self._evaluate[True](weights, gradient)

    def binary_working_response(
        mut self, coefficients: NDArray[DType.float64],
        mut response: NDArray[DType.float64], mut curvature: NDArray[DType.float64],
    ) raises:
        """Binary Newton-CD response and clipped pointwise curvature."""
        var features = self._X.shape[1]
        for i in range(self._X.shape[0]):
            Self._project_row(
                self._X, i * features, features, self._fit_intercept,
                coefficients, self._scores,
            )
            var raw = self._scores[0]
            var sign = Float64(1 if self._targets[i] == 0 else -1)
            var margin = sign * raw
            var tail = exp(-abs(margin))
            var denominator = 1 + tail
            var h = max(0.0, self._weights.unsafe_get(i) * (tail / (denominator * denominator)))
            if h > 0:
                raw -= sign * (denominator if margin <= 0 else denominator / tail)
            response.unsafe_set(i, raw)
            curvature.unsafe_set(i, h)

    @staticmethod
    def _project_row(
        X: NDArray[Self.source_dtype], row: Int, features: Int, fit_intercept: Bool,
        coefficients: NDArray[DType.float64], mut projected: List[Float64],
    ):
        var outputs = len(projected)
        for k in range(outputs):
            projected[k] = coefficients.unsafe_get(
                features * outputs + k
            ) if fit_intercept else 0
        for j in range(features):
            var value = Float64(X.unsafe_get(row + j))
            for k in range(outputs):
                projected[k] += value * coefficients.unsafe_get(j * outputs + k)

    def _accumulate_row(
        self, row: Int, features: Int, projected: List[Float64],
        mut result: NDArray[DType.float64],
    ):
        var outputs = len(projected)
        for j in range(features):
            var value = Float64(self._X.unsafe_get(row + j))
            for k in range(outputs):
                var index = j * outputs + k
                result.unsafe_set(index, result.unsafe_get(index) + value * projected[k])
        if self._fit_intercept:
            for k in range(outputs):
                var index = features * outputs + k
                result.unsafe_set(index, result.unsafe_get(index) + projected[k])

    @staticmethod
    def _exp_shifted_scores(mut scores: List[Float64]) -> Tuple[Int, Float64, Float64]:
        var peak = 0
        for k in range(1, len(scores)):
            if scores[k] > scores[peak]:
                peak = k
        var shift = scores[peak]
        var tail = Float64(0)
        for k in range(len(scores)):
            scores[k] = exp(scores[k] - shift)
            if k != peak:
                tail += scores[k]
        return (peak, shift, tail)

    def hessian_product(
        mut self, weights: NDArray[DType.float64], direction: NDArray[DType.float64],
        mut result: NDArray[DType.float64], mut projected: List[Float64],
    ) raises:
        var features = self._X.shape[1]
        var outputs = len(self._scores)
        for j in range(features * outputs):
            result.unsafe_set(j, self._l2 * direction.unsafe_get(j))
        for j in range(features * outputs, result.size):
            result.unsafe_set(j, 0)
        for i in range(self._X.shape[0]):
            var weight = self._weights.unsafe_get(i)
            if weight == 0:
                continue
            var row = i * features
            Self._project_row(self._X, row, features, self._fit_intercept, direction, projected)
            Self._project_row(self._X, row, features, self._fit_intercept, weights, self._scores)
            if outputs == 1:
                var tail = exp(-abs(self._scores[0]))
                var denominator = 1 + tail
                projected[0] *= weight * (tail / (denominator * denominator))
            else:
                var (peak, _, tail) = Self._exp_shifted_scores(self._scores)
                var center = projected[peak]
                var inverse_sum = 1 / (1 + tail)
                var mean_delta = Float64(0)
                for k in range(outputs):
                    projected[k] -= center
                    mean_delta += (self._scores[k] * inverse_sum) * projected[k]
                var other_mass = Float64(0)
                for k in range(outputs):
                    if k != peak:
                        projected[k] = (
                            weight * (self._scores[k] * inverse_sum)
                            * (projected[k] - mean_delta)
                        )
                        other_mass += projected[k]
                # The softmax covariance annihilates the all-ones vector.
                projected[peak] = -other_mass
            self._accumulate_row(row, features, projected, result)

    def _evaluate[
        with_hessian: Bool
    ](
        mut self,
        weights: NDArray[DType.float64],
        mut gradient: NDArray[DType.float64],
    ) raises -> Float64:
        var features = self._X.shape[1]
        var outputs = len(self._scores)
        var intercept_start = features * outputs
        var loss = Float64(0)
        var nonpositive_mass = Float64(0)
        var order = len(self._active)
        comptime if with_hessian:
            for i in range(order):
                for j in range(i + 1):
                    self.hessian.unsafe_set(i * order + j, 0)
                if self._active[i] < intercept_start:
                    self.hessian.unsafe_set(i * order + i, self._l2)
        for j in range(intercept_start):
            var coefficient = weights.unsafe_get(j)
            var regularizer = self._l2 * coefficient
            gradient.unsafe_set(j, regularizer)
            loss += 0.5 * regularizer * coefficient
        for k in range(intercept_start, weights.size):
            gradient.unsafe_set(k, 0)

        for i in range(self._X.shape[0]):
            var weight = self._weights.unsafe_get(i)
            if weight == 0:
                continue
            var row = i * features
            Self._project_row(self._X, row, features, self._fit_intercept, weights, self._scores)

            # Replace raw scores with weighted pointwise gradients.
            var target = self._targets[i]
            var binary_curvature = Float64(0)
            var peak = 0
            var peak_complement = Float64(0)
            if outputs == 1:
                var sign = Float64(1 if target == 0 else -1)
                var margin = sign * self._scores[0]
                var tail = exp(-abs(margin))
                loss += weight * (max(0.0, margin) + log1p(tail))
                var numerator = tail if margin < 0 else 1
                self._scores[0] = weight * sign * (numerator / (1 + tail))
                comptime if with_hessian:
                    var denominator = 1 + tail
                    var signed_curvature = weight * (
                        tail / (denominator * denominator)
                    )
                    if signed_curvature <= 0:
                        nonpositive_mass += weight
                    binary_curvature = abs(signed_curvature)
            else:
                var target_score = self._scores[target]
                var shift: Float64
                var tail: Float64
                peak, shift, tail = Self._exp_shifted_scores(self._scores)
                var target_margin = shift - target_score
                loss += weight * (target_margin + log1p(tail))
                var scale = weight / (1 + tail)
                var other_mass = Float64(0)
                for k in range(outputs):
                    comptime if with_hessian:
                        self._probabilities[k] = self._scores[k] / (1 + tail)
                    self._scores[k] *= scale
                    if k != target:
                        other_mass += self._scores[k]
                self._scores[target] = -other_mass
                comptime if with_hessian:
                    peak_complement = tail / (1 + tail)

            comptime if with_hessian:
                for a in range(order):
                    var j = self._active[a] // outputs
                    var k = self._active[a] % outputs
                    var xj = (
                        Float64(self._X.unsafe_get(row + j)) if j
                        < features else 1
                    )
                    for b in range(a + 1):
                        var m = self._active[b] // outputs
                        var l = self._active[b] % outputs
                        var xm = (
                            Float64(self._X.unsafe_get(row + m)) if m
                            < features else 1
                        )
                        var curvature = binary_curvature
                        if outputs > 1:
                            var complement = -self._probabilities[l]
                            if k == l:
                                complement = (
                                    peak_complement if k
                                    == peak else 1 - self._probabilities[k]
                                )
                            curvature = (
                                weight * self._probabilities[k]
                            ) * complement
                        var index = a * order + b
                        self.hessian.unsafe_set(
                            index,
                            self.hessian.unsafe_get(index)
                            + (curvature * xj) * xm,
                        )

            self._accumulate_row(row, features, self._scores, gradient)
        comptime if with_hessian:
            # The native symmetric factorization consumes the lower triangle.
            self.hessian_warning = nonpositive_mass > 0.25
        return loss
