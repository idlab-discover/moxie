"""Threshold policies, independent of score computation (sklearn BSD-3-Clause)."""
import numojo as nm
from std.math import isnan, isfinite, inf
from moxie.feature_selection._contracts import ScoreResult, SelectionPolicy

struct UnivariatePolicy[dt: DType](SelectionPolicy, Copyable, Movable):
    comptime dtype = Self.dt
    var mode: String
    var param: Float64
    var all_features: Bool

    def __init__(out self, mode: String = "percentile", param: Float64 = 1e-5, all_features: Bool = False):
        self.mode = mode
        self.param = param
        self.all_features = all_features

    def validate(self) raises:
        if self.mode != "k_best" and self.mode != "percentile" and self.mode != "fpr" and self.mode != "fdr" and self.mode != "fwe":
            raise Error("ValueError: unknown selection mode")
        if self.all_features:
            if self.mode != "k_best":
                raise Error("ValueError: all requires k_best")
            return
        if not isfinite(self.param) or self.param < 0:
            raise Error("ValueError: invalid selection parameter")
        if self.mode == "k_best":
            if self.param > Float64(Int.MAX) or self.param != Float64(Int(self.param)):
                raise Error("ValueError: k must be an integer")
        elif self.mode == "percentile":
            if self.param > 100:
                raise Error("ValueError: percentile must be in [0, 100]")
        elif self.param > 1:
            raise Error("ValueError: alpha must be in [0, 1]")

    def select(self, scores: ScoreResult[Self.dtype]) raises -> List[Bool]:
        self.validate()
        var n = scores.scores.shape[0]
        if n == 0:
            raise Error("ValueError: empty scores")
        var mask = List[Bool]()
        for _ in range(n):
            mask.append(False)
        if self.mode == "k_best" or self.mode == "percentile":
            if self.all_features or (self.mode == "k_best" and self.param >= Float64(n)) or (self.mode == "percentile" and self.param == 100):
                for i in range(n):
                    mask[i] = True
                return mask^
            if self.param == 0:
                return mask^
            var clean = nm.zeros[Self.dtype](nm.Shape(n))
            for i in range(n):
                var value = scores.scores.unsafe_get(i * Int(scores.scores.strides[0]))
                if isnan(value):
                    value = SIMD[Self.dtype, 1](-3.4028234663852886e38) if Self.dtype == DType.float32 else SIMD[Self.dtype, 1](-1.7976931348623157e308)
                clean.unsafe_set(i, value)
            var ordered = nm.sort(clean)
            var keep = Int(self.param)
            var threshold = ordered.unsafe_get(n - keep) if self.mode == "k_best" else ordered.unsafe_get(0)
            if self.mode == "percentile":
                var position = Float64(n - 1) * (100 - self.param) / 100
                var lower = Int(position)
                var upper = min(lower + 1, n - 1)
                var weight = position - Float64(lower)
                var a = ordered.unsafe_get(lower)
                var b = ordered.unsafe_get(upper)
                var difference = b - a
                threshold = a + difference * SIMD[Self.dtype, 1](weight)
                if weight >= 0.5:
                    threshold = b - difference * SIMD[Self.dtype, 1](1 - weight)
                keep = Int(Float64(n) * self.param / 100)
            var selected = 0
            for i in range(n):
                mask[i] = clean.unsafe_get(i) > threshold
                selected += Int(mask[i])
            var remaining = keep - selected
            for offset in range(n):
                var i = n - 1 - offset if self.mode == "k_best" else offset
                if remaining > 0 and clean.unsafe_get(i) == threshold:
                    mask[i] = True
                    remaining -= 1
            return mask^
        if not scores.pvalues:
            raise Error("ValueError: this policy requires pvalues")
        ref pvalues = scores.pvalues.value()
        var cutoff = self.param
        if self.mode == "fwe":
            cutoff /= Float64(n)
        elif self.mode == "fdr":
            var clean = nm.zeros[Self.dtype](nm.Shape(n))
            for i in range(n):
                var value = pvalues.unsafe_get(i * Int(pvalues.strides[0]))
                if isnan(value):
                    value = inf[Self.dtype]()
                clean.unsafe_set(i, value)
            var ordered = nm.sort(clean)
            var found = False
            for i in range(n):
                var value = Float64(ordered.unsafe_get(i))
                if value <= self.param / Float64(n) * Float64(i + 1):
                    cutoff = value
                    found = True
            if not found:
                return mask^
        for i in range(n):
            var value = Float64(pvalues.unsafe_get(i * Int(pvalues.strides[0])))
            mask[i] = value <= cutoff if self.mode == "fdr" else value < cutoff
        return mask^
