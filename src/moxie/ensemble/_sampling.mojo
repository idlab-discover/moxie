# scikit-learn developers, BSD-3-Clause; see LICENSES/scikit-learn.txt.
# Summation order follows NumPy's pairwise reduction; see LICENSES/NumPy.txt.
"""Ensemble row/feature sampling and class weighting for forests and bagging.

Ports sklearn `_bootstrap._get_n_samples_bootstrap`, `_forest._generate_sample_indices`
/ `_generate_unsampled_indices`, `_bagging._generate_bagging_indices` and the
`compute_class_weight`/`compute_sample_weight` paths forests use. Every draw is
consumed from the shared `moxie._random.RandomState` in the reference order.
"""
from std.math import isfinite
from std.utils import Variant
import numojo as nm
from moxie._random import RandomState
from moxie.ensemble._records import Values, Indices, SamplingPlan, SampleSelection
from moxie.ensemble._engine_api import EnsembleSampler
from moxie.ensemble._plumbing import validate_sampling, warn

comptime MaxSamples = Variant[NoneType, Int, Float64]
"""sklearn `max_samples`: None (all rows), a count, or a positive fraction."""


def _at(x: Values, i: Int) raises -> Float64:
    return x.unsafe_get(i * Int(x.strides[0]))


def _pairwise(x: Values, start: Int, count: Int) raises -> Float64:
    # NumPy loops_utils.h.src pairwise_sum: eight accumulators over leaves of at
    # most 128 elements, splitting on multiples of eight.
    if count < 8:
        var result = Float64(-0.0)
        for i in range(start, start + count):
            result += _at(x, i)
        return result
    if count <= 128:
        var r = SIMD[DType.float64, 8](0)
        for j in range(8):
            r[j] = _at(x, start + j)
        var i = 8
        while i < count - count % 8:
            for j in range(8):
                r[j] += _at(x, start + i + j)
            i += 8
        var result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < count:
            result += _at(x, start + i)
            i += 1
        return result
    var half = count // 2
    half -= half % 8
    return _pairwise(x, start, half) + _pairwise(x, start + half, count - half)


def numpy_sum(x: Values) raises -> Float64:
    """`np.sum` of a Float64 vector, bit-identical to NumPy's reduction order."""
    if x.ndim != 1:
        raise Error("ValueError: numpy_sum expects a vector")
    return _pairwise(x, 0, x.size)


def check_sample_weight(sample_weight: Values, n_samples: Int) raises:
    """sklearn `_check_sample_weight`: a finite Float64 vector with one entry per row."""
    if sample_weight.ndim != 1 or sample_weight.size != n_samples:
        raise Error("ValueError: sample_weight.shape == (" + String(sample_weight.size) + ",), expected (" + String(n_samples) + ",)")
    for i in range(n_samples):
        if not isfinite(_at(sample_weight, i)):
            raise Error("ValueError: Input sample_weight contains NaN or infinity.")


def get_n_samples_bootstrap(n_samples: Int, max_samples: MaxSamples, sample_weight: Optional[Values], mut warnings: List[String]) raises -> Int:
    """Rows drawn per bootstrap; a fraction scales the total sample weight when given."""
    if n_samples < 1:
        raise Error("ValueError: empty sampling population")
    if max_samples.isa[NoneType]():
        return n_samples
    if max_samples.isa[Int]():
        if max_samples[Int] < 1:
            raise Error("ValueError: max_samples must be an int in [1, inf)")
        return max_samples[Int]
    var fraction = max_samples[Float64]
    if not (fraction > 0) or not isfinite(fraction):
        raise Error("ValueError: max_samples must be a float in (0, inf)")
    var weighted = Float64(n_samples)
    var message = "the number of samples is " + String(n_samples) + " "
    if sample_weight:
        weighted = numpy_sum(sample_weight.value())
        message = "the total sum of sample weights is " + String(weighted) + " "
    var product = fraction * weighted
    if not isfinite(product):
        raise Error("ValueError: max_samples times the total sample weight must be finite")
    var count = max(Int(product), 1)  # Python int() truncates toward zero
    if Float64(count) < max(Float64(10), Float64(n_samples) ** (1.0 / 3.0)):
        warn(warnings, "UserWarning: Using the fractional value max_samples=" + String(fraction) + " when " + message
            + "results in a low number (" + String(count) + ") of bootstrap samples. We recommend passing `max_samples` as an integer instead.")
    return count


def normalized_weights(sample_weight: Values, n_samples: Int) raises -> Values:
    """`sample_weight / np.sum(sample_weight)`: the sampling mass for weighted choice.

    Like NumPy, validity is judged on the normalized mass: `choice` rejects
    negative, nonfinite or non-normalized probabilities (a zero total yields NaN).
    """
    if sample_weight.ndim != 1 or sample_weight.size != n_samples:
        raise Error("ValueError: sampling weights need one entry per sample")
    var total = numpy_sum(sample_weight)
    var p = nm.zeros[DType.float64](nm.Shape(n_samples))
    for i in range(n_samples):
        p.unsafe_set(i, _at(sample_weight, i) / total)
    return p^


def draw_indices(mut rng: RandomState, bootstrap: Bool, n_population: Int, n_draws: Int) raises -> Indices:
    """sklearn bagging `_generate_indices`: randint with, or sample_without_replacement without, replacement."""
    if bootstrap:
        if n_population < 1:
            raise Error("ValueError: high <= 0")
        if n_draws < 0:
            raise Error("ValueError: negative dimensions are not allowed")
        var result = nm.zeros[DType.int64](nm.Shape(n_draws))
        for i in range(n_draws):
            result.unsafe_set(i, Int64(rng.randint(0, n_population)))
        return result^
    return rng.sample_without_replacement(n_population, n_draws)


def draw_sample_indices(mut rng: RandomState, n_samples: Int, n_samples_bootstrap: Int, sample_weight: Optional[Values]) raises -> Indices:
    """Forest bootstrap rows: uniform randint, or choice(p=w/sum(w)) with weights."""
    if not sample_weight:
        return draw_indices(rng, True, n_samples, n_samples_bootstrap)
    var p = normalized_weights(sample_weight.value(), n_samples)
    return rng.choice(n_samples, p, n_samples_bootstrap, True)


def generate_sample_indices(seed: Int, n_samples: Int, n_samples_bootstrap: Int, sample_weight: Optional[Values]) raises -> Indices:
    """`_generate_sample_indices(seed, ...)`: a fresh stream per member seed, so it can be replayed."""
    var rng = RandomState(seed)
    return draw_sample_indices(rng, n_samples, n_samples_bootstrap, sample_weight)


def sample_counts(indices: Indices, n_samples: Int) raises -> Indices:
    """`np.bincount(indices, minlength=n_samples)`: draws per original row."""
    var counts = nm.zeros[DType.int64](nm.Shape(n_samples))
    for i in range(indices.size):
        var row = Int(indices.unsafe_get(i * Int(indices.strides[0])))
        if row < 0 or row >= n_samples:
            raise Error("ValueError: sample index outside population")
        counts.unsafe_set(row, counts.unsafe_get(row) + 1)
    return counts^


def count_weights(counts: Indices) raises -> Values:
    """Bootstrap counts as Float64 tree sample weights; rows are never copied."""
    var result = nm.zeros[DType.float64](nm.Shape(counts.size))
    for i in range(counts.size):
        result.unsafe_set(i, Float64(counts.unsafe_get(i * Int(counts.strides[0]))))
    return result^


def unsampled_rows(counts: Indices) raises -> Indices:
    """Sorted original rows drawn zero times (out-of-bag rows)."""
    var missing = 0
    for i in range(counts.size):
        if counts.unsafe_get(i * Int(counts.strides[0])) == 0:
            missing += 1
    var result = nm.zeros[DType.int64](nm.Shape(missing))
    var at = 0
    for i in range(counts.size):
        if counts.unsafe_get(i * Int(counts.strides[0])) == 0:
            result.unsafe_set(at, Int64(i))
            at += 1
    return result^


def generate_unsampled_indices(seed: Int, n_samples: Int, n_samples_bootstrap: Int, sample_weight: Optional[Values]) raises -> Indices:
    """`_generate_unsampled_indices`: replays a member's bootstrap and keeps the rows it never drew."""
    return unsampled_rows(sample_counts(generate_sample_indices(seed, n_samples, n_samples_bootstrap, sample_weight), n_samples))


def generate_bagging_indices(mut rng: RandomState, bootstrap_features: Bool, bootstrap_samples: Bool, n_features: Int,
        n_samples: Int, max_features: Int, max_samples: Int, sample_weight: Optional[Values]) raises -> Tuple[Indices, Indices]:
    """`_generate_bagging_indices`: features first, then rows (weighted choice when weights are given)."""
    var features = draw_indices(rng, bootstrap_features, n_features, max_features)
    if not sample_weight:
        return (features^, draw_indices(rng, bootstrap_samples, n_samples, max_samples))
    var p = normalized_weights(sample_weight.value(), n_samples)
    return (features^, rng.choice(n_samples, p, max_samples, bootstrap_samples))


def _identity(n: Int) raises -> Indices:
    var result = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        result.unsafe_set(i, Int64(i))
    return result^


@fieldwise_init
struct BootstrapSampler(EnsembleSampler):
    """Forest sampling: bootstrap rows with replacement, every feature, no feature draws."""

    def select(self, plan: SamplingPlan, weights: Optional[Values], mut rng: RandomState) raises -> SampleSelection:
        validate_sampling(plan)
        if not plan.bootstrap_samples or plan.bootstrap_features or plan.max_features != plan.n_features:
            raise Error("ValueError: forest bootstrap draws rows with replacement and keeps every feature")
        var rows = draw_sample_indices(rng, plan.n_samples, plan.max_samples, weights)
        var counts = sample_counts(rows, plan.n_samples)
        return SampleSelection(rows^, _identity(plan.n_features), counts^)

    def unsampled(self, selection: SampleSelection) raises -> Indices:
        return unsampled_rows(selection.multiplicities)


@fieldwise_init
struct BaggingSampler(EnsembleSampler):
    """Bagging sampling: features then rows, each with or without replacement."""

    def select(self, plan: SamplingPlan, weights: Optional[Values], mut rng: RandomState) raises -> SampleSelection:
        validate_sampling(plan)
        var drawn = generate_bagging_indices(rng, plan.bootstrap_features, plan.bootstrap_samples, plan.n_features,
            plan.n_samples, plan.max_features, plan.max_samples, weights)
        var counts = sample_counts(drawn[1], plan.n_samples)
        return SampleSelection(drawn[1].copy(), drawn[0].copy(), counts^)

    def unsampled(self, selection: SampleSelection) raises -> Indices:
        return unsampled_rows(selection.multiplicities)


# Class weighting on encoded targets. Forest members train on Float64 class
# codes 0..K-1 per output (np.unique return_inverse), so codes stand for labels.

def _codes(y: Values) raises -> Tuple[Int, Int]:
    if y.ndim == 1:
        return (y.shape[0], 1)
    if y.ndim == 2:
        return (y.shape[0], y.shape[1])
    raise Error("ValueError: targets must be a vector or matrix")


def _code(y: Values, row: Int, output: Int) raises -> Int:
    var offset = row * Int(y.strides[0])
    if y.ndim == 2:
        offset += output * Int(y.strides[1])
    var value = y.unsafe_get(offset)
    var code = Int(value)
    if Float64(code) != value or code < 0:
        raise Error("ValueError: class codes must be nonnegative integers")
    return code


def _class_count(y: Values, output: Int, n: Int) raises -> Int:
    var count = 0
    for i in range(n):
        count = max(count, _code(y, i, output) + 1)
    return count


def expand_class_weight(y: Values, factors: List[Values]) raises -> Values:
    """Per-sample product over outputs of per-class factors (`np.prod(axis=0)` order)."""
    var shape = _codes(y)
    var n = shape[0]
    if len(factors) != shape[1]:
        raise Error("ValueError: class_weight needs one entry per output")
    var result = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        var weight = Float64(0)
        for k in range(shape[1]):
            var code = _code(y, i, k)
            if code >= factors[k].size:
                raise Error("ValueError: class code outside class_weight")
            var factor = _at(factors[k], code)
            weight = factor if k == 0 else weight * factor
        result.unsafe_set(i, weight)
    return result^


def balanced_class_factors(y: Values, output: Int, sample_weight: Optional[Values]) raises -> Values:
    """`compute_class_weight("balanced", ...)`: sum(counts) / (n_classes * counts), weighted counts if given."""
    var n = _codes(y)[0]
    var count = _class_count(y, output, n)
    var counts = nm.zeros[DType.float64](nm.Shape(count))
    for i in range(n):  # np.bincount accumulates in row order
        var code = _code(y, i, output)
        counts.unsafe_set(code, counts.unsafe_get(code) + (_at(sample_weight.value(), i) if sample_weight else Float64(1)))
    var total = numpy_sum(counts)
    var factors = nm.zeros[DType.float64](nm.Shape(count))
    for c in range(count):
        factors.unsafe_set(c, total / (Float64(count) * counts.unsafe_get(c)))
    return factors^


def balanced_class_weight(y: Values, sample_weight: Optional[Values]) raises -> Values:
    """Forest `class_weight="balanced"`: per-output balanced factors, multiplied per sample."""
    var outputs = _codes(y)[1]
    var factors = List[Values]()
    for k in range(outputs):
        factors.append(balanced_class_factors(y, k, sample_weight))
    return expand_class_weight(y, factors)


def balanced_subsample_weight(y: Values, indices: Indices) raises -> Values:
    """`compute_sample_weight("balanced", y, indices=indices)`; classes absent from the draw weigh zero."""
    var shape = _codes(y)
    var n = shape[0]
    var result = nm.zeros[DType.float64](nm.Shape(n))
    for k in range(shape[1]):
        var count = _class_count(y, k, n)
        var drawn = nm.zeros[DType.float64](nm.Shape(count))
        for r in range(indices.size):
            var row = Int(indices.unsafe_get(r * Int(indices.strides[0])))
            if row < 0 or row >= n:
                raise Error("ValueError: sample index outside population")
            var code = _code(y, row, k)
            drawn.unsafe_set(code, drawn.unsafe_get(code) + 1)
        var present = 0
        for c in range(count):
            if drawn.unsafe_get(c) > 0:
                present += 1
        var total = numpy_sum(drawn)
        for i in range(n):
            var c = _code(y, i, k)
            var factor = Float64(0)
            if drawn.unsafe_get(c) > 0:
                factor = total / (Float64(present) * drawn.unsafe_get(c))
            result.unsafe_set(i, factor if k == 0 else result.unsafe_get(i) * factor)
    return result^


def class_weight_factors[dtype: DType](classes: nm.NDArray[dtype], weights: Dict[Scalar[dtype], Float64]) raises -> Values:
    """Per-class factors for one output from a `{label: weight}` mapping (missing labels weigh one)."""
    var factors = nm.full[DType.float64](nm.Shape(classes.size), 1)
    var matched = 0
    var unweighted = String()
    for c in range(classes.size):
        var label = classes.unsafe_get(c * Int(classes.strides[0]))
        if label in weights:
            factors.unsafe_set(c, weights[label])
            matched += 1
        else:
            unweighted += (", " if unweighted.byte_length() > 0 else "") + String(label)
    if matched < classes.size and matched != len(weights):
        raise Error("ValueError: The classes, [" + unweighted + "], are not in class_weight")
    return factors^


def class_weight_factors(classes: List[String], weights: Dict[String, Float64]) raises -> Values:
    var factors = nm.full[DType.float64](nm.Shape(len(classes)), 1)
    var matched = 0
    var unweighted = String()
    for c in range(len(classes)):
        if classes[c] in weights:
            factors.unsafe_set(c, weights[classes[c]])
            matched += 1
        else:
            unweighted += (", " if unweighted.byte_length() > 0 else "") + repr(classes[c])
    if matched < len(classes) and matched != len(weights):
        raise Error("ValueError: The classes, [" + unweighted + "], are not in class_weight")
    return factors^
