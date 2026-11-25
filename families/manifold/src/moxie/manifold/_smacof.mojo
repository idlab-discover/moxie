# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense stress majorization with bounded per-restart workspace."""
from max.algorithm import parallelize
from numojo import NDArray, Shape, zeros
from std.math import abs, isfinite, isnan, sqrt
from std.sys import num_logical_cores
from std.utils import Variant
from moxie._random import RandomState, check_random_state
from moxie.isotonic import IsotonicRegression
from moxie.manifold._plumbing import validate_matrix, validate_pairwise
from moxie.neighbors._distance import at
from moxie.warnings import FitWarning


comptime NormalizedStress = Variant[Bool, String]
comptime _Matrix = NDArray[DType.float64]


@fieldwise_init
struct SMACOFResult(Copyable, Movable):
    """Owned coordinates, final stress, optional iteration count and fit warnings."""

    var embedding: _Matrix
    var stress: Float64
    var n_iter: Optional[Int]
    var fit_warnings: List[FitWarning]


@fieldwise_init
struct _Run(Copyable, Movable):
    var embedding: _Matrix
    var stress: Float64
    var n_iter: Int

    def take_embedding(deinit self) -> _Matrix:
        return self.embedding^


@fieldwise_init
struct _Options(Copyable, Movable):
    var metric: Bool
    var max_iter: Int
    var eps: Float64
    var verbose: Int
    var normalized: Bool


def _validate_dissimilarities[dtype: DType](dissimilarities: NDArray[dtype]) raises:
    validate_pairwise(dissimilarities, symmetric=False)
    for i in range(dissimilarities.shape[0]):
        for j in range(i):
            var a = at(dissimilarities, i, j)
            var b = at(dissimilarities, j, i)
            # NumPy allclose applies the relative tolerance in both directions.
            if abs(a - b) > 1e-10 + 1e-5 * min(abs(a), abs(b)):
                raise Error("ValueError: array must be symmetric")


def _row_major_cast[dtype: DType, //, target: DType](
    values: NDArray[dtype],
) raises -> NDArray[target]:
    if values.is_c_contiguous():
        return values.astype[target]()
    # NuMojo astype reads linear storage, so walk the logical noncontiguous layout.
    var result = zeros[target](Shape(values.shape[0], values.shape[1]))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            result.unsafe_set(i * values.shape[1] + j, at(values, i, j).cast[target]())
    return result^


def _copy_initial[dtype: DType](init: NDArray[dtype]) raises -> _Matrix:
    validate_matrix(init)
    return _row_major_cast[DType.float64](init)


def _random_coordinates(n: Int, components: Int, mut rng: RandomState) raises -> _Matrix:
    var coordinates = zeros[DType.float64](Shape(n, components))
    for i in range(coordinates.size):
        coordinates.unsafe_set(i, rng.random_sample())
    return coordinates^


def _distances(coordinates: _Matrix, mut distances: _Matrix) raises:
    var n = coordinates.shape[0]
    var width = coordinates.shape[1]
    for i in range(n):
        for j in range(i):
            var square: Float64 = 0
            for c in range(width):
                var delta = coordinates.unsafe_get(i * width + c) - coordinates.unsafe_get(j * width + c)
                square += delta * delta
            var distance = sqrt(square)
            if not isfinite(distance):
                raise Error("ValueError: SMACOF pairwise distance overflow")
            distances.unsafe_set(i * n + j, distance)
            distances.unsafe_set(j * n + i, distance)


def _guttman(coordinates: _Matrix, distances: _Matrix, disparities: _Matrix, mut next: _Matrix) raises:
    var n = coordinates.shape[0]
    var width = coordinates.shape[1]
    for i in range(n):
        for c in range(width):
            next.unsafe_set(i * width + c, 0)
        for j in range(n):
            if i == j:
                continue
            var distance = distances.unsafe_get(i * n + j)
            var ratio = disparities.unsafe_get(i * n + j) / (1e-5 if distance == 0 else distance)
            for c in range(width):
                var index = i * width + c
                var delta = coordinates.unsafe_get(index) - coordinates.unsafe_get(j * width + c)
                next.unsafe_set(index, next.unsafe_get(index) + ratio * delta)
        for c in range(width):
            var index = i * width + c
            next.unsafe_set(index, next.unsafe_get(index) / Float64(n))


struct _Disparities(Movable):
    var values: _Matrix
    var _pairs: List[Tuple[Int, Int]]
    var _constraints: List[Float64]
    var _targets: List[Float64]

    def __init__(out self, dissimilarities: _Matrix) raises:
        var n = dissimilarities.shape[0]
        self.values = zeros[DType.float64](Shape(n, n))
        self._pairs = []
        self._constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                var value = dissimilarities.unsafe_get(i * n + j)
                if value != 0:
                    self._pairs.append((i, j))
                    self._constraints.append(value)
        self._targets = List[Float64](length=len(self._pairs), fill=0.0)

    def update(mut self, distances: _Matrix, first: Bool) raises:
        var n = distances.shape[0]
        var fitted: List[Float64]
        if first:
            fitted = self._constraints.copy()
        else:
            for k in range(len(self._pairs)):
                var pair = self._pairs[k]
                self._targets[k] = distances.unsafe_get(pair[0] * n + pair[1])
            var isotonic = IsotonicRegression(out_of_bounds="clip")
            _ = isotonic.fit(self._constraints, self._targets)
            fitted = isotonic.predict(self._constraints)
        var sum_squares: Float64 = 0
        for value in fitted:
            sum_squares += value * value
        var scale = sqrt((Float64(n) * Float64(n - 1) / 2) / sum_squares)
        if not isfinite(scale):
            raise Error("ValueError: non-metric MDS disparity normalization is undefined")
        for k in range(len(self._pairs)):
            var pair = self._pairs[k]
            var value = fitted[k] * scale
            self.values.unsafe_set(pair[0] * n + pair[1], value)
            self.values.unsafe_set(pair[1] * n + pair[0], value)


def _majorize(
    coordinates: _Matrix, disparities: _Matrix, mut next: _Matrix, mut distances: _Matrix,
) raises -> Tuple[Float64, Float64]:
    _guttman(coordinates, distances, disparities, next)
    _distances(next, distances)
    var stress: Float64 = 0
    var scale: Float64 = 0
    for i in range(distances.size):
        var distance = distances.unsafe_get(i)
        var residual = distance - disparities.unsafe_get(i)
        stress += residual * residual
        scale += distance * distance
    return (stress / 2, scale / 2)


def _smacof_single(
    dissimilarities: _Matrix, var coordinates: _Matrix, options: _Options,
) raises -> _Run:
    var n = coordinates.shape[0]
    var distances = zeros[DType.float64](Shape(n, n))
    var next = zeros[DType.float64](Shape(n, coordinates.shape[1]))
    var nonmetric: Optional[_Disparities] = None
    if not options.metric:
        nonmetric = _Disparities(dissimilarities)
    _distances(coordinates, distances)
    var stress: Float64 = 0
    var scale: Float64 = 0
    var iterations = 0
    for it in range(options.max_iter):
        var old_stress = stress
        var objective: Tuple[Float64, Float64]
        if nonmetric:
            nonmetric.value().update(distances, it == 0)
            objective = _majorize(coordinates, nonmetric.value().values, next, distances)
        else:
            objective = _majorize(coordinates, dissimilarities, next, distances)
        var previous = coordinates^
        coordinates = next^
        next = previous^
        stress = objective[0]
        scale = objective[1]
        iterations = it + 1
        if options.verbose >= 2:
            print("Iteration", it, "stress", stress)
        if it > 0 and (old_stress - stress) / scale < options.eps:
            if options.verbose > 0:
                print("Convergence criterion reached (iteration", it, ").")
            break
    if options.normalized:
        stress = sqrt(stress / scale)
    return _Run(coordinates^, stress, iterations)


def _best_serial(
    dissimilarities: _Matrix, options: _Options,
    n_components: Int, starts: Int, mut rng: RandomState,
) raises -> _Run:
    var best: Optional[_Run] = None
    for _ in range(starts):
        var coordinates = _random_coordinates(
            dissimilarities.shape[0], n_components, rng,
        )
        var result = _smacof_single(dissimilarities, coordinates^, options)
        if not best or result.stress < best.value().stress:
            best = result^
    return best.take()


def _prefer_parallel(stress: Float64, incumbent: Float64) -> Bool:
    return not isnan(incumbent) and (isnan(stress) or stress < incumbent)


def _best_parallel(
    dissimilarities: _Matrix, options: _Options, n_components: Int,
    seeds: List[Int], workers: Int,
) raises -> _Run:
    var results = List[Optional[_Run]](length=workers, fill=None)
    var errors = List[Optional[String]](length=workers, fill=None)
    var slots = results.unsafe_ptr()
    var error_slots = errors.unsafe_ptr()
    def work(i: Int) {slots, error_slots, dissimilarities, options, n_components, seeds, workers}:
        var best: Optional[_Run] = None
        try:
            for run in range(len(seeds) * i // workers, len(seeds) * (i + 1) // workers):
                var rng = RandomState(seeds[run])
                var coordinates = _random_coordinates(dissimilarities.shape[0], n_components, rng)
                var result = _smacof_single(dissimilarities, coordinates^, options)
                if not best or _prefer_parallel(result.stress, best.value().stress):
                    best = result^
            slots[unsafe_offset=i] = best^
        except error:
            error_slots[unsafe_offset=i] = String(error)
    if workers == 1:
        work(0)
    else:
        parallelize(work, workers, workers)
    var best: Optional[_Run] = None
    for i in range(workers):
        if errors[i]:
            raise Error(errors[i].value())
        var candidate = results[i].take()
        if not best or _prefer_parallel(candidate.stress, best.value().stress):
            best = candidate^
    return best.take()


def _smacof(
    dissimilarities: _Matrix, var initial: Optional[_Matrix], options: _Options,
    n_components: Int, n_init: Int, n_jobs: Optional[Int], random_state: Optional[Int],
) raises -> _Run:
    var jobs = n_jobs.or_else(1)
    if jobs == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if jobs < 0:
        jobs = max(1, num_logical_cores() + 1 + jobs)
    var rng = check_random_state(random_state)
    if initial:
        if jobs > 1:
            _ = rng[].randint(2147483647)
        return _smacof_single(dissimilarities, initial.take(), options)
    if jobs == 1:
        return _best_serial(dissimilarities, options, n_components, n_init, rng[])
    var seeds = List[Int](capacity=n_init)
    for _ in range(n_init):
        seeds.append(rng[].randint(2147483647))
    return _best_parallel(dissimilarities, options, n_components, seeds, min(n_init, jobs))


def _smacof_options(
    metric: Bool, n_components: Int, n_init: Int, max_iter: Int, verbose: Int,
    eps: Float64, normalized_stress: NormalizedStress,
) raises -> _Options:
    if n_components < 1 or n_init < 1 or max_iter < 1 or verbose < 0 or not (eps >= 0):
        raise Error(
            "ValueError: expected positive n_components, n_init and max_iter, "
            "and nonnegative verbose and eps"
        )
    if normalized_stress.isa[Bool]():
        return _Options(metric, max_iter, eps, verbose, normalized_stress[Bool])
    if normalized_stress[String] != "auto":
        raise Error("ValueError: normalized_stress must be a Bool or auto")
    return _Options(metric, max_iter, eps, verbose, not metric)


def _initial_warning(n_init: Int) -> FitWarning:
    return FitWarning("UserWarning", String(
        "Explicit initial positions passed: performing only one init of the MDS instead of ", n_init,
    ))


def _smacof_result[dtype: DType](
    dissimilarities: NDArray[dtype], var initial: Optional[_Matrix], metric: Bool,
    n_components: Int, n_init: Int, n_jobs: Optional[Int], max_iter: Int,
    verbose: Int, eps: Float64, random_state: Optional[Int], return_n_iter: Bool,
    normalized_stress: NormalizedStress,
) raises -> SMACOFResult:
    var options = _smacof_options(
        metric, n_components, n_init, max_iter, verbose, eps, normalized_stress,
    )
    _validate_dissimilarities(dissimilarities)
    var warnings = List[FitWarning]()
    if initial:
        if initial.value().shape[0] != dissimilarities.shape[0]:
            raise Error("ValueError: init matrix should have one row per sample")
        if n_init != 1:
            warnings.append(_initial_warning(n_init))
    var fitted = _smacof(
        _row_major_cast[DType.float64](dissimilarities), initial^, options, n_components, n_init, n_jobs, random_state,
    )
    var iterations: Optional[Int] = None
    if return_n_iter:
        iterations = fitted.n_iter
    var stress = fitted.stress
    return SMACOFResult(fitted^.take_embedding(), stress, iterations, warnings^)


def smacof[dtype: DType](
    dissimilarities: NDArray[dtype], *, metric: Bool = True, n_components: Int = 2,
    init: NoneType = None, n_init: Int = 1, n_jobs: Optional[Int] = None,
    max_iter: Int = 300, verbose: Int = 0, eps: Float64 = 1e-6,
    random_state: Optional[Int] = None, return_n_iter: Bool = False,
    normalized_stress: NormalizedStress = "auto",
) raises -> SMACOFResult:
    """Fit metric or ordinal MDS to a symmetric dense dissimilarity matrix."""
    return _smacof_result(
        dissimilarities, None, metric, n_components, n_init, n_jobs, max_iter,
        verbose, eps, random_state, return_n_iter, normalized_stress,
    )


def smacof[dtype: DType, init_dtype: DType](
    dissimilarities: NDArray[dtype], *, metric: Bool = True, n_components: Int = 2,
    init: NDArray[init_dtype], n_init: Int = 1, n_jobs: Optional[Int] = None,
    max_iter: Int = 300, verbose: Int = 0, eps: Float64 = 1e-6,
    random_state: Optional[Int] = None, return_n_iter: Bool = False,
    normalized_stress: NormalizedStress = "auto",
) raises -> SMACOFResult:
    """Use supplied coordinates, with their column count and a single restart."""
    return _smacof_result(
        dissimilarities, _copy_initial(init), metric, n_components, n_init,
        n_jobs, max_iter, verbose, eps, random_state, return_n_iter, normalized_stress,
    )
