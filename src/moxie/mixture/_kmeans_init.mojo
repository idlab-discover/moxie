"""Minimal dense unweighted KMeans used only for mixture initialization.

Translation of sklearn.cluster (pinned 4358542a) `_kmeans_plusplus`
(n_local_trials = 2 + int(log(k))) and `KMeans(n_clusters=k, n_init=1,
random_state=rng).fit(X).labels_` with algorithm="lloyd", max_iter=300,
tol=1e-4: centering by the column mean, tolerance mean(var(X, 0)) * tol,
k-means++ seeding on the centered data, Lloyd iterations with strict label
convergence or squared center shift <= tol, empty-cluster relocation to the
farthest points, and a final E-step when convergence was not strict.

Consolidation candidate: moxie-cluster/shared/centroid_engine owns weighted
Lloyd steps but, at the time of writing, had no k-means++ seeding and was
uncommitted work of a concurrent session, so it could not be pinned. Replace
this module with the cluster owner's KMeans once it is published.

Known divergence: with several simultaneously empty clusters sklearn assigns
far points in `np.argpartition` order (unspecified among the largest); here the
largest distances are assigned in descending order, ties by lower index.
Distances use -2 x.c + |x|^2 + |c|^2 (clipped at 0) in Float64, as the reference
does for Float64 and via its Float32 upcast path.
"""
from numojo import NDArray, Shape, zeros
from moxie._random import RandomState
from moxie.mixture._gaussian_kernels import Matrix, rounded
from moxie.mixture._scalar_math import log64

comptime KMEANS_MAX_ITER = 300
comptime KMEANS_TOL = 1e-4


def _row_norms[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype]) -> List[Float64]:
    var norms = List[Float64](length=xv.rows, fill=0.0)
    for i in range(xv.rows):
        var total = 0.0
        for f in range(xv.cols):
            var x = xv.at(X, i, f)
            total += x * x
        norms[i] = total
    return norms^


def _squared_distance[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], norms: List[Float64], a: Int, b: Int) -> Float64:
    """Euclidean squared distance between rows a and b, reference formula."""
    var dot = 0.0
    for f in range(xv.cols):
        dot += xv.at(X, a, f) * xv.at(X, b, f)
    var value = -2.0 * dot + norms[a] + norms[b]
    return rounded[dtype](value if value > 0.0 else 0.0)


def _uniform_weight_choice[dtype: DType](n: Int, mut rng: RandomState) -> Int:
    """`random_state.choice(n, p=w / w.sum())` for unit weights in the dtype.

    Reproduces NumPy's legacy protocol exactly (p cast to Float64,
    cdf = cumsum(p) / cdf[-1], one random_sample, searchsorted side="right").
    Done locally because moxie._random validates p against the Float64
    tolerance only, whereas NumPy relaxes it to sqrt(eps(float32)) for Float32
    probability arrays such as float32(1/n).
    """
    var p = Float64(Scalar[dtype](1) / Scalar[dtype](n))
    var total = 0.0
    for _ in range(n):
        total += p
    var u = rng.random_sample()
    var cumulative = 0.0
    for i in range(n):
        cumulative += p
        if u < cumulative / total:
            return i
    return n - 1


def kmeans_plusplus_indices[dtype: DType](X: NDArray[dtype], n_clusters: Int, mut rng: RandomState) raises -> List[Int]:
    """Row indices chosen by greedy k-means++ (unit sample weights).

    Draw order matches sklearn: one `choice(n, p=w/w.sum())`, then per center
    `uniform(size=n_local_trials)`.
    """
    var xv = Matrix[dtype](X)
    var n = xv.rows
    if n_clusters < 1 or n < n_clusters:
        raise Error(String("ValueError: n_samples=", n, " should be >= n_clusters=", n_clusters, "."))
    var norms = _row_norms(X, xv)
    var trials = 2 + Int(log64(Float64(n_clusters)))
    var indices = List[Int]()
    var first = _uniform_weight_choice[dtype](n, rng)
    indices.append(first)
    var closest = List[Float64](length=n, fill=0.0)
    var potential = 0.0
    for i in range(n):
        closest[i] = _squared_distance(X, xv, norms, first, i)
        potential += closest[i]
    potential = rounded[dtype](potential)
    var cumulative = List[Float64](length=n, fill=0.0)
    var candidate_distances = List[Float64](length=trials * n, fill=0.0)
    for _ in range(1, n_clusters):
        # np.cumsum(sample_weight * closest_dist_sq) accumulates in the dtype.
        var running = Scalar[dtype](0)
        for i in range(n):
            running += Scalar[dtype](closest[i])
            cumulative[i] = Float64(running)
        var candidates = List[Int](length=trials, fill=0)
        for t in range(trials):
            var target = rng.random_sample() * potential
            # searchsorted(side="left"), clipped to n - 1.
            var low = 0
            var high = n
            while low < high:
                var mid = low + (high - low) // 2
                if cumulative[mid] < target:
                    low = mid + 1
                else:
                    high = mid
            candidates[t] = low if low < n else n - 1
        var best = 0
        var best_potential = 0.0
        for t in range(trials):
            var total = 0.0
            for i in range(n):
                var distance = _squared_distance(X, xv, norms, candidates[t], i)
                if closest[i] < distance:
                    distance = closest[i]
                candidate_distances[t * n + i] = distance
                total += distance
            total = rounded[dtype](total)
            if t == 0 or total < best_potential:
                best = t
                best_potential = total
        potential = best_potential
        for i in range(n):
            closest[i] = candidate_distances[best * n + i]
        indices.append(candidates[best])
    return indices^


def _centered[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    """X - X.mean(axis=0) as an owned contiguous copy (KMeans copy_x=True)."""
    var xv = Matrix[dtype](X)
    var n = xv.rows
    var d = xv.cols
    var out = zeros[dtype](Shape(n, d))
    for f in range(d):
        var total = 0.0
        for i in range(n):
            total += xv.at(X, i, f)
        var mean = rounded[dtype](total / Float64(n))
        for i in range(n):
            out.unsafe_set(i * d + f, (xv.at(X, i, f) - mean).cast[dtype]())
    return out^


def _tolerance[dtype: DType](X: NDArray[dtype], tol: Float64) raises -> Float64:
    """mean(var(X, axis=0)) * tol for contiguous X."""
    var n = X.shape[0]
    var d = X.shape[1]
    var total_variance = 0.0
    for f in range(d):
        var mean = 0.0
        for i in range(n):
            mean += Float64(X.unsafe_get(i * d + f))
        mean /= Float64(n)
        var squares = 0.0
        for i in range(n):
            var delta = Float64(X.unsafe_get(i * d + f)) - mean
            squares += delta * delta
        total_variance += rounded[dtype](squares / Float64(n))
    return rounded[dtype](total_variance / Float64(d)) * tol


def _assign[dtype: DType](X: NDArray[dtype], centers: List[Float64], k: Int, mut labels: List[Int]) raises:
    """Labels by argmin of |c|^2 - 2 x.c (first minimum wins)."""
    var n = X.shape[0]
    var d = X.shape[1]
    var center_norms = List[Float64](length=k, fill=0.0)
    for j in range(k):
        var total = 0.0
        for f in range(d):
            total += centers[j * d + f] * centers[j * d + f]
        center_norms[j] = rounded[dtype](total)
    for i in range(n):
        var best = 0
        var best_value = 0.0
        for j in range(k):
            var dot = 0.0
            for f in range(d):
                dot += Float64(X.unsafe_get(i * d + f)) * centers[j * d + f]
            var value = rounded[dtype](center_norms[j] - 2.0 * dot)
            if j == 0 or value < best_value:
                best = j
                best_value = value
        labels[i] = best


def kmeans_labels[dtype: DType](X: NDArray[dtype], n_clusters: Int, mut rng: RandomState) raises -> List[Int]:
    """Labels of KMeans(n_clusters, n_init=1, random_state=rng).fit(X)."""
    var n = X.shape[0]
    var d = X.shape[1]
    if n < n_clusters:
        raise Error(String("ValueError: n_samples=", n, " should be >= n_clusters=", n_clusters, "."))
    var centered = _centered(X)
    var tol = _tolerance(centered, KMEANS_TOL)
    var seeds = kmeans_plusplus_indices(centered, n_clusters, rng)
    var k = n_clusters
    var centers = List[Float64](length=k * d, fill=0.0)
    for j in range(k):
        for f in range(d):
            centers[j * d + f] = Float64(centered.unsafe_get(seeds[j] * d + f))
    var labels = List[Int](length=n, fill=-1)
    var previous = List[Int](length=n, fill=-1)
    var updated = List[Float64](length=k * d, fill=0.0)
    var weights = List[Float64](length=k, fill=0.0)
    var strict = False
    for _ in range(KMEANS_MAX_ITER):
        _assign(centered, centers, k, labels)
        for idx in range(k * d):
            updated[idx] = 0.0
        for j in range(k):
            weights[j] = 0.0
        for i in range(n):
            var j = labels[i]
            weights[j] += 1.0
            for f in range(d):
                updated[j * d + f] += Float64(centered.unsafe_get(i * d + f))
        _relocate_empty(centered, centers, labels, updated, weights, k)
        var largest = 0
        for j in range(1, k):
            if weights[j] > weights[largest]:
                largest = j
        # `_average_centers`: one ordered pass; an empty cluster copies the
        # heaviest cluster's current (possibly not yet averaged) row.
        for j in range(k):
            if weights[j] > 0:
                var alpha = rounded[dtype](1.0 / weights[j])
                for f in range(d):
                    updated[j * d + f] = rounded[dtype](rounded[dtype](updated[j * d + f]) * alpha)
            else:
                for f in range(d):
                    updated[j * d + f] = updated[largest * d + f]
        var shift_total = 0.0
        for j in range(k):
            var shift = 0.0
            for f in range(d):
                var delta = updated[j * d + f] - centers[j * d + f]
                shift += delta * delta
            # center_shift is stored in the dtype, then squared and summed.
            var distance = rounded[dtype](shift ** 0.5)
            shift_total += distance * distance
        for idx in range(k * d):
            centers[idx] = updated[idx]
        var same = True
        for i in range(n):
            if labels[i] != previous[i]:
                same = False
                break
        if same:
            strict = True
            break
        if rounded[dtype](shift_total) <= tol:
            break
        for i in range(n):
            previous[i] = labels[i]
    if not strict:
        _assign(centered, centers, k, labels)
    return labels^


def _relocate_empty[dtype: DType](X: NDArray[dtype], centers: List[Float64], labels: List[Int], mut updated: List[Float64], mut weights: List[Float64], k: Int) raises:
    """Move the farthest points into empty clusters (`_relocate_empty_clusters_dense`)."""
    var empty = List[Int]()
    for j in range(k):
        if weights[j] == 0:
            empty.append(j)
    if len(empty) == 0:
        return
    var n = X.shape[0]
    var d = X.shape[1]
    var distances = List[Float64](length=n, fill=0.0)
    var maximum = 0.0
    for i in range(n):
        var total = 0.0
        for f in range(d):
            var delta = rounded[dtype](Float64(X.unsafe_get(i * d + f)) - centers[labels[i] * d + f])
            total += rounded[dtype](delta * delta)
        distances[i] = rounded[dtype](total)
        if distances[i] > maximum:
            maximum = distances[i]
    if maximum == 0:
        return
    var used = List[Bool](length=n, fill=False)
    for e in range(len(empty)):
        var far = -1
        for i in range(n):
            if not used[i] and (far < 0 or distances[i] > distances[far]):
                far = i
        used[far] = True
        var new_cluster = empty[e]
        var old_cluster = labels[far]
        for f in range(d):
            var value = Float64(X.unsafe_get(far * d + f))
            updated[old_cluster * d + f] -= value
            updated[new_cluster * d + f] = value
        weights[new_cluster] = 1.0
        weights[old_cluster] -= 1.0
