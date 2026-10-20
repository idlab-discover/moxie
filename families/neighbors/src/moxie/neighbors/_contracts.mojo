"""Dense query plumbing only: no distance or search algorithm."""
import numojo as nm
from std.math import isfinite, isnan


def validate_matrix[dtype: DType](
    x: nm.NDArray[dtype], expected_columns: Int = -1,
    precomputed: Bool = False, training: Bool = False,
    allow_nan: Bool = False,
) raises:
    """Validate borrowed data; preserve strides and caller ownership."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected rank-2 dense input")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise Error("ValueError: expected nonempty samples and features")
    if expected_columns >= 0 and x.shape[1] != expected_columns:
        raise Error("ValueError: query columns do not match fitted input")
    if precomputed and training and x.shape[0] != x.shape[1]:
        raise Error("ValueError: precomputed fit distances must be square")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            if not isfinite(value) and not (allow_nan and not precomputed and isnan(value)):
                raise Error("ValueError: input contains nonfinite values")
            if precomputed and value < 0:
                raise Error("ValueError: precomputed distances must be nonnegative")


def fit_snapshot[dtype: DType](
    x: nm.NDArray[dtype], precomputed: Bool = False, allow_nan: Bool = False,
) raises -> nm.NDArray[dtype]:
    """Return an independent fitted snapshot; caller commits only after success."""
    validate_matrix(x, precomputed=precomputed, training=True, allow_nan=allow_nan)
    return x.copy()


def validate_k(k: Int, n_samples_fit: Int, query_is_train: Bool) raises:
    if n_samples_fit <= 0:
        raise Error("NotFittedError: no indexed samples")
    if k <= 0 or k > n_samples_fit - Int(query_is_train):
        raise Error("ValueError: n_neighbors exceeds available candidates or is nonpositive")


def validate_radius_options(radius: Float64, return_distance: Bool, sort_results: Bool) raises:
    if isnan(radius) or radius < 0:
        raise Error("ValueError: radius must be nonnegative")
    if sort_results and not return_distance:
        raise Error("ValueError: return_distance must be True if sort_results is True")


@fieldwise_init
struct TopKResult[dtype: DType](Copyable, Movable):
    """Flat row-major indices; optional (n_queries, k) actual distances."""
    var n_queries: Int
    var k: Int
    var indices: List[Int]
    var distances: Optional[nm.NDArray[Self.dtype]]

    def validate(self, n_samples_fit: Int, allow_nan: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if self.n_queries < 0 or self.k <= 0 or n_samples_fit <= 0:
            raise Error("ValueError: invalid top-k dimensions")
        # Division avoids overflowing n_queries * k on malformed input.
        if len(self.indices) // self.k != self.n_queries or len(self.indices) % self.k != 0:
            raise Error("ValueError: top-k index length mismatch")
        for idx in self.indices:
            if idx < 0 or idx >= n_samples_fit:
                raise Error("ValueError: neighbor index outside fitted samples")
        if self.distances:
            ref d = self.distances.value()
            if d.ndim != 2:
                raise Error("ValueError: top-k distances must be rank 2")
            if d.shape[0] != self.n_queries or d.shape[1] != self.k:
                raise Error("ValueError: top-k distance shape mismatch")
            for i in range(self.n_queries):
                var previous = Float64(-1)
                var seen_nan = False
                for j in range(self.k):
                    var value = Float64(d.unsafe_get(i * Int(d.strides[0]) + j * Int(d.strides[1])))
                    if isnan(value):
                        if not allow_nan:
                            raise Error("ValueError: undefined distance requires allow_nan")
                        seen_nan = True
                    else:
                        if seen_nan or value < 0 or value < previous:
                            raise Error("ValueError: distances must be nonnegative and nondecreasing, NaNs last")
                        previous = value


@fieldwise_init
struct RadiusResult[dtype: DType](Copyable, Movable):
    """Ragged rows as offsets and indices; optional flat actual distances."""
    var offsets: List[Int]
    var indices: List[Int]
    var distances: Optional[nm.NDArray[Self.dtype]]

    def validate(self, n_samples_fit: Int, sorted_distances: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if len(self.offsets) == 0 or n_samples_fit <= 0:
            raise Error("ValueError: invalid ragged dimensions")
        if self.offsets[0] != 0 or self.offsets[len(self.offsets) - 1] != len(self.indices):
            raise Error("ValueError: invalid ragged endpoints")
        for i in range(1, len(self.offsets)):
            if self.offsets[i] < self.offsets[i - 1]:
                raise Error("ValueError: offsets must be nondecreasing")
        for idx in self.indices:
            if idx < 0 or idx >= n_samples_fit:
                raise Error("ValueError: neighbor index outside fitted samples")
        if sorted_distances and not self.distances:
            raise Error("ValueError: sorted radius results require distances")
        if self.distances:
            ref d = self.distances.value()
            if d.ndim != 1:
                raise Error("ValueError: radius distances must be rank 1")
            if d.shape[0] != len(self.indices):
                raise Error("ValueError: radius distance length mismatch")
            for i in range(len(self.offsets) - 1):
                var previous = Float64(-1)
                for j in range(self.offsets[i], self.offsets[i + 1]):
                    var value = Float64(d.unsafe_get(j * Int(d.strides[0])))
                    if isnan(value) or value < 0 or (sorted_distances and value < previous):
                        raise Error("ValueError: invalid radius distance ordering or value")
                    previous = value


trait DenseMetric(Copyable, Movable):
    """Native callable seam. Implementations and typed parameters belong to query_engine."""
    comptime dtype: DType

    def validate_features(self, n_features: Int) raises:
        ...

    def distance(self, x: nm.NDArray[Self.dtype], x_row: Int,
                 y: nm.NDArray[Self.dtype], y_row: Int) raises -> Float64:
        ...


trait DenseQueryBackend(Movable):
    """Private backend seam; public estimator adapters retain existing traits."""
    comptime dtype: DType

    def fit(mut self, x: nm.NDArray[Self.dtype]) raises:
        ...

    def kneighbors(self, x: Optional[nm.NDArray[Self.dtype]], k: Int,
                   return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        ...

    def radius_neighbors(self, x: Optional[nm.NDArray[Self.dtype]], radius: Float64,
                         return_distance: Bool = True, sort_results: Bool = False,
                         ) raises -> RadiusResult[Self.dtype]:
        ...
