"""Dense clustering exchange contracts; no assignment, linkage or query engine."""
from numojo import NDArray

comptime Labels = NDArray[DType.int64]


def validate_matrix[dtype: DType](x: NDArray[dtype]) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected rank-2 dense samples")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise Error("ValueError: expected nonempty samples and features")


def validate_centers[dtype: DType](x: NDArray[dtype], centers: NDArray[dtype]) raises:
    validate_matrix(x)
    validate_matrix(centers)
    if centers.shape[1] != x.shape[1]:
        raise Error("ValueError: center feature count differs from input")


def snapshot_centers[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    validate_matrix(x)
    return x.copy()


def validate_labels(labels: Labels, n_samples: Int, n_clusters: Int, minimum: Int = 0) raises:
    """Minimum=0 ordinary; -1 noise; -3 HDBSCAN nonfinite rows."""
    if n_samples < 0 or n_clusters < 0 or (minimum != 0 and minimum != -1 and minimum != -3):
        raise Error("ValueError: invalid label policy")
    if labels.ndim != 1 or labels.size != n_samples:
        raise Error("ValueError: labels must have shape (n_samples,)")
    for i in range(n_samples):
        var label = Int(labels.item(i))
        if label < minimum or label >= n_clusters:
            raise Error("ValueError: cluster label outside declared range")


@fieldwise_init
struct CentroidResult[dtype: DType](Copyable, Movable):
    var labels: Labels
    var centers: NDArray[Self.dtype]
    var inertia: Float64
    var n_iter: Int


@fieldwise_init
struct MergeTree(Copyable, Movable):
    var children: NDArray[DType.int64]
    var n_leaves: Int
    var n_connected_components: Int
    var parents: Optional[NDArray[DType.int64]]
    var distances: Optional[NDArray[DType.float64]]

    def validate(self) raises:
        if self.n_leaves < 1 or self.n_connected_components < 1 or self.n_connected_components > self.n_leaves:
            raise Error("ValueError: invalid leaf/component counts")
        if self.children.ndim != 2 or self.children.shape[1] != 2:
            raise Error("ValueError: children must have shape (n_merges, 2)")
        var merges = self.children.shape[0]
        if merges >= self.n_leaves:
            raise Error("ValueError: too many merges")
        var used = List[Bool]()
        for _ in range(self.n_leaves + merges):
            used.append(False)
        for i in range(merges):
            for j in range(2):
                var child = Int(self.children.item(i, j))
                if child < 0 or child >= self.n_leaves + i:
                    raise Error("ValueError: merge child must precede its parent")
                if used[child]:
                    raise Error("ValueError: merge child used twice")
                used[child] = True
        if self.distances:
            if self.distances.value().ndim != 1 or self.distances.value().size != merges:
                raise Error("ValueError: distances must have shape (n_merges,)")
        if self.parents:
            if self.parents.value().ndim != 1 or self.parents.value().size != self.n_leaves + merges:
                raise Error("ValueError: parents must have shape (n_nodes,)")
            for i in range(self.n_leaves + merges):
                var parent = Int(self.parents.value().item(i))
                if parent < i or parent >= self.n_leaves + merges:
                    raise Error("ValueError: invalid parent index")
                if not used[i] and parent != i:
                    raise Error("ValueError: roots must parent themselves")
            for i in range(merges):
                for j in range(2):
                    var child = Int(self.children.item(i, j))
                    if Int(self.parents.value().item(child)) != self.n_leaves + i:
                        raise Error("ValueError: parents disagree with children")


@fieldwise_init
struct Neighborhoods(Copyable, Movable):
    """Ragged query indices, not a sparse matrix or a neighbor search."""
    var indices: List[List[Int]]
    var n_reference: Int

    def validate(self, n_queries: Int) raises:
        if self.n_reference < 0 or n_queries < 0 or len(self.indices) != n_queries:
            raise Error("ValueError: invalid neighborhood dimensions")
        for row in self.indices:
            for index in row:
                if index < 0 or index >= self.n_reference:
                    raise Error("ValueError: neighbor index out of bounds")


trait DenseDistanceAdapter:
    comptime dtype: DType
    def pairwise(self, x: NDArray[Self.dtype], y: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        ...


trait RadiusQueryAdapter:
    comptime dtype: DType
    def radius_neighbors(self, x: NDArray[Self.dtype], radius: Float64) raises -> Neighborhoods:
        ...


@fieldwise_init
struct AssignmentResult(Copyable, Movable):
    var labels: Labels
    var inertia: Float64


@fieldwise_init
struct CenterUpdate[dtype: DType](Copyable, Movable):
    var centers: NDArray[Self.dtype]
    var masses: NDArray[Self.dtype]


trait CentroidAssignment:
    comptime dtype: DType
    def assign(self, x: NDArray[Self.dtype], centers: NDArray[Self.dtype], weights: NDArray[Self.dtype]) raises -> AssignmentResult:
        ...


trait CentroidUpdate:
    comptime dtype: DType
    def update(self, x: NDArray[Self.dtype], labels: Labels, weights: NDArray[Self.dtype], previous: NDArray[Self.dtype]) raises -> CenterUpdate[Self.dtype]:
        ...
