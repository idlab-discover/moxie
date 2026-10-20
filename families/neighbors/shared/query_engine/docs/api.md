# Private dense neighbor queries

The engine is under development; read [scope](scope.md) for validation limits.
Run `pixi run start` for [the native example](../examples/query.mojo).

```mojo
var metric = Metric[DType.float64]("euclidean")
var engine = BruteForce(metric, chunk_size=256)
engine.fit(x)
var other_neighbors = engine.kneighbors(None, 3)
var explicit_neighbors = engine.kneighbors(x.view(), 3)
```

Import Metric from `moxie.neighbors._distance` and BruteForce from
`moxie.neighbors._query_engine`. No package-level public estimators are exported.
Use `mojo -I builds/deps` only after dependency assembly; direct source include
paths do not compose the assembled namespace. The locked toolchain is Mojo 1.0.0,
MAX 26.5.0 and NuMojo 0.10.0.

Fit validates and copies the input; caller mutation cannot change the snapshot.
Failed validation preserves the preceding fit. `clone()` copies configuration
into an unfitted engine. Metric parameters are copied at construction. Native
callable structs implement DenseMetric and Deinitable; they must be deterministic
and immutable during queries. Native tests now cover a callable metric, negative returned distances and a
strided offset input. Public estimator callable metrics remain unfinished.

`None` queries omit each fitted row's own index while retaining duplicates.
Explicit arrays include self; use `.view()` to transfer an optional query handle
without a full input copy. Equal distances are ordered by fitted index. This is
a deterministic native adaptation of sklearn's backend-dependent ties.
Distances accumulate in Float64 and results retain the input dtype. Undefined
nan-Euclidean distances sort last; they never satisfy a radius comparison.

Top-k stores O(q*k) output and O(k) numeric scratch; candidates stream in blocks
of `chunk_size`. The current insertion selector costs O(q*n*k). Radius first
collects indices, then recomputes selected distances into an exactly sized array,
so numeric working storage is output-sized. Sorted radius uses in-place insertion
sorting. These choices bound working storage but have not been benchmarked.
The snapshot may copy a strided view's entire backing allocation, per NuMojo.

Set `precomputed=True` on BruteForce for dense distance matrices; fitting requires
a nonnegative finite square matrix and query columns must equal fitted rows.
Set `allow_nan=True` with Metric("nan_euclidean"). These separate typed controls
are provisional; automatic configuration and incompatible-option rejection need
completion. Errors use descriptive Mojo Error strings. Radius must be
nonnegative (positive infinity allowed); sorting requires returned distances.

`distance_weights(flat_distances, offsets)` returns inverse-distance weights for
ragged rows. If a row contains zeros, only those zeros receive weight one.
Uniform weighting can be represented by absence in an estimator adapter;
`NeighborWeights` and `BuiltinWeights` also live at this owner. The public
KNN/radius adapters flatten fixed-width/ragged distances for native callable
weighting; family prediction parity covers these adapters.
