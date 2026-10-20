# Private dense centroid kernels

**Private dense engine:** build, native contracts and numerical parity pass
against the local reference. Public estimators and their fitting policies are
separate work. No full scikit-learn or performance parity is claimed.

Import from `moxie.cluster._centroid_engine`. Package-level public exports are
empty. All numeric inputs are `numojo.NDArray[dtype]`, where dtype is Float32
or Float64. Labels use the family's Int64 `Labels` type. Inputs are immutable
borrows, results own independent storage, and all functions can raise Mojo
`Error` with a `ValueError` prefix. No Python is used in the library runtime.

| API | Behavior |
| --- | --- |
| `distances(x, centers, squared=True)` | Owned `(n,k)` squared or ordinary Euclidean distances |
| `assign(x, centers, weights)` | `AssignmentResult(labels, inertia)` using Lloyd norm/dot scores |
| `inertia(x, centers, weights, labels, single_label=-1)` | Weighted squared-distance sum, optionally restricted to one cluster |
| `update(x, labels, weights, previous)` | `CenterUpdate(centers, masses)` with weighted sums, relocation and normalization |
| `center_shift(old, new)` | Euclidean displacement vector `(k,)` |
| `minibatch_update(x, labels, weights, previous, prior_masses)` | New centers and cumulative masses; zero batch mass retains old centers |
| `Lloyd[dtype]()` | Implements the existing `CentroidAssignment` and `CentroidUpdate` seams |
| `ElkanBounds[dtype](x, centers)` | Owns labels, upper/lower bounds and center snapshot |
| `bounds.assign(x, weights)` | Pruned assignment for the **same unchanged samples**, returning an independent label snapshot |
| `bounds.advance(new_centers)` | Moves the center snapshot and adjusts all bounds for the next assignment |

X has shape `(n,d)`, centers `(k,d)`, weights `(n,)`, labels `(n,)` and
prior masses `(k,)`. Feature and center counts must be positive. Empty sample
batches are allowed for assignment/distances and mini-batch updates; ordinary
center updates require `n >= k >= 1`. Prediction can use `k > n`. Shape, label
bounds and finite inputs are checked. Signed sample weights are accepted at
this private seam, matching the low-level reference; estimator-level weight
rules belong to the wrapper. Concurrent mutation is outside the contract.

Initialize fresh Elkan bounds when samples or their order change, even if the
shape is the same. Bounds are private scratch, not fitted estimator attributes;
callers must not mutate their contents directly. Center snapshots and returned
results may be modified without changing the original caller inputs. Arbitrary
valid NuMojo strides and offsets are read logically; outputs are contiguous,
except the independent Elkan center snapshot retains NuMojo copy layout.

These are typed native return records instead of Cython output-argument arrays.
The engine executes sequentially; it has no `n_threads` knob. Weighted scalar
reductions stay in input dtype, with Float64 inertia at the family seam.
Lloyd center norms use the shared norm kernel for Float64 and same-dtype
Float32 accumulation to preserve assignment ties after streaming updates.
For a three-feature center equal to `55 * Float32(1 / 55)`, widened norms
change a tie on unit rows; the MiniBatchKMeans suite retains this regression. Floating-point
contraction is disabled. These arithmetic differences need bounded numerical
tolerances, not bitwise claims. Public initialization, convergence, RNG draws,
mini-batch reassignment schedules, EWA statistics and estimator lifecycle remain
with KMeans/MiniBatchKMeans, as assigned by the family plumbing.

Relocation preserves `np.argpartition(residuals, -n_empty)` followed by reversal
of the selected suffix. It does not sort all selected samples by distance.
The private `_centroid_selection` module reproduces the four-lane AVX2 ordering
used by the locked NumPy 2.5.3 oracle, with portable Mojo operations: sorting
networks for at most 256 entries, median-of-four partitioning above that cutoff,
and an introsort fallback with the oracle's GNU `std::sort` tie policy. Empty
clusters receive samples in ascending cluster-ID order; sample weights do not
change selection, and assignment labels remain unchanged during relocation.

`count_order` reuses these primitives for ascending cluster-weight sorting,
including the already-sorted fast path. MiniBatchKMeans uses it to cap
low-count reassignments. Both sorts share the same tie-order limitation.

NumPy does not specify partition ordering within either side, and CPU dispatch
can change it. This engine fixes that ordering independently of the host CPU;
exact tie/cluster-ID parity is validated against the four-lane reference, not
promised for other NumPy versions, SIMD backends or C++ standard libraries.
For X=[1,1,0,0], zero centers and weights=[1,2,1,1], the three cluster masses
are [2,1,2]. Input buffers remain borrowed and unchanged. Native runtime code
uses no Python or C++; the small C++ fallback oracle is test tooling only.

Build: `PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run --locked build`.
Run the example after building with `builds/weighted_step`.

The private `update` kernel accepts `allow_singleton=True` for the specific
BisectingKMeans case of one row and two centers. Default shape validation is
unchanged. This path still performs weighted normalization and relocation,
preserving floating-point behavior for fractional singleton weights.
