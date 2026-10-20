# Dense centroid engine scope

Status: dense private kernels pass the documented contract and numerical
parity suites. Public clustering estimators and full sklearn parity are not
provided by this engine.

Implemented private Float32/Float64 dense Euclidean distances, weighted Lloyd
assignment, weighted center updates, empty-cluster relocation, center shifts,
Elkan initialization/pruning/bound advancement, and mini-batch center/mass
statistics. The native contracts cover weights, ties, zero-mass duplicate
clusters, shape/rank/label/nonfinite errors, empty prediction, k>n prediction,
repeated mini-batches and independent snapshots. Differential tests cover
positive/negative-stride and offset inputs, four iterations, one center,
coincident centers, zero weights, and fractional weights. See [api.md](api.md).

Sparse inputs, CSR/CSC/COO conversions and the sparse branches in sklearn
cluster/_k_means_common.pyx, _k_means_lloyd.pyx, _k_means_elkan.pyx and
_k_means_minibatch.pyx are deferred by session-plan/SPARSITY-SCOPE.md.
The inherited numerical and cluster contracts impose the same restriction.
No sparse import or automatic densification is provided. The negative import
contract verifies that `assign_sparse` is absent.

This engine exports no public estimator constructors; there are no unsupported
estimator defaults or required dense constructor selections here. Initialization,
fit convergence and mini-batch sampling/reassignment schedules belong to the
assigned downstream estimator projects.

Executed commands use `PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache` and the immutable
Mojo 1.0.0/MAX 26.5.0 environment:

- `pixi run --locked build`: compilation against current Moxie sources.
- `pixi run --locked test`: native contracts and two isolated consumers pass;
  verifies shared owners, rejected missing sources/module conflicts, and absent
  sparse API.
- `pixi run --locked parity`: 140 dense kernel cases, four iterations each;
  2,738 exact selection/weighted-relocation cases; and 352 emergency sorting
  fallback cases. Both Float32 and Float64 are exercised. The local Cython
  kernels and NumPy run unchanged, with one thread. Labels, selected indices
  and relocation masses compare exactly. Other bounded numerical outputs use
  rtol=atol=3e-5 for Float32 and 1e-12 for Float64 to allow reduction rounding.

Selection cases exhaust four-element duplicate/tie patterns and cover zeros,
all-equal residuals, sorted/reversed/rotated/organ-pipe data and random ties,
including sizes around vector and 256-entry network boundaries through 4,097
samples. Unequal fractional weights make selection order observable in masses
and centers. The standard-library fallback is compared with a test-only C++
`std::sort` program. A C++17 compiler is therefore required for parity tooling.
See [api.md](api.md) for the NumPy backend ordering scope.

The minimized regression also runs as a native contract in both dtypes:
X=[1,1,0,0], three zero centers, weights=[1,2,1,1] produces masses=[2,1,2].
`pixi run --locked python tools/repro_relocation.py` compares it directly with
the Cython oracle after building. Performance benchmarks and broader numerical
exception/cost measurements remain separate work; no performance claim is made.
