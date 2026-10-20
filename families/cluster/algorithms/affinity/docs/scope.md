# Dense-only scope

Assigned exports: `moxie.cluster.AffinityPropagation` and
`moxie.cluster.affinity_propagation`. Supported modes are dense Float32/Float64
Euclidean features and precomputed similarities, scalar/vector/median
preferences, damping, convergence/max-iteration limits, seeded or explicitly
borrowed RNGs, both copy modes, `fit`, `fit_predict`, Euclidean `predict`, typed
parameter updates and unfitted cloning. [api.md](api.md) specifies language
adaptations and diagnostics. No statistical score is defined by this estimator.

Deferred: sparse Euclidean training and prediction inputs (CSR conversion in
sklearn `_affinity_propagation.py` fit/predict); all shared CSR/CSC/COO storage,
sparse conversions and arithmetic, per `session-plan/SPARSITY-SCOPE.md`.
Precomputed sparse similarities are rejected by sklearn itself. The public
native API accepts only NuMojo dense arrays; it never silently densifies.
There are no deferred public exports. The cluster and numerical owners
have the same dense-only restriction.

The default Euclidean mode is supported without a dense-mode override.
`random_state=None` requires an explicit retained application RNG argument;
use `random_state=0` for a self-contained deterministic call. Precomputed
`copy=False` requires the mutable `fit_inplace` method. These are language
adaptations, not excluded numerical branches. Python bindings/reflection and
arbitrary array-like conversion are separate interface work.

Required commands (consumer-local cache):

```sh
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run build
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run test
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run parity
```

Native contracts cover dtype, shape, unfitted/error handling, refits, clone and
preference ownership, mutable RNG advancement, copy mutation, negative strides,
nonconvergence with/without exemplars, singleton and all-equal inputs. The
negative sparse import confirms unsupported capability. Two isolated consumers
reuse the same current dense/RNG/family modules; missing sources and export/ownership
collisions are rejected. Differential checks compare local sklearn engine and
wrapper behavior, including asymmetric similarities, preference vectors, seeds,
damping/window combinations, negative strides, dtypes and scaled data. Exact
labels/iterations are required except the tightly checked Euclidean
near-tied-exemplar adaptation documented in api.md. Build reads live sources and compiles every owned module and executable.

Executed successfully on 2026-09-18: all three commands above. Parity passed
298 Float32/Float64 cases; two Euclidean cases exercised the documented
near-tied-exemplar rule. The six-point example produced labels 0,0,0,1,1,1.
Negative capability, two-consumer composition and missing-source/conflict tests passed.
A two-process pyperf baseline and GNU time peak-RSS measurements were also
executed with seed 42, max_iter=40 and one CPU; raw results remain ignored in
benchmarks/. No sparse compatibility or general sklearn parity is claimed.
