# Dense semi-supervised extension contracts

The family owns `moxie.semi_supervised._contracts`, `_affinity`, `_probability`,
`_label_propagation`, `_self_training`, and `_gaussian_adapter`. Both graph
estimators consume the same affinity and normalization engines. The foundation
capability traits remain unchanged. [Family development](development.md) defines
the repository ownership boundary; [the API guide](api-surface.md) describes the
concrete estimator interfaces.

## Storage and labels

Features use `numojo.NDArray[float32/float64]` with samples in rows. `Shape` and
`Item` remain the numerical owner's metadata types. A fitted graph owns a deep
`.copy()` of input features; `.view()` deliberately shares storage. Strides are
in elements. Strided copies may retain the entire backing allocation. No new
generic array container is introduced. Concurrent caller mutation is unsupported.

`ClassEncoding[T]` owns a nonempty, duplicate-free vocabulary. It preserves the
supplied order. Graph fitting discovers sorted labels; self-training preserves
its estimator's exact class order. `integer_targets` maps integer `-1` to `None`;
other negative integers remain labels. String labels use `Optional[String]`,
with `None` for missing labels. The string `"-1"` is a label. Encoded classes are
`-1` or integers in `[0, n_classes)`. Decoding rejects missing or invalid codes.

`validate_features` checks dense rank, nonempty axes and target length;
`snapshot_features` additionally copies. These helpers do not check finiteness.
Graph fitting checks finite features; self-training delegates that policy to its
estimator. `validate_probability_shape` checks rank and exact row/class axes,
including empty probability batches. Public estimators require nonempty queries.

## Graph operations

`DenseAffinity.Input` is the dense feature type. Its
`affinity(train, query) -> NDArray[float64]` returns an owned matrix of shape
`(n_train, n_query)`. Training calls it with `(X, X)`. Consumers validate output
shape and preserve asymmetric values. Spreading uses column degrees excluding
the diagonal, following SciPy's normalized Laplacian convention. Affinity and
transition normalization are distinct operations.

`ProbabilityNormalizer.normalize(values, zero_mass)` returns owned Float64 data
of the same shape. `ZeroMassPolicy.PRESERVE` replaces exactly zero denominators
with one; `IEEE_DIVIDE` leaves IEEE division unchanged. Unknown policies raise.
There is no epsilon, clipping, calibration, or uniform fallback. Row sums use
`moxie._dense.sum_axis2d`; ordinary finite diffusion uses accepted rank-two
NuMojo matrix multiplication. RBF uses shared squared norms and matrix products.

## Self-training adapters

`ProbabilisticEstimatorAdapter` declares associated `Input` and categorical
`Label`, and requires:

- `clone_unfitted() raises -> Self`, recursively cloning constructor configuration.
- `fit(mut self, X, y: List[Label]) raises`, replacing learned state.
- `classes() raises -> List[Label]`, returning an owned ordered vocabulary.
- `predict_proba(X) raises -> NDArray[float64]`, returning owned probabilities.

`ScoringEstimatorAdapter` optionally adds delegated `score(X, y)`.
`PredictingEstimatorAdapter`, `LogProbabilityEstimatorAdapter`, and
`DecisionEstimatorAdapter` optionally add their namesake methods. An adapter
need not expose unsupported methods. Self-training checks its own fitted state,
even if the constructor received a fitted estimator. Probability values are
consumed directly, without renormalization. Subset matrices and target lists
are owned, and the original estimator is never fitted by the wrapper.

`GaussianNBAdapter` imports the Naive Bayes owner's implementation and the
shared label owner from their current Moxie sources. It widens Float32 probability
outputs to Float64 without changing column order. Numax is explicitly authorized
for this integration. `GraphClassifierAdapter` supports integer and string labels
and serves as another real classifier composition target.

## Assembly and verification

The primary family checks use Moxie's full `DEPENDENCIES.json` assembly.
The two isolated consumers use the focused `DEPENDENCIES.lock.json` closure;
the legacy filename does not pin source revisions. All projects follow current
local sources.
`tools/dependencies.py` reads their declared sources from the working tree and
records HEAD revisions and hashes in `builds/effective-lock.json`. No historical
pins or cached snapshots are used.

Family modules and fixture modules are live inputs; edits do not require
repinning their contents. Each consumer gets one `builds/deps` namespace.
Module and export collisions fail before replacing an existing valid assembly.
Runtime-unneeded RNG and historical readiness evidence are not build
dependencies. Published baseline tags remain unchanged.

Use Mojo 1.0.0 / MAX 26.5.0 with the current local NuMojo source. From Moxie, run
`pixi run semi-supervised-test` and `pixi run semi-supervised-parity`. The latter records oracle revisions,
source hashes, tolerances, flags, and results in ignored `builds/parity`.
