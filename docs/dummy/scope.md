# Dense-only scope

The authorized scope is dense input, dense targets and dense outputs for
`DummyClassifier` and `DummyRegressor`. All five classification strategies and
all four regression strategies are implemented, with weights, multioutput,
class ordering, seeded sampling, scoring, cloning and regression `return_std`.
Native label and compile-time single-/multioutput adaptations are specified
in [api.md](api.md). Scalar constants and flat classifier scoring are supported.
Numerical inputs use NuMojo Float32 or Float64 storage, selected by the
estimator’s compile-time `dtype` parameter (Float64 by default). This is not a claim of all Python binding parity.

Sparse input/targets, CSC/CSR/COO representations, sparse class distributions,
sparse prediction and the classifier's sparse-uniform target conversion are
**deferred**. They have no typed native API;
there is no silent densification. Source: pinned scikit-learn revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b, `sklearn/dummy.py`
(`DummyClassifier.fit`, `predict`), `sklearn/utils/multiclass.py`
(`class_distribution`), and `sklearn/tests/test_dummy.py` sparse tests.
Regression sparse constants are also deferred; numerical constants are dense
vectors. No public estimator export is deferred.

The inherited foundation excludes Python reflection, pickle serialization, pandas and
metadata routing. Shared numerical storage supports the pinned Mojo 1.0.0 /
MAX 26.5.0 / NuMojo 0.10.0 combination. Shared random state is consumed from
foundation.random; no `_sparse` capability is required or implemented.

Default prior/mean strategies work on dense data without alternate settings.
Stochastic strategies with default `random_state=None` use Moxie's shared
process-wide MT19937 stream. Integer seeds and explicit mutable contexts are
also supported; see [api.md](api.md) for ownership and synchronization.

## Validation and provenance

The implementation and schema adapters were imported from `moxie-dummy`
revision `375532accca9f942e0745271def6e2aea6afa906`. The original checkout
remains a baseline; Moxie owns the integrated sources and tests. The retained
scikit-learn license is in `LICENSES/scikit-learn.txt`. Shared RNG and external
numerical dependencies retain their existing owners and license assets.

Run `pixi run test` for native lifecycle, parameter ownership, unfitted errors,
repeat fitting, invalid inputs, dtype, strided-view, persistence and unsupported
API checks. Run `pixi run parity` for the migrated differential suite against
the pinned scikit-learn source. It covers all strategies, single/multioutput
shapes, weighted statistics, quantile endpoints, zero/signed weights, labels,
seeded/advancing streams, probabilities, scores and standard deviations.
Tolerances are documented in [api.md](api.md). Raw oracle and parity evidence
is written under ignored `docs/private/dummy.estimators/`.

The old family assembly harness and performance drivers remain in the
standalone baseline; Moxie uses its existing shared assembly engine.
No competitive-performance claim is made.

Native persistence is available through family-owned experimental schema adapters
for both dense estimators; the shared container and optional Snappy codec remain
in Moxie. See [the API guide](api.md#experimental-native-persistence). No pipeline
schema, sparse persistence or external/global RNG stream capture is supplied.
