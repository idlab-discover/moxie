# Mutual-information feature scores

`mutual_info_classif` and `mutual_info_regression` return an owned Float64
vector with one nonnegative score per feature, in input column order. Scores
are measured in nats. Each feature is scored independently; these functions do
not evaluate a combined subset or supply p-values.

```mojo
from moxie.feature_selection import mutual_info_classif, DiscreteFeatures
from moxie.feature_selection import MutualInfoScorer, SelectKBest

# X: mutable NuMojo Float32/Float64 matrix; y: numeric target vector.
var scores = mutual_info_classif(X, y, random_state=42)
var selector = SelectKBest(
    MutualInfoScorer[DType.float64](random_state=42), k=2
)
selector.fit(X, y.copy())
var selected = selector.transform(X)
```

Use `MutualInfoScorer(classification=False, ...)` for regression. It also works
with SelectPercentile and GenericUnivariateSelect's count/percentile modes.
P-value policies reject its absent p-values. Scorer configuration is copied on
clone; ordinary selector fitting replaces learned scores and selection state.
Transform and inverse_transform use the existing SelectionState implementation.

Both functions have these keyword defaults:

| Parameter | Default and behavior |
| --- | --- |
| `discrete_features` | `DiscreteFeatures(String("auto"))`, all continuous for dense X; accepts tagged Bool, List[Bool] mask or List[Int] column indices, including negative indices |
| `n_neighbors` | 3; positive integer, bounded by available samples for continuous/continuous estimation; mixed estimation caps it within each repeated label |
| `copy` | True; False overwrites continuous Float64 columns with scaled, jittered values; Float32 features are widened into independent Float64 storage |
| `random_state` | None, the shared process RandomState stream; an integer repeats the legacy NumPy sequence, or pass a mutable shared RandomState to advance it |
| `n_jobs` | None; nonzero integers accepted, computation currently serial regardless of requested job count |

`MutualInfoScorer` exposes these options except `copy`: it always preserves its
borrowed input. Its `classification` parameter defaults to True and its
`random_state` is an optional integer. Mutable-stream configuration for the
scorer is unavailable; the standalone functions accept mutable streams.

The standalone functions require `mut X` because `copy=False` can modify it.
Targets are borrowed and preserved. Features use Float32 or Float64. Regression targets use Float32 or Float64;
classification targets also accept exact Int64 arrays and native
`moxie._labels.StringTarget`. Numeric classification labels must be integral.
Standalone functions and `MutualInfoScorer` parameterize feature and target
dtypes independently. Python objects and implicit coercion are unavailable.
Targets may be rank one or a single-column matrix. Both dtypes support strided
input and target arrays; no borrowed aliases escape. Empty arrays, nonfinite
values, incompatible shapes and invalid settings raise descriptive Mojo Errors.
Python warning transport is unavailable, including warnings for continuous
values deliberately marked discrete and single-column targets.

The implementation follows the reference's contingency formula for two discrete
variables, Kraskov estimator for continuous pairs, and Ross estimator for mixed
pairs. Mixed estimation excludes singleton labels; if none remain, it raises.
Continuous variables are scaled without centering and receive small Gaussian
noise in the reference's traversal order. Integer digamma terms are evaluated
using harmonic numbers: Euler's constant cancels from the MI formulas. Scaling
uses the approved shared reductions; Float32 intermediate accumulation is
widened and may differ in low bits from NumPy. Discontinuous neighbor counts
mean numerically indistinguishable ties need not yield identical estimates on
all extreme inputs.

Nearest/radius queries consume the neighbors family's live native query engine,
including its spatial-tree imports. Shared RNG and neighbor modules consume
the declared mojo-linalg owner through their SVD/covariance support; MI does not call those solvers. There is no tree-search backend or speed
claim. The algorithm has quadratic sample-count cost; this implementation is
intended for correctness and moderate data sizes. Parallel feature evaluation
and tree acceleration remain future work. Sparse inputs remain explicitly
deferred; `auto` never silently densifies them.

Validation: `pixi run build`, `pixi run test`, `pixi run parity`. Both isolated
consumer assemblies execute the public scorer. The MI parity harness executes
unchanged functions from scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, supplying independent SciPy distance
queries and NumPy scaling because compiled sklearn extensions are unavailable.
It compares classification/regression, all three estimator paths, neighbor
counts, Float32/Float64, ties, constants, singleton labels, mixed masks, negative
strides, independently typed targets and `copy=False` buffers. Bounded-fixture
tolerance is rtol=1e-11, atol=2e-12; this accommodates summation order, not general
statistical estimation error. Native contracts separately test validation, RNG
streams, ownership, repeat fitting, cloning and selector composition. This is
not a claim of full sklearn estimator-validation or Python API parity.

References: [Kraskov et al.](https://arxiv.org/abs/cond-mat/0305641),
[Ross](https://doi.org/10.1371/journal.pone.0087357), and the local
`sklearn/feature_selection/_mutual_info.py` and its tests.

Native label examples for selector adapters:

```mojo
# Float32 features with Float64 targets (classification or regression).
var numeric = MutualInfoScorer[DType.float32, DType.float64](random_state=7)
# Exact integer class identities, even above 2**53.
var integers = MutualInfoScorer[DType.float32, DType.int64](random_state=7)
# Import StringTarget from moxie._labels.
var strings = MutualInfoScorer[DType.float32, target_type=StringTarget](random_state=7)
```

The shared label engine encodes classes before MI converts class IDs to Float64.
Both random-state overloads accept native labels. String and Int64 targets in
the adapter require `classification=True`; regression rejects them explicitly.
SelectKBest, SelectPercentile and GenericUnivariateSelect ranking modes work
with these scorers. SelectFpr, SelectFdr, SelectFwe and GenericUnivariateSelect's
p-value modes raise when support or a transform is requested, because MI does
not supply p-values. Fitting still records the MI scores.

`tests/feature_selection/parity_labels.py` adds native string/Int64 comparisons against the pinned
reference, including continuous and discrete feature estimators, both seeded
random-state overloads, exact masks and clone fitting. Native contracts cover
invalid targets, refitting, and independently typed numeric MI adapters.
