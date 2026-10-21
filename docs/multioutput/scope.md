# Dense multioutput estimators

The family implements `MultiOutputClassifier`, `MultiOutputRegressor`,
`ClassifierChain`, and `RegressorChain` in pure Mojo. Components are supplied
by the caller and must implement the shared estimator and prediction traits.
The wrappers do not contain a base learning algorithm.

Supported behavior includes independent configuration cloning, replacement on
fit, prediction assembly, typed nested parameter records, subset accuracy and
uniform-average R2. Independent wrappers conditionally support incremental
fitting. Weighted fit delegates to components with `WeightedSupervisedEstimator`.
Independent classifier probabilities are a list of matrices, one per output.

Chains support natural, explicit and integer-seeded random orders, teacher
forcing during ordinary fit, and predicted feature augmentation at inference.
Classifier chains support `predict`, `predict_proba`, `predict_log_proba`, and
`decision_function` feature responses when the component exposes the matching
trait. Probability and decision outputs are restored to original target order.
`predict_log_proba` computes the logarithm of the final positive probabilities.
CV fitting accepts an explicit `ChainResponses` provider; the provider owns
fold construction, independent fold clones and out-of-fold coverage.

## Current limits

- Inputs are nonempty dense numeric matrices. Feature dtype is a compile-time
  Float32 or Float64 parameter. Targets are numeric NDArray matrices; numeric
  classification labels retain their dtype. String/object labels are not yet
  supported and are not converted to floats.
- Independent execution is serial. `n_jobs` follows joblib: `0` raises
  `n_jobs == 0 in Parallel has no meaning`; every other value fits the
  components one after another, which gives the same model as sklearn's
  parallel fit. Sparse inputs remain deferred.
- Random chain order uses `random_order=True`, with an optional integer
  `random_state`. Mutable RNG objects and response-method preference lists
  are not supported.
- CV is passed to `fit(X, y, cv=provider)` rather than stored as a Python-style
  constructor object. Integer CV values, Python splitters and `prefit` are
  not accepted. The default `fit(X, y)` uses true preceding targets.
- Incremental sample weights, arbitrary fit metadata, Python reflection,
  metadata routing and feature-name propagation are not implemented.
- Component inputs, targets and predictions must match the wrapper's declared
  array types. Component probabilities and decisions must be Float32 or Float64
  NDArrays. Wrapper probabilities, log probabilities and decisions are Float64:
  sklearn's chains return Float64 matrices, and `MultiOutputClassifier` keeps
  each component's values exactly (a Float32 component's probabilities are
  widened, where sklearn would return Float32 arrays). Chain feature responses
  use the feature dtype; sklearn's `hstack` promotes Float32 features and
  float64 responses to a Float64 augmented matrix, so a chain over a Float32
  component that does not cast its input itself fits in Float32 here and in
  Float64 in sklearn.
  Wrapper sample weights are Float64 and are narrowed for Float32-weight
  components. Other component response or weight types fail at compile time.
  Class ordering follows sorted observed labels, or the complete class vectors
  passed to first `partial_fit`.
  Components retaining additional classes or a different probability-column
  order are unsupported because their traits do not expose fitted class metadata.

These limits mean this is not full scikit-learn parity. See
[API usage](api-surface.md) for native calls and [plumbing](plumbing.md) for the
reused layout helpers.

## Validation

Moxie root `pixi run build` compiles the assembled package, contract tests, estimator tests,
examples and two isolated consumers. `pixi run test` executes them and checks
namespace conflicts, live-source provenance and missing required capabilities.
`pixi run parity` compares predictions with the recorded local sklearn module,
including chains with two-fold out-of-fold responses. It also compares
independent and chain probabilities, chain log probabilities and probability
feature responses, seeded chain order, subset accuracy, weighted accuracy,
uniform-average R2 and weighted fitting. Predictions, probabilities and orders
must match exactly; score reductions use absolute and relative tolerances of
1e-12 for Float64 rounding. The test components are an exact one-nearest-neighbor
implementation and a weighted mean, kept exclusively in tests.
These comparisons establish wrapper behavior for the tested cases, not parity
with every possible base estimator. No performance claim is made.
