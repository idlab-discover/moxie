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
trait. A preference list selects the first supported response. Probability
and decision outputs are restored to original target order.
`predict_log_proba` computes the logarithm of the final positive probabilities.
Chain CV configuration accepts an integer fold count, a native `KFold` or
`StratifiedKFold`, fixed `ChainFolds`, a typed `ChainSplitter`, or `None`.
A per-fit override can also use a custom `ChainResponses` provider.
Native CV clones components per fold and writes held-out responses in
original sample order.

## Current limits

- Inputs are nonempty dense numeric matrices. Feature dtype is a compile-time
  Float32 or Float64 parameter. Targets are numeric NDArray matrices; numeric
  classification labels retain their dtype. String/object labels are not yet
  supported and are not converted to floats.
- Independent fits use native CPU threads. The [API](api-surface.md)
  describes worker counts, ordering and failure behavior. Sparse inputs
  remain deferred.
- Random chain order uses `random_order=True`, with an optional integer
  `random_state`. Mutable RNG objects are unsupported.
- The default `cv=None` uses true preceding targets. Native splitter and
  class-response diagnostics use Mojo errors for invalid folds and handle
  undersized classes silently.
- Incremental sample weights, arbitrary fit metadata, Python reflection,
  metadata routing and feature-name propagation are not implemented.
- Component inputs must match the wrapper's feature array type. Regressor
  component targets and predictions use the declared NDArray type. Classifier
  components also accept a list of scalars of the declared label dtype, or
  `List[Int]` for Int64 wrapper labels. Incremental classifier components
  use the same type for their targets and class declarations. A component
  using Dummy's tagged `List[Label]` accepts Int64, Float32 or Float64 labels.
  Component probabilities and log probabilities can be Float32 or Float64 NDArrays,
  or a one-element list containing a Float64 NDArray.
  Component decisions use Float32 or Float64 NDArrays. Wrapper probabilities,
  log probabilities and decisions are Float64:
  sklearn's chains return Float64 matrices, and `MultiOutputClassifier` keeps
  each component's values exactly (a Float32 component's probabilities are
  widened, where sklearn would return Float32 arrays). Chain feature responses
  use the feature dtype; sklearn's `hstack` promotes Float32 features and
  float64 responses to a Float64 augmented matrix, so a chain over a Float32
  component that does not cast its input itself fits in Float32 here and in
  Float64 in sklearn.
  Wrapper sample weights are Float64 and are narrowed for Float32-weight
  components. Classifier wrappers also accept components with List[Float64]
  weights, validating the wrapper weights as a vector before conversion.
  Class ordering follows sorted observed labels, or the complete class vectors
  passed to first `partial_fit`.
  Components retaining additional classes or a different probability-column
  order are unsupported because the wrappers derive class order from the targets.

These limits mean this is not full scikit-learn parity. See
[API usage](api-surface.md) for native calls and [plumbing](plumbing.md) for the
reused layout helpers.

## Validation

The [multioutput test guide](README.md) lists scoped contract and parity
commands. The contract checks include isolated consumers, live-source
provenance and missing required capabilities. The parity runner compares
predictions with the recorded local sklearn module,
including chains with two-fold out-of-fold responses. It also compares
independent and chain probabilities, chain log probabilities and probability
feature responses, seeded chain order, subset accuracy, weighted accuracy,
uniform-average R2 and weighted fitting. Discrete predictions and orders
must match exactly. Floating-point comparisons use the component-specific
tolerances recorded in the [multioutput test guide](README.md). Test-only
components include a one-nearest-neighbor implementation and a weighted mean.
These comparisons establish wrapper behavior for the tested cases, not parity
with every possible base estimator. No performance claim is made.
