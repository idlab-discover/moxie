# Dense neural-network plumbing

This family owns dense storage contracts, RBM and private MLP parameter helpers.
See [scope](scope.md) for supported behavior and deferrals.

## Owners and dependencies

Moxie owns the family under `families/neural_network`. Root sources supply `_base`,
`_plumbing`, `_contracts` and raising MLP declarations. `algorithms/rbm` owns
`_rbm`; `shared/mlp_engine` owns `_mlp_engine`. Forward/backpropagation,
activation/loss and optimizer implementations remain unfinished.

## Exact storage and helper interfaces

All numerical arrays use `numojo.NDArray[dtype]`, with `dtype` selected at compile
time as `DType.float32` or `DType.float64`. `Shape` and `Item` are NuMojo metadata;
there is no family array wrapper. Integer label vocabularies/string lists are
categorical metadata, not numeric tensor substitutes. Public label codecs belong
to MLP, while the private engine receives numeric encoded targets.

| Interface in `_plumbing` | Contract |
| --- | --- |
| `validate_matrix[dtype](X: NDArray[dtype]) raises` | Borrow rank-2, nonempty, finite X; respect element strides and offset; no mutation/allocation |
| `snapshot_matrix[dtype](X) raises -> NDArray[dtype]` | Validate then `.copy()`, independent backing allocation, same dtype/layout |
| `LayerLayout(units: List[Int]) raises` | Copy widths including input/output; at least two strictly positive widths; empty hidden topology is valid |
| `LayerTensors[dtype](layout, coefs: List[NDArray[dtype]], intercepts: List[NDArray[dtype]]) raises` | Check all counts/ranks/shapes, then deep-copy each numeric array |
| `validate_incremental(solver: String, early_stopping: Bool) raises` | Accept SGD/Adam without early stopping; reject every other combination |

Let `L = len(units)-1`. `LayerTensors.coefs[l]` is `(units[l], units[l+1])`;
`intercepts[l]` is `(units[l+1],)`, with exactly L arrays in each list. The same
record represents gradients, velocity and Adam moments. Metadata lists hold
layer arrays; they never replace NuMojo element storage. Engine implementations
expose estimator attributes `coefs_`/`intercepts_` from this state and avoid
retaining caller-owned mutable aliases. Helper validation does not assert finite
gradients, fitted status, class compatibility or complete estimator parameters.
The numerical engine must check post-update parameters for finite values.

Input X is `(n_samples,n_features)`; internal y is `(n_samples,n_outputs)` even
for one output; optional sample weights are a rank-1 NDArray `(n_samples,)` of
the same dtype. Absence means unit weights. Workspace has L+1 activation arrays
`(batch,units[l])` followed by L delta arrays `(batch,units[l+1])` in a single
`List[NDArray[dtype]]`. The first activation is an explicitly scoped input view;
other workspace arrays are owned and disjoint from parameters, gradients and
caller data. A future engine may privately split that list without changing the
public estimator interface, but must version a cross-owner interface change.

Arrays may have positive/negative element strides and offsets. Validation and
snapshots accept these layouts; kernels must support them or explicitly
materialize them under the engine policy. Newly allocated learning state uses
C order; RBM may use a documented F-order materialization to match its reference
initialization. Iteration and RNG draws follow reference logical order regardless
of physical layout. `.view()` aliases storage and retains lifetime; `.copy()` is
independent. Copying a strided view duplicates its backing allocation, not just
logical elements. `.T()`, reshape, slicing and astype are not free views under
the accepted NuMojo policy. No concurrent mutation/overlapping in-place writes.
Move owned results across module boundaries; borrow inputs for each call.

Metadata fields are private-engine conventions despite being accessible in Mojo:
mutating topology/shapes behind a validated record invalidates its contract.
Ordinary fit replaces learned state; clone copies constructor configuration into
an unfitted estimator, never copies LayerTensors as a substitute for clone.
A fitted model snapshot explicitly deep-copies tensors and optimizer/RNG state.
Warm start and partial fit retain exactly the state required by their reference
paths; warm-start fit must not silently preserve an incremental optimizer when
sklearn recreates it. Failed helper validation does not mutate existing state.

Memory scales as `P=sum(units[l]*units[l+1]+units[l+1])` tensor elements;
gradients add P, SGD velocity P, Adam moments 2P, best-model snapshots P.
Workspace scales with batch times the sum of widths; L-BFGS history scales with
its history size times P. Strided snapshot costs follow backing allocation size.
These are storage bounds, not measured performance claims.

## Numerical extension interfaces

`_contracts` contains abstract Mojo traits, with no executable default results:

- `MLPKernel`: associated `Matrix=NDArray[dtype]`,
  `Tensors=LayerTensors[dtype]`, `Workspace=List[NDArray[dtype]]`.
  `initialize(mut params, mut rng: RandomState, activation: String) raises` fills
  already shaped parameters in reference draw order. `forward(X, params,
  hidden: String, output: String) raises -> Matrix` returns owned activations.
  `loss_gradient(X, y, weights: Optional[Matrix], params, mut gradients,
  mut work, hidden, output, loss: String, alpha: Float64) raises -> Float64`
  computes the scalar objective and fills aligned gradients/workspace, leaving
  X/y/weights/params unchanged. The associated name Matrix also carries the
  rank-1 weights; ranks are semantic validation, not a separate buffer type.
- `MLPOptimizer`: associated `Tensors=LayerTensors[dtype]`;
  `update_params(mut self, mut params, gradients) raises`,
  `iteration_ends(mut self, time_step: Int) raises`, and
  `trigger_stopping(mut self, message: String, verbose: Bool) raises -> Bool`.
  Step count and learning-rate state belong to the optimizer. `time_step` counts
  training samples, while Adam's t counts updates. Validate aligned gradients
  before mutating params. False from trigger_stopping means adapt and continue.
- `LBFGSObjective`: `Vector=NDArray[DType.float64]` rank 1;
  `value_gradient(mut self, packed, mut gradient) raises -> Float64` is a
  full-batch objective. Pack coefficients layer-by-layer in C logical order,
  then intercepts layer-by-layer. Both vectors have length P. Float64 solver
  work may convert to the model dtype explicitly. The engine owns native
  L-BFGS and stopping/max_fun handling; a SciPy subprocess is not an engine.
- `RBMSampling`: `Input=NDArray[dtype]`, `Samples=NDArray[DType.bool]`;
  `gibbs(mut self, X) raises -> Samples` returns `(n_samples,n_features)`;
  `partial_fit(mut self, X, y: Optional[Input]=None) raises` ignores y.
  This optional trait complements preserved `RBMTransformer`; it adds no MLP
  optimizer or supervised capability to RBM.

These exact signatures compile through raising probes. Trait conformance checks
signatures; children must enforce the documented associated-type/layout rules
and behavioral contracts in their tests. The RBM child implements its sampling contract; MLP engine traits remain probes.

MLP engine acceptance includes hidden identity/logistic/tanh/relu; output
identity/exp/logistic/softmax; squared_error, poisson, log_loss and binary_log_loss;
weighted objectives and coefficient-only L2; stable probability clipping;
forward/backprop; SGD constant/invscaling/adaptive schedules, momentum/Nesterov;
Adam moment/bias corrections; full-batch L-BFGS. Preserve the pinned source's
loss normalization, including multioutput distinctions. Numerical gradient
checks, ill-scaled/nonfinite cases, zero targets for Poisson, singleton batches
and dtype-specific tolerances belong to that session. All remain required dense
branches; naming a trait is not evidence that they are implemented.

RBM owns `components_` `(n_components,n_features)`, `intercept_hidden_`
`(n_components,)`, `intercept_visible_` `(n_features,)`, and persistent
`h_samples_` `(batch_size,n_components)` plus RNG state. It owns free energy,
conditional probabilities, binary sampling, persistent contrastive divergence,
and pseudo-likelihood. Do not reinterpret RBM components as MLP weights.
RBM `partial_fit` explicitly validates/converts to Float64 in the pinned source;
its child must test this separately from Float32-preserving ordinary fit.

## Native compatibility and future public estimators

Reference: scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`,
`neural_network/__init__.py`, `_base.py`, `_multilayer_perceptron.py`,
`_stochastic_optimizers.py`, `_rbm.py`, their tests, and both neural-network guides.
The source is authoritative where older guide text differs (for example Poisson
regression and default relu). Preserve inherited accuracy/R2, TransformerMixin
fit_transform and feature-name behavior as documented by the child API.

MLP shared defaults: hidden_layer_sizes=(100,), activation='relu', solver='adam',
alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4,
verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
epsilon=1e-8, n_iter_no_change=10, max_fun=15000. MLPClassifier fixes log_loss;
MLPRegressor adds loss='squared_error' (also 'poisson'). RBM defaults:
n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0,
random_state=None. Typed parameter records must preserve these names/defaults;
'associated types' is not permission to drop solver branches.

| Future estimator | Required methods/modes | Learned state/output |
| --- | --- | --- |
| MLPClassifier | weighted fit/partial_fit, predict, predict_proba/log_proba, accuracy score; binary/multiclass/multilabel; warm start; stochastic early stopping through shared splitters | sorted classes_; coefs_/intercepts_; n_features_in_, n_layers_, n_outputs_, out_activation_, n_iter_, t_, loss_; solver-dependent loss_curve_, best_loss_, validation_scores_, best_validation_score_; label vector or multilabel indicator; binary probability `(n,2)`, multiclass `(n,k)`, multilabel `(n,k)` independent marginals |
| MLPRegressor | weighted fit/partial_fit, predict, R2 score; single/multioutput, squared_error/poisson; warm start/early stopping | same MLP state except classes_; predictions `(n,)` or `(n,o)` following target shape |
| BernoulliRBM | fit/partial_fit, transform, fit_transform, gibbs, score_samples, get_feature_names_out | RBM tensors above, n_features_in_; transform `(n,n_components)`, pseudo-log-likelihood `(n,)`, Gibbs bool matrix; generated names bernoullirbm0... |

MLP partial_fit requires SGD/Adam, rejects early_stopping, and uses classes on
first classification call; subsequent class/feature consistency matters. RBM has
no predict_proba, supervised score, or sample-weight fit. Multilabel probability
rows need not sum to one. `feature_names_in_` is conditional on named inputs;
Python/pandas containers, reflection, metadata routing and serialization are
binding work, not native solver omissions.

Fit and typed set_params return mutable self; RBM partial_fit returns nothing; params are typed,
errors are Mojo Error with descriptive prefixes, optional capabilities use traits.
None random_state means a borrowed application context from the shared RNG owner;
integer seeds reset only where sklearn resets them. Mutable RNG contexts advance
without hidden copies. `.copy()` means an explicit independent snapshot.
`RandomState` draw order, Float32 casts and arithmetic mode must be tested;
sharing a seed alone does not establish parity. The raising probes test native
signature adaptation without pretending to test estimator fit.

## Approved numerical operations

Consume [numerics](../../../shared/numerics/docs/numerics.md) directly:
NuMojo 0.10.0 at `515fb2856f0ecf3d2740a34d958fe168183e1129`, Mojo 1.0.0,
MAX 26.5.0, locked package hashes. Creation/copy/explicit views, finite elementwise
multiply/column broadcast subtraction, global sum, mean with explicit
returned_dtype, and finite rank-2 matmul are accepted. `_dense.sum_axis2d` and
`row_squared_norms` supply shared widened reductions. Exclude NuMojo general
axis sum, dot, variance/ddof, solve/inverse/unpivoted LU and random sampling.
Use `_random.RandomState` from foundation.random instead. No new operation
exception or numerical kernel is introduced here. Engine-specific activation,
loss, optimizer and RBM kernels require their own evidence over the same storage.

## Dependency assembly

Core `DEPENDENCIES.json` declares each live owner, module, export and license asset.
Focused manifests resolve to those same owners. No source path refers to the
standalone neural-network checkout. Preserve compatible Pixi locks and the oracle
revision. Run `pixi run neural-network-check` from the core root. The compiler uses
`-O3 --fp-mode=contract=off` and one assembled include root. The two isolated
fixture consumers retain missing-source, ownership/export conflict and storage
checks. MLP declarations raise at runtime; the public probe checks those errors.
