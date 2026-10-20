# Linear-model shared plumbing

The family defines dense storage, ownership and extension contracts. Existing
`_base` traits remain unchanged. `_contracts` adds native validation and ownership
helpers, not a solver. Moxie owns implementations under
`src/moxie/linear_model`; child directories retain algorithm-specific validation
and documentation.

## Interfaces and layouts

All numeric arrays are `numojo.NDArray[dtype]`, dtype Float32 or Float64 selected
at compile time; no implicit dtype conversion. `Shape` and `Item` remain NuMojo
metadata. X has shape (samples, features), y (samples,) or (samples, outputs).
Row order is stable. Valid strided arrays are accepted by shape checks;
`validate_xy(X,y)` rejects empty dimensions, wrong rank and unequal sample counts.
It deliberately does not validate finiteness or estimator-specific target domains:
those are explicit child fit checks, performed before committing fitted state.

`validate_weights(weights,n_samples,nonnegative=True)` checks a rank-1 finite
weight vector and positive mass with nonnegative values. The False option checks
only finiteness/shape and permits signed weights for estimators that support them;
callers must check their required nonzero normalization separately. It does not
normalize. Scalar weights and None are expanded by the estimator boundary to
same-dtype (samples,) arrays; an absent weight is equivalent to ones. Do not
apply square roots or rescaling to signed weights. This helper does not confer
weighted-fit capability on every estimator.

`LinearShape(n_features,n_outputs,vector_target).validate()` validates positive
sizes and requires one output when vector_target is True. The movable `LinearParameters[dtype](coef,intercept)` record validates against
LinearShape; internal coefficient arrays have shape (outputs,features), intercepts (outputs,). Single-target public
regression uses coef_ (features,), intercept_ (1,) and prediction (samples,);
multi-target preserves the second axis, even for a one-column target. Binary
classifiers use coef_ (1,features), intercept_ (1,), decision (samples,).
Multiclass rows follow sorted classes_. Ridge label encoding lives in `_ridge_labels`;
linear scores alone do not supply probabilities. GLMs apply their own inverse
link; classifiers apply their documented decision/probability mapping.

`snapshot(array)` returns `.copy()`: independent owned storage. Arguments are
borrowed, returned arrays move, and explicit `.view()` shares mutation and retains
buffer lifetime. A strided copy may copy the entire backing allocation. Never
retain a mutable caller view as learned state. New fits validate and compute into
fresh state before replacing old state; clone copies only constructor parameters.
Warm starts and partial fits are explicit per-estimator choices. Reuse the
foundation FittedState for unfitted and feature-count checks where appropriate.

## Shared contracts and implementations

`DensePreprocessor` has associated `dtype` and method
`prepare(X,y,weights,fit_intercept,rescale_with_sw) raises -> PreparedData[dtype]`.
The movable `PreparedData` record exposes owned X, y, X_offset (features,), y_offset (outputs,), X_scale (features,)
and optional sample_weight_sqrt (samples,). The vector_target Bool preserves original target rank. The family implementation computes weighted offsets when fitting
an intercept, zero offsets otherwise, and X_scale=ones. Center before optional
sqrt-weight rescaling; retain the ability to disable rescaling for engines using
weights directly. The public contract always preserves caller input; an explicit
future mutable overload may implement sklearn copy_X=False. No hidden mutation.
Intercept reconstruction is y_offset - coef @ X_offset (with any explicitly
introduced scale undone first). The common `_linear_base` implementation is shared by least-squares,
coordinate and ridge estimators.

`DenseLinearPrediction` has associated dtype and
`decision(X,coef,intercept) raises -> NDArray[dtype]`. The canonical result is
(samples,outputs) containing X @ coef.T + intercept, with checked dimensions.
Estimator wrappers expose the rank described above and enforce fitted state.
`_linear_base.linear_decision` implements the prediction kernel.

`DenseSolver` has dtype, Config: Copyable, State: Movable, Result: Movable and
`solve(X,y,weights,config,mut state) raises -> Result`. Each engine owns its concrete
config/state/result; all use the same dense arrays and explicit weights. Result
must carry coefficients, intercept or reconstruction metadata, convergence status,
iteration count and solver-specific diagnostics. No universal convergence test,
penalty normalization, probability or incremental method is imposed. State is
private scratch/warm state, never an implicit mutable input alias. Invalid input
raises a descriptive Mojo Error. Nonconvergence follows each public estimator's
reference behavior and diagnostics, never a successful placeholder.

Smooth engine owns losses, gradients/Hessian products, optimizer state, line
search and stopping. Match `_linear_loss.py` weighted average normalization,
unpenalized intercept and L1 exclusion from derivatives. Its packed parameter
layout is output-fast: offset = output + n_outputs * feature, intercept in the
last feature slot; document materialization into canonical coefficients.
Coordinate engine owns residual/Gram updates, dual gaps, path continuation,
positive constraints and multi-task group penalties. Use feature-contiguous
columns when needed, explicitly materializing layout once per fit, not per step.
SGD engine owns learning-rate counters, shuffle/RNG state, averaging, penalties,
early stopping and incremental transitions. Consume `moxie._random.RandomState`;
never use NuMojo random routines. Least-angle path state belongs to lars, not a
variant of the coordinate solver. Gram-input path APIs validate Gram/Xy metadata
and retain the reference normalization. CV owns splitting/scoring orchestration
through model_selection.validation and calls the exact matching solver.

## Implemented dense owner: `moxie.linear_model._linear_base`

The family now provides the single implementation of the extension points above
(declared by the live manifests as part of Moxie):

| Call | sklearn source | Behavior |
| --- | --- | --- |
| `preprocess_data(X, y, sample_weight, fit_intercept, rescale_with_sw=True, owner=...)` | `_preprocess_data` + `_rescale_data` | Validates X/y (shape, finiteness) and weights; weighted column means as X_offset/y_offset when fitting an intercept (zeros otherwise); centers, then multiplies rows by sqrt(sample_weight) when weights are given and rescaling is requested. Returns owned `PreparedData` with y in (samples, outputs) layout and `vector_target` recording the caller's target rank. Rescaling requires nonnegative weights; signed weights are accepted only with `rescale_with_sw=False`. |
| `set_intercept(mut coef, X_offset, y_offset, X_scale, fit_intercept)` | `LinearModel._set_intercept` | Divides (outputs, features) coef by X_scale in place and returns y_offset - coef @ X_offset, or zeros without an intercept. |
| `linear_decision(X, coef, intercept)` | `LinearModel._decision_function` | Checked X @ coef.T + intercept, (samples, outputs). |
| `public_coef`, `canonical_coef`, `public_prediction`, `target_matrix` | coef_/predict shape rules | Rank-1 target: coef_ (features,), prediction (samples,); matrix target keeps the output axis. |
| `check_predict_input`, `check_finite`, `r2_score` | `validate_data(reset=False)`, `RegressorMixin.score` | Feature-count/finiteness checks with estimator-named errors; R2 with uniform output averaging and sklearn's constant-target rules. |
| `gram_matrix`, `outer_gram`, `transpose_product` | `X.T @ X`, `X @ X.T`, `X.T @ Y` | Float64-accumulated products shared by ridge/Bayesian/OMP/LARS (exactly symmetric Gram). |
| `DenseCentering[dtype]`, `DenseDecision[dtype]` | — | Trait providers for `DensePreprocessor` / `DenseLinearPrediction`. |

Accumulations (means, intercepts, decisions, R2) use Float64 and store the input
dtype, so Float32 results can differ from NumPy's Float32 reductions in low bits.
Language adaptation: public single-target `intercept_` is a length-1 array
(`NDArray` fields cannot switch between scalar and array types; NuMojo 0-d arrays
have size 0). Contracts: `tests/test_linear_base.mojo`.

## Owners and dependencies

All rows depend on linear_model.scaffold; rows below describe intended responsibilities; docs/coverage.md identifies
which implementations exist.

| Child location | Exports | Additional dependencies |
| --- | --- | --- |
| algorithms/least_squares | LinearRegression | none |
| shared/smooth_engine | private smooth engine | none |
| shared/coordinate_engine | private coordinate engine | none |
| shared/sgd_engine | private SGD engine | none |
| algorithms/lars | Lars, LassoLars, lars_path, lars_path_gram | none |
| algorithms/omp | OrthogonalMatchingPursuit, orthogonal_mp, orthogonal_mp_gram | none |
| algorithms/bayesian | BayesianRidge, ARDRegression | none |
| algorithms/quantile | QuantileRegressor | none |
| algorithms/ridge | Ridge, RidgeClassifier, ridge_regression | smooth_engine, preprocessing.labels |
| algorithms/elastic_net | Lasso, ElasticNet, lasso_path, enet_path | coordinate_engine |
| algorithms/multitask | MultiTaskLasso, MultiTaskElasticNet | coordinate_engine |
| algorithms/logistic | LogisticRegression | smooth_engine, svm.liblinear, preprocessing.labels |
| algorithms/sgd | SGDClassifier, SGDRegressor | sgd_engine |
| algorithms/glm | PoissonRegressor, GammaRegressor, TweedieRegressor | smooth_engine |
| algorithms/huber | HuberRegressor | smooth_engine |
| algorithms/ransac | RANSACRegressor | least_squares |
| algorithms/theil_sen | TheilSenRegressor | least_squares |
| algorithms/ridge_cv | RidgeCV, RidgeClassifierCV | ridge, model_selection.validation |
| algorithms/coordinate_cv | LassoCV, ElasticNetCV, MultiTaskLassoCV, MultiTaskElasticNetCV | elastic_net, multitask, model_selection.validation |
| algorithms/lars_selection | LarsCV, LassoLarsCV, LassoLarsIC | lars, model_selection.validation |
| algorithms/omp_cv | OrthogonalMatchingPursuitCV | omp, model_selection.validation |
| algorithms/logistic_cv | LogisticRegressionCV | logistic, model_selection.validation |
| algorithms/sgd_variants | Perceptron, PassiveAggressiveClassifier, PassiveAggressiveRegressor, SGDOneClassSVM | sgd_engine, sgd |

Use sklearn's `_base`, `_linear_loss`, `_coordinate_descent`, `_stochastic_gradient`,
`_least_angle`, `_omp`, `_ridge`, `_logistic`, `_glm` and corresponding private
module boundaries. Algorithm components contribute distinct private modules;
when multiple children extend one Python module, native leaf modules use explicit
export manifests rather than source-file concatenation. `_contracts` is a deliberate
native split so helpers do not change inherited capability traits. Child sessions
must inspect constructors, exports, inherited methods, tests and cited papers,
then provide exact parameter/learned-state contracts and numerical parity evidence.

## Assembly and compatibility

Moxie's root `DEPENDENCIES.json` owns all runtime modules and exports. The
focused `DEPENDENCIES.lock.json` files retain their historical filenames but
resolve live Moxie and external library sources. They contain no source hash
gates. The compatible Pixi locks and sklearn reference revision remain intact.

Run root `pixi run linear-model-build`, `linear-model-test`, and
`linear-model-parity`. The tests retain isolated consumers, missing-source and
ownership/export collision rejection, native contracts and differential suites.
`PROVENANCE.json` records the original paths. No standalone checkout is needed.

The external linear-model linalg worktree supplies `moxie._linear_linalg`;
its namespace prevents conflicts with other families' adopted numerical
worktrees. Moxie owns `_random`, `_dense`, estimator contracts and validation.
Runtime learning remains pure Mojo. Python, NumPy, SciPy and sklearn are
validation tools only. Preserve `-O3 --fp-mode=contract=off`.

## Cyclic coordinate implementation

`shared/coordinate_engine` owns `_cd_fast.coordinate_descent` (prepared X,
vector y, owned initial coefficients, unnormalized L1/L2 penalties, max_iter,
tol and positivity). It returns owned coefficients, dual gap, iteration count
and convergence status. `algorithms/elastic_net` owns ElasticNet and Lasso:
weight normalization, lifecycle and public parameter/shape semantics. Both use
family `_linear_base` for centering, intercepts, decisions and R2. See the child
API for diagnostics and explicit random/Gram/path/multi-task limitations.
