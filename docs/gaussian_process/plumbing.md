# Gaussian-process shared plumbing

The family supplies contracts and validation/retention helpers, with dense
kernels and GaussianProcessRegressor implemented by the owners below. Root
assembly includes these implementations. GaussianProcessClassifier remains
unimplemented; sklearn's private binary Laplace class is not an additional public
estimator. See [API coverage](api-surface.md) for remaining differences.

## Owners and dependencies

Moxie owns all modules under `src/moxie/gaussian_process/`: the `_base`
capability contracts, `_plumbing` records and validators, `kernels`, `_gpr`,
and private algebra, mathematical and optimizer adapters. Classification is
unimplemented. No `_gpc` estimator is exported.

General factorizations and special functions remain in the dedicated
`../.worktrees/mojo-linalg-gaussian_process` library checkout, assembled as
`moxie._gp_linalg`. The optimizer remains in `../mojo-optimize`, assembled as
`moxie._optimize`. See [numerical ownership](README.md).

The GP algebra adapter preserves the library's Cholesky and triangular-solve
semantics. Moxie's covariance adapter has a stricter symmetry/finite-input
contract and is not substituted during this relocation.

## Storage and ownership

Use pinned NuMojo 0.10.0 at `515fb2856f0ecf3d2740a34d958fe168183e1129`,
Mojo 1.0.0 and MAX 26.5.0. `numojo.NDArray`, `Shape`, `Item` are the only
numerical storage/shape/index representation. No family buffer wrapper exists.
Float32/Float64 dense feature retention is tested; `Dense64` in `_plumbing` is
an alias for `NDArray[DType.float64]` for theta, bounds and numerical engine
results. Engines must explicitly convert input dtype where needed and validate
that behavior against the pinned oracle; this alias is not a blanket precision
parity claim.

Dense X uses `(samples, features)`. Borrow valid strided input for computation;
strides are element counts and offsets are owned by NuMojo. `retain_training(X,
copy_X_train=True)` requires nonempty rank-2 input and returns `.copy()` by default
or `.view()` for False. The argument is `mut X` because the pinned NuMojo
view constructor requires a mutable borrow even though retention does not alter
elements; this is a tested native signature adaptation. The latter deliberately preserves sklearn's caller-mutation
semantics and retains storage lifetime. This helper does not perform estimator
finite-value/target validation. GPR/GPC must perform that validation before use.
Structured native kernel inputs use `KernelContract.Input` and their own
retention implementation; they do not pass through the dense helper.

Snapshot copies are independent; repeated fit replaces them. Clone copies only
constructor configuration, recursively including kernels, and has no learned
posterior state. Normal `Copyable` is an independent value copy, not estimator
clone. No implicit cast or automatic layout materialization is promised. A copy
of a strided view can copy the entire backing allocation. Outputs own storage;
concurrent caller mutation and overlapping in-place operations are unsupported.

Reuse the numerics operation decision: finite elementwise operations, rank-2
matrix multiplication, global sum, explicit-return-dtype mean, and
`moxie._dense.sum_axis2d`/`row_squared_norms`. General NuMojo axis sum, dot,
variance/ddof, solve/inverse/unpivoted LU and random sampling are excluded.
Factorization and stable normalization need dedicated validated implementations,
not a new container. Every operation exception must retain this storage and
record correctness, cost and source/license evidence in its owning session.
No performance claim is made by these scaffold probes.

## Exact extension contracts

Import the following private names from `moxie.gaussian_process._plumbing`.
Their executable signatures are authoritative and compiled in `test_plumbing`.

| Contract / record | Interface and layout |
| --- | --- |
| `HyperparameterSpec` | `name: String`, `n_elements: Int`, `fixed: Bool`, owned raw-unit `bounds: Dense64`; constructor validates positive element count and `(n_elements,2)` or fixed `(0,2)` shape; `n_dims()` excludes fixed entries |
| `KernelContract` | Associated `Input`, `Params`; `get_params(deep=True)`, in-place `set_params`, `hyperparameters()`, `theta()`, `bounds()`, `set_theta`, `clone_with_theta`; `evaluate(X,Y=None,eval_gradient=False)`, `diag(X)`, stationary/vector flags |
| `KernelEvaluation` | Owned `covariance: Dense64`; `gradient: Optional[Dense64]`; None when not requested |
| `PositiveDefiniteAlgebra` | `cholesky(matrix)`, `solve_triangular(lower,rhs,transpose=False)`, `cho_solve(lower,rhs)`, `log_determinant(lower)`; all borrow arguments, return owned arrays or Float64 log determinant |
| `LikelihoodObjective` | Mutating `evaluate(theta,eval_gradient=True)` returns `ObjectiveEvaluation(value: Float64, gradient: Optional[Dense64])`; minimized value is negative LML |
| `HyperparameterOptimizer` | Associated `Objective: LikelihoodObjective`; `minimize(mut objective,initial_theta,bounds)` returns `OptimizationResult` |
| `OptimizerRestarts` | Same associated Objective; `optimize(mut objective,initial_theta,bounds,n_restarts,mut rng: RandomState)` returns `OptimizationResult` |
| `OptimizationResult` | Owned log-space theta, minimized value, convergence Bool and message; optimizer owner must surface nonconvergence honestly |
| `PredictionResult` | Owned mean, optional std, optional covariance; validate request flags before construction; never both uncertainty fields present |
| `LatentMoments` | Owned mean and variance vectors for binary GPC |

Traits declare extension points only; test implementations raise clearly and
are never assembled as library modules. Engine sessions implement concrete types
with these signatures. Typed Params and native trait/callable implementations
replace Python reflection and heterogeneous callable dispatch. `evaluate` is the
private native adaptation of `Kernel.__call__`; theta/bounds are methods and
`set_theta` is explicit. Public kernel names and operator ergonomics belong to
the kernel session. Generic Input permits dense arrays and native structured
objects; this does not authorize a Python runtime bridge.

### Theta, bounds and kernel algebra

Nonfixed parameters concatenate in sklearn's hyperparameter order (basic names
sorted as by `hyperparameter_...`; operators recursively k1 before k2;
Exponentiation prefixes `kernel__`). Anisotropic values flatten within each
parameter. Theta is natural-log space `(p,)`; bounds `(p,2)` use the same order
and natural logs. Fixed values contribute no optimizer/gradient coordinates.
The internal empty bounds shape `(0,2)` intentionally normalizes sklearn's empty
`(0,)` result. Kernel adapters perform this conversion. `HyperparameterSpec`
keeps raw-unit bounds, expands scalar bound pairs before construction, and
represents `"fixed"` with a Boolean plus an empty bounds matrix. These are
internal language adaptations, tested for fixed and vector-valued metadata.

`validate_theta(theta,bounds,n_dims,n_restarts=0)` checks ranks, lengths, finite
theta, non-NaN ordered bounds and nonnegative counts. Infinite bounds are allowed
without restarts; all bounds must be finite for restarts. It does not clip theta
or enforce that it starts within bounds; optimizer-specific policy owns that.
This is an internal optimizer guard, not a replacement for public kernel setters
(e.g. raw zero values whose logarithm is infinite).

Scalar covariance is `(n,m)`; self-gradient is `(n,n,p)` and differentiates
with respect to **log** theta. `diag` is `(n,)`. Cross-kernel gradients reject
Y being present, even when it aliases X. `validate_gradient` enforces that rule.
Sum adds covariance and concatenates derivative blocks; Product multiplies
covariance elementwise and applies the product rule; Exponentiation uses its
chain rule. WhiteKernel cross-covariance is zero even for identical coordinate
values; self-call has the diagonal noise contribution. The kernel engine implements these operations and tests their shapes and gradients.

CompoundKernel has an additional final kernel axis: `(n,m,k)` covariance and
`(n,n,p_child,k)` gradients when child gradient widths permit concatenation;
heterogeneous widths must follow the pinned source's errors rather than silently
padding. Compound diag is `(n,k)`, theta concatenates children. GPC rejects it
as an input kernel but may expose a learned CompoundKernel for multiclass.
Do not impose scalar-covariance shapes on this kernel class.

### Algebra, optimizer and RNG

Lower Cholesky factor L represents `A=L L.T`, `(n,n)`. RHS is always `(n,r)`;
callers adapt vector targets to a one-column matrix and restore output shape.
`solve_triangular` solves L or L.T; `cho_solve` solves A. Log determinant is
`2*sum(log(diag(L)))`. Inputs are unchanged; non-SPD, singular, dimension and
finite-value failures must be explicit. No hidden jitter, inverse fallback or
silently changed alpha is permitted. General solver implementation remains
separate from this declaration.

The first optimizer run starts at supplied theta. Each additional restart draws
uniformly in finite log bounds in coordinate order. Select the minimum objective
(first on a tie); set kernel theta once to the winner. Disabled optimizer and
p=0 bypass optimization and evaluate LML at the current kernel. A native custom
optimizer satisfies the same objective/sign/bounds contract. The kernel engine implements default L-BFGS-B, stopping rules and warning handling.

Use `moxie._random.RandomState` from foundation.random, compiling with
`--fp-mode=contract=off`. Integer seeds reset at sklearn's check_random_state
call boundaries; mutable streams are borrowed and advance; None uses the shared
application-owned entropy context described in the RNG owner's docs. Sample_y's
default is integer 0, distinct from estimator random_state=None. Never reseed
inside a kernel or numerical operation. Native multivariate-normal sampling
requires a separately validated covariance factorization and draw-order contract.

## Posterior boundary and compatibility obligations

GPR defaults: kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None,
random_state=None. Learned `kernel_`, `X_train_`, `y_train_`, `L_`, `alpha_`,
`log_marginal_likelihood_value_`, `n_features_in_` preserve sklearn meanings.
Alpha is scalar or per-sample diagonal noise, distinct from WhiteKernel.
`log_marginal_likelihood` takes a mutable receiver to honor `clone_kernel=False`.
GPR must support prior prediction before fit, scalar/multitarget y, normalization,
constant targets, posterior sampling and LML with optional gradient/clone_kernel.
For t>1 targets: mean/std `(n,t)`, covariance `(n,n,t)`, samples `(n,t,s)`;
for a single target outputs squeeze to `(n,)`, `(n,n)`, `(n,s)` even if fit y
had shape `(n,1)`. Sample axis is last. R2 scoring uses uniform target averaging.

GPC defaults: kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None,
multi_class="one_vs_rest", n_jobs=None. It owns `base_estimator_`, `kernel_`,
`classes_`, `n_classes_`, LML and feature metadata. Classes are ordered as in
sklearn; probabilities are `(n,c)`, predictions `(n,)`. GPC inference requires
fit; it has no GPR sample_y capability. Binary `latent_mean_and_variance` returns
two `(n,)` vectors and multiclass rejects it. OvO prediction supports dense
inputs but probability prediction rejects multiclass OvO. Multiclass LML averages
children, accepts shared or concatenated theta and rejects gradients. Accuracy
is its score. Warm-start mode caches belong to the Laplace posterior, not kernels.

Both lack partial_fit and sample-weighted fit; score weights do not imply fit
weights. `validate_uncertainty` rejects simultaneous std/cov;
`validate_likelihood` rejects gradients without explicit theta. Helpers use Mojo
Error with descriptive prefixes. Native fit/set_params mutate in place as in the
unchanged foundation contract. Language differences must be tested by children;
Python reflection, pandas adapters, metadata routing and serialization remain
binding work. The kernel engine implements general Matérn nu and dense string PairwiseKernel
metrics. Native custom/structured kernels remain incomplete.

## Dependency assembly and verification

Moxie's root `DEPENDENCIES.json` declares all live inputs. The assembler
materializes one compilation namespace under `builds/deps`; external library
implementations are never copied into tracked Gaussian-process sources.
Missing sources and conflicting modules or exports fail before compilation.

Run Moxie's root `pixi run build`, `pixi run test`, and `pixi run parity`.
Focused commands are `pixi run python tools/gaussian_process_checks.py test`
and `pixi run python tools/gaussian_process_checks.py parity`.

The focused suite retains family capability and plumbing probes, kernel and
regression contracts, algebra checks, negative compilation tests, and two
independently assembled external consumers. Consumers verify source provenance,
shared random-stream advancement, Float32/64 storage and regression behavior.
Assembly failures leave the previous successful output intact. Kernel, pairwise
and regression parity use the pinned scikit-learn sources and original oracles.
