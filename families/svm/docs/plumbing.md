This family is now owned by Moxie under `families/svm`. The paths below are
relative to that directory. See [integration](migration.md) for current commands.

# Dense SVM plumbing, version 1

This document defines the family contracts and module owners. Both engines and
all seven dense estimators are implemented in their assigned directories. Existing
`moxie.svm._base` traits and foundation contracts remain unchanged. See [scope.md](scope.md) for the accepted dense-only scope.

## Owners and extension graph

| Internal owner | Reserved native modules | Exports and dependencies |
| --- | --- | --- |
| families/svm | `moxie.svm._base`, `moxie.svm._contracts` | Existing capability traits; private shared records/helpers |
| svm.libsvm / shared/libsvm | `moxie.svm._libsvm`, `_kernels`, `_cache` | Private kernel engine, cache, KKT/shrinking, weights and OVO packing; scaffold |
| svm.liblinear / shared/liblinear | `moxie.svm._liblinear` | Private linear engine and solver dispatch; scaffold |
| svm.c_svm / algorithms/c_svm | `moxie.svm._c_svm`, `_classification` | SVC, SVR; scaffold + libsvm; shared classification presentation |
| svm.one_class / algorithms/one_class | `moxie.svm._one_class` | OneClassSVM; scaffold + libsvm |
| svm.linear / algorithms/linear | `moxie.svm._linear`, `_bounds` | LinearSVC, LinearSVR, l1_min_c; scaffold + liblinear |
| svm.nu_svm / algorithms/nu_svm | `moxie.svm._nu_svm` | NuSVC, NuSVR; scaffold + libsvm + c_svm |

Children contribute public symbols to the `moxie.svm` export manifest. sklearn's
`_base`, `_bounds`, LIBSVM and LIBLINEAR boundaries are retained conceptually;
its monolithic private `_classes.py` is split by owner to compose independent
contributions without competing copies. Private BaseLibSVM/BaseSVC are not
public estimators. The family assembly includes the implementation owners below.

## Shared storage and lifecycle

`moxie.svm._contracts.Dense = numojo.NDArray[DType.float64]` and
`Indices = NDArray[DType.int64]`. Shape/index metadata uses NuMojo `Shape`/`Item`.
Inputs have samples in rows. Engine numeric arrays use Float64, matching dense
sklearn conversion. Wrappers must accept supported Float32 inputs through
explicit Float64 conversion, preserving sklearn's output dtype. Index arrays are native Int64;
public sklearn `n_support_` is Int32, so wrappers must checked-convert at their
public boundary and test the adaptation. Numeric labels are not stored in
indices: wrappers retain sorted `classes_` (including native string labels)
and give engines contiguous numeric class codes 0..k-1 as a Dense vector.

Borrow inputs for calls. `TrainingSnapshot(x, precomputed=False)` checks finite,
nonempty rank-2 input and owns `x.copy()`. Its `n_train`, `n_features`, and
`precomputed` metadata drive `validate_query(x)`. Precomputed training must be
square; queries have n_train columns. It preserves NuMojo strides/offsets rather
than promising contiguous storage. `.copy()` is independent; `.view()` is an
explicit mutable alias. A strided copy may copy its whole backing allocation.
Engine implementations may materialize contiguous data once where required;
record that cost and retain the shared storage type. Results move ownership.

`validate_matrix(x, columns=-1)` and `validate_vector(x, length)` check rank,
shape and finiteness using logical strides. They impose no sample-weight sign
policy: sklearn's zero/negative weight handling is formulation-dependent and
belongs to engines. Validate lengths before optimizer allocation. These helpers
raise Mojo Error with ValueError prefixes. Input metadata must satisfy NuMojo's
own layout invariants; concurrent mutation is outside the borrowing contract.

Constructor configuration remains separate from learned state. Each successful
normal fit replaces model state; no SVM here supports partial_fit or warm_start.
Use the existing FittedState/estimator contracts for unfitted errors, changed
feature counts, typed get/set_params, in-place fit and unfitted configuration
clones. Snapshot copying is a fitted-data operation, never estimator cloning.
A failed validation must not install partial learned buffers.

## Kernel boundary

`KernelSpec(kind, gamma, degree, coef0)` is resolved configuration, not an
estimator constructor. kind is linear/poly/rbf/sigmoid/precomputed/callable;
`validate()` checks that enum, finite nonnegative gamma, nonnegative degree and
finite coef0. `KernelOptions` fields, in declaration order, are formulation,
kernel, c, nu, epsilon, tol, cache_mb, shrinking, probability, max_iter, seed.
All options are resolved by wrappers; engine validation remains mandatory.
Formulations are c_svc, nu_svc, one_class, epsilon_svr, nu_svr. The engine
checks formulation-specific C/nu/epsilon, tolerance, iteration and cache
constraints against the pinned source, including reference-permitted infinite C.

`KernelEngine.solve(x, y, sample_weight, class_weight, options) -> KernelModel`
borrows Float64 arrays, receives full sample weights `(n,)`, class weights `(k,)`
(or empty for regression/one-class), and encoded targets `(n,)`. Absent sample
weights become explicit ones; one-class targets are an ignored length-n vector.
Seed is the resolved native engine seed: wrappers draw it from the shared
RandomState at the same reference call site. Integer configuration resets the
wrapper stream per fit; mutable streams advance; None uses an application-owned
context as documented by the RNG owner. Engine-specific shuffling and seed
consumption require source-level parity; matching a seed alone is insufficient.

Class-weight expansion is engine-owned and shared by its wrappers. C-SVC uses
C * sample_weight * class_weight bounds. Nu-SVC uses sample weights as bounds
and the reference nu rescaling; its native solve_nu_svc does not consume the
class-specific C values. Preserve this reference behavior rather than imposing
C-SVC weighting on every formulation. Engines also preserve nonpositive-weight
masking, original sample indices and formulation-specific feasibility rules.
Classification encoding and balanced-class weight construction live in the
shared `_classification` wrapper helper. At the pinned revision BaseSVC uses
unweighted label frequencies for balanced weights; `_fit_liblinear` passes
sample_weight to compute_class_weight. Preserve that distinction, with explicit
weighted-frequency tests in the linear owner. Linear wrappers consume the same
array conventions, not LIBSVM solvers.

`KernelProvider.evaluate(x, train) -> Dense` is the native callable seam.
It returns a finite matrix `(n_query, n_train)`; the wrapper validates it and
passes it as `kind=precomputed` to the engine. It retains an independent training
snapshot for later cross-kernel evaluation. Arbitrary Python callables/objects
are binding work. The `callable` marker in KernelSpec is wrapper-side only;
engines must reject unresolved callable configuration.

Gamma auto = 1/n_features; scale = 1/(n_features * population_variance(X)), with
1 for zero variance. `_kernels` owns the tested calculation; NuMojo variance is
not approved. Kernels are dot, (gamma*dot+coef0)^degree, exp(-gamma*distance²),
and tanh(gamma*dot+coef0). These operations are implemented in `_kernels`.
Use shared dense norms/reductions and the approved rank-2 matmul baseline;
measure any optimized kernel replacement in its owner.

`KernelEngine.decision_values(x, model, kernel) -> Dense` returns raw pairwise
values `(m,q)` using internal signs even for binary classification. Callable
queries arrive as validated precomputed matrices.
`KernelEngine.class_probabilities(decisions, model) -> Dense` applies the owned
calibration/coupling to raw pair decisions and returns `(m,k)` in class order;
it raises when classifier calibration is absent. Public wrappers decide whether
to expose probability methods and flatten binary decisions.

## Kernel results, cache and optimizer state

`KernelModel` is an owned result with these exact fields:

| Field | Layout / invariant |
| --- | --- |
| support | Int64 `(s,)`, original training row indices; class-block order |
| support_vectors | Float64 `(s,p)`; precomputed case `(0,0)` |
| n_support | Int64 `(k,)` classification; `(1,)` regression/one-class |
| dual_coef | Float64 `(k-1,s)` classification; `(1,s)` otherwise, internal LIBSVM signs |
| rho | Float64 `(q,)`, internal thresholds; q=k(k-1)/2 or 1 |
| iterations | Int64 `(q,)`, per binary solve or one solve |
| fit_status | 0 converged; 1 iteration limit; errors raise |
| prob_a, prob_b | Float64 `(q,)` for enabled classifier calibration, otherwise empty vectors |

`ClassPacking(counts, offsets)` owns Int64 class counts `(k,)` and prefix offsets
`(k+1,)` starting at zero and ending at s. `validate(s)` checks the invariant.
`pair_index(k,i,j)` requires 0<=i<j<k and returns lexicographic OVO index. For pair
(i,j), class-i supports use dual_coef[j-1, block_i], and class-j supports use
dual_coef[i, block_j]. `_libsvm` packs once; wrappers reuse this layout.

Raw engine decision is sum(coeff*kernel)-rho. Public intercept = -rho, then
binary SVC/NuSVC invert public dual_coef/intercept/decision signs so positive
values mean classes_[1]. Multiclass decisions retain reference pair signs and
ordering. `_classification` owns OVR vote/confidence conversion, break_ties and
probability presentation. SVR uses its signed coefficients directly. OneClassSVM
has offset_=rho, intercept_=-rho, decision=score_samples-offset_, and ±1 labels.
The separate public arrays must not mutate internal signs through shared aliases.

`WorkingSetState` is a layout record, not a solver. All vectors have length l:
Float64 alpha, gradient, gradient_bar, linear_term, diagonal, upper_bound;
Int64 signs (±1), alpha_status (0 lower, 1 upper, 2 free), active_set (permutation
of 0..l-1); plus active_size in [0,l] and unshrunk Bool. SVR can use l=2*n with
its mapping owned by `_libsvm`. The solver keeps its own parallel arrays and
active permutation; cache keys remain original row indices. Reconstruct gradients before final KKT checks. The engine owns all
updates, initialization, stopping, nu rescaling and restoration to input order.

`KernelColumnCache.column(mut self,index) -> NDArray[float32]` returns a full
column keyed by the original training row. `_cache.ColumnCache` owns bounded
Float32 columns with LRU eviction. The payload budget is cache_mb*2**20, with a
minimum of two columns capped by the number of training rows. Metadata is extra.
The solver maps active indices onto original rows, so cache permutation and
prefix maintenance are unnecessary. Float32 matches upstream Qfloat;
diagonals and optimization remain Float64.

Returned arrays are read-only leases by contract. NuMojo views retain their backing
allocation after eviction; callers must not mutate them. Native tests exercise
column values, reuse and eviction lifetime. No cache performance claim is made.

## Linear boundary

`LinearOptions` fields are solver, c, epsilon, tol, fit_intercept,
intercept_scaling, max_iter, seed. `LinearEngine.solve` has the same array
arguments as KernelEngine and returns `LinearModel(coef, intercept, iterations,
fit_status)`. Coef is Float64 `(r,p)`, intercept `(r,)`, iterations Int64 `(t,)`;
t=1 for a joint Crammer-Singer solve and t=r for independent row solves;
r=1 for binary classification/regression, k for OVR or Crammer-Singer. `LinearEngine.decision_values(x, model) -> Dense` returns `(m,r)`; public
wrappers flatten single-row decisions/predictions. The
Crammer-Singer binary wrapper reduces the two engine rows to the reference
single public row. Fit status has the same convergence convention.

Solver IDs preserve sklearn's dispatch: 1 L2 squared-hinge dual, 2 its primal,
3 L2 hinge dual, 4 Crammer-Singer, 5 L1 squared-hinge primal; 11 L2 squared
 epsilon-insensitive primal, 12 its dual, 13 L2 epsilon-insensitive dual.
`dual=auto` chooses n_samples<n_features when supported, otherwise the available
primal/dual alternative. Invalid combinations raise; Crammer-Singer ignores
penalty/loss/dual as the reference does. Logistic IDs 0/6/7 are reserved for
future shared consumers, not an SVM estimator deliverable.

`_liblinear` owns intercept synthetic-feature regularization, scaling, weighted
bounds, primal/dual stopping and multiclass packing. No kernel cache or LIBSVM
shrinking state is reused by this different optimizer. The `_bounds` owner
implements l1_min_c for both squared_hinge and log, with optional intercept and
ill-posed zero denominator errors. No probability trait belongs to LinearSVC.

## Dependency assembly and extension rules

Consume live Moxie HEAD foundation, numerics and random owners directly.
`DEPENDENCIES.json` (assembly schema 1) declares source paths, modules, exports,
NuMojo and mojo-linalg assets/licenses and the assembly tool; it carries no
revision or hash pins, and every build reads the current files (including
working edits). The frozen SCAFFOLD_BASE is retained; no tag is moved.
`pixi.lock` fixes Mojo 1.0.0, MAX 26.5.0 and the resolved environment.

`tools/check.py` assembles the root Moxie manifest through the current
shared namespace validator. Isolated consumers use its live
`tools/consume_family.py` launcher, checking only its location. All output is inside
the consumer's ignored builds/deps and caches inside its .pixi. Compilation uses
one import root with -O3 --fp-mode=contract=off. There is no repin step.
Inherited source files stay intact, while assembly consumes the authoritative
foundation modules and explicit family exports. No source-path precedence or
copied tracked shared implementation is used.

For a child, copy the family manifest (rebasing relative paths) and add its own
modules. Add every actual transitive engine/algorithm and its license notices. Register unique private modules and
public symbols through manifests. The shared tool rejects conflicting ownership
and exports and permits identical baseline deduplication only with verified
Git provenance. An explicit ownership override is solely for an authorized
inherited algorithm relocation; it never licenses a shared-helper fork.

The two `tests/consumers/{a,b}` have independent manifests/build trees and each
composes both contributed probe modules in the same `moxie.svm` package. They
call the same snapshot and dense kernel, execute real linear and kernel fits and
a seeded RNG draw, and compare
module hashes/owners. Tests reject missing live sources and ambiguous
module/export manifests without changing a valid assembled tree. Probe modules test assembly separately from the real estimators. Future helpers or layout changes require an owned
family change and consumer tests, not local copies in algorithms.

## Implemented family assembly

The root and both isolated consumers flatten both engines and all four algorithm
owners alongside `svm-scaffold`. `_linear` owns LinearSVC and LinearSVR;
`_bounds` owns l1_min_c. `_c_svm`, `_nu_svm` and `_one_class` own the five kernel
estimators, sharing `_classification` and `_libsvm`. No optimizer is copied into
the wrappers. External
foundation files are read live from the Moxie root (`../../`) on every build.
