# Dense mixture plumbing

Moxie owns the contracts and concrete implementations. Both estimators use
`_em`, `_responsibilities`, `_initialization`, `_kmeans_init`, `_precision`,
`_gaussian_kernels`, and `_scalar_math`. Bayesian variational updates remain in
`_bayesian_mixture`; Gaussian estimation and common sampling remain in
`_gaussian_mixture`. The reference is sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

Global estimator contracts, dense storage and random streams remain in their
existing Moxie owners. See the [family guide](README.md) for live numerical
assembly, public consumer migration and validation commands.

## Arrays and ownership

All floating arrays use NuMojo NDArray with compile-time Float32 or Float64.
Shape/Item and storage ownership come directly from the live numerical owner.
Int64 NDArray holds output labels. Input X is `(n,d)`, means `(k,d)`, weights
`(k)`, component log probabilities/responsibilities/log responsibilities `(n,k)`,
and per-row log likelihood `(n)`. No class sorting: columns keep component order.
`sample` returns owned Float64 points and Int64 labels, `(points: (m,d), labels: (m))`, grouped by component as in
sklearn. `predict` returns `(n)` argmax labels; ties use the first component.

| covariance_type | covariances_, precisions_, precisions_cholesky_ shape |
| --- | --- |
| full | `(k,d,d)` |
| tied | `(d,d)` |
| diag | `(k,d)` |
| spherical | `(k)` |

`covariance_shape(kind, components, features) -> Shape` validates kind and positive
sizes. `check_shape(array, expected)` checks exact rank/extents, not numerical
positivity. `check_input(X, components=1, fitting=True, features=-1)` requires
rank 2, finite values, positive features, at least two training rows and n>=k;
inference requires at least one row and checks a supplied fitted feature count.
A shape helper is never evidence of positive definiteness.

Borrow arrays immutably at engine boundaries. All result buffers and fitted
snapshots are owned. `snapshot(X)` uses NuMojo `.copy()` and prevents caller
mutation from changing learned state. `.view()` explicitly aliases and retains
buffer lifetime. Input validation honors element strides and offset, including
negative strides. Returned engine arrays shall be contiguous row-major; learned
snapshots may preserve input layout. `.reshape`, slicing, `.T` and `.astype`
can allocate; a strided `.copy()` can copy its full backing buffer. No implicit
view/copy or dtype conversion is permitted at a child boundary. Concurrent
mutation and overlapping in-place operations remain unsupported.

Storage costs: X O(nd), responsibilities O(nk), means O(kd); covariance storage
O(kd²), O(d²), O(kd), O(k) in the table's order. Best-restart state requires an
independent copy, as does clone configuration containing initialization arrays.
Array ownership is shared family policy, not a child-selectable representation.

## Exact native extension points

Import these private traits from `moxie.mixture._contracts`. Each engine inherits one `DenseMixtureEngine` dtype contract and declares
`comptime dtype: DType`; every NDArray below uses that same dtype. All methods
raise on failure. There are no default numerical bodies.

* `EMHooks`: `initialize(mut self, X, resp)`, `estimate_log_prob(self, X) -> array`,
  `estimate_log_weights(self) -> array`, `m_step(mut self, X, log_resp)`,
  `lower_bound(self, log_resp, mean_log_prob: Float64) -> Float64`,
  `get_state(self) -> State`, `set_state(mut self, var state: State)`.
  `State: Movable & Deinitable` is algorithm-owned and includes all learned arrays needed to
  restore the selected restart; get_state makes independent snapshots.
* `ResponsibilityEngine.normalize(self, weighted_log_prob) -> Tuple[array,array]`
  returns `(log_prob_norm: (n), log_resp: (n,k))`. Weighted log probability is
  log density plus log weight. Use row-wise stable logsumexp; never exponentiate
  raw large logits. Gaussian uses log(weights); Bayesian uses expected log weights.
  Zero weights may yield -infinity; impossible rows and NaNs must match the
  pinned reference's numerical behavior, not silently become uniform rows.
* `PrecisionEngine.from_covariances(self, covariance, kind: String) -> array`
  and `from_precisions(self, precision, kind: String) -> array` return the above
  factor layout. Full/tied U is upper triangular with precision `U @ U.T`;
  log-density uses `(X-mean) @ U`. Diag/spherical use positive inverse standard
  deviations. Reject non-SPD or invalid factors. These are distinct entry points:
  taking Cholesky of precision naively gives the wrong orientation for this use.
* `InitializationEngine.responsibilities(self, X, components: Int,
  method: String, mut rng: RandomState) -> array` returns `(n,k)`. KMeans uses
  n_init=1 and one-hot assignments; k-means++ and random_from_data seed only
  selected rows; random fills uniforms and normalizes each row. Initial seed
  matrices are not always row-normalized, so do not impose that invariant here.

`EMControl` stores n_components=1, covariance_type="full", tol=1e-3,
reg_covar=1e-6, max_iter=100, n_init=1, init_params="kmeans", warm_start=False,
verbose=0, verbose_interval=10. `validate()` checks ranges/modes;
`restart_count(fitted)` gives one restart for a fitted warm start, else n_init.
These are internal controls, not a new estimator constructor. Validation rejects
nonfinite tol/reg_covar explicitly, a stricter native control contract; public
estimator parameter validation must be compared separately to sklearn.

## Lifecycle and public behavior

The shared `_em` driver validates input and parameters, initializes each restart,
then alternates E-step, M-step, objective evaluation. Compare absolute objective
change strictly below tol; do not clamp decreases. Gaussian objective is mean
log likelihood; Bayesian lower bound is its separate variational objective.
Keep the first strictly best restart and its convergence flag, iteration count,
objective history, and owned state. Warm start resumes the learned objective,
skips initialization and uses one restart. Normal fit replaces learned state.
max_iter=0 performs initialization, restores the final initialized state, reports
n_iter_=0, lower_bound_=-infinity, empty lower_bounds_, converged_=False and no
convergence warning. Always run a final E-step so fit_predict equals fit.predict.
Nonconvergence with positive max_iter emits a diagnostic while retaining fitted
state; the native warning/reporting mechanism is tested by the estimator suites.

Both estimators expose fit, fit_predict, predict, predict_proba, score_samples,
score, sample, typed get_params/set_params and clone. score is the mean of
score_samples, not a sum. `y` is accepted and ignored. No partial_fit, transform,
sample_weight fitting or public predict_log_proba capability is implied.
AIC/BIC is Gaussian-only. MixtureDensity retains its existing capability traits;
responsibilities are not classifier probabilities and no classes_ is introduced.
Learned common attributes: weights_, means_, covariances_, precisions_,
precisions_cholesky_, converged_, n_iter_, lower_bound_, lower_bounds_,
n_features_in_. Names/feature_names_in_ require a future binding policy.
Unfitted prediction/scoring/sampling must raise; parameter clones contain no
learned arrays and copying fitted state is a separate operation.

Gaussian adds weights_init, means_init, precisions_init (all default None).
Bayesian adds weight_concentration_prior_type="dirichlet_process" and
weight_concentration_prior, mean_precision_prior, mean_prior,
degrees_of_freedom_prior, covariance_prior (all default None). Resolved Bayesian
prior defaults follow reference data-dependent validation: 1/k concentration,
1 mean precision, feature mean, d degrees of freedom, empirical covariance.
Both dirichlet_process and dirichlet_distribution and all covariance modes stay
in scope. Posterior mean_precision_ and degrees_of_freedom_ are `(k)` except
shared tied degrees of freedom; concentration is `(k)` for Dirichlet and a pair
of `(k)` arrays for stick-breaking. Prior attributes remain distinct from
posterior state. Bayesian special functions/updates stay in its own module.

Native adaptations retain fluent in-place fit, typed parameter records and descriptive
Mojo Error prefixes from the foundation. Dtype is static; Python reflection,
metadata routing, pandas, serialization and plotting are separate binding work.
random_state integer resets where sklearn check_random_state does; mutable
RandomState is borrowed and advanced; None uses `entropy_state()` per call. Gaussian sampling must validate its exact
normal draw/factorization sequence; sharing a seed alone does not prove parity.
