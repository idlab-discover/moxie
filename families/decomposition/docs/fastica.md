# Dense FastICA

`from moxie.decomposition import FastICA, fastica` separates linearly mixed
non-Gaussian sources (Hyvarinen et al., *Independent Component Analysis*,
Tables 8.3/8.4), following local scikit-learn
`sklearn/decomposition/_fastica.py` at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Float32 and Float64 input are supported; learned arrays keep the input dtype.

```mojo
var ica = FastICA(n_components=2, random_state=0)
var sources = ica.fit_transform(X)            # (n_samples, 2)
var remixed = ica.inverse_transform(sources)  # (n_samples, n_features)
```

See [the example](../examples/fastica.mojo).

## API

Defaults follow sklearn: `n_components=None`, `algorithm="parallel"`
(or `"deflation"`), `whiten="unit-variance"` (`WhitenOption`, a
`Variant[String, Bool]`: `"unit-variance"`, `"arbitrary-variance"` or `False`),
`fun="logcosh"` (`"exp"`, `"cube"`), `fun_args=None`
(`Dict[String, Float64]`, key `"alpha"` in [1, 2]), `max_iter=200`, `tol=1e-4`,
`w_init=None` (Float64 `(k, k)`), `whiten_solver="svd"` (or `"eigh"`),
`random_state=None`.
Learned: `components_ (k, p)`, `mixing_ (p, k)`, `mean_`, `whitening_ (k, p)`
(absent when `whiten=False`), `n_iter_`, `n_features_in_`.
Methods: `fit`, `fit_transform` (sources), `transform`, `inverse_transform`
(each also with sklearn's `copy` keyword, which is a no-op because native
inputs are never mutated), `get_feature_names_out` (`fastica0`, ...), typed
`get_params`/`set_params` (`FastICAParams`), `__sklearn_clone__`.

`fastica(X, ...)` returns a `FastICAResult` record with `K`, `W`, `S`,
`X_mean` and `n_iter` (the reference's positional list); `return_X_mean` and
`return_n_iter` are accepted for signature parity and the record always holds
both. `compute_sources=False` leaves `S` empty.

Behavior matches the reference, including: `n_components` clipped to
`min(n_samples, n_features)` and ignored for `whiten=False`; whitening by
economy SVD (shared engine) or eigen-decomposition of `Xc^T X` (mojo-linalg,
eigenvalues floored at `10 eps`) with the `u *= sign(u[0])` convention;
`w_init` drawn from the seeded NumPy-compatible RandomState; symmetric
decorrelation `(W W^T)^{-1/2} W` via mojo-linalg `symmetric_eigh`;
unit-variance rescaling; `mixing_ = pinv(components_)` via mojo-linalg SVD
with scipy's default cutoff.

Not supported natively: Python callables for `fun` (binding work), the
`ConvergenceWarning` (check `n_iter_ == max_iter`) and the `n_components`
warnings. `whiten=False` requires `n_samples >= n_features`, where the
reference fails with a shape error.

## Validation

`pixi run parity` runs 180 cases against the pinned source: both algorithms,
all three nonlinearities, both whitening modes and solvers, `alpha=1.5`,
truncated/tight `tol`/`max_iter`, custom `w_init`, `fit` versus
`fit_transform`, the `fastica` function, clipped `n_components` and
`whiten=False`, in Float64 and Float32. Sources, unmixing, mixing, whitening
and round trips are compared directly (same RNG draws and sign conventions, no
permutation matching). Tolerances: Float64 1e-6, Float32 2e-2 relative and
absolute (scaled by the data magnitude); Float64 iteration counts must match,
Float32 may differ by two sweeps because its stopping test compares a
rounding-level change with `tol`. Data use four non-Gaussian sources so the
problem is identifiable.
