# Dense TruncatedSVD

`from moxie.decomposition import TruncatedSVD` reduces dense Float32/Float64
`numojo.NDArray` matrices by a truncated SVD of the **uncentered** data (latent
semantic analysis). `TruncatedSVD()` uses Float64; `TruncatedSVD[DType.float32]()`
keeps Float32 arrays.

```mojo
var svd = TruncatedSVD(n_components=2, random_state=42)
var topics = svd.fit_transform(counts)       # (n_samples, 2)
var restored = svd.inverse_transform(topics) # (n_samples, n_features)
```

See [the example](../examples/truncated_svd.mojo). Reference: local scikit-learn
`sklearn/decomposition/_truncated_svd.py` at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## API

Constructor names/defaults follow sklearn: `n_components=2`,
`algorithm="randomized"`, `n_iter=5`, `n_oversamples=10`,
`power_iteration_normalizer="auto"`, `random_state=None`, `tol=0.0`.
Learned fields `components_ (k, p)`, `explained_variance_`,
`explained_variance_ratio_`, `singular_values_` are `Optional[NDArray]`
(absent before fit; use `.value()`), and `n_features_in_` is an `Int`.
Methods: `fit`, `fit_transform`, `transform` (`X @ components_.T`),
`inverse_transform` (`X @ components_`), `get_feature_names_out`
(`truncatedsvd0`, ...), `get_params`/`set_params` (typed
`TruncatedSVDParams` record or named partial updates), `__sklearn_clone__`,
`set_output(transform="default")`.

- `randomized` runs the shared Halko randomized SVD with sklearn's
  integer-seeded RandomState draws; fitted values agree with sklearn to
  rounding for the same seed. `n_components` must not exceed `n_features`.
- `arpack` is the dense equivalent of scipy ARPACK `svds`: the exact leading
  triplets from the shared Jacobi SVD, which satisfies any `tol`. It requires
  `n_components < min(n_samples, n_features)`, like the reference.
- Signs follow sklearn (`svd_flip(u_based_decision=False)`). As in sklearn,
  `fit_transform` returns `U * Sigma` for exact ARPACK fits and `X @ V` otherwise;
  `explained_variance_` is the population variance of the transformed data and
  the ratio divides by the summed feature variances (NaN for all-zero data).
- Input must be finite, nonempty and have at least two features. Errors are
  Mojo `Error`s prefixed `ValueError:`/`NotFittedError:`.

Sparse input (the common LSA storage) is deferred by the dense-only scope;
dense data exercises every algorithm branch. Python metadata routing,
feature-name inference and pandas output are binding work.

## Validation

`pixi run test` runs lifecycle/ownership/invalid-parameter contracts in both
dtypes (unfitted errors, repeat-fit determinism, clone independence, strided
input, arpack/randomized rank limits, zero data). `pixi run parity` compares
358 cases against the pinned source: tall/wide/small shapes, every normalizer,
n_iter 0/3/5, arpack with `tol` 0 and 1e-8, count data, full-rank randomized
and rank-deficient inputs (null directions compared by subspace). Tolerances:
Float64 `3e-9`, Float32 `6e-4` (relative and absolute, scaled by the input
magnitude), with 30x for randomized fits whose sketch passes through extra
products and QR/LU normalization.
