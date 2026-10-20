# Dense IncrementalPCA

`IncrementalPCA` accepts finite dense NuMojo Float32/Float64 arrays. It updates
running means and population variances, then factors the retained scaled basis,
centered batch, and mean-correction row. Only the requested components survive
each update, so batch order and size can affect the approximation.

```mojo
from moxie.decomposition import IncrementalPCA
from moxie_numax import NumaxIncrementalPCABackend

var model = IncrementalPCA[DType.float64, NumaxIncrementalPCABackend[32, 8, 3]](
    n_components=3, batch_size=32, whiten=True,
)
model.partial_fit(first_batch)
model.partial_fit(next_batch)
var scores = model.transform(X)
var restored = model.inverse_transform(scores)
```

The Numax environment uses Mojo 1.1 / MAX 26.6. The [streaming example](../examples/incremental_pca.mojo) creates its own batches. Run:

```sh
pixi run -e numax python tools/check_incremental_pca.py test
pixi run -e numax python tools/check_incremental_pca.py parity
```

`NumaxIncrementalPCABackend[batch_size, n_features, n_components]` reserves
`batch_size + 2*n_components` SVD rows. Use matching estimator settings, or a
larger capacity. The adapter rejects an augmented matrix exceeding this capacity.
It accepts shorter batches by zero-padding the SVD input; padded rows never
increase `n_samples_seen_` or enter noise-variance averaging. The extra capacity
also accommodates `fit` merging its final undersized batch. Inference row counts
remain dynamic. Single-feature Float32 batches use Numax Float64 SVD internally
to avoid a MAX Float32 GEMV crash, then cast factors back. Without a backend parameter, the existing runtime SVD fallback
supports dynamic batch and feature dimensions.

Constructor defaults follow sklearn: `n_components=None`, `whiten=False`,
`copy=True`, `batch_size=None`. `None` infers the rank from the first batch;
`fit` defaults to batches of `5*n_features`. `partial_fit` processes its entire
input as one batch, independent of `batch_size`. Its first batch must contain
at least `n_components` rows; later batches may contain just one row. Feature
and component counts must remain fixed across partial updates. `fit` resets
state, while `clone` copies only constructor settings. Failed updates preserve
the previously fitted model.

Learned fields include `components_`, `singular_values_`, `mean_`, `var_`,
`explained_variance_`, `explained_variance_ratio_`, `noise_variance_`,
`n_components_`, `n_features_in_`, `n_samples_seen_`, and `batch_size_` (set by
`fit`). Array fields are optional before fitting. Means and population variances
use Float64 accumulation and storage; spectral and transformed arrays retain the
input dtype. This differs from sklearn's possible Float32-to-Float64 promotion
on subsequent updates. Zero-variance ratios and one-sample explained variances
can be NaN, as in the reference.

Projection, inverse projection, whitening and covariance reuse PCA's helpers.
`get_precision`, `get_feature_names_out`, typed `get_params` / `set_params`, and
`set_output(transform="default")` are included. `IncrementalPCAParams.copy_input`
represents constructor `copy`; nested optional keyword wrappers distinguish
omission from clearing `n_components` or `batch_size`.

Public `fit`, `partial_fit` and `set_params` return a mutable reference to self.
The explicit in-place helper methods retain their existing return conventions. `copy=False` requires
`fit_inplace`, `partial_fit_inplace`, or `fit_transform_inplace`. They center the
caller's data by batch. As in sklearn, `fit_transform_inplace(copy=False)` then
projects that modified input; use `copy=True` to project the original data.
`check_input=False` is accepted on partial fitting but native shape/finiteness
validation is always retained. Sparse input, implicit dtype conversion, labeled
input, Python metadata routing, pandas/polars output, and GPU execution are not
implemented. No performance improvement is claimed.

The differential runner executes the actual local sklearn IncrementalPCA and
incremental-moment methods at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. It compares moments, spectra,
projections, reconstruction, covariance, input mutation and short-batch behavior.
Float64 tolerances are `3e-9`; Float32 tolerances are `8e-4`, accounting for
retained Float32 factorization and native reduction order. Degenerate bases are
compared through reconstruction/covariance rather than arbitrary null vectors.

Validation passed 318 differential cases, native lifecycle/ownership contracts,
and two isolated assembled consumers. The shared inference refactor also passed
the existing 290-case Numax PCA differential suite. These bounded checks do not
claim complete sklearn parity or arbitrary-conditioning guarantees.
