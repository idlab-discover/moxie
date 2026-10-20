# Dense threshold sparse coding

The public `sparse_encode` function and `SparseCoder` estimator support the
`threshold` algorithm over Float32/Float64 NuMojo arrays. The code and dictionary
are dense arrays; mathematical coefficient sparsity does not require sparse
input containers. The implementation lives in `shared/sparse_coding`, assembled
once into `moxie.decomposition._sparse_coding`.

```mojo
from moxie.decomposition import SparseCoder, sparse_encode
var code = sparse_encode(X, dictionary, algorithm="threshold", alpha=0.5)
var coder = SparseCoder(dictionary, transform_algorithm="threshold",
                        transform_alpha=0.5, positive_code=True)
var positive_code = coder.transform(X)
var reconstruction = coder.inverse_transform(positive_code)
```

Run [the example](../examples/sparse_coding.mojo) and contracts with
`MOXIE_ESTIMATORS=sparse_coding pixi run test`; parity uses
`MOXIE_ESTIMATORS=sparse_coding pixi run parity`.

`dictionary` has shape (atoms, features), input is (samples, features), and code
is (samples, atoms). Thresholding computes sign(D X.T) max(abs(D X.T)-alpha, 0),
then transposes; positive=True clips negative coefficients. Alpha=None selects
1.0. Optional precomputed covariance has shape (atoms, samples), and is honored
without mutating it. Gram, init, copy_cov and n_nonzero_coefs are accepted but
unused by this reference branch. Native validation remains enabled even with
check_input=False. n_jobs is accepted for API compatibility; zero is invalid.

SparseCoder can transform and reconstruct before fit. Fit validates inputs and
sets n_components_/n_features_in_; it does not learn a dictionary. Construction,
clone and parameter records own dictionary copies. Set_params accepts a typed
record or named updates. Clone has no fit metadata but remains usable as a
stateless coder. No sample weights or randomness apply to threshold coding.

Float64 supports split_sign=True: output concatenates positive and negative
parts, and inverse_transform subtracts them before multiplication. Float32
split_sign=True raises explicitly because sklearn promotes this output to
Float64 whereas the native transform trait has a fixed output dtype. Use a
Float64 dictionary/input for this mode. Unsplit outputs preserve input dtype.
Feature-name behavior follows the reference, including its atom-count names
for split-sign output. Fit/set_params mutate in place and return a mutable reference to self.

**Incomplete dense solver modes:** lasso_lars, lasso_cd, lars and omp raise
NotImplementedError. Defaults remain lasso_lars for sparse_encode and omp for
SparseCoder; callers must explicitly select threshold. They are not silently
replaced with thresholding. The external linear-model solver dependencies must
be integrated before these branches, DictionaryLearning/MiniBatchDictionaryLearning,
the dict_learning functions, SparsePCA and MiniBatchSparsePCA can be completed.
Sparse input containers and Python metadata/output facilities remain deferred.

Parity compares functional codes (including perturbed precomputed covariance),
stateless transforms, fit_transform and reconstructions against the pinned
sklearn _dict_learning.py. Tolerances are 2e-12 in Float64 and 2e-6 in Float32,
allowing matrix-product accumulation differences on bounded fixtures. These are
threshold-branch tests, not evidence for the unavailable regression solvers.
