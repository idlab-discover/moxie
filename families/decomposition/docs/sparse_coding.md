# Dense sparse coding

The public `sparse_encode` function and `SparseCoder` estimator support the
`threshold` and `lasso_cd` algorithms over dense Float32/Float64 NuMojo
arrays. The implementation lives in `shared/sparse_coding`, assembled
once into `moxie.decomposition._sparse_coding`.

```mojo
from moxie.decomposition import SparseCoder, sparse_encode
var code = sparse_encode(X, dictionary, algorithm="threshold", alpha=0.5)
var coder = SparseCoder(dictionary, transform_algorithm="lasso_cd",
                        transform_alpha=0.5, positive_code=True)
var positive_code = coder.transform(X)
var reconstruction = coder.inverse_transform(positive_code)
```

Run [the example](../examples/sparse_coding.mojo) and contracts with
`MOXIE_ESTIMATORS=sparse_coding pixi run test`; parity uses
`MOXIE_ESTIMATORS=sparse_coding pixi run parity`.

`dictionary` has shape (atoms, features), input is (samples, features), and code
is (samples, atoms). Thresholding computes sign(D X.T) max(abs(D X.T)-alpha, 0),
and transposes the result. `positive=True` clips negative coefficients.
`alpha=None` selects 1.0. Optional precomputed covariance has shape (atoms, samples), and is honored
without mutating it. Gram, init, copy_cov and n_nonzero_coefs are accepted but
unused by this reference branch.

The `lasso_cd` algorithm uses the linear-model family’s cyclic Gram
coordinate descent solver with tolerance 1e-8. Each input row minimizes half
its squared reconstruction error plus `alpha` times the L1 norm of its code.
`alpha=None` selects 1.0 and `positive=True` constrains coefficients to be
nonnegative. Supply a symmetric dictionary Gram matrix of shape (atoms, atoms)
or omit it to compute `dictionary @ dictionary.T`. The solver computes
correlations from the dictionary and input, even when `cov` is supplied.
`init` provides warm-start coefficients in (samples, atoms) order, or a
vector of length `atoms` for one sample. The solver borrows `gram` and `init`
and works on owned buffers. `max_iter` must be positive. At the iteration limit
the call returns the last code without a convergence warning. Both coding
modes run serially for every nonzero `n_jobs` and preserve dtype for unsplit
output. Shape and finite-value checks remain enabled with `check_input=False`.
With one sample the native code remains rank 2 so transform and inverse
transform accept the same layouts for every row count.
The pinned serial Lasso branch returns a rank-1 coefficient vector in that case.

SparseCoder can transform and reconstruct before fit. Fit validates inputs and
sets n_components_/n_features_in_; it does not learn a dictionary. Construction,
clone and parameter records own dictionary copies. Set_params accepts a typed
record or named updates. Clone has no fit metadata but remains usable as a
stateless coder. Both algorithms are deterministic.

Float64 supports split_sign=True: output concatenates positive and negative
parts, and inverse_transform subtracts them before multiplication. Float32
split_sign=True raises explicitly because sklearn promotes this output to
Float64 whereas the native transform trait has a fixed output dtype. Use a
Float64 dictionary/input for this mode. Unsplit outputs preserve input dtype.
Feature-name behavior follows the reference, including its atom-count names
for split-sign output. Fit/set_params mutate in place and return a mutable reference to self.

The dense `lasso_lars`, `lars` and `omp` modes raise NotImplementedError.
The reference defaults are `lasso_lars` for `sparse_encode` and `omp` for
SparseCoder. Select an implemented algorithm explicitly.

Parity runs pinned sklearn's decomposition, Lasso and Cython Gram solver.
The 144 threshold and 124 Lasso CD cases compare functional codes, stateless
transforms, fit_transform and reconstructions, including transforms after
fitted and unfitted serialization. Threshold tolerances are
2e-12 in Float64 and 2e-6 in Float32. Lasso CD tolerances are 3e-8 in Float64
and 2e-5 in Float32, allowing for solver stopping and BLAS reduction order.
