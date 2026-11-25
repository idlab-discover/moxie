# Dense sparse coding

The public `sparse_encode` function and `SparseCoder` estimator support the
`threshold`, `lasso_cd`, `omp`, `lars`, and `lasso_lars` algorithms over dense
Float32 and Float64 NuMojo arrays. The implementation lives in `shared/sparse_coding`, assembled
once into `moxie.decomposition._sparse_coding`.

```mojo
from moxie.decomposition import SparseCoder, sparse_encode
var code = sparse_encode(X, dictionary)  # Lasso LARS with alpha=1
var pursuit = sparse_encode(X, dictionary, algorithm="omp", n_nonzero_coefs=2)
var default_coder = SparseCoder(dictionary)
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
the call returns the last code without a convergence warning. The pinned
serial Lasso branch can return a rank-1 coefficient vector in the single-sample
case.

The `omp` algorithm selects atoms by absolute residual correlation and refits
their coefficients jointly. Its default count is one tenth of the input
feature count, rounded down and bounded between one and the dictionary size.
An explicit count cannot exceed the number of atoms. Supply an optional Gram
matrix (atoms, atoms) and covariance matrix (atoms, samples) independently.
Omitted matrices are computed as `dictionary @ dictionary.T()` and
`dictionary @ X.T()`. The Gram must represent a symmetric positive
semidefinite system and the covariance must come from the same dictionary.
Both matrices are read only, including when
`copy_cov=False`. A small correlation or dependent atom ends the search and
returns the current solution without a warning. OMP rejects `positive=True` and uses only the
atom count to control regularization. `alpha`, `max_iter` and `init` do not
drive its solver.

The least-angle modes share the linear-model Gram solver. `lars` stops after
`n_nonzero_coefs` path steps, or earlier if the path ends. Its default count
matches OMP's. It rejects `positive=True` and ignores `alpha` and `max_iter`. `lasso_lars` uses the same reconstruction objective and default
`alpha` as `lasso_cd`. It follows Lasso coefficient drops and interpolates to
the requested penalty. Positive Lasso LARS follows sklearn's knot interpolation
when remaining inactive correlations become negative. `max_iter` caps path
steps, including zero steps, and `n_nonzero_coefs` does not drive its solver.
Both modes accept the same borrowed Gram and covariance layouts as OMP and
ignore `init`. They return the current iterate at an early stop and skip
rejected Cholesky pivots without emitting sklearn's warnings. The native
solver accepts `verbose` without printing progress.

All supported modes run serially for every nonzero `n_jobs` and preserve the
dtype for unsplit output. Shape and finite-value checks remain enabled with
`check_input=False`. Native codes always have rank 2, including for one sample
or one atom.

SparseCoder can transform and reconstruct before fit. Fit validates inputs and
sets n_components_/n_features_in_; it does not learn a dictionary. Construction,
clone and parameter records own dictionary copies. Set_params accepts a typed
record or named updates. Clone has no fit metadata but remains usable as a
stateless coder. All five algorithms are deterministic.

Float64 supports split_sign=True: output concatenates positive and negative
parts, and inverse_transform subtracts them before multiplication. Float32
split_sign=True raises explicitly because sklearn promotes this output to
Float64 whereas the native transform trait has a fixed output dtype. Use a
Float64 dictionary/input for this mode. Unsplit outputs preserve input dtype.
Feature-name behavior follows the reference, including its atom-count names
for split-sign output. Fit/set_params mutate in place and return a mutable reference to self.

`SparseCoder(dictionary)` defaults to OMP and `sparse_encode(X, dictionary)`
defaults to Lasso LARS.

Parity runs pinned sklearn's decomposition, Lasso, Cython Gram solver and
`orthogonal_mp_gram`. The threshold, Lasso CD and OMP cases compare functional
codes, stateless transforms, fit_transform and reconstructions, including transforms after
fitted and unfitted serialization. Threshold tolerances are
2e-12 in Float64 and 2e-6 in Float32. Lasso CD tolerances are 3e-8 in Float64
and 2e-5 in Float32, allowing for solver stopping and BLAS reduction order.
OMP compares shapes, singleton cases, dependent dictionaries, correlation
ties and supplied statistics. Its tolerances are 2e-12 in Float64 and
2e-5 in Float32 for BLAS and triangular solve rounding. Near the dependence
threshold, rounding can also change which atoms are selected.

Least-angle parity also executes the pinned `_least_angle.py` and its
`arrayfuncs.pyx` helper. It covers count and penalty stopping, Lasso coefficient
drops and re-entry, positive constraints, supplied statistics, ties, and
single-row and overcomplete dictionaries. Tests use 2e-10 in Float64 and
1e-4 in Float32 to allow for Gram products, triangular solves, and successive
correlation updates. Near tied or degenerate path events, rounding can change
which atom joins the active set. The overcomplete random fixtures check LARS
counts below the rank. Oversized counts are tested on full-rank and exactly
dependent dictionaries.
