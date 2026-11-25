# Batch dictionary learning

`dict_learning(X, n_components, alpha=...)` borrows a rank-2 Float32 or Float64
NuMojo array and returns an owned `DictionaryLearningResult`. Its fields are
`code` (samples by components), `dictionary` (components by features),
`errors` (Float64 objectives, one per iteration), and `n_iter`. The result
always carries the iteration count. Pass `None` for `n_components` to use
the number of input features.

```mojo
from moxie.decomposition import dict_learning

var result = dict_learning(X, 4, alpha=0.2, method="cd", random_state=0)
var reconstruction = result.code @ result.dictionary
```

The batch solver alternates sparse coding with sequential dictionary updates
that project each atom onto the unit ball. It initializes with thin SVD and
the scikit-learn sign convention, or uses `code_init` and `dict_init` when both
are supplied. The initializers are borrowed and truncate or zero-pad to the
requested count. `max_iter=0` returns these initial factors with an empty
error history. Iterations stop on scikit-learn's relative objective criterion.

Both `method="lars"` (default) and `method="cd"` support `positive_dict`.
`positive_code` requires `method="cd"`. Coding is serial and follows the
[dense sparse coding contract](sparse_coding.md). With zero CD solver steps
(`method_max_iter=0`), the dictionary update uses the current code.
An integer `random_state` uses the shared MT19937 stream to select replacement
samples and add Gaussian noise to unused atoms. `None` seeds a fresh native stream with entropy.
`verbose` prints native iteration summaries.

An optional thin Mojo callback declared with `raises` receives
`(iteration, code, dictionary, cost)` with borrowed matrices and a Float64 objective. It runs at zero-based
iterations 0, 5, 10, and so on, after the stopping check. The callback can raise
to interrupt fitting. The factors always retain the input dtype, including
when padding. The pinned reference pads with Float64 zeros, which can
introduce mixed buffers in the Float32 CD path.

## Numerical operations and validation

The implementation reuses `golub_kahan_svd`, `sparse_encode`, and rank-2
NuMojo `@` for the Gram matrix, cross product and reconstruction. Numax
`linalg.eigen.svd` requires a statically shaped tall TensorLike and
`linalg.blas.matmul` requires TensorLike, which would require adapting
NuMojo storage to a second tensor interface. The existing runtime-shape
NuMojo product works with the family storage contract. The sequential atom
updates read previously updated rows and compute one row at a time. Objective
sums, projection norms and seeded noise scaling use Float64 accumulation.

The family runner registers the dictionary contracts and differential driver
under `dictionary`. The oracle in `tests/dictionary/parity.py` uses local
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
It checks both factors, reconstruction, per-iteration objective, iteration
count and callback cadence. Absolute and relative tolerances are 4e-5 for
Float32 and 3e-7 for Float64 to account for SVD, matrix products and the
underlying CD stopping tolerance. Float32 padding cases compare with a
Float64 reference using the same Float32-rounded inputs. For padded CD,
the parity cases use eight solver sweeps per update. On nearly dependent
atoms, longer Float32 trajectories can distribute coefficients differently
from the Float64 run.
