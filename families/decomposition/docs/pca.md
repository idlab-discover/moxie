# Dense PCA

`from moxie.decomposition import PCA` provides native CPU PCA over runtime-sized
`numojo.NDArray` matrices. `PCA()` uses Float64; `PCA[DType.float32]()` preserves
Float32 input/output arrays. Use the same dtype for fit and inference.

```mojo
var model = PCA(n_components=2, whiten=True)
var projected = model.fit_transform(X)
var restored = model.inverse_transform(projected)
var variance = model.explained_variance_.value()
```

The [example](../examples/pca.mojo) creates its own input. From this repository:

```sh
pixi run build
pixi run test
pixi run parity
pixi run start
```

## Compile-time shapes with Numax

Use the Numax backend when the training matrix dimensions are known at compile
time (the component count remains a runtime constructor argument):

```mojo
from moxie.decomposition import PCA
from moxie_numax import NumaxPCABackend

var model = PCA[DType.float64, NumaxPCABackend[100, 8]](
    n_components=3, svd_solver="full", whiten=True,
)
var projected = model.fit_transform(X)  # X must be 100 x 8
```

The explicit backend is the second type parameter. Constructor defaults,
learned attributes, and output arrays stay the same. Fit rejects mismatched
training dimensions; transform and score accept any number of rows with the
same feature count. Tall, square, and wide training matrices are supported.

Run the [Numax example](../examples/numax_pca.mojo) and validation with:

```sh
pixi run numax-start
pixi run numax-test
pixi run numax-parity
```

This environment pins Mojo 1.1.0 / MAX 26.6.0. The default environment retains
Mojo 1.0.0 / MAX 26.5.0 and the runtime backend.

## API and behavior

The reference is local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. All 20 decomposition exports and
their inherited public method signatures were checked against its Python AST
before implementing PCA. Dense [IncrementalPCA](incremental_pca.md) also reuses PCA inference helpers;
other estimator/function declarations remain raising scaffolds.

Constructor names/defaults are `n_components=None`, `copy=True`, `whiten=False`,
`svd_solver="auto"`, `tol=0.0`, `iterated_power="auto"`, `n_oversamples=10`,
`power_iteration_normalizer="auto"`, `random_state=None`. Component count accepts
an integer (including zero), a variance fraction in `(0, 1)`, `"mle"`, or the
native `NoneType()` marker. Integer and string constructors work directly;
use `Float64(0.95)` for a fraction. Omission selects all available components.

- `fit`, `fit_transform`, `transform`, and `inverse_transform` operate on dense
  finite arrays. Fit replaces learned state on success; failure can retain the
  previous fit. Fit returns a mutable reference to self, following the shared native contract.
- `arpack` is the dense equivalent of scipy's ARPACK `svds`: it computes the
  exact leading singular triplets with the Jacobi SVD, so every `tol` is met
  (ARPACK's `tol=0` means machine precision). The reference's restrictions
  apply: an integer `1 <= n_components < min(n_samples, n_features)`, and an
  omitted count selects `min(n_samples, n_features) - 1`. Singular values,
  variances, components (V-based signs) and scores agree with `svds` to
  rounding; sklearn's ARPACK start vector draw is not needed by the exact solver.
- `full`, `covariance_eigh`, and `randomized` are implemented. `auto` uses the
  reference shape/rank dispatch. Full SVD uses a runtime Jacobi factorization;
  randomized uses the existing family range finder and sklearn-compatible
  integer-seeded RNG. These are native algorithms, not LAPACK driver calls.
- Whitening, V-based sign orientation, fractional/MLE rank selection, sample
  variance, explained-variance ratios and probabilistic-PCA noise are included.
  Repeated or zero singular values can have different valid bases from LAPACK.
- `get_covariance`, `get_precision`, `score_samples`, and `score` implement the
  reference generative-model formulas, including the whitening convention.
  Singular inverses raise; no regularization is silently added.
- Learned array fields are `Optional[NDArray]`, absent before fitting; access
  through `.value()`. Counts are zero before fit. Arrays own their data, and
  later caller mutations do not change the model. `get_feature_names_out`
  returns `pca0`, `pca1`, … and validates an optional input-name list's length.
- `get_params` returns `PCAParams`; `set_params` accepts that record or named
  partial updates. Its `copy_input` field represents the constructor's `copy`
  parameter because the record's native `copy()` operation uses that name.
  Optional update wrappers distinguish omission from clearing a setting.
  `clone` copies configuration into an unfitted estimator.

`copy=False` requires `fit_inplace` or `fit_transform_inplace`, which accept a
mutable input. The ordinary fit methods reject it rather than mutating an
immutable borrow. Full SVD leaves centered input; randomized leaves squared
centered input as the reference does; covariance-eigen leaves input untouched.
`copy=True` always preserves input. Positive/negative-stride views are accepted.
Output arrays are native contiguous arrays; sklearn's Fortran-order
`fit_transform` allocation is not reproduced.

## Numerical dependencies

The compile-time Numax backend delegates full SVD, symmetric eigen decomposition,
matrix products, inverse and log-determinant to Numax on an explicit CPU context.
Wide SVD transposes the input, then exchanges the left/right factors. Eigenvalues
are reversed to sklearn's descending order. Runtime `n_components` and inference
row counts use Numax's dynamic matrix-product API. Singleton output dimensions
are padded to two and cropped after GEMM: the pinned MAX dynamic GEMV path
crashes for `(24, 5) @ (5, 1)`. No reduction dimension is padded. Centering retains the shared
NuMojo operations. Scalar square roots, logarithms and Minka's log-gamma use the
Mojo standard library in both backends.

NuMojo remains the public array storage. The backend copies logical matrix
entries (including strided inputs) into owned Numax tensors and copies results
back. It does not borrow pointers across runtimes. These adapters favor clarity
and correctness; no speedup is claimed.

Randomized SVD still uses `shared/svd_engine`, including its mojo-linalg QR/SVD:
its runtime component count and sketch width do not satisfy Numax's static
factorization signatures. The default runtime backend uses mojo-linalg for
SVD/eigen/precision factorizations and NuMojo matrix products.

The Numax build applies [a recorded compatibility patch](../tools/numojo_mojo11.patch)
only to separate, generated NuMojo and shared RNG copies under `builds/numax-deps`. It updates
`Atomic`'s type argument and the MAX parallel callback syntax/captures, retaining
NuMojo storage and ownership. Original shared sources are verified before copying
and remain unchanged. This is an isolated family integration, not a migration of
other consumers or a replacement for upstream NuMojo compatibility work.

Moxie's root `DEPENDENCIES.json` assembles current owner sources into one import
root. Source snapshots are historical provenance; compatible toolchain locks and
the sklearn oracle revision remain fixed. The optional backend uses the family's
minimal `DEPENDENCIES.json` with current Moxie framework, RNG and kernel owners.
Numax is imported only by the Mojo 1.1 environment; its static factorizations
require compile-time dimensions.

## Remaining scope

Dense `svd_solver="arpack"` does not run Lanczos iterations; see the solver
list above for why the exact factorization is an equivalent. Sparse inputs, GPU,
Python metadata routing, pandas/polars output, implicit integer promotion and
Python RNG objects are not supported. `random_state=None` creates an entropy
stream for each randomized fit; it does not share Python's global RNG state.
`set_output(transform="default")` is supported; other containers raise.

Edge cases that deliberately differ from the pinned reference: a fractional
`n_components` within rounding of 1 (for example `nextafter(1, 0)`) selects
all available components, where sklearn's `searchsorted` runs one past the end
and reports `n_components_ = min + 1` with fewer rows in `components_`;
`get_precision` inverts `get_covariance` directly instead of sklearn's
inversion lemma, so in degenerate fits (`noise_variance_` ~ 1e-33, e.g.
`n_components="mle"` on square random data) `score` is `-inf` where sklearn
returns a huge finite value; and `tol=inf` is rejected.
Feature-name inference needs a labeled input type; `feature_names_in_` remains
absent for native arrays.

Validation executes the actual local sklearn PCA/base/extmath method bodies
with NumPy validation plumbing, rather than substituting an installed PCA.
Native checks cover ownership, cloning, repeated fits, strided input, invalid
settings, unfitted inference and explicit unsupported modes. Differential checks
cover Float32/Float64, tall/wide/singleton, rank-deficient and constant matrices,
all implemented solvers, automatic dispatch, whitening, rank selection,
covariance, precision and likelihood. Tolerances account for native reductions
and Jacobi/LAPACK rounding; degenerate bases are compared by subspace.

## Validation record

The runtime backend passed **349 cases** against the pinned local source,
including 58 dense-ARPACK cases (tall, wide, k from 1 to min-1 and omitted,
whitening, copy=False) compared with the reference's `scipy.sparse.linalg.svds`. The
compile-time Numax backend passed **290 cases**, covering eight training shapes
in both dtypes. Both backends pass the native lifecycle/ownership contracts.
The extra runtime case selects randomized SVD on `(20, 501)`; the static driver
does not instantiate that shape. Numax validation includes the singleton-product
crash regression described above, MLE/fractional ranks, whitening, covariance,
precision and scoring.
Float64 comparisons use `rtol=atol=3e-9`; Float32 uses `6e-4`, with ten times
those bounds for inverse/likelihood quantities. Covariance-eigen near-zero singular
values use an `O(sqrt(eps) * largest_singular_value)` absolute allowance because
they are square roots of rounded eigenvalues. Arbitrary null-space directions
are excluded from component-by-component comparison; reconstruction and the
nonzero subspace are checked instead. This is bounded validation, not full
scikit-learn compatibility or an extreme-conditioning guarantee.

A small native fit baseline is reproducible with:

```sh
pixi run python tools/benchmark_pca.py --processes 1 --values 3 --warmups 1 \
  --min-time 0.01 -o benchmarks/pca-initial.json
```

Raw pyperf measurements include source/dependency hashes, compiler flags and
hardware metadata. They measure the fit region including allocations, excluding
compilation and process startup. This initial implementation favors correctness
and makes multiple input/factor copies; optimization remains separate work.
