# Random projections

This repository implements `GaussianRandomProjection`, `SparseRandomProjection`,
and `johnson_lindenstrauss_min_dim` in pure Mojo under `moxie.random_projection`.
Both estimators implement `RandomProjector` using the current shared Moxie
estimator contracts.

## Example

```mojo
from moxie.random_projection import GaussianRandomProjection, ProjectionMatrix

var X = ProjectionMatrix(2, 3, List[Float64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
var model = GaussianRandomProjection(n_components=2, random_state=42)
var projected = model.fit_transform(X)
var reconstructed = model.inverse_transform(projected)
```

Run `builds/examples/random_projection` for the complete dense and sparse example.

## Parameters and learned state

| Estimator | Constructor defaults |
| --- | --- |
| `GaussianRandomProjection` | `n_components="auto"`, `eps=0.1`, `compute_inverse_components=False`, `random_state=None` |
| `SparseRandomProjection` | Same, plus `density="auto"`, `dense_output=False` |

`ComponentCount = Variant[Int, String]` represents a positive component count or
`"auto"`. `ProjectionDensity = Variant[Float64, String]` represents a density in
`(0, 1]` or `"auto"`. The variants permit the reference constructor spelling.
Invalid values are rejected at fit. `eps` must be finite and positive; automatic
dimensionality additionally requires `eps < 1`.

Fit validates finite, nonempty input and uses its shape, not its values. Targets
are ignored. Successful fit replaces all learned state. A failed fit leaves the
previous learned state intact. Integer seeds restart the same stream at every
fit. Repeated transforms use the fitted matrix without drawing random numbers.

Public learned fields are `n_components_`, `n_features_in_`, `components_`, and
`inverse_components_`; sparse projections also expose `density_`. Before fit,
counts are zero and components are empty. `get_n_features_in()` mirrors
sklearn's `n_features_in_` attribute: it raises an `AttributeError`-prefixed
error before fit, as the attribute is absent there. Both estimators declare the
shared `FeatureNamesOut` and `InputFeatureCount` traits. `inverse_components_` is an
`Optional[ProjectionMatrix[dtype]]`, populated only when inverse computation was
requested during the last successful fit. Inference and feature-name queries
raise `NotFittedError` before fit.

`get_params(deep=True)` returns a typed `GaussianRandomProjectionParams` or
`SparseRandomProjectionParams` record with constructor configuration only.
`set_params(params)` and named partial updates are supported. Parameter changes
take effect on the next fit, except `dense_output` and
`compute_inverse_components`, which govern inference immediately. Enabling
inverse caching after a fit without a cache raises a clear error; refit or turn
caching off. Use the full Params record to reset `random_state` to `None`, since
`None` in a named partial update means "leave unchanged."

`clone(model)` creates an unfitted estimator with independent configuration;
`model.copy()` also copies its learned buffers. `fit` and `set_params` return
a mutable borrow, following the [shared native contract](../interfaces.md).

## Matrix representation and dtypes

`ProjectionMatrix[dtype]` is family-local owned storage, not a new shared array
ABI. It supports `DType.float64` (default) and `DType.float32`:

- `ProjectionMatrix(rows, cols, data)` consumes a row-major list.
- `ProjectionMatrix(rows, cols)` creates a zero-filled dense matrix.
- `ProjectionMatrix[dtype].from_csr(rows, cols, data, indices, indptr)` consumes
  CSR buffers. Unsorted indices and duplicates are accepted; duplicates add.
- `to_dense()`, `to_csr()`, and `transpose()` return independent storage.
  `transpose()` currently returns dense storage.
- `rows`, `cols`, `data`, `indices`, `indptr`, and `is_sparse` expose storage.
  Estimator boundaries validate shapes, CSR pointers/indices, and finite values,
  including after users edit public buffers. `matrix[row, col]` reads a value.

Match the estimator dtype explicitly for Float32:

```mojo
var model = GaussianRandomProjection[DType.float32](2, random_state=42)
var X = ProjectionMatrix[DType.float32](2, 3,
    List[Float32]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
var projected = model.fit_transform(X)
```

Float32 inputs produce Float32 components and outputs. Gaussian output is always
dense. Sparse projection output is CSR only if **both** its input and components
are sparse and `dense_output=False`. With `density=1`, components and output are
dense, as in the reference implementation. Projection multiplication does not
densify CSR inputs. Inverse transforms always return dense matrices.

## Dimensions, sampling, and inverses

`johnson_lindenstrauss_min_dim` uses the reference formula
`4 * log(n_samples) / (eps**2 / 2 - eps**3 / 3)`, truncated to an integer. It
supports scalar Float64 inputs and one-dimensional lists, with scalar and
length-one broadcasting. Counts must be finite and at least one, and
`0 < eps < 1`. The helper returns zero for one sample; auto-fit rejects that
zero dimension. Auto-fit also rejects a bound exceeding the original feature
count. Explicit component counts may exceed it. Array inputs follow the scalar domain: the
reference validates the `n_samples >= 1` interval only for scalars and returns
negative bounds for array entries in `(0, 1)`; the native helper rejects them.

Gaussian coefficients have variance `1 / n_components`. Sparse coefficients
are zero with probability `1-density`, and each sign has probability
`density/2`, with magnitude `sqrt(1/density) / sqrt(n_components)`.
Automatic density is `1 / sqrt(n_features)`.

The native RNG reproduces legacy NumPy MT19937 integer seeding and its Gaussian
polar, binomial inversion/BTPE, and sampling-without-replacement algorithms.
Integer seeds in `[0, 2**32-1]` are supported. The tests compare reference matrices
for multiple seeds, including MT state refills and the sparse sampling branches.
Floating-point coefficients are compared with explicit tolerances; bitwise
identity of transcendental arithmetic across platforms is not promised.
`random_state=None` leases Moxie's process-global MT19937 stream through
`check_random_state`. It advances across estimators and fits, unlike the old
root and child's per-fit entropy initialization. `seed_global_random_state(seed)`
resets it reproducibly. Its initial seed follows the shared monotonic-clock
entropy policy, not NumPy's OS-entropy initialization. Python RandomState objects
are not accepted. Configure `random_state=None` and call `fit(X, rng=state)` to
borrow an application-owned advancing `RandomState` or `entropy_state()`.
An integer seed with an explicit context raises. Validation failures before
sampling consume no draws; a later inverse failure can consume draws while
preserving fitted state. Clone copies configuration, not the borrowed stream.

Inverse components use a scaled one-sided Jacobi SVD, with the reference rank
cutoff `max(shape) * dtype_epsilon * largest_singular_value`. The implementation
supports rectangular, singular, rank-deficient, and all-zero matrices, and raises
on nonconvergence or an unrepresentable result. It does not form normal equations.
With `compute_inverse_components=True`, fit stores the inverse; otherwise each
inverse call recomputes it. Reconstruction is a pseudoinverse reconstruction,
not recovery of information discarded by dimensionality reduction.

`get_feature_names_out()` returns `gaussianrandomprojection0`, etc. Optional
input names are checked for the fitted feature count. Names are generated from
`n_components_`, including in auto mode. The pinned Python source currently uses
its constructor value for this helper and fails for `"auto"`; the native helper
uses the documented number of output features.

## Explicit native limits

- Inputs use the owned dense/CSR Float32 or Float64 type. CSC/COO adapters,
  integer-array conversion, dataframe feature metadata, Python bindings,
  arbitrary NumPy broadcasting, and dynamic metadata routing are not included.
- `ProjectionMatrix` is not NuMojo `NDArray`, the storage every other Moxie
  estimator and meta-estimator uses. The projections therefore cannot be
  composed yet: `make_pipeline(SparseRandomProjection(...), RidgeClassifier())`
  fails to compile with `pipeline intermediate types must match`, and
  `FeatureUnion` and `ColumnTransformer` require `NDArray` steps. Scored
  cross-validation or search over a projection needs such a pipeline. Convert explicitly between the two storages until a shared
  decision lands (see `notebooks/random_projection/`).
- Sparse projection supports at most `2**32-1` input features. Matrix sizes and
  JL bounds that exceed native integer limits raise clear errors.
- Python's dimensionality warning for explicit expansion is not emitted;
  expansion itself is supported. Nonfinite inputs/parameters and nonfinite
  computed results are rejected.
- Fits commit learned state atomically; a failed refit preserves the previous
  state. Successful refits clear an old inverse cache when caching is disabled.
- CSR structural ordering is not an API guarantee. For example, SciPy sorts
  columns when casting to Float32; native storage preserves sampling order.
  Dense values and mathematical results remain equivalent.
- Pseudoinverse computation densifies components. This implementation prioritizes
  correctness and has no large-matrix performance claim.

## Validation and references

```sh
pixi install --locked
pixi run build
pixi run test
builds/examples/random_projection
```

The native suite covers generic trait dispatch, fitted-state errors, refits,
clones, ownership, parameter updates, output storage, validation, automatic
bounds, singular inverses, and matrix-distribution properties. Original scaffold
contract and negative compilation tests remain enabled.

Reference comparisons require a local scikit-learn checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` (1.10.dev0):

```sh
pixi run parity
# Or only this family:
pixi run python tools/random_projection_checks.py parity
```

The runner accepts `--sklearn-root PATH`, verifies the source revision and clean
reference files, and loads the pinned local `random_projection.py`. Installed
sklearn provides supporting utilities, including compiled sampling code; NumPy
and SciPy provide test oracles only. The runner reports those versions and saves
fixtures under ignored `builds/random_projection/parity/`. The library never calls Python or foreign
numerical libraries.

The original 365 fixtures include both dtypes, five seeds, sparse densities and sampling
branches, storage choices, auto dimensions, cached/uncached inverse paths,
rank-deficient matrices, and extreme inverse scales. Float64 Gaussian matrices
use `3e-13 * (1 + abs(expected))`; transform/inverse cases use `3e-10`. Float32
uses `2e-6` for matrices and `3e-5` for transforms/inverses to allow single-precision
reductions and SVD rounding. Shapes and output storage are exact; Float64 sparse
CSR indices and row pointers are also checked exactly. This is evidence for the
native scope above, not full Python API parity.

Primary references:

- Pinned `sklearn/random_projection.py`, `sklearn/tests/test_random_projection.py`,
  `sklearn/utils/_random.pyx`, and `doc/modules/random_projection.rst`.
- Dasgupta and Gupta, *An Elementary Proof of a Theorem of Johnson and
  Lindenstrauss*, DOI [10.1002/rsa.10073](https://doi.org/10.1002/rsa.10073).
- Li, Hastie, and Church, [Very Sparse Random Projections](https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf), KDD 2006.
- [NumPy v2.3.3 legacy RNG sources](https://github.com/numpy/numpy/tree/v2.3.3/numpy/random/src),
  revision `f2a77a76e08719556527e0819182073fe9b5f1c3`.

Source attributions and redistribution licenses are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Current estimator contracts

`fit` and both `set_params` overloads return a mutable borrow of the estimator.
Concrete and generic calls can chain `model.set_params(params).fit(X).transform(X)`.
`get_params` returns `GaussianRandomProjectionParams` or
`SparseRandomProjectionParams`. Named partial updates interpret None as unchanged;
use the full record to reset `random_state=None`. `clone` returns an unfitted
estimator; `.copy()` independently copies fitted buffers. Modifying components
does not invalidate an existing inverse cache, matching the reference. Refitting
regenerates consistent components and inverse. Persistence is not implemented.
