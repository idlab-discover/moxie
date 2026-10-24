# Implemented dense algorithms

`ClassicalMDS` and `trustworthiness` run entirely in Mojo. Other public estimator
names currently remain declarations that raise `NotImplementedError`.

## Classical MDS

```mojo
from moxie.manifold import ClassicalMDS
from moxie.base import clone

var model = ClassicalMDS[DType.float64](n_components=2)
var coordinates = model.fit_transform(X)
var fresh = clone(model)
```

`X` is a finite, nonempty rank-two NuMojo float32 or float64 array. The estimator
supports `euclidean`, `l2`, `sqeuclidean`, `manhattan`, `cityblock`, `l1`,
`chebyshev`, `cosine`, `minkowski`, and `precomputed`. `metric_params` is an
optional `Dict[String, Float64]`; only Minkowski `p` is supported. It defaults
to 2, must be positive, and may be infinite (Chebyshev), as in SciPy. Other
metrics, callables and parameter entries raise `ValueError`. The default
metric is `euclidean`, with two components.

The estimator squares the dissimilarities, double-centers them, and calls the
live decomposition `symmetric_eigh` engine. Eigenvalues descend and each
coordinate column has a nonnegative largest-magnitude entry before scaling.
As in the reference, requesting more components than samples returns at most
one component per sample. Negative eigenvalues produce NaN coordinates;
classical scaling does not repair non-Euclidean dissimilarities.

`embedding_`, `eigenvalues_`, and `dissimilarity_matrix_` are optional owned arrays.
They are absent before fitting; access fitted arrays through `.value()`.
Coordinates and eigenvalues use float64, matching the reference's explicit
promotion after squaring. Dissimilarities retain the input dtype in this native
interface, including metrics for which SciPy promotes float32 input.
`n_features_in_` records the input column count. Returned coordinates and learned
arrays are independent of caller-owned data. A successful fit replaces learned
state; clone copies only configuration. `fit` and `set_params` mutate in place.
`get_params` returns `ClassicalMDSParams`. Named partial updates support
`n_components` and `metric`; use the typed record to update `metric_params`.
There is no out-of-sample `transform`. NDArray inputs carry no column names, so
`feature_names_in_` is absent; metadata routing (`get_metadata_routing`) is
outside the native interface. `ClassicalMDS` cannot end a `moxie.pipeline`
pipeline yet: the pipeline's final step must be a supervised predictor.

## Trustworthiness

```mojo
from moxie.manifold import trustworthiness

var score = trustworthiness(X, coordinates, n_neighbors=5)
```

The inputs may have different floating dtypes but must have the same sample
count. `n_neighbors` defaults to 5 and must be positive and strictly less than
half the sample count. Original-space metrics are `euclidean`, `l2`,
`sqeuclidean`, `manhattan`, `cityblock`, `l1`, `cosine`, and `precomputed`.
Embedding neighbors always use Euclidean distance. A precomputed matrix must
be square, finite and nonnegative; it may contain distances or squared distances.
Neither input is mutated, and self-neighbors are excluded by index.

Equal distances use increasing sample index. Scikit-learn's unstable sort and
neighbor backend can choose different tied neighbors, so exact parity is not
claimed for ties. Distance arithmetic uses float64, which can also distinguish
pairs that become ties in float32 reference arithmetic. Unsupported metrics and
callables raise or fail to type-check. Intermediate distance overflow raises.
Row-wise standard-library sorting uses linear auxiliary storage. It does not
retain the full pairwise distance or rank matrices.

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity`. The parity task
loads the recorded local scikit-learn source and verifies its bytes against Git
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. An installed sklearn package
supplies validation utilities; estimator code comes from the local reference.
Python is validation tooling only.

Trustworthiness compares untied ranks across both dtypes, five metrics and
multiple neighborhood sizes, with absolute score tolerance `1e-14` for scalar
rounding. ClassicalMDS compares eigenvalues and sign-fixed coordinates at
`2e-11` for float64, `2e-8` for float64 Minkowski powers other than 1, 2
and infinity, and `5e-6` for float32. The locked Mojo power operation differs
from SciPy by about `1e-10` relative, even for square roots: the `p=1.5`
fixtures have distance errors below `2.7e-9`. Minkowski `p=1`, `p=2` and
`p=inf` therefore use exact absolute sums, `sqrt` and maxima. The parity suite
checks every supported metric, these intermediate distances, eigenvalues and
coordinates, plus edge fixtures: more components than samples (shape, leading
spectrum and coordinates; null-space columns are roundoff), zero rows under
cosine, duplicate rows, constant data, a single sample, asymmetric precomputed
input and an unsupported metric.
Float32 tolerance accommodates
rounding in dissimilarities before the float64 eigensolve. Native tests cover
invalid inputs, clone, repeat fit, returned-array ownership and negative strides.
Raw comparison results remain under ignored `builds/`.

The build uses the current Moxie assembly tool and live dependency sources,
including working edits. `NUMERICAL-DEPENDENCIES.json` declares the decomposition
engine. Generated receipts record source hashes; no historical sources are restored.
