# Dense t-SNE

```mojo
from moxie.manifold import TSNE

var model = TSNE[DType.float64](perplexity=20, random_state=7)
var embedding = model.fit_transform(X)
print(model.kl_divergence_, model.n_iter_)
```

Pass a finite rank-two NuMojo `NDArray` with samples in rows, including
valid strided views. `TSNE` defaults to Float64 input. Both Float32 and
Float64 inputs produce Float32 coordinates by default, matching sklearn's
built-in initializers. Specify `TSNE[input_dtype, DType.float64]` with a
Float64 `init` array to keep its precision. Supply a named array with
`init=coordinates.copy()` to retain it, or move it with `init=coordinates^`.
Construction, cloning and parameter updates snapshot array configuration.
`fit_transform` returns coordinates independently owned from `embedding_`.

The default `method="barnes_hut"` supports one to three output dimensions.
Its `angle` parameter sets the tree opening threshold. At zero, the tree
walk visits every leaf and the input probabilities still use the fixed-k
neighborhood. `method="exact"` uses all input pairs and supports more than
three output dimensions. `perplexity` must be positive and smaller than
the sample count. `init="pca"` uses the decomposition family's PCA
with automatic solver selection, then
rescales the Float32 scores to a first-component population standard
deviation of `1e-4`. Zero first-component variance raises. `init="random"`
uses the shared MT19937 normal stream and converts each draw to Float32
before scaling it by `1e-4`. An integer seed resets the stream at each fit.
`None` borrows the native global stream for random initialization. The
optional seed is an integer, and NumPy's mutable `RandomState` is a Python
binding type. The Float64 embedding option widens the built-in
initializations after Float32 scaling.

The neighbors engine supplies `euclidean`, `l2`, `sqeuclidean`,
`manhattan`, `cityblock`, `l1`, `cosine`, `chebyshev`, `minkowski`,
`canberra`, `correlation`, `hamming`, `jaccard`, `dice`,
`rogerstanimoto`, `russellrao`, `sokalsneath`, `yule`, `seuclidean`,
`mahalanobis`, `nan_euclidean` and `haversine`. Use `TSNEMetricParams[dtype]`
for Minkowski `p` and `w`, standardized Euclidean `V`, Mahalanobis `VI`,
or `nan_euclidean`'s finite missing-value sentinel. TSNE validates finite
input before metric evaluation. When `metric="precomputed"`, pass
a square nonnegative distance matrix. The algorithm squares it internally,
excludes the diagonal, and uses both input triangles when building the
conditional distributions. The input is borrowed and remains intact. Precomputed
distances require random or explicit array initialization. Unsupported metric
names raise `ValueError`.

Both methods accept a native metric through
`metric="callable", metric_params=TSNEMetricParams[dtype](distance=my_distance)`.
The callback type is
`def(NDArray[dtype], Int, NDArray[dtype], Int) thin raises -> Float64`.
The two integers select rows in the corresponding borrowed arrays. The
callback can use their strides to read features without making row copies.
Supply a pure, symmetric distance that returns finite nonnegative values.
Exact t-SNE mirrors the upper triangle and evaluates the diagonal. Barnes–Hut
queries only distinct rows through the bounded neighbor search and stores the
selected distances in the input dtype before squaring. Native callbacks have
a fixed signature and use the typed parameter record in place of Python
callables and keyword dictionaries.

Fitting records `embedding_`, `kl_divergence_`, `n_iter_`,
`learning_rate_` and `n_features_in_` after success. `n_iter_` is the
reference optimizer's last iteration index. If all 250 exploration
iterations ran and `max_iter=250`, sklearn starts an empty second phase
and exposes its maximum-Float64 loss sentinel and index 250. A successful
refit replaces the learned attributes. `get_params` returns a typed
`TSNEParams` record, `set_params` accepts the record or keyword updates,
and `clone` returns an unfitted estimator. Use the record to reset optional
parameters. `fit` and `set_params` return a mutable borrow. An NDArray
carries no input column names. `get_feature_names_out` produces
`tsne0`, `tsne1` and so on after fitting. Output uses native arrays.
`n_jobs` retains the input configuration and the dense loops run serially.

Both methods use a Gaussian perplexity search, symmetrized pair probabilities
and a Student t KL objective. Distances enter the perplexity search as Float32
values, with Float64 probability arithmetic. Numax `core.libm.exp` and `log`
supply the Float64 elementary functions.

Exact t-SNE uses quadratic memory, reuses condensed Student t weights and
accumulates both endpoint gradients in one pass over pairs. Barnes–Hut uses
the neighbors owner's dense `BruteForce.kneighbors`, with
`k = min(n_samples - 1, floor(3 * perplexity + 1))`. Its owned `TopKResult`
carries the same fixed-k arrays as `DenseNeighborhood` and avoids a second
snapshot of the result. The query snapshots the input for the duration of the
probability calculation and resolves distance ties by sample index.
Symmetrization merges unordered pair weights in a private list of at most
`n_samples * k` entries. The tree traverses one to three dimensions for
repulsive forces and uses Float32 coordinates and forces, even when the
optimizer's output dtype is Float64. As in sklearn, the tree groups positions
within `1e-6` per coordinate and excludes matching leaves from repulsion.

Numax pairwise tensor operations would materialize all coordinate differences.
The neighbors query, private pair list, and tree avoid that allocation.
The input neighbor scan still examines all candidate pairs.

The locked MAX layout lacks `DefaultEngine`, which
`numax.stats.statistics.stddev` requires. PCA normalization uses
two passes over the first coordinate column with Float32 accumulation and
constant scratch space. The shared numerical policy excludes NuMojo variance.
[Shared numerical policy](../../../shared/numerics/docs/numerics.md)

Run the scoped contracts and parity from the Moxie root with the locked environment.
The parity command requires Cython, setuptools and a C compiler to build
the pinned perplexity, quad-tree and Barnes–Hut extensions under the ignored
family `builds/`. The harness specializes the pinned sklearn allocator for
tree cells to isolate it from allocator ABI differences in the installed wheel.

```sh
python families/manifold/tools/check.py tsne
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python families/manifold/tools/tsne_parity.py
```

Parity checks the probabilities, KL and gradient separately before comparing
complete fits. Float64 explicit-initialization fixtures use coordinate
tolerances of `3e-9` to account for the amplification of reduction-order
roundoff across iterations. Float32 tolerances account for the state and
gradient rounding in every update. The random initializer is checked
bit for bit, and seeded fits compare KL and neighborhood preservation
across seeds. Barnes–Hut comparisons cover fixed-k probabilities, KL, and
forces for one, two and three dimensions, angles from zero to 0.85, and
coincident points. Barnes–Hut's Float32 summation order differs from sklearn's
CSR row reductions. Explicit-initialization fit tolerances account for that
rounding across iterations. Callable-distance comparisons cover both input
dtypes and both optimization methods. SciPy, NumPy and scikit-learn are used
in validation only.
The estimator and perplexity oracle sources come from sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
