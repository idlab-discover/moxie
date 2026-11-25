# Dense MDS and SMACOF

`MDS[dtype]` takes a finite Float32 or Float64 NuMojo matrix. Pass feature
rows with `metric="euclidean"` or a supported named distance, or pass symmetric
dissimilarities with `metric="precomputed"`. The estimator stores
`dissimilarity_matrix_` in the input dtype, `embedding_` in Float64,
and learns `stress_`, `n_iter_`, and `n_features_in_`. A successful fit owns
its learned arrays and the returned embedding has independent storage. A fitted
copy also owns its arrays and warning records.

```mojo
from moxie.manifold import MDS, smacof

var model = MDS(init="classical_mds", metric_mds=False)
var embedding = model.fit_transform(X)
var result = smacof(D, metric=False, init=embedding, return_n_iter=True)
print(result.stress, result.n_iter.value())
```

- `fit` returns a mutable borrow. `fit_transform` implements the shared
  `EmbeddingEstimator` contract. Both accept named `init=` with a Float32
  or Float64 matrix. Its column count selects the embedding dimension and
  its presence selects one restart. The estimator has no out-of-sample transform.
- The default `init="warn"` resolves to random initialization and records
  the pinned deprecation warning. `init="classical_mds"` uses the native
  ClassicalMDS implementation. Legacy `dissimilarity` and boolean `metric`
  keep their pinned meanings and warnings. A conflicting distance name raises.
- The shared RNG supplies NumPy-compatible uniform draws for integer seeds.
  `None` uses the shared process stream. `n_jobs=1` draws serial
  starts from one stream. Multiple jobs draw each restart's seed from the
  primary stream and run on bounded MAX CPU workers, with the first minimum
  selected in restart order. Native negative job counts use the logical core count.
- `metric_mds=False` fits the observed upper triangle with
  `moxie.isotonic.IsotonicRegression(out_of_bounds="clip")` from the second
  iteration onward. Zero dissimilarities represent missing constraints.
  `normalized_stress="auto"` selects raw stress for metric MDS and Stress-1
  for non-metric MDS. A boolean overrides this default. Undefined non-metric
  disparity normalization raises. A collapsed metric embedding follows IEEE
  division when reporting Stress-1.
- `smacof` returns an owned `SMACOFResult` containing `embedding`, `stress`,
  `n_iter: Optional[Int]`, and `fit_warnings`. `return_n_iter=True`
  populates the field. Estimator warnings use `model.fit_warnings` and
  are replaced on each successful fit.
- Get or replace the full configuration with `MDSParams[dtype]`. Named
  `set_params` updates accept string metric names and a typed
  `NormalizedStress` value. Use the record to set a legacy boolean metric
  or clear optional values. Clone copies the configuration into an unfitted model.
- Named metrics use the live neighbors distance engine and typed
  `MDSMetricParams[dtype]` (`p`, `w`, `V`, `VI`, `missing_values`).
  Generic Python metric callables and Python RNG objects are outside this native
  interface. The input matrix must be finite, and every evaluated dissimilarity
  must be finite. Comparisons assume the engine's Float32 input-dtype rounding
  for the stored distance matrix, including metrics scikit-learn widens.

The Guttman update fuses ratio formation and row accumulation into
`sum_j ratio[i,j] * (X[i] - X[j]) / n`. Numax's `linalg.matmul` and
NuMojo's rank-two `@` would consume a materialized dense Guttman matrix.
The fused update uses the existing distance matrix and two coordinate buffers,
with no separate ratio or Guttman allocation. Non-metric fits also maintain
the disparities and observed pairs. NuMojo `astype` casts contiguous inputs.
Strided inputs are materialized by logical indexing because `astype` reads
linear storage. Bray–Curtis uses SciPy's
`sum(abs(x + y))` denominator to handle signed rows.

From the Moxie root with the locked environment active:

```sh
python -u families/manifold/tools/check.py mds
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u families/manifold/tools/mds_parity.py
```

The oracle uses scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The comparisons require exact shape, iteration counts and warning records for
the selected fixtures. Float64 comparisons use `3e-10` absolute and relative
tolerances for the direct-distance kernel and fused summation. Float32
comparisons allow `1e-4` for input-distance rounding and the Float64
optimization and output used for explicit Float32 initial coordinates.
Run `python -u families/manifold/tools/mds_parity.py --notebook` to execute only
the MDS comparisons in `notebooks/manifold/plot_lle_digits.ipynb`.
