# Dense decomposition model files

Use the experimental [Moxie model format](../../../docs/serialization.md) to
save configuration and learned state. Import the schema from
`moxie.decomposition_serialization` and select the model's input dtype at
compile time. The schema dtype defaults to Float64.

| Estimator | Schema |
| --- | --- |
| PCA with `RuntimePCABackend` | `PCASchema` |
| IncrementalPCA with `RuntimePCABackend` | `IncrementalPCASchema` |
| TruncatedSVD | `TruncatedSVDSchema` |
| FactorAnalysis | `FactorAnalysisSchema` |
| FastICA | `FastICASchema` |
| KernelPCA | `KernelPCASchema` |
| NMF | `NMFSchema` |
| MiniBatchNMF | `MiniBatchNMFSchema` |
| LatentDirichletAllocation | `LatentDirichletAllocationSchema` |
| SparseCoder | `SparseCoderSchema` |

```mojo
from moxie.decomposition import IncrementalPCA
from moxie.decomposition_serialization import IncrementalPCASchema
from moxie.serialization import save_model, load_model

var model = IncrementalPCA[DType.float32](2)
_ = model.partial_fit(first_batch)
save_model[IncrementalPCASchema[DType.float32]](model, "projection.moxie")
var restored = load_model[IncrementalPCASchema[DType.float32]]("projection.moxie")
_ = restored.partial_fit(next_batch)
```

All schemas restore independent learned arrays in logical row order and accept
unfitted instances. Current settings and learned state are stored separately
so parameter changes made after fitting survive a round trip.
IncrementalPCA stores its running mean and variance in Float64 for both model
dtypes and restores the sample count needed for the next update. A model fitted
with `partial_fit` can have `batch_size_ == 0`.
FactorAnalysis stores its learned arrays and likelihood history in Float64,
including for Float32 inputs. Schema version 2 also preserves its fit warning.
Version 1 files load with an empty `fit_warnings` list.
FastICA restores its unmixing matrix and any fitted whitening and centering
state. It preserves the optional Float64 `w_init` configuration in logical
row order.
KernelPCA restores the fitted training data, eigenspace and kernel centering
statistics, including for precomputed kernels and zero-component fits. It
stores any learned inverse map independently of the current
`fit_inverse_transform` setting.
NMF and MiniBatchNMF preserve the learned rank, reconstruction error, and
iteration count. MiniBatchNMF also restores the numerator and denominator
accumulators, step count, and fit-time settings needed to continue
`partial_fit`. A MiniBatchNMF model trained only by `partial_fit` can have
`n_steps_ > 0` and `n_iter_ == 0`.
LatentDirichletAllocation restores both topic matrices, the fitted priors,
iteration counters, bound and its owned MT19937 stream, including any cached
normal. This preserves the next `partial_fit` update even after a change to
the configured seed.
SparseCoder stores its finite, nonempty dictionary even before fitting because
it can transform without fitting. The fitted feature and component counts are
stored independently of the current dictionary. Coding algorithm support and
the Float64 requirement for `split_sign` follow the estimator contract.

The schemas support Float32 and Float64 inputs. Loading checks
the estimator identity, schema version, dtype, array shapes and fitted metadata.
Undefined variance ratios on constant data and undefined explained variance
after a single sample retain their NaNs. TruncatedSVD preserves positive
infinity in variance ratios from division by zero. FactorAnalysis preserves
positive infinity in likelihood history when a singular value is zero.

From the Moxie root, run the focused check:

```sh
pixi run python families/decomposition/tools/family.py serialization-check
```

It checks raw file round trips, resumption, strided learned arrays and invalid
state. The differential drivers pass restored models through the pinned
scikit-learn comparisons. The IncrementalPCA, MiniBatchNMF and
LatentDirichletAllocation drivers also restore state between partial-fit
batches. Set `MOXIE_ESTIMATORS` to a
comma-separated list of supported estimator names to select the differential
runs.
