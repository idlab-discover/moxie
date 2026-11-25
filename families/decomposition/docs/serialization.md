# PCA and IncrementalPCA model files

Use the experimental [Moxie model format](../../../docs/serialization.md) to
save configuration and learned state for runtime PCA and IncrementalPCA. Select
the schema and dtype at compile time. The schema dtype defaults to Float64.

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

For PCA, select `PCASchema` with the model's dtype. Both schemas restore
independent learned arrays in logical row order and accept unfitted instances.
Learned rank, fitted solver, and learned batch size are stored separately from
the current settings so subsequent parameter changes survive a round trip.
IncrementalPCA stores its running mean and variance in Float64 for both model
dtypes and restores the sample count needed for the next update. A model fitted
with `partial_fit` can have `batch_size_ == 0`.

Version 1 targets Float32 and Float64 `RuntimePCABackend` models. Loading checks
the estimator identity, schema version, dtype, array shapes and fitted metadata.
Undefined variance ratios on constant data and undefined explained variance
after a single sample retain their NaNs. The loader rejects infinite learned
values. Files use Moxie's native format, with the model schema chosen by the
caller.

From the Moxie root, run the focused check:

```sh
pixi run python families/decomposition/tools/family.py serialization-check
```

It checks raw file round trips, resumption, strided learned arrays and invalid
state. The PCA and IncrementalPCA differential drivers send restored models
through the existing pinned scikit-learn comparisons. The IncrementalPCA driver
also restores state between each pair of partial-fit batches.
