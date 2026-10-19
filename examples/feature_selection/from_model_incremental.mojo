"""Select columns from PCA loadings learned across successive batches."""
import numojo as nm
from moxie.decomposition import IncrementalPCA
from moxie.feature_selection import SelectFromModel, IncrementalPCAImportance, IncrementalPCAUpdateParams

def main() raises:
    var batch = nm.zeros[DType.float64](nm.Shape(6, 3))
    for i in range(6):
        batch.unsafe_set(i * 3, Float64(i * 3))
        batch.unsafe_set(i * 3 + 1, Float64(i % 2))
        batch.unsafe_set(i * 3 + 2, Float64(i % 3))
    var selector = SelectFromModel(
        IncrementalPCAImportance(IncrementalPCA[DType.float64](1)),
        importance_getter="components_",
    )
    _ = selector.partial_fit(batch)
    _ = selector.partial_fit(batch, partial_fit_params=IncrementalPCAUpdateParams())
    print("Rows learned:", selector.estimator_().model.n_samples_seen_)
    print("Selected columns:", selector.get_support(True)[List[Int]])
    print("Transformed width:", selector.transform(batch).shape[1])
