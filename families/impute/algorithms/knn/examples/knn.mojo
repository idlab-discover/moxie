"""Fit missing-aware neighbors and impute a dense table."""
import numojo as nm
from moxie.impute import KNNImputer
from moxie.impute._plumbing import MissingValue


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 3))
    var values: List[Float64] = [1, 2, -1, 3, 4, 3, -1, 6, 5, 8, 8, 7]
    for i in range(len(values)): X.unsafe_set(i, values[i])
    var model = KNNImputer(n_neighbors=2, missing_values=MissingValue[DType.float64](-1), add_indicator=True)
    model.fit_named(X, ["first", "second", "third"])
    var filled = model.transform(X)
    print("Columns:", model.get_feature_names_out())
    for row in range(filled.shape[0]):
        for col in range(filled.shape[1]):
            print(filled.unsafe_get(row * filled.shape[1] + col), end=" ")
        print()
