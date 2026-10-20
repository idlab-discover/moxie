"""Impute integer measurements while retaining fractional means."""
import numojo as nm
from moxie.impute import IntegerSimpleImputer
from moxie.impute._plumbing import MissingValue

def main() raises:
    var X = nm.zeros[DType.int64](nm.Shape(3, 1))
    X.unsafe_set(0, 1)
    X.unsafe_set(1, 4)
    var model = IntegerSimpleImputer[DType.int64](missing_values=MissingValue[DType.int64](0))
    var means = model.fit_transform(X)
    print(means[nm.NDArray[DType.float64]])
    _ = model.set_params(strategy="constant", fill_value=Int64(7))
    var constants = model.fit_transform(X)
    print(constants[nm.NDArray[DType.int64]])
