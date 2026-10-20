"""Inspect missingness and layout before an algorithm fills values."""
import numojo as nm
from moxie.impute._plumbing import MissingValue, missing_mask, FeatureLayout

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(2, 2))
    x.unsafe_set(0, -1)
    var mask = missing_mask(x, MissingValue[DType.float64](-1))
    var layout = FeatureLayout()
    layout.fit(mask, error_on_new=False)
    print("Original feature indices with missing values:", layout.indicator_features)
