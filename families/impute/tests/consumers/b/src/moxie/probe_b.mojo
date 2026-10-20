"""Second independent consumer of the same pinned family implementation."""
import numojo as nm
from moxie.impute._plumbing import FeatureLayout, MissingValue, missing_mask
from moxie._dense import row_squared_norms

struct ProbeB:
    var layout: FeatureLayout
    def __init__(out self):
        self.layout = FeatureLayout()
    def fit(mut self, x: nm.NDArray[DType.float64]) raises:
        var mask = missing_mask(x, MissingValue[DType.float64](-1))
        self.layout.fit(mask, error_on_new=False)
    def transform(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.bool]:
        var mask = missing_mask(x, MissingValue[DType.float64](-1))
        return self.layout.indicator(mask)
    def norms(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return row_squared_norms(x)
