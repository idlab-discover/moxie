"""Estimator-shaped ownership fixture; deliberately no imputation algorithm."""
import numojo as nm
from moxie.impute._plumbing import FeatureLayout, MissingValue, missing_mask

struct ProbeA:
    var layout: FeatureLayout
    def __init__(out self):
        self.layout = FeatureLayout()
    def fit(mut self, x: nm.NDArray[DType.float64]) raises:
        var mask = missing_mask(x, MissingValue[DType.float64](-1))
        self.layout.fit(mask, error_on_new=False)
    def __sklearn_clone__(self) -> Self:
        return Self()
    def transform(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.bool]:
        var mask = missing_mask(x, MissingValue[DType.float64](-1))
        return self.layout.indicator(mask)
