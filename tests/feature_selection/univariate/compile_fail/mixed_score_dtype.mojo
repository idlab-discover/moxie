"""A scorer must return its declared score dtype."""
import numojo as nm
from moxie.feature_selection import UnivariateScorer, ScoreResult, f_regression

struct Float32Regression(UnivariateScorer):
    comptime dtype = DType.float32
    comptime score_dtype = DType.float32
    comptime Target = nm.NDArray[DType.float32]
    def score(self, x: nm.NDArray[DType.float32], y: Optional[Self.Target]) raises -> ScoreResult[DType.float64]:
        return f_regression(x, y.value())

def main():
    pass
