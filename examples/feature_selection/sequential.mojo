"""Select features with real Lasso fits and five-fold cross-validation."""
import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection import SequentialFeatureSelector, DenseSupervisedCV, SequentialCount


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(30, 3))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X.unsafe_set(i * 3, Float64(i))
        X.unsafe_set(i * 3 + 1, Float64(i % 3))
        X.unsafe_set(i * 3 + 2, Float64(i % 2))
        y.unsafe_set(i, Float64(2 * i))
    comptime CV = DenseSupervisedCV[Lasso[DType.float64]]
    var selector = SequentialFeatureSelector[CV](Lasso(alpha=0.01), n_features_to_select=SequentialCount(1))
    _ = selector.fit(X, y)
    print("Selected columns:", selector.get_support(True)[List[Int]])
    print("Selected shape:", selector.transform(X).shape)
