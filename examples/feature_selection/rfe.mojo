"""Select two useful columns and predict with the final refitted model."""
import numojo as nm
from moxie.feature_selection import RFE, FeatureCount, LassoImportance
from moxie.linear_model import Lasso


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(8, 4))
    var y = nm.zeros[DType.float64](nm.Shape(8))
    var coefficients: List[Float64] = [6, -3, 0.5, 0]
    for row in range(8):
        X.unsafe_set(row * 4 + row % 4, 2)
        y.unsafe_set(row, 2 * coefficients[row % 4])
    var selector = RFE(
        LassoImportance(Lasso(alpha=0.1, fit_intercept=False)),
        n_features_to_select=FeatureCount(2),
    )
    selector.fit_named(X, y, [String("signal"), String("inverse_signal"), String("weak"), String("unused")])
    var names = selector.get_feature_names_out()
    print("Selected:", names[0], names[1])
    print("Original feature count:", selector.n_features_in_())
    print("Selected feature count:", selector.n_features_())
    print("First prediction:", selector.predict(X).unsafe_get(0))
    print("Training R2:", selector.score(X, y))
