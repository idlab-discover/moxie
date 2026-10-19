"""Choose a feature count by cross-validation, then predict with the final fit."""
import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection import RFECV, DenseSupervisedCV, LassoImportance


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(40, 4))
    var y = nm.zeros[DType.float64](nm.Shape(40))
    for i in range(40):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, Float64(6 if i % 4 == 0 else 0))
    comptime CV = DenseSupervisedCV[LassoImportance[DType.float64]]
    var selector = RFECV[CV](LassoImportance(Lasso(alpha=0.1, fit_intercept=False)))
    _ = selector.fit(X, y)
    print("Selected features:", selector.get_support(True)[List[Int]])
    print("Training score:", selector.score(X, y))
    var results = selector.cv_results_()
    var counts = results["n_features"][List[Int]].copy()
    ref means = results["mean_test_score"][nm.NDArray[DType.float64]]
    for i in range(len(counts)):
        print("Features:", counts[i], "mean validation score:", means.unsafe_get(i))
