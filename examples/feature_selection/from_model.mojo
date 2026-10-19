"""Select features from a fitted Lasso without changing its optimizer."""
import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection import SelectFromModel, LassoImportance, MaxFeatures

def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(8, 4))
    var y = nm.zeros[DType.float64](nm.Shape(8))
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, 6.0 if i % 4 == 0 else 0.0)
    var selector = SelectFromModel(
        LassoImportance(Lasso(alpha=0.1, fit_intercept=False)),
        max_features=MaxFeatures(2),
    )
    _ = selector.fit(X, y^)
    print(selector.get_feature_names_out())
    print(selector.transform(X))
