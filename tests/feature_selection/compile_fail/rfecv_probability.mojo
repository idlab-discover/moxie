from moxie.feature_selection import RFECV, DenseSupervisedCV, LassoImportance
from moxie.linear_model import Lasso
import numojo as nm


def main() raises:
    comptime CV = DenseSupervisedCV[LassoImportance[DType.float64]]
    var selector = RFECV[CV](LassoImportance(Lasso()))
    _ = selector.predict_proba(nm.ones[DType.float64](nm.Shape(2, 2)))
