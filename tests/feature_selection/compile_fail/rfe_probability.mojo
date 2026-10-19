from moxie.feature_selection import RFE, LassoImportance
from moxie.linear_model import Lasso
import numojo as nm


def main() raises:
    var selector = RFE(LassoImportance(Lasso()))
    _ = selector.predict_proba(nm.ones[DType.float64](nm.Shape(2, 2)))
