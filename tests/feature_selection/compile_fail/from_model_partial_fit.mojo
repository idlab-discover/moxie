import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection import SelectFromModel, LassoImportance

def main() raises:
    var selector = SelectFromModel(LassoImportance(Lasso[DType.float64]()))
    _ = selector.partial_fit(nm.zeros[DType.float64](nm.Shape(4, 2)))
