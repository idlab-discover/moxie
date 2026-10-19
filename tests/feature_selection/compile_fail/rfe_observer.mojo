import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection import RFE, LassoImportance
from rfe_path import HeldOutPath


def main() raises:
    var X32 = nm.ones[DType.float32](nm.Shape(4, 2))
    var y32 = nm.ones[DType.float32](nm.Shape(4))
    var observer = HeldOutPath[LassoImportance[DType.float32]](X32, y32)
    var selector = RFE(LassoImportance(Lasso[DType.float64]()))
    selector._fit_with_observer(X32.astype[DType.float64](), y32.astype[DType.float64](), observer)
