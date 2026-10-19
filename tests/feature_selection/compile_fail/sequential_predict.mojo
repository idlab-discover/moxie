from moxie.feature_selection import SequentialFeatureSelector, DenseSupervisedCV
from moxie.linear_model import Lasso
import numojo as nm


def main() raises:
    var selector = SequentialFeatureSelector[DenseSupervisedCV[Lasso[]]](Lasso())
    _ = selector.predict(nm.ones[DType.float64](nm.Shape(2, 2)))
