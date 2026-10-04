"""Store supplied dense estimates without computing an estimator."""
import numojo as nm
from moxie.covariance._state import CovarianceState

def main() raises:
    var covariance = nm.ones[DType.float64](nm.Shape(1, 1))
    var location = nm.zeros[DType.float64](nm.Shape(1))
    var state = CovarianceState[DType.float64]()
    state.replace(location, covariance)
    covariance.unsafe_set(0, 2)
    print("Independent stored covariance:", state.covariance_.value().unsafe_get(0))
