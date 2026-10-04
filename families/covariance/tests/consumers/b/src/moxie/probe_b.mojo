"""Ownership probe, not a covariance estimator implementation."""
import numojo as nm
from moxie.covariance._state import CovarianceState
from moxie._dense import row_squared_norms

struct ProbeB:
    @staticmethod
    def snapshot[dtype: DType](x: nm.NDArray[dtype]) raises -> CovarianceState[dtype]:
        var state = CovarianceState[dtype]()
        state.replace(nm.zeros[dtype](nm.Shape(x.shape[1])), x)
        return state^

    @staticmethod
    def norms[dtype: DType](x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
        return row_squared_norms(x)

    @staticmethod
    def share[dtype: DType](mut x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
        return x.view()
