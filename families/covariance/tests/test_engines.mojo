import numojo as nm
from std.testing import assert_raises
from moxie.covariance._engines import CovarianceEngine, ShrinkageEngine, CenteredData, ShrinkageStatistics

struct RaisingEngine(CovarianceEngine, ShrinkageEngine):
    comptime dtype = DType.float64

    @staticmethod
    def center(x: nm.NDArray[Self.dtype], assume_centered: Bool) raises -> CenteredData[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def empirical(x: nm.NDArray[Self.dtype], assume_centered: Bool) raises -> nm.NDArray[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def precision(covariance: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def logdet(precision: nm.NDArray[Self.dtype]) raises -> Float64:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def mahalanobis(x: nm.NDArray[Self.dtype], location: nm.NDArray[Self.dtype], precision: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def likelihood(empirical: nm.NDArray[Self.dtype], precision: nm.NDArray[Self.dtype]) raises -> Float64:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def error_norm(covariance: nm.NDArray[Self.dtype], comparison: nm.NDArray[Self.dtype], norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def shrink(empirical: nm.NDArray[Self.dtype], shrinkage: Float64 = 0.1) raises -> nm.NDArray[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")



    @staticmethod
    def statistics(centered: nm.NDArray[Self.dtype], block_size: Int = 1000) raises -> ShrinkageStatistics[Self.dtype]:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def ledoit_wolf(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        raise Error("Contract probe: numerical engine is not implemented")

    @staticmethod
    def oas(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        raise Error("Contract probe: numerical engine is not implemented")

def dispatch[E: CovarianceEngine & ShrinkageEngine]() raises:
    var x = nm.ones[E.dtype](nm.Shape(2, 2))
    var loc = nm.zeros[E.dtype](nm.Shape(2))
    var stats = ShrinkageStatistics[E.dtype](x.copy(), 2, 2, 4, 8)
    with assert_raises():
        _ = E.center(x, False)
    with assert_raises():
        _ = E.empirical(x, True)
    with assert_raises():
        _ = E.precision(x)
    with assert_raises():
        _ = E.logdet(x)
    with assert_raises():
        _ = E.mahalanobis(x, loc, x)
    with assert_raises():
        _ = E.likelihood(x, x)
    with assert_raises():
        _ = E.error_norm(x, x)
    with assert_raises():
        _ = E.shrink(x)
    with assert_raises():
        _ = E.statistics(x)
    with assert_raises():
        _ = E.ledoit_wolf(stats)
    with assert_raises():
        _ = E.oas(stats)

def main() raises:
    dispatch[RaisingEngine]()
    print("PASS: raising engine extension contracts")
