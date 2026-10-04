"""Extension contracts only. Dedicated children supply numerical engines."""
import numojo as nm
from ._state import CovarianceState


@fieldwise_init
struct CenteredData[dtype: DType](Copyable, Movable):
    var values: nm.NDArray[Self.dtype]
    var location: nm.NDArray[Self.dtype]


@fieldwise_init
struct ShrinkageStatistics[dtype: DType](Copyable, Movable):
    var empirical: nm.NDArray[Self.dtype]
    var n_samples: Int
    var trace: Float64
    var squared_frobenius: Float64
    var fourth_moment_sum: Float64


@fieldwise_init
struct RobustCovarianceState[dtype: DType](Copyable, Movable):
    var estimate: CovarianceState[Self.dtype]
    var raw_location_: nm.NDArray[Self.dtype]
    var raw_covariance_: nm.NDArray[Self.dtype]
    var raw_support_: nm.NDArray[DType.bool]
    var support_: nm.NDArray[DType.bool]
    var dist_: nm.NDArray[Self.dtype]


trait DenseCovarianceKernel:
    comptime dtype: DType


trait CovarianceEngine(DenseCovarianceKernel):

    @staticmethod
    def center(x: nm.NDArray[Self.dtype], assume_centered: Bool) raises -> CenteredData[Self.dtype]:
        ...

    @staticmethod
    def empirical(x: nm.NDArray[Self.dtype], assume_centered: Bool) raises -> nm.NDArray[Self.dtype]:
        ...

    @staticmethod
    def precision(covariance: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        ...

    @staticmethod
    def logdet(precision: nm.NDArray[Self.dtype]) raises -> Float64:
        ...

    @staticmethod
    def mahalanobis(x: nm.NDArray[Self.dtype], location: nm.NDArray[Self.dtype], precision: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        ...

    @staticmethod
    def likelihood(empirical: nm.NDArray[Self.dtype], precision: nm.NDArray[Self.dtype]) raises -> Float64:
        ...

    @staticmethod
    def error_norm(covariance: nm.NDArray[Self.dtype], comparison: nm.NDArray[Self.dtype], norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        ...

    @staticmethod
    def shrink(empirical: nm.NDArray[Self.dtype], shrinkage: Float64 = 0.1) raises -> nm.NDArray[Self.dtype]:
        ...


trait ShrinkageEngine(DenseCovarianceKernel):

    @staticmethod
    def statistics(centered: nm.NDArray[Self.dtype], block_size: Int = 1000) raises -> ShrinkageStatistics[Self.dtype]:
        ...

    @staticmethod
    def ledoit_wolf(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        ...

    @staticmethod
    def oas(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        ...
