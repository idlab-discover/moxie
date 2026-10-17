"""Must not compile: a covariance estimator lacking fit/covariance (sklearn's
KMeans-as-covariance_estimator ValueError becomes a compile-time trait error)."""
from numojo import NDArray, Shape, zeros
from moxie.discriminant_analysis import LinearDiscriminantAnalysis


@fieldwise_init
struct NotACovarianceEstimator(Copyable, Movable):
    var n_clusters: Int


def main() raises:
    var lda = LinearDiscriminantAnalysis[DType.float64, Int, NotACovarianceEstimator](solver="lsqr")
    lda.fit(zeros[DType.float64](Shape(4, 2)), [0, 0, 1, 1])
