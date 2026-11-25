"""Density and novelty capabilities are forwarded through an optional-target prefix."""
from moxie.base import ClusterMixin, DensityMixin, OutlierMixin, Predictor, clone
from moxie.neighbors import KernelDensity, LocalOutlierFactor
from moxie.pipeline import UnsupervisedPipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import Shape, zeros
from parity_io import ResultLine
from std.testing import assert_false, assert_raises
from unsupervised_fixtures import query_data, training_data


def density_score[D: DensityMixin](
    pipe: D, X: D.Input, y: Optional[D.Target],
) raises -> Float64:
    return pipe.score(X, y)


def decision[O: OutlierMixin](pipe: O, X: O.Input) raises -> O.Decision:
    return pipe.decision_function(X)


def main() raises:
    comptime KDE = UnsupervisedPipeline[StandardScaler[], KernelDensity[]]
    comptime assert not conforms_to(KDE, Predictor)
    comptime assert not conforms_to(KDE, ClusterMixin)
    var X = training_data[DType.float64]()
    var query = query_data[DType.float64]()
    var density = make_pipeline(StandardScaler(), KernelDensity(bandwidth=0.45))
    with assert_raises(contains="NotFittedError"):
        _ = density.score_samples(query)
    _ = density.fit(X)
    with assert_raises():
        _ = density.score(zeros[DType.float64](Shape(3, 1)))
    var line = ResultLine("density")
    line.add("samples", density.score_samples(query))
    line.scalar("score", density_score(density, query, None))
    var fresh = clone(density)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.bandwidth = 0.9
    line.add("refit", fresh.set_params(params).fit(X).score_samples(query))
    line.add("original", density.score_samples(query))
    line.emit()

    var novelty = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=2, novelty=True))
    with assert_raises(contains="NotFittedError"):
        _ = decision(novelty, query)
    _ = novelty.fit(X)
    line = ResultLine("novelty")
    line.add("prediction", novelty.predict(query))
    line.add("decision", decision(novelty, query))
    line.add("samples", novelty.score_samples(query))
    line.emit()
