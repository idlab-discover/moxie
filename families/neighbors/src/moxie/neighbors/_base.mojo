"""Interfaces for the neighbors family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClassifierMixin,
    DensityMixin,
    InputEstimator,
    OutlierMixin,
    ProbabilityPredictor,
    RegressorMixin,
    SampleScorer,
)


trait KNeighborsQuery(InputEstimator):
    comptime NeighborResult: Movable

    def kneighbors(
        self,
        X: Optional[Self.Input] = None,
        n_neighbors: Optional[Int] = None,
        return_distance: Bool = True,
    ) raises -> Self.NeighborResult:
        ...


trait RadiusNeighborsQuery(InputEstimator):
    comptime NeighborResult: Movable

    def radius_neighbors(
        self,
        X: Optional[Self.Input] = None,
        radius: Optional[Float64] = None,
        return_distance: Bool = True,
        sort_results: Bool = False,
    ) raises -> Self.NeighborResult:
        ...


trait KNeighborsClassifierAPI(
    ClassifierMixin, KNeighborsQuery, ProbabilityPredictor
):
    pass


trait KNeighborsRegressorAPI(KNeighborsQuery, RegressorMixin):
    pass


trait RadiusNeighborsClassifierAPI(
    ClassifierMixin, ProbabilityPredictor, RadiusNeighborsQuery
):
    pass


trait RadiusNeighborsRegressorAPI(RadiusNeighborsQuery, RegressorMixin):
    pass


trait NeighborDensity(DensityMixin):
    pass


trait NoveltyDetector(OutlierMixin, SampleScorer):
    pass
