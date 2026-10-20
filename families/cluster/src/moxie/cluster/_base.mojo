"""Interfaces for the cluster family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClusterMixin,
    Predictor,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait Clusterer(ClusterMixin):
    pass


trait PredictiveClusterer(Clusterer, Predictor):
    pass


trait ClusterDistanceTransformer(Clusterer, TransformerMixin):
    pass


trait IncrementalClusterer(Clusterer):
    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> ref[self] Self:
        ...


trait Biclusterer(UnsupervisedEstimator):
    comptime Indices: Movable
    comptime Submatrix: Movable

    def get_indices(self, i: Int) raises -> Self.Indices:
        ...

    def get_shape(self, i: Int) raises -> Tuple[Int, Int]:
        ...

    def get_submatrix(self, i: Int, data: Self.Input) raises -> Self.Submatrix:
        ...
