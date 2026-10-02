"""Extension points only. Concrete vocabulary/statistics engines have child owners."""
from numojo import NDArray
from moxie.base import BaseEstimator


trait LabelTransformer(BaseEstimator):
    """Target-only fit(y), distinct from TargetEncoder.fit(X, y)."""
    comptime Labels: Copyable & Movable
    comptime Encoded: Copyable & Movable
    def fit(mut self, y: Self.Labels) raises: ...
    def transform(self, y: Self.Labels) raises -> Self.Encoded: ...
    def fit_transform(mut self, y: Self.Labels) raises -> Self.Encoded: ...
    def inverse_transform(self, y: Self.Encoded) raises -> Self.Labels: ...


trait CategoryVocabulary(Copyable, Movable):
    """One typed column; selection/grouping/unknown policy stays with the encoder."""
    comptime Value: Copyable & Movable
    def size(self) -> Int: ...
    def category(self, index: Int) raises -> Self.Value: ...
    def lookup(self, value: Self.Value, *, is_missing: Bool = False) raises -> Int: ...


struct FeatureMoments(Copyable, Movable):
    """Owned feature vectors, not a moments algorithm. Counts may be weighted."""
    var count: NDArray[DType.float64]
    var mean: NDArray[DType.float64]
    var variance: NDArray[DType.float64]

    def __init__(out self, count: NDArray[DType.float64],
                 mean: NDArray[DType.float64], variance: NDArray[DType.float64]) raises:
        if count.ndim != 1 or mean.ndim != 1 or variance.ndim != 1:
            raise Error("ValueError: moments must be feature vectors")
        if count.shape[0] != mean.shape[0] or count.shape[0] != variance.shape[0]:
            raise Error("ValueError: moments have different feature counts")
        self.count = count.copy()
        self.mean = mean.copy()
        self.variance = variance.copy()


trait FeatureStatistics(Copyable, Movable):
    """Optional engine interface for compatible population-moment consumers."""
    def compute[dtype: DType](self, x: NDArray[dtype],
        weights: Optional[NDArray[DType.float64]] = None) raises -> FeatureMoments: ...
    def merge(self, left: FeatureMoments, right: FeatureMoments) raises -> FeatureMoments: ...


trait DenseOutputBuilder(Copyable, Movable):
    """Optional explicit output construction; no implicit sparse conversion."""
    def build[dtype: DType](self, n_samples: Int, n_features_out: Int) raises -> NDArray[dtype]: ...
