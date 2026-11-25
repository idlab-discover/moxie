"""Dense model-selection adapter for Gaussian process regression."""
from numojo import NDArray
from moxie.model_selection import (
    EstimatorScore,
    IndexArray,
    IndexSplit,
    KFold,
    SearchAdapter,
    select_rows,
)
from ._gpr import FminLBFGSB, GPROptimizer, GenericGaussianProcessRegressor


struct GaussianProcessSearchAdapter[
    dtype: DType = DType.float64,
    Optimizer: GPROptimizer = FminLBFGSB,
](SearchAdapter):
    """Search scalar alpha and nested numeric kernel parameters by name."""

    comptime Estimator = GenericGaussianProcessRegressor[Self.dtype, Self.Optimizer]
    comptime Dense = NDArray[Self.dtype]
    comptime Weight = List[Float64]
    comptime Scorer = EstimatorScore[Self.Estimator, Self.Dense, Self.Dense, Self.Weight]
    comptime Value = Float64

    def __init__(out self):
        pass

    def n_samples(self, X: Self.Dense, y: Self.Dense) raises -> Int:
        if X.ndim != 2 or (y.ndim != 1 and y.ndim != 2):
            raise Error("ValueError: expected rank-two features and rank-one or rank-two targets")
        if X.shape[0] != y.shape[0]:
            raise Error("ValueError: inconsistent sample counts")
        return X.shape[0]

    def input_rows(self, X: Self.Dense, rows: IndexArray, train: IndexArray) raises -> Self.Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Self.Dense, rows: IndexArray) raises -> Self.Dense:
        return select_rows(y, rows)

    def validate_metadata(self, metadata: Optional[Self.Weight], n_samples: Int) raises:
        if metadata and len(metadata.value()) != n_samples:
            raise Error("ValueError: sample_weight length does not match samples")

    def metadata_rows(self, metadata: Optional[Self.Weight], rows: IndexArray) raises -> Optional[Self.Weight]:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Self.Estimator, X: Self.Dense, y: Self.Dense, metadata: Optional[Self.Weight]) raises:
        if metadata:
            raise Error("ValueError: GaussianProcessRegressor does not support weighted fit")
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: Self.Estimator, params: Dict[String, Float64]) raises:
        if len(params) == 0:
            return
        var configuration = estimator.get_params()
        comptime prefix = "kernel__"
        for key in params.keys():
            if key == "alpha":
                configuration.alpha = params[key]
            elif key.startswith(prefix):
                if not configuration.kernel:
                    raise Error("ValueError: kernel parameters require an explicit kernel")
                var name = String(key[byte=prefix.byte_length() : key.byte_length()])
                configuration.kernel.value().set_param(name, params[key])
            else:
                raise Error("ValueError: unsupported GaussianProcessRegressor search parameter " + key)
        _ = estimator.set_params(configuration)

    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()

    def default_cv(self, X: Self.Dense, y: Self.Dense) raises -> List[IndexSplit]:
        var cursor = KFold().split(self.n_samples(X, y))
        var splits = List[IndexSplit]()
        while True:
            var split = cursor.next()
            if not split:
                return splits^
            splits.append(split.take())
