"""GaussianNB composition using the native classifier owned by Moxie."""
from numojo import NDArray
from moxie._nb_gaussian import GaussianNB
from moxie._naive_bayes_dense import IntListDomain
from ._contracts import ScoringEstimatorAdapter
from ._self_training import PredictingEstimatorAdapter, LogProbabilityEstimatorAdapter

struct GaussianNBAdapter[dtype: DType = DType.float64](ScoringEstimatorAdapter, PredictingEstimatorAdapter, LogProbabilityEstimatorAdapter):
    comptime Input = NDArray[Self.dtype]
    comptime Label = Int
    comptime Estimator = GaussianNB[IntListDomain, Self.dtype]
    var estimator: Self.Estimator
    def __init__(out self, var estimator: Self.Estimator):
        self.estimator = estimator^
    def clone_unfitted(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())
    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        self.estimator.fit(X, y)
    def classes(self) raises -> List[Int]:
        return self.estimator.classes_.copy()
    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_proba(X).astype[DType.float64]()
    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_log_proba(X).astype[DType.float64]()
    def predict(self, X: Self.Input) raises -> List[Int]:
        return self.estimator.predict(X)
    def score(self, X: Self.Input, y: List[Int]) raises -> Float64:
        return self.estimator.score(X, y)
