"""Configured mutual-information callable for the univariate selectors."""
import numojo as nm
from moxie._labels import StringTarget
from moxie.feature_selection._contracts import ScoreResult
from moxie.feature_selection._univariate_selection import ConfiguredScorer
from moxie.feature_selection._mutual_info import DiscreteFeatures, mutual_info_classif, mutual_info_regression

struct MutualInfoScorer[dt: DType = DType.float64, target_dt: DType = dt, target_type: Copyable = nm.NDArray[target_dt]](ConfiguredScorer):
    comptime dtype = Self.dt
    comptime score_dtype = DType.float64
    comptime Target = Self.target_type
    var classification: Bool
    var discrete_features: DiscreteFeatures
    var n_neighbors: Int
    var random_state: Optional[Int]
    var n_jobs: Optional[Int]

    def __init__(out self):
        self.classification = True
        self.discrete_features = DiscreteFeatures(String("auto"))
        self.n_neighbors = 3
        self.random_state = None
        self.n_jobs = None

    def __init__(out self, *, classification: Bool = True, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None):
        self.classification = classification
        self.discrete_features = discrete_features.copy()
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

    def clone_config(self) -> Self:
        return Self(classification=self.classification, discrete_features=self.discrete_features.copy(), n_neighbors=self.n_neighbors, random_state=self.random_state, n_jobs=self.n_jobs)

    def score(self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target]) raises -> ScoreResult[Self.score_dtype]:
        comptime assert Self.Target == StringTarget or Self.Target == nm.NDArray[Self.target_dt]
        if not y:
            raise Error("ValueError: mutual information requires y")
        var work = X.copy()
        var scores = nm.zeros[DType.float64](nm.Shape(0))
        comptime if Self.Target == StringTarget:
            if not self.classification:
                raise Error("ValueError: string targets require classification")
            scores = mutual_info_classif(work, rebind[StringTarget](y.value()), discrete_features=self.discrete_features.copy(), n_neighbors=self.n_neighbors, random_state=self.random_state, n_jobs=self.n_jobs)
        else:
            ref target = rebind[nm.NDArray[Self.target_dt]](y.value())
            if self.classification:
                scores = mutual_info_classif(work, target, discrete_features=self.discrete_features.copy(), n_neighbors=self.n_neighbors, random_state=self.random_state, n_jobs=self.n_jobs)
            else:
                comptime if Self.target_dt == DType.int64:
                    raise Error("ValueError: mutual information regression requires floating targets")
                else:
                    scores = mutual_info_regression(work, target, discrete_features=self.discrete_features.copy(), n_neighbors=self.n_neighbors, random_state=self.random_state, n_jobs=self.n_jobs)
        return ScoreResult[Self.score_dtype](scores^)
