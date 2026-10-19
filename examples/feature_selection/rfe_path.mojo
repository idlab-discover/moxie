"""Observe a single RFE path on held-out data; this is not RFECV."""
import numojo as nm
from moxie.base import SupervisedScorer
from moxie.linear_model import Lasso
from moxie.feature_selection import RFE, FeatureCount, LassoImportance, SelectionState, SelectionEstimator
from moxie.feature_selection._rfe import _RFEStepObserver


@fieldwise_init
struct PathStep(Movable):
    var n_features: Int
    var score: Float64
    var support: List[Bool]
    var ranking: List[Int]


struct HeldOutPath[E: SelectionEstimator & SupervisedScorer](_RFEStepObserver):
    """Example observer: owned validation data and independent step snapshots."""
    comptime Estimator = Self.E
    comptime Dense = nm.NDArray[Self.E.dtype]
    var X: Self.Dense
    var y: Self.Dense
    var score_weight: Optional[Self.Dense]
    var steps: List[PathStep]

    def __init__(out self, X: Self.Dense, y: Self.Dense,
                 score_weight: Optional[Self.Dense] = None):
        comptime assert Self.E.Input == Self.Dense
        comptime assert Self.E.Target == Self.Dense
        comptime assert Self.E.SampleWeight == Self.Dense
        self.X = X.copy()
        self.y = y.copy()
        self.score_weight = score_weight.copy()
        self.steps = List[PathStep]()

    def observe(mut self, estimator: Self.E,
                selection: SelectionState, ranking: List[Int]) raises:
        var reduced = selection.transform(self.X)
        var score = estimator.score(rebind[Self.E.Input](reduced), rebind[Self.E.Target](self.y),
                                    rebind[Optional[Self.E.SampleWeight]](self.score_weight))
        self.steps.append(PathStep(selection.selected_count(), score,
                                   selection.get_support()[List[Bool]].copy(), ranking.copy()))


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(8, 4))
    var y = nm.zeros[DType.float64](nm.Shape(8))
    var beta: List[Float64] = [6, -3, 0.5, 0]
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, 2 * beta[i % 4])
    var X_test = nm.zeros[DType.float64](nm.Shape(4, 4))
    var y_test = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X_test.unsafe_set(i * 4 + i, 3)
        y_test.unsafe_set(i, 3 * beta[i] + 0.25)
    var selector = RFE(LassoImportance(Lasso(alpha=0.1, fit_intercept=False)),
                       n_features_to_select=FeatureCount(1), step=FeatureCount(2))
    var observer = HeldOutPath[LassoImportance[DType.float64]](X_test, y_test)
    selector._fit_with_observer(X, y, observer)
    for i in range(len(observer.steps)):
        print("features", observer.steps[i].n_features, "validation R2", observer.steps[i].score)
