"""Raising probes validate signatures without pretending to score or fit."""
import numojo as nm
from std.testing import assert_raises
from moxie.feature_selection import UnivariateScorer, SelectionPolicy, ImportanceProvider, ScoreResult

struct Scorer(UnivariateScorer):
    def __init__(out self):
        pass
    comptime dtype = DType.float64
    comptime score_dtype = DType.float64
    comptime Target = nm.NDArray[DType.float64]
    def score(self, x: nm.NDArray[Self.dtype], y: Optional[Self.Target]) raises -> ScoreResult[Self.dtype]:
        raise Error("probe: no score engine")

struct Policy(SelectionPolicy):
    def __init__(out self):
        pass
    comptime dtype = DType.float64
    def select(self, scores: ScoreResult[Self.dtype]) raises -> List[Bool]:
        raise Error("probe: no policy engine")

struct Provider(ImportanceProvider):
    def __init__(out self):
        pass
    comptime dtype = DType.float64
    def feature_importances(self) raises -> nm.NDArray[Self.dtype]:
        raise Error("probe: no fitted estimator")

def main() raises:
    var scorer = Scorer()
    var policy = Policy()
    var provider = Provider()
    var x = nm.ones[DType.float64](nm.Shape(2, 3))
    var result = ScoreResult[DType.float64](nm.ones[DType.float64](nm.Shape(3)))
    with assert_raises():
        _ = scorer.score(x, None)
    with assert_raises():
        _ = policy.select(result)
    with assert_raises():
        _ = provider.feature_importances()
