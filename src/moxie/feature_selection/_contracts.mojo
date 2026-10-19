"""Extension signatures only; no scoring, importance aggregation or CV engines."""
import numojo as nm

struct ScoreResult[dtype: DType](Movable):
    var scores: nm.NDArray[Self.dtype]
    var pvalues: Optional[nm.NDArray[Self.dtype]]

    def __init__(out self, var scores: nm.NDArray[Self.dtype], var pvalues: Optional[nm.NDArray[Self.dtype]] = None) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if scores.ndim != 1:
            raise Error("ValueError: scores must have rank one")
        if pvalues:
            if pvalues.value().ndim != 1 or pvalues.value().shape[0] != scores.shape[0]:
                raise Error("ValueError: pvalues must match scores")
        self.scores = scores^
        self.pvalues = pvalues^

trait UnivariateScorer:
    comptime dtype: DType
    comptime score_dtype: DType
    comptime Target: Movable
    def score(self, x: nm.NDArray[Self.dtype], y: Optional[Self.Target]) raises -> ScoreResult[Self.score_dtype]:
        ...

trait SelectionPolicy:
    comptime dtype: DType
    def select(self, scores: ScoreResult[Self.dtype]) raises -> List[Bool]:
        ...

trait ImportanceProvider:
    comptime dtype: DType
    def feature_importances(self) raises -> nm.NDArray[Self.dtype]:
        ...
