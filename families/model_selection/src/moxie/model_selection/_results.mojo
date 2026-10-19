"""Owned result columns and refit extension points."""
from numojo import NDArray

@fieldwise_init
struct ScoreTable(Movable):
    # Shape (candidate, split, metric); Float64 even for Float32 estimators.
    var test_scores: NDArray[DType.float64]
    var train_scores: Optional[NDArray[DType.float64]]
    var metric_names: List[String]

    def validate(self) raises:
        if self.test_scores.ndim != 3:
            raise Error("ValueError: scores require candidate, split, metric axes")
        if self.test_scores.shape[2] != len(self.metric_names):
            raise Error("ValueError: metric names do not match scores")
        for i in range(len(self.metric_names)):
            if self.metric_names[i].byte_length() == 0:
                raise Error("ValueError: metric names must be nonempty")
            for j in range(i):
                if self.metric_names[i] == self.metric_names[j]:
                    raise Error("ValueError: duplicate metric name")
        if self.train_scores:
            if self.train_scores.value().ndim != 3:
                raise Error("ValueError: train scores require rank three")
            for axis in range(3):
                if self.train_scores.value().shape[axis] != self.test_scores.shape[axis]:
                    raise Error("ValueError: train/test score shapes differ")

@fieldwise_init
struct FitStatus(Copyable, Movable):
    var failed: Bool
    var message: String
    var fit_time: Float64
    var score_time: Float64

@fieldwise_init
struct ErrorPolicy(Copyable, Movable):
    var raise_on_error: Bool
    var error_score: Float64

trait RefitSelector(Movable):
    comptime Results: Movable
    def select(self, results: Self.Results) raises -> Int:
        ...


def validate_refit_index(index: Int, n_candidates: Int) raises:
    if index < 0 or index >= n_candidates:
        raise Error("ValueError: refit selection is outside candidate range")


def require_refit(enabled: Bool, fitted: Bool) raises:
    if not enabled:
        raise Error("ValueError: prediction requires refit=True")
    if not fitted:
        raise Error("NotFittedError: selected estimator has not been fitted")
