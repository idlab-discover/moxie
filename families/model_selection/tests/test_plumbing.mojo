from std.testing import assert_equal, assert_false, assert_raises
from numojo import NDArray, Shape, full, Item
from moxie.model_selection import IndexSplit, validate_indices, ScoreTable, validate_refit_index, require_refit, Candidate

def main() raises:
    var indices = full[DType.int64](Shape(2), 0)
    indices[Item(1)] = 2
    validate_indices(indices, 3)
    var split = IndexSplit(indices.copy(), indices.copy())
    var independent = split.copy()
    independent.train[Item(0)] = 1
    assert_equal(Int(split.train[Item(0)]), 0)
    split.validate(3)
    with assert_raises():
        validate_indices(indices, 2)
    indices[Item(0)] = -1
    with assert_raises():
        validate_indices(indices, 3)
    with assert_raises():
        validate_indices(full[DType.int64](Shape(1, 2), 0), 3)
    validate_indices(full[DType.int64](Shape(0), 0), 0)
    with assert_raises():
        validate_indices(full[DType.int64](Shape(0), 0), -1)
    var names: List[String] = ["score"]
    var table = ScoreTable(full[DType.float64](Shape(2, 3, 1), 0), None, names^)
    table.validate()
    table.metric_names.append("extra")
    with assert_raises():
        table.validate()
    validate_refit_index(0, 1)
    with assert_raises():
        validate_refit_index(-1, 1)
    with assert_raises():
        validate_refit_index(1, 1)
    with assert_raises():
        require_refit(False, True)
    with assert_raises():
        require_refit(True, False)
    require_refit(True, True)
    var candidate = Candidate[Int](7, 0)
    candidate.validate()
    candidate.ordinal = -1
    with assert_raises():
        candidate.validate()
    var cursor = CursorProbe()
    with assert_raises():
        _ = cursor.next()
    var candidates = CandidatesProbe()
    with assert_raises():
        _ = candidates.next()
    var random_state = RandomState(42)
    with assert_raises():
        _ = DistributionProbe().sample(random_state)
    with assert_raises():
        _ = SelectorProbe().select(table)
    var duplicate_names: List[String] = ["score", "score"]
    var duplicate = ScoreTable(full[DType.float64](Shape(1, 1, 2), 0), None, duplicate_names^)
    with assert_raises():
        duplicate.validate()
    var empty_name: List[String] = [""]
    var bad_name = ScoreTable(full[DType.float64](Shape(1, 1, 1), 0), None, empty_name^)
    with assert_raises():
        bad_name.validate()
    var one_name: List[String] = ["score"]
    var train = full[DType.float64](Shape(1, 2, 1), 0)
    var mismatch = ScoreTable(full[DType.float64](Shape(1, 1, 1), 0), train^, one_name^)
    with assert_raises():
        mismatch.validate()
    var good_names: List[String] = ["score"]
    var good_train = full[DType.float64](Shape(1, 1, 1), 0)
    var good = ScoreTable(full[DType.float64](Shape(1, 1, 1), 0), good_train^, good_names^)
    good.validate()
    print("Dense plumbing checks passed")

# Signature-only extension probes raise: they implement no numerical engine.
from moxie.model_selection import SplitCursor, ParameterCandidates, ParameterDistribution, RefitSelector, IndexArray
from moxie._random import RandomState

@fieldwise_init
struct CursorProbe(SplitCursor):
    def next(mut self) raises -> Optional[IndexSplit]:
        raise Error("Contract probe: no splitter implemented")

@fieldwise_init
struct CandidatesProbe(ParameterCandidates):
    comptime Params = Int
    def next(mut self) raises -> Optional[Int]:
        raise Error("Contract probe: no parameter generator implemented")

@fieldwise_init
struct DistributionProbe(ParameterDistribution):
    comptime Value = Float64
    def sample(self, mut random_state: RandomState) raises -> Float64:
        raise Error("Contract probe: no distribution implemented")

@fieldwise_init
struct SelectorProbe(RefitSelector):
    comptime Results = ScoreTable
    def select(self, results: ScoreTable) raises -> Int:
        raise Error("Contract probe: no selection algorithm implemented")
