"""Capability-only probes for curve execution order and failure isolation."""
from lifecycle import Probe, ProbeAdapter, Dense
from moxie.model_selection import CVAdapter, ParameterAdapter, TypedScorer, validation_curve, IndexArray, IndexSplit, KFold, ErrorPolicy
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_raises


@fieldwise_init
struct CurveScorer(TypedScorer):
    comptime Estimator = Probe
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Int
    var fail: Bool
    def score(self, estimator: Probe, X: Dense, y: Dense, metadata: Int) raises -> Float64:
        assert_equal(estimator.fit_count, 1)
        if self.fail and y[Item(0)] == 0:
            raise Error("intentional score failure for one side of a fold")
        return Float64(estimator.parameter + metadata) + y[Item(0)]


@fieldwise_init
struct CurveAdapter(CVAdapter, ParameterAdapter):
    comptime Scorer = CurveScorer
    comptime Value = Int
    var trace: Bool
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return ProbeAdapter().n_samples(X, y)
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return ProbeAdapter().input_rows(X, rows, train)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return ProbeAdapter().target_rows(y, rows)
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        return metadata
    def configure(self, mut estimator: Probe, params: Dict[String, Int]) raises:
        assert_equal(estimator.fit_count, 0)
        for key in params:
            if key != "parameter":
                raise Error("ValueError: unknown parameter")
        _ = estimator.set_params(params["parameter"])
    def fit(self, mut estimator: Probe, X: Dense, y: Dense, metadata: Int) raises:
        if self.trace:
            var total = 0
            for i in range(y.shape[0]):
                total += Int(y[Item(i)])
            print("order", total * 10 + estimator.parameter)
        ProbeAdapter().fit(estimator, X, y, metadata)


def main() raises:
    var X = zeros[DType.float64](Shape(8, 1))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i)
    var model = Probe(7, 99)
    var cursor = KFold(4).split(8)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var adapter = CurveAdapter(True)
    var values: List[Int] = [3, 1, 3]
    var result = validation_curve[CurveAdapter](model, X, y, adapter, CurveScorer(False), splits, 0, 10, param_name="parameter", param_range=values)
    for fold in range(4):
        for parameter in range(3):
            assert_equal(result[0][Item(parameter, fold)], Float64(values[parameter] + 10 + (2 if fold == 0 else 0)))
            assert_equal(result[1][Item(parameter, fold)], Float64(values[parameter] + 10 + 2 * fold))
    adapter.trace = False
    var partial = validation_curve[CurveAdapter](model, X, y, adapter, CurveScorer(False), splits, 1, 10, param_name="parameter", param_range=values, error_score=ErrorPolicy(False, -999))
    var scoring = validation_curve[CurveAdapter](model, X, y, adapter, CurveScorer(True), splits, 0, 10, param_name="parameter", param_range=values, error_score=ErrorPolicy(False, -456))
    for parameter in range(3):
        for fold in range(4):
            assert_equal(partial[0][Item(parameter, fold)], result[0][Item(parameter, fold)] if fold == 0 else -999.0)
            assert_equal(partial[1][Item(parameter, fold)], result[1][Item(parameter, fold)] if fold == 0 else -999.0)
            assert_equal(scoring[0][Item(parameter, fold)], result[0][Item(parameter, fold)] if fold == 0 else -456.0)
            assert_equal(scoring[1][Item(parameter, fold)], -456.0 if fold == 0 else result[1][Item(parameter, fold)])
            print("partial", parameter, fold, partial[0][Item(parameter, fold)], partial[1][Item(parameter, fold)])
            print("scoring", parameter, fold, scoring[0][Item(parameter, fold)], scoring[1][Item(parameter, fold)])
    assert_equal(model.parameter, 7)
    assert_equal(model.fit_count, 99)
    # Invalid indices must fail before any fit; they are not numeric fit failures.
    adapter.trace = True
    splits[0].test[Item(0)] = 8
    with assert_raises():
        _ = validation_curve[CurveAdapter](model, X, y, adapter, CurveScorer(False), splits, 0, 10, param_name="parameter", param_range=values)
    print("PASS curve clone isolation, split-first order, partial fit/score failures and index validation")
