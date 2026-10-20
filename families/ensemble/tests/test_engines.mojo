"""Concrete compile-only engine probes: every algorithm call raises."""
from moxie.base import BaseEstimator, InputEstimator
from moxie._random import RandomState
from moxie.ensemble._records import (
    Values, Indices, Bins, Gradients, SamplingPlan, SampleSelection,
    ClassAlignment, OOBAccumulator, BinMetadata, Histogram,
    HistogramSplit, HistogramPredictor, GrowerConfig,
)

from moxie.ensemble._engine_api import EnsembleSampler, EnsembleAggregator, HistogramBinner, HistogramGrower
from std.testing import assert_raises, assert_equal
from numojo import full, Shape

@fieldwise_init
struct ProbeEnsembleSampler(EnsembleSampler):
    def select(self, plan: SamplingPlan, weights: Optional[Values], mut rng: RandomState) raises -> SampleSelection:
        raise Error("Scaffold: engine not implemented")
    def unsampled(self, selection: SampleSelection) raises -> Indices:
        raise Error("Scaffold: engine not implemented")

@fieldwise_init
struct ProbeEnsembleAggregator(EnsembleAggregator):
    def add_regression(self, predictions: Values, weight: Float64, mut sums: Values) raises:
        raise Error("Scaffold: engine not implemented")
    def add_probability(self, probabilities: Values, alignment: ClassAlignment, weight: Float64, mut sums: Values) raises:
        raise Error("Scaffold: engine not implemented")
    def add_oob(self, rows: Indices, predictions: Values, mut state: OOBAccumulator) raises:
        raise Error("Scaffold: engine not implemented")
    def normalize(self, sums: Values, denominators: Values) raises -> Values:
        raise Error("Scaffold: engine not implemented")

@fieldwise_init
struct ProbeHistogramBinner(HistogramBinner):
    def fit(mut self, X: Values, weights: Optional[Values], max_bins: Int, categorical: Indices, mut rng: RandomState) raises -> BinMetadata:
        raise Error("Scaffold: engine not implemented")
    def transform(self, X: Values, metadata: BinMetadata, training: Bool) raises -> Bins:
        raise Error("Scaffold: engine not implemented")
    def accumulate(self, X: Bins, rows: Indices, gradients: Gradients, hessians: Gradients, constant_hessian: Bool, n_bins: Int) raises -> Histogram:
        raise Error("Scaffold: engine not implemented")
    def subtract(self, parent: Histogram, sibling: Histogram) raises -> Histogram:
        raise Error("Scaffold: engine not implemented")

@fieldwise_init
struct ProbeHistogramGrower(HistogramGrower):
    def find_split(self, hist: Histogram, metadata: BinMetadata, config: GrowerConfig, monotonic: Indices, allowed_features: Indices, lower: Float64, upper: Float64) raises -> HistogramSplit:
        raise Error("Scaffold: engine not implemented")
    def grow(self, X: Bins, metadata: BinMetadata, gradients: Gradients, hessians: Gradients, constant_hessian: Bool, config: GrowerConfig, monotonic: Indices, interactions: List[Indices], mut rng: RandomState) raises -> HistogramPredictor:
        raise Error("Scaffold: engine not implemented")
    def predict(self, tree: HistogramPredictor, X: Values, metadata: BinMetadata) raises -> Values:
        raise Error("Scaffold: engine not implemented")
    def predict_binned(self, tree: HistogramPredictor, X: Bins, missing_bin: UInt8) raises -> Values:
        raise Error("Scaffold: engine not implemented")

def sampler[T: EnsembleSampler](engine: T) raises:
    var rng = RandomState(42)
    with assert_raises():
        _ = engine.select(SamplingPlan(2, 1, 2, 1, True, False), None, rng)

def binner[T: HistogramBinner](mut engine: T) raises:
    var X = full[DType.float64](Shape(2, 1), 1)
    var categories = full[DType.int64](Shape(0), 0)
    var rng = RandomState(42)
    with assert_raises():
        _ = engine.fit(X, None, 255, categories, rng)

def aggregator[T: EnsembleAggregator](engine: T) raises:
    var predictions = full[DType.float64](Shape(2, 1), 1)
    var sums = full[DType.float64](Shape(2, 1), 0)
    with assert_raises():
        engine.add_regression(predictions, 1, sums)

def grower[T: HistogramGrower](engine: T) raises:
    var hist = Histogram(full[DType.float64](Shape(1, 2), 0),
                         full[DType.float64](Shape(1, 2), 0),
                         full[DType.uint32](Shape(1, 2), 0))
    var metadata = BinMetadata(List[Values](), full[DType.int64](Shape(1), 1),
        full[DType.uint8](Shape(1), 0), List[Values](), UInt8(1))
    var config = GrowerConfig(31, -1, 20, 0, 0.001, 0, 0.1, 1)
    var signs = full[DType.int64](Shape(1), 0)
    var allowed = full[DType.int64](Shape(1), 0)
    with assert_raises():
        _ = engine.find_split(hist, metadata, config, signs, allowed, -1, 1)

def main() raises:
    sampler(ProbeEnsembleSampler())
    var bins = ProbeHistogramBinner()
    binner(bins)
    aggregator(ProbeEnsembleAggregator())
    grower(ProbeHistogramGrower())
    print("PASS raising engine extension probes")
