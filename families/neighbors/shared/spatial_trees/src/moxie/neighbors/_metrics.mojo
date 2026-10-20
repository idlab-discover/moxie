"""Implemented dense backend names; parameter constraints are checked at fit."""
from moxie.neighbors._distance import Metric

struct MetricNames[ball: Bool = False](ImplicitlyCopyable, Sized):
    def __init__(out self): pass
    def __len__(self) -> Int:
        return 22 if Self.ball else 9
    def __getitem__(self, index: Int) raises -> String:
        var names = VALID_METRICS["ball_tree" if Self.ball else "kd_tree"]
        if index < 0 or index >= len(names): raise Error("IndexError: metric index")
        return names[index]

struct MetricRegistry(ImplicitlyCopyable, Sized):
    def __init__(out self): pass
    def __len__(self) -> Int: return 3
    def keys(self) -> List[String]: return ["ball_tree", "kd_tree", "brute"]
    def __contains__(self, backend: String) -> Bool:
        return backend == "ball_tree" or backend == "kd_tree" or backend == "brute"
    def __getitem__(self, backend: String) raises -> List[String]:
        if backend == "kd_tree": return ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"]
        if backend == "ball_tree": return ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity", "seuclidean", "mahalanobis", "hamming", "canberra", "braycurtis", "jaccard", "dice", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "haversine", "pyfunc"]
        if backend == "brute": return ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "haversine", "jaccard", "l1", "l2", "mahalanobis", "manhattan", "minkowski", "nan_euclidean", "precomputed", "rogerstanimoto", "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule"]
        raise Error("KeyError: unknown neighbors backend")

comptime VALID_METRICS = MetricRegistry()

def kd_eligible[dtype: DType](metric: Metric[dtype]) -> Bool:
    return not metric.w and (metric.name == "euclidean" or metric.name == "manhattan" or metric.name == "chebyshev" or (metric.name == "minkowski" and metric.p >= 1))

def ball_eligible[dtype: DType](metric: Metric[dtype]) raises -> Bool:
    if metric.name == "minkowski" and (metric.p < 1 or metric.p == Float64("inf")): return False
    for name in VALID_METRICS["ball_tree"]:
        if metric.name == name: return True
    return False

def ball_prunable[dtype: DType](metric: Metric[dtype]) -> Bool:
    # Nonmetrics and Boolean formulas evaluated at numeric centroids cannot use
    # triangle bounds. Mahalanobis accepts arbitrary VI: no SPD assumption.
    return kd_eligible(metric) or metric.name == "seuclidean" or metric.name == "haversine" or metric.name == "canberra" or metric.name == "hamming" or ((metric.name == "minkowski" or metric.name == "wminkowski") and metric.p >= 1)
