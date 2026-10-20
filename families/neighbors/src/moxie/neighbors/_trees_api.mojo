# API declarations adapted from scikit-learn (BSD-3-Clause).
# Authors: The scikit-learn developers; binary trees by Jake Vanderplas.
"""Deferred public spatial-tree API from neighbors/_binary_tree.pxi.tp."""
from ._api_types import APIValue

struct DeferredMetricRegistry(ImplicitlyCopyable):
    """Named registry placeholder; no metric support is claimed by this scaffold."""

    def __init__(out self):
        pass

    def __getitem__(self, algorithm: String) raises -> APIValue:
        raise Error("NotImplementedError: neighbor metric registry is an API scaffold")

struct DeferredMetricNames(ImplicitlyCopyable):
    """Placeholder for the reference tree's list of supported metric names."""

    def __init__(out self):
        pass

    def __getitem__(self, index: Int) raises -> String:
        raise Error("NotImplementedError: tree metric names are an API scaffold")

    def __len__(self) raises -> Int:
        raise Error("NotImplementedError: tree metric names are an API scaffold")

comptime VALID_METRICS = DeferredMetricRegistry()
comptime VALID_METRICS_SPARSE = DeferredMetricRegistry()

struct BallTree(Movable):
    """Public BallTree declarations; construction and all operations are deferred."""

    comptime valid_metrics = DeferredMetricNames()

    var data: APIValue
    var sample_weight: APIValue
    var sum_weight: APIValue
    var idx_array: APIValue
    var node_data: APIValue
    var node_bounds: APIValue

    def __init__(out self, data: Some[Movable], leaf_size: Some[Movable] = 40, metric: Some[Movable] = "minkowski", sample_weight: Some[Movable] = None, kwargs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, data, leaf_size=40, metric='minkowski', sample_weight=None, **kwargs)."
        raise Error("NotImplementedError: BallTree.__init__ is an API scaffold")

    def get_tree_stats(mut self) raises -> APIValue:
        "Scikit-learn signature: get_tree_stats(self)."
        raise Error("NotImplementedError: BallTree.get_tree_stats is an API scaffold")

    def reset_n_calls(mut self) raises -> APIValue:
        "Scikit-learn signature: reset_n_calls(self)."
        raise Error("NotImplementedError: BallTree.reset_n_calls is an API scaffold")

    def get_n_calls(mut self) raises -> APIValue:
        "Scikit-learn signature: get_n_calls(self)."
        raise Error("NotImplementedError: BallTree.get_n_calls is an API scaffold")

    def get_arrays(mut self) raises -> APIValue:
        "Scikit-learn signature: get_arrays(self)."
        raise Error("NotImplementedError: BallTree.get_arrays is an API scaffold")

    def query(mut self, X: Some[Movable], k: Some[Movable] = 1, return_distance: Some[Movable] = True, dualtree: Some[Movable] = False, breadth_first: Some[Movable] = False, sort_results: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False, sort_results=True)."
        raise Error("NotImplementedError: BallTree.query is an API scaffold")

    def query_radius(mut self, X: Some[Movable], r: Some[Movable], return_distance: Some[Movable] = False, count_only: Some[Movable] = False, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: query_radius(self, X, r, return_distance=False, count_only=False, sort_results=False)."
        raise Error("NotImplementedError: BallTree.query_radius is an API scaffold")

    def kernel_density(mut self, X: Some[Movable], h: Some[Movable], kernel: Some[Movable] = "gaussian", atol: Some[Movable] = 0, rtol: Some[Movable] = 1e-08, breadth_first: Some[Movable] = True, return_log: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1e-08, breadth_first=True, return_log=False)."
        raise Error("NotImplementedError: BallTree.kernel_density is an API scaffold")

    def two_point_correlation(mut self, X: Some[Movable], r: Some[Movable], dualtree: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: two_point_correlation(self, X, r, dualtree=False)."
        raise Error("NotImplementedError: BallTree.two_point_correlation is an API scaffold")

struct KDTree(Movable):
    """Public KDTree declarations; construction and all operations are deferred."""

    comptime valid_metrics = DeferredMetricNames()

    var data: APIValue
    var sample_weight: APIValue
    var sum_weight: APIValue
    var idx_array: APIValue
    var node_data: APIValue
    var node_bounds: APIValue

    def __init__(out self, data: Some[Movable], leaf_size: Some[Movable] = 40, metric: Some[Movable] = "minkowski", sample_weight: Some[Movable] = None, kwargs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, data, leaf_size=40, metric='minkowski', sample_weight=None, **kwargs)."
        raise Error("NotImplementedError: KDTree.__init__ is an API scaffold")

    def get_tree_stats(mut self) raises -> APIValue:
        "Scikit-learn signature: get_tree_stats(self)."
        raise Error("NotImplementedError: KDTree.get_tree_stats is an API scaffold")

    def reset_n_calls(mut self) raises -> APIValue:
        "Scikit-learn signature: reset_n_calls(self)."
        raise Error("NotImplementedError: KDTree.reset_n_calls is an API scaffold")

    def get_n_calls(mut self) raises -> APIValue:
        "Scikit-learn signature: get_n_calls(self)."
        raise Error("NotImplementedError: KDTree.get_n_calls is an API scaffold")

    def get_arrays(mut self) raises -> APIValue:
        "Scikit-learn signature: get_arrays(self)."
        raise Error("NotImplementedError: KDTree.get_arrays is an API scaffold")

    def query(mut self, X: Some[Movable], k: Some[Movable] = 1, return_distance: Some[Movable] = True, dualtree: Some[Movable] = False, breadth_first: Some[Movable] = False, sort_results: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False, sort_results=True)."
        raise Error("NotImplementedError: KDTree.query is an API scaffold")

    def query_radius(mut self, X: Some[Movable], r: Some[Movable], return_distance: Some[Movable] = False, count_only: Some[Movable] = False, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: query_radius(self, X, r, return_distance=False, count_only=False, sort_results=False)."
        raise Error("NotImplementedError: KDTree.query_radius is an API scaffold")

    def kernel_density(mut self, X: Some[Movable], h: Some[Movable], kernel: Some[Movable] = "gaussian", atol: Some[Movable] = 0, rtol: Some[Movable] = 1e-08, breadth_first: Some[Movable] = True, return_log: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1e-08, breadth_first=True, return_log=False)."
        raise Error("NotImplementedError: KDTree.kernel_density is an API scaffold")

    def two_point_correlation(mut self, X: Some[Movable], r: Some[Movable], dualtree: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: two_point_correlation(self, X, r, dualtree=False)."
        raise Error("NotImplementedError: KDTree.two_point_correlation is an API scaffold")
