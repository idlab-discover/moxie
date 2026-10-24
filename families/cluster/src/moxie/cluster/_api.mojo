# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.cluster._api_types import APIValue
from std.memory import bitcast
from moxie.cluster._api_types import APIExpressionDefault

struct HDBSCAN(Movable):

    "API scaffold for sklearn.cluster._hdbscan.hdbscan.HDBSCAN. Operations are deferred."

    var min_cluster_size: APIValue

    var min_samples: APIValue

    var cluster_selection_epsilon: APIValue

    var max_cluster_size: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var alpha: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var n_jobs: APIValue

    var cluster_selection_method: APIValue

    var allow_single_cluster: APIValue

    var store_centers: APIValue

    var copy: APIValue

    var labels_: APIValue

    var probabilities_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var centroids_: APIValue

    var medoids_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HDBSCAN.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HDBSCAN.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HDBSCAN.set_params is an API scaffold")

    def fit_predict(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_predict(self, X, y=None)."
        raise Error("NotImplementedError: HDBSCAN.fit_predict is an API scaffold")

    def __init__(out self, min_cluster_size: Some[Movable] = 5, min_samples: Some[Movable] = None, cluster_selection_epsilon: Some[Movable] = 0.0, max_cluster_size: Some[Movable] = None, metric: Some[Movable] = "euclidean", metric_params: Some[Movable] = None, alpha: Some[Movable] = 1.0, algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 40, n_jobs: Some[Movable] = None, cluster_selection_method: Some[Movable] = "eom", allow_single_cluster: Some[Movable] = False, store_centers: Some[Movable] = None, copy: Some[Movable] = "warn") raises:
        "Scikit-learn signature: __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto', leaf_size=40, n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False, store_centers=None, copy='warn')."
        raise Error("NotImplementedError: HDBSCAN.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: HDBSCAN.fit is an API scaffold")

    def dbscan_clustering(mut self, cut_distance: Some[Movable], min_cluster_size: Some[Movable] = 5) raises -> APIValue:
        "Scikit-learn signature: dbscan_clustering(self, cut_distance, min_cluster_size=5)."
        raise Error("NotImplementedError: HDBSCAN.dbscan_clustering is an API scaffold")

struct SpectralBiclustering(Movable):

    "API scaffold for sklearn.cluster._bicluster.SpectralBiclustering. Operations are deferred."

    var n_clusters: APIValue

    var method: APIValue

    var n_components: APIValue

    var n_best: APIValue

    var svd_method: APIValue

    var n_svd_vecs: APIValue

    var mini_batch: APIValue

    var init: APIValue

    var n_init: APIValue

    var random_state: APIValue

    var rows_: APIValue

    var columns_: APIValue

    var row_labels_: APIValue

    var column_labels_: APIValue

    var biclusters_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: SpectralBiclustering.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: SpectralBiclustering.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: SpectralBiclustering.set_params is an API scaffold")

    def get_indices(mut self, i: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_indices(self, i)."
        raise Error("NotImplementedError: SpectralBiclustering.get_indices is an API scaffold")

    def get_shape(mut self, i: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_shape(self, i)."
        raise Error("NotImplementedError: SpectralBiclustering.get_shape is an API scaffold")

    def get_submatrix(mut self, i: Some[Movable], data: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_submatrix(self, i, data)."
        raise Error("NotImplementedError: SpectralBiclustering.get_submatrix is an API scaffold")

    def __init__(out self, n_clusters: Some[Movable] = 3, *, method: Some[Movable] = "bistochastic", n_components: Some[Movable] = 6, n_best: Some[Movable] = 3, svd_method: Some[Movable] = "randomized", n_svd_vecs: Some[Movable] = None, mini_batch: Some[Movable] = False, init: Some[Movable] = "k-means++", n_init: Some[Movable] = 10, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_clusters=3, *, method='bistochastic', n_components=6, n_best=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None)."
        raise Error("NotImplementedError: SpectralBiclustering.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: SpectralBiclustering.fit is an API scaffold")

struct SpectralClustering(Movable):

    "API scaffold for sklearn.cluster._spectral.SpectralClustering. Operations are deferred."

    var n_clusters: APIValue

    var eigen_solver: APIValue

    var n_components: APIValue

    var random_state: APIValue

    var n_init: APIValue

    var gamma: APIValue

    var affinity: APIValue

    var n_neighbors: APIValue

    var eigen_tol: APIValue

    var assign_labels: APIValue

    var degree: APIValue

    var coef0: APIValue

    var kernel_params: APIValue

    var n_jobs: APIValue

    var verbose: APIValue

    var affinity_matrix_: APIValue

    var labels_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: SpectralClustering.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: SpectralClustering.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: SpectralClustering.set_params is an API scaffold")

    def fit_predict(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_predict(self, X, y=None)."
        raise Error("NotImplementedError: SpectralClustering.fit_predict is an API scaffold")

    def __init__(out self, n_clusters: Some[Movable] = 8, *, eigen_solver: Some[Movable] = None, n_components: Some[Movable] = None, random_state: Some[Movable] = None, n_init: Some[Movable] = 10, gamma: Some[Movable] = 1.0, affinity: Some[Movable] = "rbf", n_neighbors: Some[Movable] = 10, eigen_tol: Some[Movable] = "auto", assign_labels: Some[Movable] = "kmeans", degree: Some[Movable] = 3, coef0: Some[Movable] = 1, kernel_params: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol='auto', assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)."
        raise Error("NotImplementedError: SpectralClustering.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: SpectralClustering.fit is an API scaffold")

struct SpectralCoclustering(Movable):

    "API scaffold for sklearn.cluster._bicluster.SpectralCoclustering. Operations are deferred."

    var n_clusters: APIValue

    var svd_method: APIValue

    var n_svd_vecs: APIValue

    var mini_batch: APIValue

    var init: APIValue

    var n_init: APIValue

    var random_state: APIValue

    var rows_: APIValue

    var columns_: APIValue

    var row_labels_: APIValue

    var column_labels_: APIValue

    var biclusters_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: SpectralCoclustering.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: SpectralCoclustering.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: SpectralCoclustering.set_params is an API scaffold")

    def get_indices(mut self, i: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_indices(self, i)."
        raise Error("NotImplementedError: SpectralCoclustering.get_indices is an API scaffold")

    def get_shape(mut self, i: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_shape(self, i)."
        raise Error("NotImplementedError: SpectralCoclustering.get_shape is an API scaffold")

    def get_submatrix(mut self, i: Some[Movable], data: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: get_submatrix(self, i, data)."
        raise Error("NotImplementedError: SpectralCoclustering.get_submatrix is an API scaffold")

    def __init__(out self, n_clusters: Some[Movable] = 3, *, svd_method: Some[Movable] = "randomized", n_svd_vecs: Some[Movable] = None, mini_batch: Some[Movable] = False, init: Some[Movable] = "k-means++", n_init: Some[Movable] = 10, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_clusters=3, *, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None)."
        raise Error("NotImplementedError: SpectralCoclustering.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: SpectralCoclustering.fit is an API scaffold")

def spectral_clustering(affinity: Some[Movable], *, n_clusters: Some[Movable] = 8, n_components: Some[Movable] = None, eigen_solver: Some[Movable] = None, random_state: Some[Movable] = None, n_init: Some[Movable] = 10, eigen_tol: Some[Movable] = "auto", assign_labels: Some[Movable] = "kmeans", verbose: Some[Movable] = False) raises -> APIValue:
    "Scikit-learn signature: spectral_clustering(affinity, *, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol='auto', assign_labels='kmeans', verbose=False)."
    raise Error("NotImplementedError: spectral_clustering is an API scaffold")

