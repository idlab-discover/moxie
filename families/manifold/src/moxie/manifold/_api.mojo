# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deferred public scikit-learn API declarations.

ClassicalMDS and trustworthiness are implemented in their own modules.
"""
from moxie.manifold._api_types import APIValue

struct MDS(Movable):

    "API scaffold for sklearn.manifold._mds.MDS. Operations are deferred."

    var n_components: APIValue

    var metric_mds: APIValue

    var n_init: APIValue

    var init: APIValue

    var max_iter: APIValue

    var verbose: APIValue

    var eps: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var dissimilarity: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var normalized_stress: APIValue

    var embedding_: APIValue

    var stress_: APIValue

    var dissimilarity_matrix_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_iter_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: MDS.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: MDS.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: MDS.set_params is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = 2, *, metric_mds: Some[Movable] = True, n_init: Some[Movable] = 1, init: Some[Movable] = "warn", max_iter: Some[Movable] = 300, verbose: Some[Movable] = 0, eps: Some[Movable] = 1e-06, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, dissimilarity: Some[Movable] = "deprecated", metric: Some[Movable] = "euclidean", metric_params: Some[Movable] = None, normalized_stress: Some[Movable] = "auto") raises:
        "Scikit-learn signature: __init__(self, n_components=2, *, metric_mds=True, n_init=1, init='warn', max_iter=300, verbose=0, eps=1e-06, n_jobs=None, random_state=None, dissimilarity='deprecated', metric='euclidean', metric_params=None, normalized_stress='auto')."
        raise Error("NotImplementedError: MDS.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, init: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, init=None)."
        raise Error("NotImplementedError: MDS.fit is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, init: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, init=None)."
        raise Error("NotImplementedError: MDS.fit_transform is an API scaffold")

struct TSNE(Movable):

    "API scaffold for sklearn.manifold._t_sne.TSNE. Operations are deferred."

    var n_components: APIValue

    var perplexity: APIValue

    var early_exaggeration: APIValue

    var learning_rate: APIValue

    var max_iter: APIValue

    var n_iter_without_progress: APIValue

    var min_grad_norm: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var init: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var method: APIValue

    var angle: APIValue

    var n_jobs: APIValue

    var embedding_: APIValue

    var kl_divergence_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var learning_rate_: APIValue

    var n_iter_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: TSNE.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: TSNE.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: TSNE.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: TSNE.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: TSNE.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: TSNE.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = 2, *, perplexity: Some[Movable] = 30.0, early_exaggeration: Some[Movable] = 12.0, learning_rate: Some[Movable] = "auto", max_iter: Some[Movable] = 1000, n_iter_without_progress: Some[Movable] = 300, min_grad_norm: Some[Movable] = 1e-07, metric: Some[Movable] = "euclidean", metric_params: Some[Movable] = None, init: Some[Movable] = "pca", verbose: Some[Movable] = 0, random_state: Some[Movable] = None, method: Some[Movable] = "barnes_hut", angle: Some[Movable] = 0.5, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)."
        raise Error("NotImplementedError: TSNE.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: TSNE.fit is an API scaffold")

struct Isomap(Movable):

    "API scaffold for sklearn.manifold._isomap.Isomap. Operations are deferred."

    var n_neighbors: APIValue

    var radius: APIValue

    var n_components: APIValue

    var eigen_solver: APIValue

    var tol: APIValue

    var max_iter: APIValue

    var path_method: APIValue

    var neighbors_algorithm: APIValue

    var n_jobs: APIValue

    var metric: APIValue

    var p: APIValue

    var metric_params: APIValue

    var embedding_: APIValue

    var kernel_pca_: APIValue

    var nbrs_: APIValue

    var dist_matrix_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: Isomap.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: Isomap.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: Isomap.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: Isomap.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: Isomap.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: Isomap.get_feature_names_out is an API scaffold")

    def __init__(out self, *, n_neighbors: Some[Movable] = 5, radius: Some[Movable] = None, n_components: Some[Movable] = 2, eigen_solver: Some[Movable] = "auto", tol: Some[Movable] = 0, max_iter: Some[Movable] = None, path_method: Some[Movable] = "auto", neighbors_algorithm: Some[Movable] = "auto", n_jobs: Some[Movable] = None, metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, n_neighbors=5, radius=None, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='minkowski', p=2, metric_params=None)."
        raise Error("NotImplementedError: Isomap.__init__ is an API scaffold")

    def reconstruction_error(mut self) raises -> APIValue:
        "Scikit-learn signature: reconstruction_error(self)."
        raise Error("NotImplementedError: Isomap.reconstruction_error is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: Isomap.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: Isomap.transform is an API scaffold")

struct LocallyLinearEmbedding(Movable):

    "API scaffold for sklearn.manifold._locally_linear.LocallyLinearEmbedding. Operations are deferred."

    var n_neighbors: APIValue

    var n_components: APIValue

    var reg: APIValue

    var eigen_solver: APIValue

    var tol: APIValue

    var max_iter: APIValue

    var method: APIValue

    var hessian_tol: APIValue

    var modified_tol: APIValue

    var neighbors_algorithm: APIValue

    var random_state: APIValue

    var n_jobs: APIValue

    var embedding_: APIValue

    var reconstruction_error_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var nbrs_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.get_feature_names_out is an API scaffold")

    def __init__(out self, *, n_neighbors: Some[Movable] = 5, n_components: Some[Movable] = 2, reg: Some[Movable] = 0.001, eigen_solver: Some[Movable] = "auto", tol: Some[Movable] = 1e-06, max_iter: Some[Movable] = 100, method: Some[Movable] = "standard", hessian_tol: Some[Movable] = 0.0001, modified_tol: Some[Movable] = 1e-12, neighbors_algorithm: Some[Movable] = "auto", random_state: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, n_neighbors=5, n_components=2, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=None)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: LocallyLinearEmbedding.transform is an API scaffold")

struct SpectralEmbedding(Movable):

    "API scaffold for sklearn.manifold._spectral_embedding.SpectralEmbedding. Operations are deferred."

    var n_components: APIValue

    var affinity: APIValue

    var gamma: APIValue

    var random_state: APIValue

    var eigen_solver: APIValue

    var eigen_tol: APIValue

    var n_neighbors: APIValue

    var n_jobs: APIValue

    var embedding_: APIValue

    var affinity_matrix_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_neighbors_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: SpectralEmbedding.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: SpectralEmbedding.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: SpectralEmbedding.set_params is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = 2, *, affinity: Some[Movable] = "nearest_neighbors", gamma: Some[Movable] = None, random_state: Some[Movable] = None, eigen_solver: Some[Movable] = None, eigen_tol: Some[Movable] = "auto", n_neighbors: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_components=2, *, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, eigen_tol='auto', n_neighbors=None, n_jobs=None)."
        raise Error("NotImplementedError: SpectralEmbedding.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: SpectralEmbedding.fit is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: SpectralEmbedding.fit_transform is an API scaffold")

def locally_linear_embedding(X: Some[Movable], *, n_neighbors: Some[Movable], n_components: Some[Movable], reg: Some[Movable] = 0.001, eigen_solver: Some[Movable] = "auto", tol: Some[Movable] = 1e-06, max_iter: Some[Movable] = 100, method: Some[Movable] = "standard", hessian_tol: Some[Movable] = 0.0001, modified_tol: Some[Movable] = 1e-12, random_state: Some[Movable] = None, n_jobs: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: locally_linear_embedding(X, *, n_neighbors, n_components, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, random_state=None, n_jobs=None)."
    raise Error("NotImplementedError: locally_linear_embedding is an API scaffold")

def smacof(dissimilarities: Some[Movable], *, metric: Some[Movable] = True, n_components: Some[Movable] = 2, init: Some[Movable] = None, n_init: Some[Movable] = 1, n_jobs: Some[Movable] = None, max_iter: Some[Movable] = 300, verbose: Some[Movable] = 0, eps: Some[Movable] = 1e-06, random_state: Some[Movable] = None, return_n_iter: Some[Movable] = False, normalized_stress: Some[Movable] = "auto") raises -> APIValue:
    "Scikit-learn signature: smacof(dissimilarities, *, metric=True, n_components=2, init=None, n_init=1, n_jobs=None, max_iter=300, verbose=0, eps=1e-06, random_state=None, return_n_iter=False, normalized_stress='auto')."
    raise Error("NotImplementedError: smacof is an API scaffold")

def spectral_embedding(adjacency: Some[Movable], *, n_components: Some[Movable] = 8, eigen_solver: Some[Movable] = None, random_state: Some[Movable] = None, eigen_tol: Some[Movable] = "auto", norm_laplacian: Some[Movable] = True, drop_first: Some[Movable] = True) raises -> APIValue:
    "Scikit-learn signature: spectral_embedding(adjacency, *, n_components=8, eigen_solver=None, random_state=None, eigen_tol='auto', norm_laplacian=True, drop_first=True)."
    raise Error("NotImplementedError: spectral_embedding is an API scaffold")
