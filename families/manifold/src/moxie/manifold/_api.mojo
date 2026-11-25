# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deferred public scikit-learn API declarations.

Implemented estimators and functions live in their own modules.
"""
from moxie.manifold._api_types import APIValue

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

def spectral_embedding(adjacency: Some[Movable], *, n_components: Some[Movable] = 8, eigen_solver: Some[Movable] = None, random_state: Some[Movable] = None, eigen_tol: Some[Movable] = "auto", norm_laplacian: Some[Movable] = True, drop_first: Some[Movable] = True) raises -> APIValue:
    "Scikit-learn signature: spectral_embedding(adjacency, *, n_components=8, eigen_solver=None, random_state=None, eigen_tol='auto', norm_laplacian=True, drop_first=True)."
    raise Error("NotImplementedError: spectral_embedding is an API scaffold")
