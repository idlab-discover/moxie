# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.neighbors._api_types import APIValue

struct KNeighborsClassifier(Movable):

    "API scaffold for sklearn.neighbors._classification.KNeighborsClassifier. Operations are deferred."

    var n_neighbors: APIValue

    var weights: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var p: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var classes_: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    var outputs_2d_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: KNeighborsClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: KNeighborsClassifier.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: KNeighborsClassifier.set_params is an API scaffold")

    def __init__(out self, n_neighbors: Some[Movable] = 5, *, weights: Some[Movable] = "uniform", algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, p: Some[Movable] = 2, metric: Some[Movable] = "minkowski", metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: KNeighborsClassifier.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: KNeighborsClassifier.score is an API scaffold")

    def kneighbors(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, return_distance: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: kneighbors(self, X=None, n_neighbors=None, return_distance=True)."
        raise Error("NotImplementedError: KNeighborsClassifier.kneighbors is an API scaffold")

    def kneighbors_graph(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, mode: Some[Movable] = "connectivity") raises -> APIValue:
        "Scikit-learn signature: kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')."
        raise Error("NotImplementedError: KNeighborsClassifier.kneighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: KNeighborsClassifier.fit is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: KNeighborsClassifier.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: KNeighborsClassifier.predict_proba is an API scaffold")

struct KNeighborsRegressor(Movable):

    "API scaffold for sklearn.neighbors._regression.KNeighborsRegressor. Operations are deferred."

    var n_neighbors: APIValue

    var weights: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var p: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: KNeighborsRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: KNeighborsRegressor.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: KNeighborsRegressor.set_params is an API scaffold")

    def __init__(out self, n_neighbors: Some[Movable] = 5, *, weights: Some[Movable] = "uniform", algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, p: Some[Movable] = 2, metric: Some[Movable] = "minkowski", metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: KNeighborsRegressor.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: KNeighborsRegressor.score is an API scaffold")

    def kneighbors(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, return_distance: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: kneighbors(self, X=None, n_neighbors=None, return_distance=True)."
        raise Error("NotImplementedError: KNeighborsRegressor.kneighbors is an API scaffold")

    def kneighbors_graph(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, mode: Some[Movable] = "connectivity") raises -> APIValue:
        "Scikit-learn signature: kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')."
        raise Error("NotImplementedError: KNeighborsRegressor.kneighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: KNeighborsRegressor.fit is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: KNeighborsRegressor.predict is an API scaffold")

struct KNeighborsTransformer(Movable):

    "API scaffold for sklearn.neighbors._graph.KNeighborsTransformer. Operations are deferred."

    var mode: APIValue

    var n_neighbors: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var metric: APIValue

    var p: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: KNeighborsTransformer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: KNeighborsTransformer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: KNeighborsTransformer.set_params is an API scaffold")

    def __init__(out self, *, mode: Some[Movable] = "distance", n_neighbors: Some[Movable] = 5, algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, mode='distance', n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: KNeighborsTransformer.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: KNeighborsTransformer.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: KNeighborsTransformer.fit_transform is an API scaffold")

    def kneighbors(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, return_distance: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: kneighbors(self, X=None, n_neighbors=None, return_distance=True)."
        raise Error("NotImplementedError: KNeighborsTransformer.kneighbors is an API scaffold")

    def kneighbors_graph(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, mode: Some[Movable] = "connectivity") raises -> APIValue:
        "Scikit-learn signature: kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')."
        raise Error("NotImplementedError: KNeighborsTransformer.kneighbors_graph is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: KNeighborsTransformer.get_feature_names_out is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: KNeighborsTransformer.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: KNeighborsTransformer.transform is an API scaffold")

struct KernelDensity(Movable):

    "API scaffold for sklearn.neighbors._kde.KernelDensity. Operations are deferred."

    var bandwidth: APIValue

    var algorithm: APIValue

    var kernel: APIValue

    var metric: APIValue

    var atol: APIValue

    var rtol: APIValue

    var breadth_first: APIValue

    var leaf_size: APIValue

    var metric_params: APIValue

    var n_features_in_: APIValue

    var tree_: APIValue

    var feature_names_in_: APIValue

    var bandwidth_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: KernelDensity.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: KernelDensity.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: KernelDensity.set_params is an API scaffold")

    def __init__(out self, *, bandwidth: Some[Movable] = 1.0, algorithm: Some[Movable] = "auto", kernel: Some[Movable] = "gaussian", metric: Some[Movable] = "euclidean", atol: Some[Movable] = 0, rtol: Some[Movable] = 0, breadth_first: Some[Movable] = True, leaf_size: Some[Movable] = 40, metric_params: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean', atol=0, rtol=0, breadth_first=True, leaf_size=40, metric_params=None)."
        raise Error("NotImplementedError: KernelDensity.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, sample_weight: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, sample_weight=None)."
        raise Error("NotImplementedError: KernelDensity.fit is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: KernelDensity.score_samples is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y=None)."
        raise Error("NotImplementedError: KernelDensity.score is an API scaffold")

    def sample(mut self, n_samples: Some[Movable] = 1, random_state: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: sample(self, n_samples=1, random_state=None)."
        raise Error("NotImplementedError: KernelDensity.sample is an API scaffold")

struct LocalOutlierFactor(Movable):

    "API scaffold for sklearn.neighbors._lof.LocalOutlierFactor. Operations are deferred."

    var n_neighbors: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var metric: APIValue

    var p: APIValue

    var metric_params: APIValue

    var contamination: APIValue

    var novelty: APIValue

    var n_jobs: APIValue

    var negative_outlier_factor_: APIValue

    var n_neighbors_: APIValue

    var offset_: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LocalOutlierFactor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: LocalOutlierFactor.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: LocalOutlierFactor.set_params is an API scaffold")

    def __init__(out self, n_neighbors: Some[Movable] = 20, *, algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, contamination: Some[Movable] = "auto", novelty: Some[Movable] = False, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None)."
        raise Error("NotImplementedError: LocalOutlierFactor.__init__ is an API scaffold")

    def fit_predict(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_predict(self, X, y=None)."
        raise Error("NotImplementedError: LocalOutlierFactor.fit_predict is an API scaffold")

    def kneighbors(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, return_distance: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: kneighbors(self, X=None, n_neighbors=None, return_distance=True)."
        raise Error("NotImplementedError: LocalOutlierFactor.kneighbors is an API scaffold")

    def kneighbors_graph(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, mode: Some[Movable] = "connectivity") raises -> APIValue:
        "Scikit-learn signature: kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')."
        raise Error("NotImplementedError: LocalOutlierFactor.kneighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: LocalOutlierFactor.fit is an API scaffold")

    def predict(mut self, X: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict(self, X=None)."
        raise Error("NotImplementedError: LocalOutlierFactor.predict is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: LocalOutlierFactor.decision_function is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: LocalOutlierFactor.score_samples is an API scaffold")

struct NearestCentroid(Movable):

    "API scaffold for sklearn.neighbors._nearest_centroid.NearestCentroid. Operations are deferred."

    var metric: APIValue

    var shrink_threshold: APIValue

    var priors: APIValue

    var centroids_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var deviations_: APIValue

    var within_class_std_dev_: APIValue

    var class_prior_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: NearestCentroid.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: NearestCentroid.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: NearestCentroid.set_params is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: NearestCentroid.score is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: NearestCentroid.decision_function is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: NearestCentroid.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: NearestCentroid.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: NearestCentroid.predict_log_proba is an API scaffold")

    def __init__(out self, metric: Some[Movable] = "euclidean", *, shrink_threshold: Some[Movable] = None, priors: Some[Movable] = "uniform") raises:
        "Scikit-learn signature: __init__(self, metric='euclidean', *, shrink_threshold=None, priors='uniform')."
        raise Error("NotImplementedError: NearestCentroid.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: NearestCentroid.fit is an API scaffold")

struct NearestNeighbors(Movable):

    "API scaffold for sklearn.neighbors._unsupervised.NearestNeighbors. Operations are deferred."

    var n_neighbors: APIValue

    var radius: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var metric: APIValue

    var p: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: NearestNeighbors.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: NearestNeighbors.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: NearestNeighbors.set_params is an API scaffold")

    def __init__(out self, *, n_neighbors: Some[Movable] = 5, radius: Some[Movable] = 1.0, algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: NearestNeighbors.__init__ is an API scaffold")

    def radius_neighbors(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, return_distance: Some[Movable] = True, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False)."
        raise Error("NotImplementedError: NearestNeighbors.radius_neighbors is an API scaffold")

    def radius_neighbors_graph(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, mode: Some[Movable] = "connectivity", sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', sort_results=False)."
        raise Error("NotImplementedError: NearestNeighbors.radius_neighbors_graph is an API scaffold")

    def kneighbors(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, return_distance: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: kneighbors(self, X=None, n_neighbors=None, return_distance=True)."
        raise Error("NotImplementedError: NearestNeighbors.kneighbors is an API scaffold")

    def kneighbors_graph(mut self, X: Some[Movable] = None, n_neighbors: Some[Movable] = None, mode: Some[Movable] = "connectivity") raises -> APIValue:
        "Scikit-learn signature: kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')."
        raise Error("NotImplementedError: NearestNeighbors.kneighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: NearestNeighbors.fit is an API scaffold")

struct NeighborhoodComponentsAnalysis(Movable):

    "API scaffold for sklearn.neighbors._nca.NeighborhoodComponentsAnalysis. Operations are deferred."

    var n_components: APIValue

    var init: APIValue

    var warm_start: APIValue

    var max_iter: APIValue

    var tol: APIValue

    var callback: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var components_: APIValue

    var n_features_in_: APIValue

    var n_iter_: APIValue

    var random_state_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = None, *, init: Some[Movable] = "auto", warm_start: Some[Movable] = False, max_iter: Some[Movable] = 50, tol: Some[Movable] = 1e-05, callback: Some[Movable] = None, verbose: Some[Movable] = 0, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_components=None, *, init='auto', warm_start=False, max_iter=50, tol=1e-05, callback=None, verbose=0, random_state=None)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: NeighborhoodComponentsAnalysis.transform is an API scaffold")

struct RadiusNeighborsClassifier(Movable):

    "API scaffold for sklearn.neighbors._classification.RadiusNeighborsClassifier. Operations are deferred."

    var radius: APIValue

    var weights: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var p: APIValue

    var metric: APIValue

    var outlier_label: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var classes_: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    var outlier_label_: APIValue

    var outputs_2d_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.set_params is an API scaffold")

    def __init__(out self, radius: Some[Movable] = 1.0, *, weights: Some[Movable] = "uniform", algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, p: Some[Movable] = 2, metric: Some[Movable] = "minkowski", outlier_label: Some[Movable] = None, metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, radius=1.0, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.score is an API scaffold")

    def radius_neighbors(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, return_distance: Some[Movable] = True, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.radius_neighbors is an API scaffold")

    def radius_neighbors_graph(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, mode: Some[Movable] = "connectivity", sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.radius_neighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.fit is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: RadiusNeighborsClassifier.predict_proba is an API scaffold")

struct RadiusNeighborsRegressor(Movable):

    "API scaffold for sklearn.neighbors._regression.RadiusNeighborsRegressor. Operations are deferred."

    var radius: APIValue

    var weights: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var p: APIValue

    var metric: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.set_params is an API scaffold")

    def __init__(out self, radius: Some[Movable] = 1.0, *, weights: Some[Movable] = "uniform", algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, p: Some[Movable] = 2, metric: Some[Movable] = "minkowski", metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, radius=1.0, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.score is an API scaffold")

    def radius_neighbors(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, return_distance: Some[Movable] = True, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.radius_neighbors is an API scaffold")

    def radius_neighbors_graph(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, mode: Some[Movable] = "connectivity", sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.radius_neighbors_graph is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable]) raises:
        "Scikit-learn signature: fit(self, X, y)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.fit is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: RadiusNeighborsRegressor.predict is an API scaffold")

struct RadiusNeighborsTransformer(Movable):

    "API scaffold for sklearn.neighbors._graph.RadiusNeighborsTransformer. Operations are deferred."

    var mode: APIValue

    var radius: APIValue

    var algorithm: APIValue

    var leaf_size: APIValue

    var metric: APIValue

    var p: APIValue

    var metric_params: APIValue

    var n_jobs: APIValue

    var effective_metric_: APIValue

    var effective_metric_params_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_samples_fit_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.set_params is an API scaffold")

    def __init__(out self, *, mode: Some[Movable] = "distance", radius: Some[Movable] = 1.0, algorithm: Some[Movable] = "auto", leaf_size: Some[Movable] = 30, metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, n_jobs: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, mode='distance', radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.fit_transform is an API scaffold")

    def radius_neighbors(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, return_distance: Some[Movable] = True, sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.radius_neighbors is an API scaffold")

    def radius_neighbors_graph(mut self, X: Some[Movable] = None, radius: Some[Movable] = None, mode: Some[Movable] = "connectivity", sort_results: Some[Movable] = False) raises -> APIValue:
        "Scikit-learn signature: radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', sort_results=False)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.radius_neighbors_graph is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.get_feature_names_out is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: RadiusNeighborsTransformer.transform is an API scaffold")

def kneighbors_graph(X: Some[Movable], n_neighbors: Some[Movable], *, mode: Some[Movable] = "connectivity", metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, include_self: Some[Movable] = False, n_jobs: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: kneighbors_graph(X, n_neighbors, *, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)."
    raise Error("NotImplementedError: kneighbors_graph is an API scaffold")

def radius_neighbors_graph(X: Some[Movable], radius: Some[Movable], *, mode: Some[Movable] = "connectivity", metric: Some[Movable] = "minkowski", p: Some[Movable] = 2, metric_params: Some[Movable] = None, include_self: Some[Movable] = False, n_jobs: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: radius_neighbors_graph(X, radius, *, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)."
    raise Error("NotImplementedError: radius_neighbors_graph is an API scaffold")

def sort_graph_by_row_values(graph: Some[Movable], copy: Some[Movable] = False, warn_when_not_sorted: Some[Movable] = True) raises -> APIValue:
    "Scikit-learn signature: sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=True)."
    raise Error("NotImplementedError: sort_graph_by_row_values is an API scaffold")
