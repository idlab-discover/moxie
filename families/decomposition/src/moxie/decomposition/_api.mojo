# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.decomposition._api_types import APIValue

struct DictionaryLearning(Movable):

    "API scaffold for sklearn.decomposition._dict_learning.DictionaryLearning. Operations are deferred."

    var n_components: APIValue

    var alpha: APIValue

    var max_iter: APIValue

    var tol: APIValue

    var fit_algorithm: APIValue

    var transform_algorithm: APIValue

    var transform_n_nonzero_coefs: APIValue

    var transform_alpha: APIValue

    var n_jobs: APIValue

    var code_init: APIValue

    var dict_init: APIValue

    var callback: APIValue

    var verbose: APIValue

    var split_sign: APIValue

    var random_state: APIValue

    var positive_code: APIValue

    var positive_dict: APIValue

    var transform_max_iter: APIValue

    var components_: APIValue

    var error_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_iter_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: DictionaryLearning.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: DictionaryLearning.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: DictionaryLearning.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: DictionaryLearning.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: DictionaryLearning.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: DictionaryLearning.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = None, *, alpha: Some[Movable] = 1, max_iter: Some[Movable] = 1000, tol: Some[Movable] = 1e-08, fit_algorithm: Some[Movable] = "lars", transform_algorithm: Some[Movable] = "omp", transform_n_nonzero_coefs: Some[Movable] = None, transform_alpha: Some[Movable] = None, n_jobs: Some[Movable] = None, code_init: Some[Movable] = None, dict_init: Some[Movable] = None, callback: Some[Movable] = None, verbose: Some[Movable] = False, split_sign: Some[Movable] = False, random_state: Some[Movable] = None, positive_code: Some[Movable] = False, positive_dict: Some[Movable] = False, transform_max_iter: Some[Movable] = 1000) raises:
        "Scikit-learn signature: __init__(self, n_components=None, *, alpha=1, max_iter=1000, tol=1e-08, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None, dict_init=None, callback=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000)."
        raise Error("NotImplementedError: DictionaryLearning.__init__ is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: DictionaryLearning.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: DictionaryLearning.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: DictionaryLearning.fit is an API scaffold")

struct MiniBatchDictionaryLearning(Movable):

    "API scaffold for sklearn.decomposition._dict_learning.MiniBatchDictionaryLearning. Operations are deferred."

    var n_components: APIValue

    var alpha: APIValue

    var max_iter: APIValue

    var fit_algorithm: APIValue

    var n_jobs: APIValue

    var batch_size: APIValue

    var shuffle: APIValue

    var dict_init: APIValue

    var transform_algorithm: APIValue

    var transform_n_nonzero_coefs: APIValue

    var transform_alpha: APIValue

    var verbose: APIValue

    var split_sign: APIValue

    var random_state: APIValue

    var positive_code: APIValue

    var positive_dict: APIValue

    var transform_max_iter: APIValue

    var callback: APIValue

    var tol: APIValue

    var max_no_improvement: APIValue

    var components_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_iter_: APIValue

    var n_steps_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = None, *, alpha: Some[Movable] = 1, max_iter: Some[Movable] = 1000, fit_algorithm: Some[Movable] = "lars", n_jobs: Some[Movable] = None, batch_size: Some[Movable] = 256, shuffle: Some[Movable] = True, dict_init: Some[Movable] = None, transform_algorithm: Some[Movable] = "omp", transform_n_nonzero_coefs: Some[Movable] = None, transform_alpha: Some[Movable] = None, verbose: Some[Movable] = False, split_sign: Some[Movable] = False, random_state: Some[Movable] = None, positive_code: Some[Movable] = False, positive_dict: Some[Movable] = False, transform_max_iter: Some[Movable] = 1000, callback: Some[Movable] = None, tol: Some[Movable] = 0.001, max_no_improvement: Some[Movable] = 10) raises:
        "Scikit-learn signature: __init__(self, n_components=None, *, alpha=1, max_iter=1000, fit_algorithm='lars', n_jobs=None, batch_size=256, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000, callback=None, tol=0.001, max_no_improvement=10)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.__init__ is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.fit is an API scaffold")

    def partial_fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: partial_fit(self, X, y=None)."
        raise Error("NotImplementedError: MiniBatchDictionaryLearning.partial_fit is an API scaffold")

struct MiniBatchSparsePCA(Movable):

    "API scaffold for sklearn.decomposition._sparse_pca.MiniBatchSparsePCA. Operations are deferred."

    var n_components: APIValue

    var alpha: APIValue

    var ridge_alpha: APIValue

    var max_iter: APIValue

    var callback: APIValue

    var batch_size: APIValue

    var verbose: APIValue

    var shuffle: APIValue

    var n_jobs: APIValue

    var method: APIValue

    var random_state: APIValue

    var tol: APIValue

    var max_no_improvement: APIValue

    var components_: APIValue

    var n_components_: APIValue

    var n_iter_: APIValue

    var mean_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = None, *, alpha: Some[Movable] = 1, ridge_alpha: Some[Movable] = 0.01, max_iter: Some[Movable] = 1000, callback: Some[Movable] = None, batch_size: Some[Movable] = 3, verbose: Some[Movable] = False, shuffle: Some[Movable] = True, n_jobs: Some[Movable] = None, method: Some[Movable] = "lars", random_state: Some[Movable] = None, tol: Some[Movable] = 0.001, max_no_improvement: Some[Movable] = 10) raises:
        "Scikit-learn signature: __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, callback=None, batch_size=3, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None, tol=0.001, max_no_improvement=10)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: MiniBatchSparsePCA.inverse_transform is an API scaffold")

struct SparsePCA(Movable):

    "API scaffold for sklearn.decomposition._sparse_pca.SparsePCA. Operations are deferred."

    var n_components: APIValue

    var alpha: APIValue

    var ridge_alpha: APIValue

    var max_iter: APIValue

    var tol: APIValue

    var method: APIValue

    var n_jobs: APIValue

    var U_init: APIValue

    var V_init: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var components_: APIValue

    var error_: APIValue

    var n_components_: APIValue

    var n_iter_: APIValue

    var mean_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: SparsePCA.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: SparsePCA.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: SparsePCA.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: SparsePCA.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: SparsePCA.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: SparsePCA.get_feature_names_out is an API scaffold")

    def __init__(out self, n_components: Some[Movable] = None, *, alpha: Some[Movable] = 1, ridge_alpha: Some[Movable] = 0.01, max_iter: Some[Movable] = 1000, tol: Some[Movable] = 1e-08, method: Some[Movable] = "lars", n_jobs: Some[Movable] = None, U_init: Some[Movable] = None, V_init: Some[Movable] = None, verbose: Some[Movable] = False, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None)."
        raise Error("NotImplementedError: SparsePCA.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: SparsePCA.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: SparsePCA.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: SparsePCA.inverse_transform is an API scaffold")

def dict_learning(X: Some[Movable], n_components: Some[Movable], *, alpha: Some[Movable], max_iter: Some[Movable] = 100, tol: Some[Movable] = 1e-08, method: Some[Movable] = "lars", n_jobs: Some[Movable] = None, dict_init: Some[Movable] = None, code_init: Some[Movable] = None, callback: Some[Movable] = None, verbose: Some[Movable] = False, random_state: Some[Movable] = None, return_n_iter: Some[Movable] = False, positive_dict: Some[Movable] = False, positive_code: Some[Movable] = False, method_max_iter: Some[Movable] = 1000) raises -> APIValue:
    "Scikit-learn signature: dict_learning(X, n_components, *, alpha, max_iter=100, tol=1e-08, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000)."
    raise Error("NotImplementedError: dict_learning is an API scaffold")

def dict_learning_online(X: Some[Movable], n_components: Some[Movable] = 2, *, alpha: Some[Movable] = 1, max_iter: Some[Movable] = 100, return_code: Some[Movable] = True, dict_init: Some[Movable] = None, callback: Some[Movable] = None, batch_size: Some[Movable] = 256, verbose: Some[Movable] = False, shuffle: Some[Movable] = True, n_jobs: Some[Movable] = None, method: Some[Movable] = "lars", random_state: Some[Movable] = None, positive_dict: Some[Movable] = False, positive_code: Some[Movable] = False, method_max_iter: Some[Movable] = 1000, tol: Some[Movable] = 0.001, max_no_improvement: Some[Movable] = 10) raises -> APIValue:
    "Scikit-learn signature: dict_learning_online(X, n_components=2, *, alpha=1, max_iter=100, return_code=True, dict_init=None, callback=None, batch_size=256, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None, positive_dict=False, positive_code=False, method_max_iter=1000, tol=0.001, max_no_improvement=10)."
    raise Error("NotImplementedError: dict_learning_online is an API scaffold")
