# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.neural_network._api_types import APIValue

from ._rbm import BernoulliRBM

struct MLPClassifier(Movable):

    "API scaffold for sklearn.neural_network._multilayer_perceptron.MLPClassifier. Operations are deferred."

    var hidden_layer_sizes: APIValue

    var activation: APIValue

    var solver: APIValue

    var alpha: APIValue

    var batch_size: APIValue

    var learning_rate: APIValue

    var learning_rate_init: APIValue

    var power_t: APIValue

    var max_iter: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    var tol: APIValue

    var verbose: APIValue

    var warm_start: APIValue

    var momentum: APIValue

    var nesterovs_momentum: APIValue

    var early_stopping: APIValue

    var validation_fraction: APIValue

    var beta_1: APIValue

    var beta_2: APIValue

    var epsilon: APIValue

    var n_iter_no_change: APIValue

    var max_fun: APIValue

    var classes_: APIValue

    var loss_: APIValue

    var best_loss_: APIValue

    var loss_curve_: APIValue

    var validation_scores_: APIValue

    var best_validation_score_: APIValue

    var t_: APIValue

    var coefs_: APIValue

    var intercepts_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_iter_: APIValue

    var n_layers_: APIValue

    var n_outputs_: APIValue

    var out_activation_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: MLPClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: MLPClassifier.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: MLPClassifier.set_params is an API scaffold")

    def __init__(out self, hidden_layer_sizes: Some[Movable] = (100,), activation: Some[Movable] = "relu", *, solver: Some[Movable] = "adam", alpha: Some[Movable] = 0.0001, batch_size: Some[Movable] = "auto", learning_rate: Some[Movable] = "constant", learning_rate_init: Some[Movable] = 0.001, power_t: Some[Movable] = 0.5, max_iter: Some[Movable] = 200, shuffle: Some[Movable] = True, random_state: Some[Movable] = None, tol: Some[Movable] = 0.0001, verbose: Some[Movable] = False, warm_start: Some[Movable] = False, momentum: Some[Movable] = 0.9, nesterovs_momentum: Some[Movable] = True, early_stopping: Some[Movable] = False, validation_fraction: Some[Movable] = 0.1, beta_1: Some[Movable] = 0.9, beta_2: Some[Movable] = 0.999, epsilon: Some[Movable] = 1e-08, n_iter_no_change: Some[Movable] = 10, max_fun: Some[Movable] = 15000) raises:
        "Scikit-learn signature: __init__(self, hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)."
        raise Error("NotImplementedError: MLPClassifier.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: MLPClassifier.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: MLPClassifier.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: MLPClassifier.predict is an API scaffold")

    def partial_fit(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, classes: Some[Movable] = None) raises:
        "Scikit-learn signature: partial_fit(self, X, y, sample_weight=None, classes=None)."
        raise Error("NotImplementedError: MLPClassifier.partial_fit is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: MLPClassifier.predict_log_proba is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: MLPClassifier.predict_proba is an API scaffold")

struct MLPRegressor(Movable):

    "API scaffold for sklearn.neural_network._multilayer_perceptron.MLPRegressor. Operations are deferred."

    var loss: APIValue

    var hidden_layer_sizes: APIValue

    var activation: APIValue

    var solver: APIValue

    var alpha: APIValue

    var batch_size: APIValue

    var learning_rate: APIValue

    var learning_rate_init: APIValue

    var power_t: APIValue

    var max_iter: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    var tol: APIValue

    var verbose: APIValue

    var warm_start: APIValue

    var momentum: APIValue

    var nesterovs_momentum: APIValue

    var early_stopping: APIValue

    var validation_fraction: APIValue

    var beta_1: APIValue

    var beta_2: APIValue

    var epsilon: APIValue

    var n_iter_no_change: APIValue

    var max_fun: APIValue

    var loss_: APIValue

    var best_loss_: APIValue

    var loss_curve_: APIValue

    var validation_scores_: APIValue

    var best_validation_score_: APIValue

    var t_: APIValue

    var coefs_: APIValue

    var intercepts_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_iter_: APIValue

    var n_layers_: APIValue

    var n_outputs_: APIValue

    var out_activation_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: MLPRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: MLPRegressor.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: MLPRegressor.set_params is an API scaffold")

    def __init__(out self, loss: Some[Movable] = "squared_error", hidden_layer_sizes: Some[Movable] = (100,), activation: Some[Movable] = "relu", *, solver: Some[Movable] = "adam", alpha: Some[Movable] = 0.0001, batch_size: Some[Movable] = "auto", learning_rate: Some[Movable] = "constant", learning_rate_init: Some[Movable] = 0.001, power_t: Some[Movable] = 0.5, max_iter: Some[Movable] = 200, shuffle: Some[Movable] = True, random_state: Some[Movable] = None, tol: Some[Movable] = 0.0001, verbose: Some[Movable] = False, warm_start: Some[Movable] = False, momentum: Some[Movable] = 0.9, nesterovs_momentum: Some[Movable] = True, early_stopping: Some[Movable] = False, validation_fraction: Some[Movable] = 0.1, beta_1: Some[Movable] = 0.9, beta_2: Some[Movable] = 0.999, epsilon: Some[Movable] = 1e-08, n_iter_no_change: Some[Movable] = 10, max_fun: Some[Movable] = 15000) raises:
        "Scikit-learn signature: __init__(self, loss='squared_error', hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)."
        raise Error("NotImplementedError: MLPRegressor.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: MLPRegressor.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: MLPRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: MLPRegressor.predict is an API scaffold")

    def partial_fit(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises:
        "Scikit-learn signature: partial_fit(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: MLPRegressor.partial_fit is an API scaffold")
