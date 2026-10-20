# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.ensemble._api_types import APIValue

struct AdaBoostClassifier(Movable):

    "API scaffold for sklearn.ensemble._weight_boosting.AdaBoostClassifier. Operations are deferred."

    var estimator: APIValue

    var n_estimators: APIValue

    var learning_rate: APIValue

    var random_state: APIValue

    var estimator_: APIValue

    var estimators_: APIValue

    var classes_: APIValue

    var n_classes_: APIValue

    var estimator_weights_: APIValue

    var estimator_errors_: APIValue

    var feature_importances_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: AdaBoostClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: AdaBoostClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: AdaBoostClassifier.set_params is an API scaffold")

    def __init__(out self, estimator: Some[Movable] = None, *, n_estimators: Some[Movable] = 50, learning_rate: Some[Movable] = 1.0, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, random_state=None)."
        raise Error("NotImplementedError: AdaBoostClassifier.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: AdaBoostClassifier.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: AdaBoostClassifier.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: AdaBoostClassifier.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostClassifier.fit is an API scaffold")

    def staged_score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: staged_score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostClassifier.staged_score is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostClassifier.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.staged_predict is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.decision_function is an API scaffold")

    def staged_decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_decision_function(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.staged_decision_function is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.predict_proba is an API scaffold")

    def staged_predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict_proba(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.staged_predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: AdaBoostClassifier.predict_log_proba is an API scaffold")

struct AdaBoostRegressor(Movable):

    "API scaffold for sklearn.ensemble._weight_boosting.AdaBoostRegressor. Operations are deferred."

    var estimator: APIValue

    var n_estimators: APIValue

    var learning_rate: APIValue

    var loss: APIValue

    var random_state: APIValue

    var estimator_: APIValue

    var estimators_: APIValue

    var estimator_weights_: APIValue

    var estimator_errors_: APIValue

    var feature_importances_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: AdaBoostRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: AdaBoostRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: AdaBoostRegressor.set_params is an API scaffold")

    def __init__(out self, estimator: Some[Movable] = None, *, n_estimators: Some[Movable] = 50, learning_rate: Some[Movable] = 1.0, loss: Some[Movable] = "linear", random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)."
        raise Error("NotImplementedError: AdaBoostRegressor.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: AdaBoostRegressor.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: AdaBoostRegressor.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: AdaBoostRegressor.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostRegressor.fit is an API scaffold")

    def staged_score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: staged_score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostRegressor.staged_score is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: AdaBoostRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: AdaBoostRegressor.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: AdaBoostRegressor.staged_predict is an API scaffold")

struct BaggingClassifier(Movable):

    "API scaffold for sklearn.ensemble._bagging.BaggingClassifier. Operations are deferred."

    var estimator: APIValue

    var n_estimators: APIValue

    var max_samples: APIValue

    var max_features: APIValue

    var bootstrap: APIValue

    var bootstrap_features: APIValue

    var oob_score: APIValue

    var warm_start: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var verbose: APIValue

    var estimator_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var estimators_: APIValue

    var estimators_samples_: APIValue

    var estimators_features_: APIValue

    var classes_: APIValue

    var n_classes_: APIValue

    var oob_score_: APIValue

    var oob_decision_function_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: BaggingClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: BaggingClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: BaggingClassifier.set_params is an API scaffold")

    def __init__(out self, estimator: Some[Movable] = None, n_estimators: Some[Movable] = 10, *, max_samples: Some[Movable] = None, max_features: Some[Movable] = 1.0, bootstrap: Some[Movable] = True, bootstrap_features: Some[Movable] = False, oob_score: Some[Movable] = False, warm_start: Some[Movable] = False, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimator=None, n_estimators=10, *, max_samples=None, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)."
        raise Error("NotImplementedError: BaggingClassifier.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: BaggingClassifier.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: BaggingClassifier.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: BaggingClassifier.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, **fit_params)."
        raise Error("NotImplementedError: BaggingClassifier.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: BaggingClassifier.score is an API scaffold")

    def predict(mut self, X: Some[Movable], params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict(self, X, **params)."
        raise Error("NotImplementedError: BaggingClassifier.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable], params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X, **params)."
        raise Error("NotImplementedError: BaggingClassifier.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable], params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X, **params)."
        raise Error("NotImplementedError: BaggingClassifier.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable], params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X, **params)."
        raise Error("NotImplementedError: BaggingClassifier.decision_function is an API scaffold")

struct BaggingRegressor(Movable):

    "API scaffold for sklearn.ensemble._bagging.BaggingRegressor. Operations are deferred."

    var estimator: APIValue

    var n_estimators: APIValue

    var max_samples: APIValue

    var max_features: APIValue

    var bootstrap: APIValue

    var bootstrap_features: APIValue

    var oob_score: APIValue

    var warm_start: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var verbose: APIValue

    var estimator_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var estimators_: APIValue

    var estimators_samples_: APIValue

    var estimators_features_: APIValue

    var oob_score_: APIValue

    var oob_prediction_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: BaggingRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: BaggingRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: BaggingRegressor.set_params is an API scaffold")

    def __init__(out self, estimator: Some[Movable] = None, n_estimators: Some[Movable] = 10, *, max_samples: Some[Movable] = None, max_features: Some[Movable] = 1.0, bootstrap: Some[Movable] = True, bootstrap_features: Some[Movable] = False, oob_score: Some[Movable] = False, warm_start: Some[Movable] = False, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimator=None, n_estimators=10, *, max_samples=None, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)."
        raise Error("NotImplementedError: BaggingRegressor.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: BaggingRegressor.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: BaggingRegressor.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: BaggingRegressor.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, **fit_params)."
        raise Error("NotImplementedError: BaggingRegressor.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: BaggingRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable], params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict(self, X, **params)."
        raise Error("NotImplementedError: BaggingRegressor.predict is an API scaffold")

struct GradientBoostingClassifier(Movable):

    "API scaffold for sklearn.ensemble._gb.GradientBoostingClassifier. Operations are deferred."

    var loss: APIValue

    var learning_rate: APIValue

    var n_estimators: APIValue

    var subsample: APIValue

    var criterion: APIValue

    var min_samples_split: APIValue

    var min_samples_leaf: APIValue

    var min_weight_fraction_leaf: APIValue

    var max_depth: APIValue

    var min_impurity_decrease: APIValue

    var init: APIValue

    var random_state: APIValue

    var max_features: APIValue

    var verbose: APIValue

    var max_leaf_nodes: APIValue

    var warm_start: APIValue

    var validation_fraction: APIValue

    var n_iter_no_change: APIValue

    var tol: APIValue

    var ccp_alpha: APIValue

    var n_estimators_: APIValue

    var n_trees_per_iteration_: APIValue

    var feature_importances_: APIValue

    var oob_improvement_: APIValue

    var oob_scores_: APIValue

    var oob_score_: APIValue

    var train_score_: APIValue

    var init_: APIValue

    var estimators_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_classes_: APIValue

    var max_features_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: GradientBoostingClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: GradientBoostingClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: GradientBoostingClassifier.set_params is an API scaffold")

    def __init__(out self, *, loss: Some[Movable] = "log_loss", learning_rate: Some[Movable] = 0.1, n_estimators: Some[Movable] = 100, subsample: Some[Movable] = 1.0, criterion: Some[Movable] = "deprecated", min_samples_split: Some[Movable] = 2, min_samples_leaf: Some[Movable] = 1, min_weight_fraction_leaf: Some[Movable] = 0.0, max_depth: Some[Movable] = 3, min_impurity_decrease: Some[Movable] = 0.0, init: Some[Movable] = None, random_state: Some[Movable] = None, max_features: Some[Movable] = None, verbose: Some[Movable] = 0, max_leaf_nodes: Some[Movable] = None, warm_start: Some[Movable] = False, validation_fraction: Some[Movable] = 0.1, n_iter_no_change: Some[Movable] = None, tol: Some[Movable] = 0.0001, ccp_alpha: Some[Movable] = 0.0) raises:
        "Scikit-learn signature: __init__(self, *, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='deprecated', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)."
        raise Error("NotImplementedError: GradientBoostingClassifier.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: GradientBoostingClassifier.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: GradientBoostingClassifier.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: GradientBoostingClassifier.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, monitor: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, monitor=None)."
        raise Error("NotImplementedError: GradientBoostingClassifier.fit is an API scaffold")

    def apply(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: apply(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.apply is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: GradientBoostingClassifier.score is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.decision_function is an API scaffold")

    def staged_decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_decision_function(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.staged_decision_function is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.staged_predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.predict_log_proba is an API scaffold")

    def staged_predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict_proba(self, X)."
        raise Error("NotImplementedError: GradientBoostingClassifier.staged_predict_proba is an API scaffold")

struct GradientBoostingRegressor(Movable):

    "API scaffold for sklearn.ensemble._gb.GradientBoostingRegressor. Operations are deferred."

    var loss: APIValue

    var learning_rate: APIValue

    var n_estimators: APIValue

    var subsample: APIValue

    var criterion: APIValue

    var min_samples_split: APIValue

    var min_samples_leaf: APIValue

    var min_weight_fraction_leaf: APIValue

    var max_depth: APIValue

    var min_impurity_decrease: APIValue

    var init: APIValue

    var random_state: APIValue

    var max_features: APIValue

    var alpha: APIValue

    var verbose: APIValue

    var max_leaf_nodes: APIValue

    var warm_start: APIValue

    var validation_fraction: APIValue

    var n_iter_no_change: APIValue

    var tol: APIValue

    var ccp_alpha: APIValue

    var n_estimators_: APIValue

    var n_trees_per_iteration_: APIValue

    var feature_importances_: APIValue

    var oob_improvement_: APIValue

    var oob_scores_: APIValue

    var oob_score_: APIValue

    var train_score_: APIValue

    var init_: APIValue

    var estimators_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var max_features_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: GradientBoostingRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: GradientBoostingRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: GradientBoostingRegressor.set_params is an API scaffold")

    def __init__(out self, *, loss: Some[Movable] = "squared_error", learning_rate: Some[Movable] = 0.1, n_estimators: Some[Movable] = 100, subsample: Some[Movable] = 1.0, criterion: Some[Movable] = "deprecated", min_samples_split: Some[Movable] = 2, min_samples_leaf: Some[Movable] = 1, min_weight_fraction_leaf: Some[Movable] = 0.0, max_depth: Some[Movable] = 3, min_impurity_decrease: Some[Movable] = 0.0, init: Some[Movable] = None, random_state: Some[Movable] = None, max_features: Some[Movable] = None, alpha: Some[Movable] = 0.9, verbose: Some[Movable] = 0, max_leaf_nodes: Some[Movable] = None, warm_start: Some[Movable] = False, validation_fraction: Some[Movable] = 0.1, n_iter_no_change: Some[Movable] = None, tol: Some[Movable] = 0.0001, ccp_alpha: Some[Movable] = 0.0) raises:
        "Scikit-learn signature: __init__(self, *, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='deprecated', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)."
        raise Error("NotImplementedError: GradientBoostingRegressor.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: GradientBoostingRegressor.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: GradientBoostingRegressor.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: GradientBoostingRegressor.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, monitor: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, monitor=None)."
        raise Error("NotImplementedError: GradientBoostingRegressor.fit is an API scaffold")

    def apply(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: apply(self, X)."
        raise Error("NotImplementedError: GradientBoostingRegressor.apply is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: GradientBoostingRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: GradientBoostingRegressor.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: GradientBoostingRegressor.staged_predict is an API scaffold")

struct HistGradientBoostingClassifier(Movable):

    "API scaffold for sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier. Operations are deferred."

    var loss: APIValue

    var learning_rate: APIValue

    var max_iter: APIValue

    var max_leaf_nodes: APIValue

    var max_depth: APIValue

    var min_samples_leaf: APIValue

    var l2_regularization: APIValue

    var max_features: APIValue

    var max_bins: APIValue

    var categorical_features: APIValue

    var monotonic_cst: APIValue

    var interaction_cst: APIValue

    var warm_start: APIValue

    var early_stopping: APIValue

    var scoring: APIValue

    var validation_fraction: APIValue

    var n_iter_no_change: APIValue

    var tol: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var class_weight: APIValue

    var classes_: APIValue

    var do_early_stopping_: APIValue

    var n_iter_: APIValue

    var n_trees_per_iteration_: APIValue

    var train_score_: APIValue

    var validation_score_: APIValue

    var is_categorical_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.set_params is an API scaffold")

    def __init__(out self, loss: Some[Movable] = "log_loss", *, learning_rate: Some[Movable] = 0.1, max_iter: Some[Movable] = 100, max_leaf_nodes: Some[Movable] = 31, max_depth: Some[Movable] = None, min_samples_leaf: Some[Movable] = 20, l2_regularization: Some[Movable] = 0.0, max_features: Some[Movable] = 1.0, max_bins: Some[Movable] = 255, categorical_features: Some[Movable] = "from_dtype", monotonic_cst: Some[Movable] = None, interaction_cst: Some[Movable] = None, warm_start: Some[Movable] = False, early_stopping: Some[Movable] = "auto", scoring: Some[Movable] = "loss", validation_fraction: Some[Movable] = 0.1, n_iter_no_change: Some[Movable] = 10, tol: Some[Movable] = 1e-07, verbose: Some[Movable] = 0, random_state: Some[Movable] = None, class_weight: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, loss='log_loss', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None, class_weight=None)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.__init__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, *, X_val: Some[Movable] = None, y_val: Some[Movable] = None, sample_weight_val: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None, sample_weight_val=None)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.staged_predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.predict_proba is an API scaffold")

    def staged_predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict_proba(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.staged_predict_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.decision_function is an API scaffold")

    def staged_decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_decision_function(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingClassifier.staged_decision_function is an API scaffold")

struct HistGradientBoostingRegressor(Movable):

    "API scaffold for sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor. Operations are deferred."

    var loss: APIValue

    var quantile: APIValue

    var learning_rate: APIValue

    var max_iter: APIValue

    var max_leaf_nodes: APIValue

    var max_depth: APIValue

    var min_samples_leaf: APIValue

    var l2_regularization: APIValue

    var max_features: APIValue

    var max_bins: APIValue

    var categorical_features: APIValue

    var monotonic_cst: APIValue

    var interaction_cst: APIValue

    var warm_start: APIValue

    var early_stopping: APIValue

    var scoring: APIValue

    var validation_fraction: APIValue

    var n_iter_no_change: APIValue

    var tol: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var do_early_stopping_: APIValue

    var n_iter_: APIValue

    var n_trees_per_iteration_: APIValue

    var train_score_: APIValue

    var validation_score_: APIValue

    var is_categorical_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.set_params is an API scaffold")

    def __init__(out self, loss: Some[Movable] = "squared_error", *, quantile: Some[Movable] = None, learning_rate: Some[Movable] = 0.1, max_iter: Some[Movable] = 100, max_leaf_nodes: Some[Movable] = 31, max_depth: Some[Movable] = None, min_samples_leaf: Some[Movable] = 20, l2_regularization: Some[Movable] = 0.0, max_features: Some[Movable] = 1.0, max_bins: Some[Movable] = 255, categorical_features: Some[Movable] = "from_dtype", monotonic_cst: Some[Movable] = None, interaction_cst: Some[Movable] = None, warm_start: Some[Movable] = False, early_stopping: Some[Movable] = "auto", scoring: Some[Movable] = "loss", validation_fraction: Some[Movable] = 0.1, n_iter_no_change: Some[Movable] = 10, tol: Some[Movable] = 1e-07, verbose: Some[Movable] = 0, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, loss='squared_error', *, quantile=None, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.__init__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None, *, X_val: Some[Movable] = None, y_val: Some[Movable] = None, sample_weight_val: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None, sample_weight_val=None)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.predict is an API scaffold")

    def staged_predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: staged_predict(self, X)."
        raise Error("NotImplementedError: HistGradientBoostingRegressor.staged_predict is an API scaffold")

struct IsolationForest(Movable):

    "API scaffold for sklearn.ensemble._iforest.IsolationForest. Operations are deferred."

    var n_estimators: APIValue

    var max_samples: APIValue

    var contamination: APIValue

    var max_features: APIValue

    var bootstrap: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var verbose: APIValue

    var warm_start: APIValue

    var estimator_: APIValue

    var estimators_: APIValue

    var estimators_features_: APIValue

    var estimators_samples_: APIValue

    var max_samples_: APIValue

    var offset_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: IsolationForest.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: IsolationForest.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: IsolationForest.set_params is an API scaffold")

    def __init__(out self, *, n_estimators: Some[Movable] = 100, max_samples: Some[Movable] = "auto", contamination: Some[Movable] = "auto", max_features: Some[Movable] = 1.0, bootstrap: Some[Movable] = False, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, verbose: Some[Movable] = 0, warm_start: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, *, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)."
        raise Error("NotImplementedError: IsolationForest.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: IsolationForest.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: IsolationForest.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: IsolationForest.__iter__ is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable] = None, sample_weight: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y=None, sample_weight=None)."
        raise Error("NotImplementedError: IsolationForest.fit is an API scaffold")

    def fit_predict(mut self, X: Some[Movable], y: Some[Movable] = None, kwargs: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_predict(self, X, y=None, **kwargs)."
        raise Error("NotImplementedError: IsolationForest.fit_predict is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: IsolationForest.predict is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: IsolationForest.decision_function is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: IsolationForest.score_samples is an API scaffold")

struct RandomTreesEmbedding(Movable):

    "API scaffold for sklearn.ensemble._forest.RandomTreesEmbedding. Operations are deferred."

    var n_estimators: APIValue

    var max_depth: APIValue

    var min_samples_split: APIValue

    var min_samples_leaf: APIValue

    var min_weight_fraction_leaf: APIValue

    var max_leaf_nodes: APIValue

    var min_impurity_decrease: APIValue

    var sparse_output: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var verbose: APIValue

    var warm_start: APIValue

    var estimator_: APIValue

    var estimators_: APIValue

    var feature_importances_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_outputs_: APIValue

    var one_hot_encoder_: APIValue

    var estimators_samples_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RandomTreesEmbedding.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: RandomTreesEmbedding.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: RandomTreesEmbedding.set_params is an API scaffold")

    def __init__(out self, n_estimators: Some[Movable] = 100, *, max_depth: Some[Movable] = 5, min_samples_split: Some[Movable] = 2, min_samples_leaf: Some[Movable] = 1, min_weight_fraction_leaf: Some[Movable] = 0.0, max_leaf_nodes: Some[Movable] = None, min_impurity_decrease: Some[Movable] = 0.0, sparse_output: Some[Movable] = True, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, verbose: Some[Movable] = 0, warm_start: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, sparse_output=True, n_jobs=None, random_state=None, verbose=0, warm_start=False)."
        raise Error("NotImplementedError: RandomTreesEmbedding.__init__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: RandomTreesEmbedding.__len__ is an API scaffold")

    def __getitem__(mut self, index: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, index)."
        raise Error("NotImplementedError: RandomTreesEmbedding.__getitem__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: RandomTreesEmbedding.__iter__ is an API scaffold")

    def apply(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: apply(self, X)."
        raise Error("NotImplementedError: RandomTreesEmbedding.apply is an API scaffold")

    def decision_path(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_path(self, X)."
        raise Error("NotImplementedError: RandomTreesEmbedding.decision_path is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable] = None, sample_weight: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y=None, sample_weight=None)."
        raise Error("NotImplementedError: RandomTreesEmbedding.fit is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: RandomTreesEmbedding.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, sample_weight=None)."
        raise Error("NotImplementedError: RandomTreesEmbedding.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: RandomTreesEmbedding.get_feature_names_out is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: RandomTreesEmbedding.transform is an API scaffold")

struct StackingClassifier(Movable):

    "API scaffold for sklearn.ensemble._stacking.StackingClassifier. Operations are deferred."

    var estimators: APIValue

    var final_estimator: APIValue

    var cv: APIValue

    var stack_method: APIValue

    var n_jobs: APIValue

    var passthrough: APIValue

    var verbose: APIValue

    var classes_: APIValue

    var estimators_: APIValue

    var named_estimators_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var final_estimator_: APIValue

    var stack_method_: APIValue

    var named_estimators: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: StackingClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: StackingClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: StackingClassifier.set_params is an API scaffold")

    def __init__(out self, estimators: Some[Movable], final_estimator: Some[Movable] = None, *, cv: Some[Movable] = None, stack_method: Some[Movable] = "auto", n_jobs: Some[Movable] = None, passthrough: Some[Movable] = False, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0)."
        raise Error("NotImplementedError: StackingClassifier.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: StackingClassifier.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: StackingClassifier.fit_transform is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, **fit_params)."
        raise Error("NotImplementedError: StackingClassifier.fit is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: StackingClassifier.get_feature_names_out is an API scaffold")

    def predict(mut self, X: Some[Movable], predict_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict(self, X, **predict_params)."
        raise Error("NotImplementedError: StackingClassifier.predict is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: StackingClassifier.score is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: StackingClassifier.predict_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: StackingClassifier.decision_function is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: StackingClassifier.transform is an API scaffold")

struct StackingRegressor(Movable):

    "API scaffold for sklearn.ensemble._stacking.StackingRegressor. Operations are deferred."

    var estimators: APIValue

    var final_estimator: APIValue

    var cv: APIValue

    var n_jobs: APIValue

    var passthrough: APIValue

    var verbose: APIValue

    var estimators_: APIValue

    var named_estimators_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var final_estimator_: APIValue

    var stack_method_: APIValue

    var named_estimators: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: StackingRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: StackingRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: StackingRegressor.set_params is an API scaffold")

    def __init__(out self, estimators: Some[Movable], final_estimator: Some[Movable] = None, *, cv: Some[Movable] = None, n_jobs: Some[Movable] = None, passthrough: Some[Movable] = False, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimators, final_estimator=None, *, cv=None, n_jobs=None, passthrough=False, verbose=0)."
        raise Error("NotImplementedError: StackingRegressor.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: StackingRegressor.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable], fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y, **fit_params)."
        raise Error("NotImplementedError: StackingRegressor.fit_transform is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, **fit_params)."
        raise Error("NotImplementedError: StackingRegressor.fit is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: StackingRegressor.get_feature_names_out is an API scaffold")

    def predict(mut self, X: Some[Movable], predict_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: predict(self, X, **predict_params)."
        raise Error("NotImplementedError: StackingRegressor.predict is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: StackingRegressor.score is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: StackingRegressor.transform is an API scaffold")

struct VotingClassifier(Movable):

    "API scaffold for sklearn.ensemble._voting.VotingClassifier. Operations are deferred."

    var estimators: APIValue

    var voting: APIValue

    var weights: APIValue

    var n_jobs: APIValue

    var flatten_transform: APIValue

    var verbose: APIValue

    var estimators_: APIValue

    var named_estimators_: APIValue

    var le_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var named_estimators: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: VotingClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: VotingClassifier.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: VotingClassifier.set_params is an API scaffold")

    def __init__(out self, estimators: Some[Movable], *, voting: Some[Movable] = "hard", weights: Some[Movable] = None, n_jobs: Some[Movable] = None, flatten_transform: Some[Movable] = True, verbose: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)."
        raise Error("NotImplementedError: VotingClassifier.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: VotingClassifier.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: VotingClassifier.fit_transform is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, **fit_params)."
        raise Error("NotImplementedError: VotingClassifier.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: VotingClassifier.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: VotingClassifier.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: VotingClassifier.predict_proba is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: VotingClassifier.transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: VotingClassifier.get_feature_names_out is an API scaffold")

struct VotingRegressor(Movable):

    "API scaffold for sklearn.ensemble._voting.VotingRegressor. Operations are deferred."

    var estimators: APIValue

    var weights: APIValue

    var n_jobs: APIValue

    var verbose: APIValue

    var estimators_: APIValue

    var named_estimators_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var named_estimators: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: VotingRegressor.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: VotingRegressor.get_params is an API scaffold")

    def set_params[origin: MutOrigin](ref[origin] self, params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: VotingRegressor.set_params is an API scaffold")

    def __init__(out self, estimators: Some[Movable], *, weights: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False)."
        raise Error("NotImplementedError: VotingRegressor.__init__ is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: VotingRegressor.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: VotingRegressor.fit_transform is an API scaffold")

    def fit[origin: MutOrigin](ref[origin] self, X: Some[Movable], y: Some[Movable], fit_params: Some[Movable] = None) raises -> ref[self] Self:
        "Scikit-learn signature: fit(self, X, y, **fit_params)."
        raise Error("NotImplementedError: VotingRegressor.fit is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: VotingRegressor.score is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: VotingRegressor.predict is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: VotingRegressor.transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: VotingRegressor.get_feature_names_out is an API scaffold")


from moxie.ensemble._forest import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
