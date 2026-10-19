# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.model_selection._api_types import APIValue
from std.memory import bitcast

struct BaseCrossValidator(Movable):

    "API scaffold for sklearn.model_selection._split.BaseCrossValidator. Operations are deferred."

    def __init__(out self) raises:

        raise Error("NotImplementedError: BaseCrossValidator is an API scaffold")

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: BaseCrossValidator.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: BaseCrossValidator.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: BaseCrossValidator.get_n_splits is an API scaffold")

struct BaseShuffleSplit(Movable):

    "API scaffold for sklearn.model_selection._split.BaseShuffleSplit. Operations are deferred."

    var n_splits: APIValue

    var test_size: APIValue

    var train_size: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: BaseShuffleSplit.get_metadata_routing is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 10, *, test_size: Some[Movable] = None, train_size: Some[Movable] = None, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None)."
        raise Error("NotImplementedError: BaseShuffleSplit.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: BaseShuffleSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: BaseShuffleSplit.get_n_splits is an API scaffold")

struct FixedThresholdClassifier(Movable):

    "API scaffold for sklearn.model_selection._classification_threshold.FixedThresholdClassifier. Operations are deferred."

    var estimator: APIValue

    var threshold: APIValue

    var pos_label: APIValue

    var response_method: APIValue

    var estimator_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: FixedThresholdClassifier.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: FixedThresholdClassifier.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: FixedThresholdClassifier.set_params is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: FixedThresholdClassifier.score is an API scaffold")

    def __init__(out self, estimator: Some[Movable], *, threshold: Some[Movable] = "auto", pos_label: Some[Movable] = None, response_method: Some[Movable] = "auto") raises:
        "Scikit-learn signature: __init__(self, estimator, *, threshold='auto', pos_label=None, response_method='auto')."
        raise Error("NotImplementedError: FixedThresholdClassifier.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable], params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y, **params)."
        raise Error("NotImplementedError: FixedThresholdClassifier.fit is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: FixedThresholdClassifier.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: FixedThresholdClassifier.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: FixedThresholdClassifier.decision_function is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: FixedThresholdClassifier.predict is an API scaffold")

struct GridSearchCV(Movable):

    "API scaffold for sklearn.model_selection._search.GridSearchCV. Operations are deferred."

    var estimator: APIValue

    var param_grid: APIValue

    var scoring: APIValue

    var n_jobs: APIValue

    var refit: APIValue

    var cv: APIValue

    var verbose: APIValue

    var pre_dispatch: APIValue

    var error_score: APIValue

    var return_train_score: APIValue

    var cv_results_: APIValue

    var best_estimator_: APIValue

    var best_score_: APIValue

    var best_params_: APIValue

    var best_index_: APIValue

    var scorer_: APIValue

    var n_splits_: APIValue

    var refit_time_: APIValue

    var multimetric_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: GridSearchCV.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: GridSearchCV.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: GridSearchCV.set_params is an API scaffold")

    def set_callbacks(mut self, *callbacks: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: set_callbacks(self, *callbacks)."
        raise Error("NotImplementedError: GridSearchCV.set_callbacks is an API scaffold")

    def __init__(out self, estimator: Some[Movable], param_grid: Some[Movable], *, scoring: Some[Movable] = None, n_jobs: Some[Movable] = None, refit: Some[Movable] = True, cv: Some[Movable] = None, verbose: Some[Movable] = 0, pre_dispatch: Some[Movable] = "2*n_jobs", error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), return_train_score: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False)."
        raise Error("NotImplementedError: GridSearchCV.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y=None, **params)."
        raise Error("NotImplementedError: GridSearchCV.score is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: GridSearchCV.score_samples is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: GridSearchCV.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: GridSearchCV.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: GridSearchCV.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: GridSearchCV.decision_function is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: GridSearchCV.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: GridSearchCV.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, **params)."
        raise Error("NotImplementedError: GridSearchCV.fit is an API scaffold")

struct GroupKFold(Movable):

    "API scaffold for sklearn.model_selection._split.GroupKFold. Operations are deferred."

    var n_splits: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: GroupKFold.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: GroupKFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: GroupKFold.get_n_splits is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, *, shuffle: Some[Movable] = False, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, *, shuffle=False, random_state=None)."
        raise Error("NotImplementedError: GroupKFold.__init__ is an API scaffold")

struct GroupShuffleSplit(Movable):

    "API scaffold for sklearn.model_selection._split.GroupShuffleSplit. Operations are deferred."

    var n_splits: APIValue

    var test_size: APIValue

    var train_size: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: GroupShuffleSplit.get_metadata_routing is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, *, test_size: Some[Movable] = None, train_size: Some[Movable] = None, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, *, test_size=None, train_size=None, random_state=None)."
        raise Error("NotImplementedError: GroupShuffleSplit.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: GroupShuffleSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: GroupShuffleSplit.get_n_splits is an API scaffold")

struct HalvingGridSearchCV(Movable):

    "API scaffold for sklearn.model_selection._search_successive_halving.HalvingGridSearchCV. Operations are deferred."

    var estimator: APIValue

    var param_grid: APIValue

    var factor: APIValue

    var resource: APIValue

    var max_resources: APIValue

    var min_resources: APIValue

    var aggressive_elimination: APIValue

    var cv: APIValue

    var scoring: APIValue

    var refit: APIValue

    var error_score: APIValue

    var return_train_score: APIValue

    var random_state: APIValue

    var n_jobs: APIValue

    var verbose: APIValue

    var n_resources_: APIValue

    var n_candidates_: APIValue

    var n_remaining_candidates_: APIValue

    var max_resources_: APIValue

    var min_resources_: APIValue

    var n_iterations_: APIValue

    var n_possible_iterations_: APIValue

    var n_required_iterations_: APIValue

    var cv_results_: APIValue

    var all_cv_results_: APIValue

    var best_estimator_: APIValue

    var best_score_: APIValue

    var best_params_: APIValue

    var best_index_: APIValue

    var scorer_: APIValue

    var n_splits_: APIValue

    var refit_time_: APIValue

    var multimetric_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HalvingGridSearchCV.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HalvingGridSearchCV.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HalvingGridSearchCV.set_params is an API scaffold")

    def set_callbacks(mut self, *callbacks: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: set_callbacks(self, *callbacks)."
        raise Error("NotImplementedError: HalvingGridSearchCV.set_callbacks is an API scaffold")

    def __init__(out self, estimator: Some[Movable], param_grid: Some[Movable], *, factor: Some[Movable] = 3, resource: Some[Movable] = "n_samples", max_resources: Some[Movable] = "auto", min_resources: Some[Movable] = "exhaust", aggressive_elimination: Some[Movable] = False, cv: Some[Movable] = 5, scoring: Some[Movable] = None, refit: Some[Movable] = True, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), return_train_score: Some[Movable] = True, random_state: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimator, param_grid, *, factor=3, resource='n_samples', max_resources='auto', min_resources='exhaust', aggressive_elimination=False, cv=5, scoring=None, refit=True, error_score=np.nan, return_train_score=True, random_state=None, n_jobs=None, verbose=0)."
        raise Error("NotImplementedError: HalvingGridSearchCV.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y=None, **params)."
        raise Error("NotImplementedError: HalvingGridSearchCV.score is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.score_samples is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.decision_function is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: HalvingGridSearchCV.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, **params)."
        raise Error("NotImplementedError: HalvingGridSearchCV.fit is an API scaffold")

struct HalvingRandomSearchCV(Movable):

    "API scaffold for sklearn.model_selection._search_successive_halving.HalvingRandomSearchCV. Operations are deferred."

    var estimator: APIValue

    var param_distributions: APIValue

    var n_candidates: APIValue

    var factor: APIValue

    var resource: APIValue

    var max_resources: APIValue

    var min_resources: APIValue

    var aggressive_elimination: APIValue

    var cv: APIValue

    var scoring: APIValue

    var refit: APIValue

    var error_score: APIValue

    var return_train_score: APIValue

    var random_state: APIValue

    var n_jobs: APIValue

    var verbose: APIValue

    var n_resources_: APIValue

    var n_candidates_: APIValue

    var n_remaining_candidates_: APIValue

    var max_resources_: APIValue

    var min_resources_: APIValue

    var n_iterations_: APIValue

    var n_possible_iterations_: APIValue

    var n_required_iterations_: APIValue

    var cv_results_: APIValue

    var all_cv_results_: APIValue

    var best_estimator_: APIValue

    var best_score_: APIValue

    var best_params_: APIValue

    var best_index_: APIValue

    var scorer_: APIValue

    var n_splits_: APIValue

    var refit_time_: APIValue

    var multimetric_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.set_params is an API scaffold")

    def set_callbacks(mut self, *callbacks: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: set_callbacks(self, *callbacks)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.set_callbacks is an API scaffold")

    def __init__(out self, estimator: Some[Movable], param_distributions: Some[Movable], *, n_candidates: Some[Movable] = "exhaust", factor: Some[Movable] = 3, resource: Some[Movable] = "n_samples", max_resources: Some[Movable] = "auto", min_resources: Some[Movable] = "smallest", aggressive_elimination: Some[Movable] = False, cv: Some[Movable] = 5, scoring: Some[Movable] = None, refit: Some[Movable] = True, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), return_train_score: Some[Movable] = True, random_state: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, estimator, param_distributions, *, n_candidates='exhaust', factor=3, resource='n_samples', max_resources='auto', min_resources='smallest', aggressive_elimination=False, cv=5, scoring=None, refit=True, error_score=np.nan, return_train_score=True, random_state=None, n_jobs=None, verbose=0)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y=None, **params)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.score is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.score_samples is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.decision_function is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, **params)."
        raise Error("NotImplementedError: HalvingRandomSearchCV.fit is an API scaffold")

struct KFold(Movable):

    "API scaffold for sklearn.model_selection._split.KFold. Operations are deferred."

    var n_splits: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: KFold.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: KFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: KFold.get_n_splits is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, *, shuffle: Some[Movable] = False, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, *, shuffle=False, random_state=None)."
        raise Error("NotImplementedError: KFold.__init__ is an API scaffold")

struct LearningCurveDisplay(Movable):

    "API scaffold for sklearn.model_selection._plot.LearningCurveDisplay. Operations are deferred."

    var train_sizes: APIValue

    var train_scores: APIValue

    var test_scores: APIValue

    var score_name: APIValue

    var ax_: APIValue

    var figure_: APIValue

    var errorbar_: APIValue

    var lines_: APIValue

    var fill_between_: APIValue

    def __init__(out self, *, train_sizes: Some[Movable], train_scores: Some[Movable], test_scores: Some[Movable], score_name: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, train_sizes, train_scores, test_scores, score_name=None)."
        raise Error("NotImplementedError: LearningCurveDisplay.__init__ is an API scaffold")

    def plot(mut self, ax: Some[Movable] = None, *, negate_score: Some[Movable] = False, score_name: Some[Movable] = None, score_type: Some[Movable] = "both", std_display_style: Some[Movable] = "fill_between", line_kw: Some[Movable] = None, fill_between_kw: Some[Movable] = None, errorbar_kw: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: plot(self, ax=None, *, negate_score=False, score_name=None, score_type='both', std_display_style='fill_between', line_kw=None, fill_between_kw=None, errorbar_kw=None)."
        raise Error("NotImplementedError: LearningCurveDisplay.plot is an API scaffold")

    @staticmethod
    def from_estimator(estimator: Some[Movable], X: Some[Movable], y: Some[Movable], *, groups: Some[Movable] = None, train_sizes: Some[Movable] = List[Float64]([0.1, 0.325, 0.55, 0.775, 1.0]), cv: Some[Movable] = None, scoring: Some[Movable] = None, exploit_incremental_learning: Some[Movable] = False, n_jobs: Some[Movable] = None, pre_dispatch: Some[Movable] = "all", verbose: Some[Movable] = 0, shuffle: Some[Movable] = False, random_state: Some[Movable] = None, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), fit_params: Some[Movable] = None, ax: Some[Movable] = None, negate_score: Some[Movable] = False, score_name: Some[Movable] = None, score_type: Some[Movable] = "both", std_display_style: Some[Movable] = "fill_between", line_kw: Some[Movable] = None, fill_between_kw: Some[Movable] = None, errorbar_kw: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: from_estimator(cls, estimator, X, y, *, groups=None, train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', verbose=0, shuffle=False, random_state=None, error_score=np.nan, fit_params=None, ax=None, negate_score=False, score_name=None, score_type='both', std_display_style='fill_between', line_kw=None, fill_between_kw=None, errorbar_kw=None)."
        raise Error("NotImplementedError: LearningCurveDisplay.from_estimator is an API scaffold")

struct LeaveOneGroupOut(Movable):

    "API scaffold for sklearn.model_selection._split.LeaveOneGroupOut. Operations are deferred."

    def __init__(out self) raises:

        raise Error("NotImplementedError: LeaveOneGroupOut is an API scaffold")

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LeaveOneGroupOut.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeaveOneGroupOut.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: LeaveOneGroupOut.get_n_splits is an API scaffold")

struct LeaveOneOut(Movable):

    "API scaffold for sklearn.model_selection._split.LeaveOneOut. Operations are deferred."

    def __init__(out self) raises:

        raise Error("NotImplementedError: LeaveOneOut is an API scaffold")

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LeaveOneOut.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeaveOneOut.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeaveOneOut.get_n_splits is an API scaffold")

struct LeavePGroupsOut(Movable):

    "API scaffold for sklearn.model_selection._split.LeavePGroupsOut. Operations are deferred."

    var n_groups: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LeavePGroupsOut.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeavePGroupsOut.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: LeavePGroupsOut.get_n_splits is an API scaffold")

    def __init__(out self, n_groups: Some[Movable]) raises:
        "Scikit-learn signature: __init__(self, n_groups)."
        raise Error("NotImplementedError: LeavePGroupsOut.__init__ is an API scaffold")

struct LeavePOut(Movable):

    "API scaffold for sklearn.model_selection._split.LeavePOut. Operations are deferred."

    var p: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: LeavePOut.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeavePOut.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: LeavePOut.get_n_splits is an API scaffold")

    def __init__(out self, p: Some[Movable]) raises:
        "Scikit-learn signature: __init__(self, p)."
        raise Error("NotImplementedError: LeavePOut.__init__ is an API scaffold")

struct ParameterGrid(Movable):

    "API scaffold for sklearn.model_selection._search.ParameterGrid. Operations are deferred."

    var param_grid: APIValue

    def __init__(out self, param_grid: Some[Movable]) raises:
        "Scikit-learn signature: __init__(self, param_grid)."
        raise Error("NotImplementedError: ParameterGrid.__init__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: ParameterGrid.__iter__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: ParameterGrid.__len__ is an API scaffold")

    def __getitem__(mut self, ind: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: __getitem__(self, ind)."
        raise Error("NotImplementedError: ParameterGrid.__getitem__ is an API scaffold")

struct ParameterSampler(Movable):

    "API scaffold for sklearn.model_selection._search.ParameterSampler. Operations are deferred."

    var param_distributions: APIValue

    var n_iter: APIValue

    var random_state: APIValue

    def __init__(out self, param_distributions: Some[Movable], n_iter: Some[Movable], *, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, param_distributions, n_iter, *, random_state=None)."
        raise Error("NotImplementedError: ParameterSampler.__init__ is an API scaffold")

    def __iter__(mut self) raises -> APIValue:
        "Scikit-learn signature: __iter__(self)."
        raise Error("NotImplementedError: ParameterSampler.__iter__ is an API scaffold")

    def __len__(mut self) raises -> APIValue:
        "Scikit-learn signature: __len__(self)."
        raise Error("NotImplementedError: ParameterSampler.__len__ is an API scaffold")

struct PredefinedSplit(Movable):

    "API scaffold for sklearn.model_selection._split.PredefinedSplit. Operations are deferred."

    var test_fold: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: PredefinedSplit.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: PredefinedSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: PredefinedSplit.get_n_splits is an API scaffold")

    def __init__(out self, test_fold: Some[Movable]) raises:
        "Scikit-learn signature: __init__(self, test_fold)."
        raise Error("NotImplementedError: PredefinedSplit.__init__ is an API scaffold")

struct RandomizedSearchCV(Movable):

    "API scaffold for sklearn.model_selection._search.RandomizedSearchCV. Operations are deferred."

    var estimator: APIValue

    var param_distributions: APIValue

    var n_iter: APIValue

    var scoring: APIValue

    var n_jobs: APIValue

    var refit: APIValue

    var cv: APIValue

    var verbose: APIValue

    var pre_dispatch: APIValue

    var random_state: APIValue

    var error_score: APIValue

    var return_train_score: APIValue

    var cv_results_: APIValue

    var best_estimator_: APIValue

    var best_score_: APIValue

    var best_params_: APIValue

    var best_index_: APIValue

    var scorer_: APIValue

    var n_splits_: APIValue

    var refit_time_: APIValue

    var multimetric_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RandomizedSearchCV.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: RandomizedSearchCV.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: RandomizedSearchCV.set_params is an API scaffold")

    def set_callbacks(mut self, *callbacks: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: set_callbacks(self, *callbacks)."
        raise Error("NotImplementedError: RandomizedSearchCV.set_callbacks is an API scaffold")

    def __init__(out self, estimator: Some[Movable], param_distributions: Some[Movable], *, n_iter: Some[Movable] = 10, scoring: Some[Movable] = None, n_jobs: Some[Movable] = None, refit: Some[Movable] = True, cv: Some[Movable] = None, verbose: Some[Movable] = 0, pre_dispatch: Some[Movable] = "2*n_jobs", random_state: Some[Movable] = None, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), return_train_score: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=np.nan, return_train_score=False)."
        raise Error("NotImplementedError: RandomizedSearchCV.__init__ is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y=None, **params)."
        raise Error("NotImplementedError: RandomizedSearchCV.score is an API scaffold")

    def score_samples(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: score_samples(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.score_samples is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.predict is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.decision_function is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: RandomizedSearchCV.inverse_transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, **params)."
        raise Error("NotImplementedError: RandomizedSearchCV.fit is an API scaffold")

struct RepeatedKFold(Movable):

    "API scaffold for sklearn.model_selection._split.RepeatedKFold. Operations are deferred."

    var n_splits: APIValue

    var n_repeats: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RepeatedKFold.get_metadata_routing is an API scaffold")

    def __init__(out self, *, n_splits: Some[Movable] = 5, n_repeats: Some[Movable] = 10, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, n_splits=5, n_repeats=10, random_state=None)."
        raise Error("NotImplementedError: RepeatedKFold.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: RepeatedKFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: RepeatedKFold.get_n_splits is an API scaffold")

struct RepeatedStratifiedKFold(Movable):

    "API scaffold for sklearn.model_selection._split.RepeatedStratifiedKFold. Operations are deferred."

    var n_splits: APIValue

    var n_repeats: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: RepeatedStratifiedKFold.get_metadata_routing is an API scaffold")

    def __init__(out self, *, n_splits: Some[Movable] = 5, n_repeats: Some[Movable] = 10, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, n_splits=5, n_repeats=10, random_state=None)."
        raise Error("NotImplementedError: RepeatedStratifiedKFold.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable], groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y, groups=None)."
        raise Error("NotImplementedError: RepeatedStratifiedKFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: RepeatedStratifiedKFold.get_n_splits is an API scaffold")

struct ShuffleSplit(Movable):

    "API scaffold for sklearn.model_selection._split.ShuffleSplit. Operations are deferred."

    var n_splits: APIValue

    var test_size: APIValue

    var train_size: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: ShuffleSplit.get_metadata_routing is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 10, *, test_size: Some[Movable] = None, train_size: Some[Movable] = None, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None)."
        raise Error("NotImplementedError: ShuffleSplit.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: ShuffleSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: ShuffleSplit.get_n_splits is an API scaffold")

struct StratifiedGroupKFold(Movable):

    "API scaffold for sklearn.model_selection._split.StratifiedGroupKFold. Operations are deferred."

    var n_splits: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: StratifiedGroupKFold.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: StratifiedGroupKFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: StratifiedGroupKFold.get_n_splits is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, shuffle: Some[Movable] = False, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, shuffle=False, random_state=None)."
        raise Error("NotImplementedError: StratifiedGroupKFold.__init__ is an API scaffold")

struct StratifiedKFold(Movable):

    "API scaffold for sklearn.model_selection._split.StratifiedKFold. Operations are deferred."

    var n_splits: APIValue

    var shuffle: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: StratifiedKFold.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable], groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y, groups=None)."
        raise Error("NotImplementedError: StratifiedKFold.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: StratifiedKFold.get_n_splits is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, *, shuffle: Some[Movable] = False, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, *, shuffle=False, random_state=None)."
        raise Error("NotImplementedError: StratifiedKFold.__init__ is an API scaffold")

struct StratifiedShuffleSplit(Movable):

    "API scaffold for sklearn.model_selection._split.StratifiedShuffleSplit. Operations are deferred."

    var n_splits: APIValue

    var test_size: APIValue

    var train_size: APIValue

    var random_state: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: StratifiedShuffleSplit.get_metadata_routing is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 10, *, test_size: Some[Movable] = None, train_size: Some[Movable] = None, random_state: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None)."
        raise Error("NotImplementedError: StratifiedShuffleSplit.__init__ is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable], groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y, groups=None)."
        raise Error("NotImplementedError: StratifiedShuffleSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: StratifiedShuffleSplit.get_n_splits is an API scaffold")

struct TimeSeriesSplit(Movable):

    "API scaffold for sklearn.model_selection._split.TimeSeriesSplit. Operations are deferred."

    var n_splits: APIValue

    var max_train_size: APIValue

    var test_size: APIValue

    var gap: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: TimeSeriesSplit.get_metadata_routing is an API scaffold")

    def split(mut self, X: Some[Movable], y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: split(self, X, y=None, groups=None)."
        raise Error("NotImplementedError: TimeSeriesSplit.split is an API scaffold")

    def get_n_splits(mut self, X: Some[Movable] = None, y: Some[Movable] = None, groups: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_n_splits(self, X=None, y=None, groups=None)."
        raise Error("NotImplementedError: TimeSeriesSplit.get_n_splits is an API scaffold")

    def __init__(out self, n_splits: Some[Movable] = 5, *, max_train_size: Some[Movable] = None, test_size: Some[Movable] = None, gap: Some[Movable] = 0) raises:
        "Scikit-learn signature: __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0)."
        raise Error("NotImplementedError: TimeSeriesSplit.__init__ is an API scaffold")

struct TunedThresholdClassifierCV(Movable):

    "API scaffold for sklearn.model_selection._classification_threshold.TunedThresholdClassifierCV. Operations are deferred."

    var estimator: APIValue

    var scoring: APIValue

    var response_method: APIValue

    var thresholds: APIValue

    var cv: APIValue

    var refit: APIValue

    var n_jobs: APIValue

    var random_state: APIValue

    var store_cv_results: APIValue

    var estimator_: APIValue

    var best_threshold_: APIValue

    var best_score_: APIValue

    var cv_results_: APIValue

    var classes_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.set_params is an API scaffold")

    def score(mut self, X: Some[Movable], y: Some[Movable], sample_weight: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: score(self, X, y, sample_weight=None)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.score is an API scaffold")

    def __init__(out self, estimator: Some[Movable], *, scoring: Some[Movable] = "balanced_accuracy", response_method: Some[Movable] = "auto", thresholds: Some[Movable] = 100, cv: Some[Movable] = None, refit: Some[Movable] = True, n_jobs: Some[Movable] = None, random_state: Some[Movable] = None, store_cv_results: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimator, *, scoring='balanced_accuracy', response_method='auto', thresholds=100, cv=None, refit=True, n_jobs=None, random_state=None, store_cv_results=False)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable], params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y, **params)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.fit is an API scaffold")

    def predict_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_proba(self, X)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.predict_proba is an API scaffold")

    def predict_log_proba(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict_log_proba(self, X)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.predict_log_proba is an API scaffold")

    def decision_function(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decision_function(self, X)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.decision_function is an API scaffold")

    def predict(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: predict(self, X)."
        raise Error("NotImplementedError: TunedThresholdClassifierCV.predict is an API scaffold")

struct ValidationCurveDisplay(Movable):

    "API scaffold for sklearn.model_selection._plot.ValidationCurveDisplay. Operations are deferred."

    var param_name: APIValue

    var param_range: APIValue

    var train_scores: APIValue

    var test_scores: APIValue

    var score_name: APIValue

    var ax_: APIValue

    var figure_: APIValue

    var errorbar_: APIValue

    var lines_: APIValue

    var fill_between_: APIValue

    def __init__(out self, *, param_name: Some[Movable], param_range: Some[Movable], train_scores: Some[Movable], test_scores: Some[Movable], score_name: Some[Movable] = None) raises:
        "Scikit-learn signature: __init__(self, *, param_name, param_range, train_scores, test_scores, score_name=None)."
        raise Error("NotImplementedError: ValidationCurveDisplay.__init__ is an API scaffold")

    def plot(mut self, ax: Some[Movable] = None, *, negate_score: Some[Movable] = False, score_name: Some[Movable] = None, score_type: Some[Movable] = "both", std_display_style: Some[Movable] = "fill_between", line_kw: Some[Movable] = None, fill_between_kw: Some[Movable] = None, errorbar_kw: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: plot(self, ax=None, *, negate_score=False, score_name=None, score_type='both', std_display_style='fill_between', line_kw=None, fill_between_kw=None, errorbar_kw=None)."
        raise Error("NotImplementedError: ValidationCurveDisplay.plot is an API scaffold")

    @staticmethod
    def from_estimator(estimator: Some[Movable], X: Some[Movable], y: Some[Movable], *, param_name: Some[Movable], param_range: Some[Movable], groups: Some[Movable] = None, cv: Some[Movable] = None, scoring: Some[Movable] = None, n_jobs: Some[Movable] = None, pre_dispatch: Some[Movable] = "all", verbose: Some[Movable] = 0, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), fit_params: Some[Movable] = None, ax: Some[Movable] = None, negate_score: Some[Movable] = False, score_name: Some[Movable] = None, score_type: Some[Movable] = "both", std_display_style: Some[Movable] = "fill_between", line_kw: Some[Movable] = None, fill_between_kw: Some[Movable] = None, errorbar_kw: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: from_estimator(cls, estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0, error_score=np.nan, fit_params=None, ax=None, negate_score=False, score_name=None, score_type='both', std_display_style='fill_between', line_kw=None, fill_between_kw=None, errorbar_kw=None)."
        raise Error("NotImplementedError: ValidationCurveDisplay.from_estimator is an API scaffold")

def check_cv(cv: Some[Movable] = 5, y: Some[Movable] = None, *, classifier: Some[Movable] = False, shuffle: Some[Movable] = False, random_state: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: check_cv(cv=5, y=None, *, classifier=False, shuffle=False, random_state=None)."
    raise Error("NotImplementedError: check_cv is an API scaffold")

def cross_val_predict(estimator: Some[Movable], X: Some[Movable], y: Some[Movable] = None, *, groups: Some[Movable] = None, cv: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = 0, params: Some[Movable] = None, pre_dispatch: Some[Movable] = "2*n_jobs", method: Some[Movable] = "predict") raises -> APIValue:
    "Scikit-learn signature: cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', method='predict')."
    raise Error("NotImplementedError: cross_val_predict is an API scaffold")

def cross_val_score(estimator: Some[Movable], X: Some[Movable], y: Some[Movable] = None, *, groups: Some[Movable] = None, scoring: Some[Movable] = None, cv: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = 0, params: Some[Movable] = None, pre_dispatch: Some[Movable] = "2*n_jobs", error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000))) raises -> APIValue:
    "Scikit-learn signature: cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', error_score=np.nan)."
    raise Error("NotImplementedError: cross_val_score is an API scaffold")

def cross_validate(estimator: Some[Movable], X: Some[Movable], y: Some[Movable] = None, *, groups: Some[Movable] = None, scoring: Some[Movable] = None, cv: Some[Movable] = None, n_jobs: Some[Movable] = None, verbose: Some[Movable] = 0, params: Some[Movable] = None, pre_dispatch: Some[Movable] = "2*n_jobs", return_train_score: Some[Movable] = False, return_estimator: Some[Movable] = False, return_indices: Some[Movable] = False, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000))) raises -> APIValue:
    "Scikit-learn signature: cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', return_train_score=False, return_estimator=False, return_indices=False, error_score=np.nan)."
    raise Error("NotImplementedError: cross_validate is an API scaffold")

def learning_curve(estimator: Some[Movable], X: Some[Movable], y: Some[Movable], *, groups: Some[Movable] = None, train_sizes: Some[Movable] = List[Float64]([0.1, 0.325, 0.55, 0.775, 1.0]), cv: Some[Movable] = None, scoring: Some[Movable] = None, exploit_incremental_learning: Some[Movable] = False, n_jobs: Some[Movable] = None, pre_dispatch: Some[Movable] = "all", verbose: Some[Movable] = 0, shuffle: Some[Movable] = False, random_state: Some[Movable] = None, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), return_times: Some[Movable] = False, params: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: learning_curve(estimator, X, y, *, groups=None, train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', verbose=0, shuffle=False, random_state=None, error_score=np.nan, return_times=False, params=None)."
    raise Error("NotImplementedError: learning_curve is an API scaffold")

def permutation_test_score(estimator: Some[Movable], X: Some[Movable], y: Some[Movable], *, groups: Some[Movable] = None, cv: Some[Movable] = None, n_permutations: Some[Movable] = 100, n_jobs: Some[Movable] = None, random_state: Some[Movable] = 0, verbose: Some[Movable] = 0, scoring: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: permutation_test_score(estimator, X, y, *, groups=None, cv=None, n_permutations=100, n_jobs=None, random_state=0, verbose=0, scoring=None, params=None)."
    raise Error("NotImplementedError: permutation_test_score is an API scaffold")

def train_test_split(*arrays: Some[Movable], test_size: Some[Movable] = None, train_size: Some[Movable] = None, random_state: Some[Movable] = None, shuffle: Some[Movable] = True, stratify: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)."
    raise Error("NotImplementedError: train_test_split is an API scaffold")

def validation_curve(estimator: Some[Movable], X: Some[Movable], y: Some[Movable], *, param_name: Some[Movable], param_range: Some[Movable], groups: Some[Movable] = None, cv: Some[Movable] = None, scoring: Some[Movable] = None, n_jobs: Some[Movable] = None, pre_dispatch: Some[Movable] = "all", verbose: Some[Movable] = 0, error_score: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), params: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: validation_curve(estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0, error_score=np.nan, params=None)."
    raise Error("NotImplementedError: validation_curve is an API scaffold")
