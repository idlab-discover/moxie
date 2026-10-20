# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.impute._api_types import APIValue
from std.memory import bitcast

struct IterativeImputer(Movable):

    "API scaffold for sklearn.impute._iterative.IterativeImputer. Operations are deferred."

    var estimator: APIValue

    var missing_values: APIValue

    var sample_posterior: APIValue

    var max_iter: APIValue

    var tol: APIValue

    var n_nearest_features: APIValue

    var initial_strategy: APIValue

    var fill_value: APIValue

    var imputation_order: APIValue

    var skip_complete: APIValue

    var min_value: APIValue

    var max_value: APIValue

    var verbose: APIValue

    var random_state: APIValue

    var add_indicator: APIValue

    var keep_empty_features: APIValue

    var initial_imputer_: APIValue

    var imputation_sequence_: APIValue

    var n_iter_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    var n_features_with_missing_: APIValue

    var indicator_: APIValue

    var random_state_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: IterativeImputer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: IterativeImputer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: IterativeImputer.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: IterativeImputer.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **params)."
        raise Error("NotImplementedError: IterativeImputer.fit_transform is an API scaffold")

    def __init__(out self, estimator: Some[Movable] = None, *, missing_values: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff8000000000000)), sample_posterior: Some[Movable] = False, max_iter: Some[Movable] = 10, tol: Some[Movable] = 0.001, n_nearest_features: Some[Movable] = None, initial_strategy: Some[Movable] = "mean", fill_value: Some[Movable] = None, imputation_order: Some[Movable] = "ascending", skip_complete: Some[Movable] = False, min_value: Some[Movable] = bitcast[DType.float64](UInt64(0xfff0000000000000)), max_value: Some[Movable] = bitcast[DType.float64](UInt64(0x7ff0000000000000)), verbose: Some[Movable] = 0, random_state: Some[Movable] = None, add_indicator: Some[Movable] = False, keep_empty_features: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, estimator=None, *, missing_values=np.nan, sample_posterior=False, max_iter=10, tol=0.001, n_nearest_features=None, initial_strategy='mean', fill_value=None, imputation_order='ascending', skip_complete=False, min_value=-np.inf, max_value=np.inf, verbose=0, random_state=None, add_indicator=False, keep_empty_features=False)."
        raise Error("NotImplementedError: IterativeImputer.__init__ is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: IterativeImputer.transform is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: IterativeImputer.fit is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: IterativeImputer.get_feature_names_out is an API scaffold")
