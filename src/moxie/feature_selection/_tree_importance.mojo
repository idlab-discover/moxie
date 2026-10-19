"""Dense single-output adapter for the adopted native decision-tree classifier."""
import numojo as nm
from std.os import abort
from moxie.base import ClassifierMixin, LogProbabilityPredictor, WeightedSupervisedEstimator
from moxie.tree import DecisionTreeClassifier
from moxie.tree._records import TreeData
from moxie._labels import StringTarget
from ._model_importance import SelectionClassLabels


struct DecisionTreeClassifierImportance[dt: DType = DType.float64, label_dtype: DType = DType.int64, string_labels: Bool = False](SelectionClassLabels, ClassifierMixin, LogProbabilityPredictor, WeightedSupervisedEstimator):
    comptime dtype = Self.dt
    comptime Model = DecisionTreeClassifier[Self.label_dtype, Self.string_labels]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Model.Target
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.Input
    comptime Probabilities = nm.NDArray[DType.float64]
    comptime Classes = Self.Model.ClassLabels
    comptime Params = Self.Model.Params
    var model: Self.Model

    def __init__(out self, var model: Self.Model):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.model = model^

    def __init__(out self, *, copy: Self):
        # Copyable cannot raise; default construction has no user parameters.
        try:
            self.model = Self.Model()
        except error:
            abort("DecisionTreeClassifierImportance: copy allocation failed: " + String(error))
        # Copy the adopted model snapshot without refitting or retaining aliases.
        self.model.params = copy.model.params.copy()
        self.model.classes_ = copy.model.classes_.copy()
        self.model.n_classes_ = copy.model.n_classes_.copy()
        self.model.feature_names_in_ = copy.model.feature_names_in_.copy()
        self.model.feature_importances_ = copy.model.feature_importances_.copy()
        self.model._encoder = copy.model._encoder.copy()
        self.model.is_categorical_ = copy.model.is_categorical_.copy()
        self.model.n_outputs_ = copy.model.n_outputs_
        self.model.n_features_in_ = copy.model.n_features_in_
        self.model.max_features_ = copy.model.max_features_
        self.model._fitted = copy.model._fitted
        self.model.tree_ = TreeData(
            copy.model.tree_.n_features, copy.model.tree_.n_classes.copy(),
            copy.model.tree_.n_categories.copy(), copy.model.tree_.n_outputs,
            copy.model.tree_.max_n_classes, copy.model.tree_.max_depth,
            copy.model.tree_.node_count, copy.model.tree_.capacity,
            copy.model.tree_.nodes.copy(), copy.model.tree_.value.copy(),
        )

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.model.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.__sklearn_is_fitted__()

    def _single_output(self) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying decision tree is not fitted")
        if self.model.n_outputs_ != 1:
            raise Error("ValueError: selection adapter requires a single-output classifier")

    def fit_selection(mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: Optional[Self.Input]) raises:
        if not y:
            raise Error("ValueError: DecisionTreeClassifier requires y")
        comptime if Self.string_labels:
            if rebind[StringTarget](y.value()).rank() != 1:
                raise Error("ValueError: selection adapter requires rank-one y")
        else:
            if rebind[nm.NDArray[Self.label_dtype]](y.value()).ndim != 1:
                raise Error("ValueError: selection adapter requires rank-one y")
        var weights: Optional[nm.NDArray[DType.float64]] = None
        if sample_weight:
            weights = sample_weight.value().contiguous().astype[DType.float64]()
        _ = self.model.fit(X.contiguous().astype[DType.float32](), y.value(), weights)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.fit_selection(X, y.copy(), None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        self.fit_selection(X, y.copy(), sample_weight.copy())
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._single_output()
        return self.model.predict(X.contiguous().astype[DType.float32]())

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        self._single_output()
        return self.model.predict_proba(X.contiguous().astype[DType.float32]())[0].copy()

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        self._single_output()
        return self.model.predict_log_proba(X.contiguous().astype[DType.float32]())[0].copy()

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        self._single_output()
        var weights: Optional[nm.NDArray[DType.float64]] = None
        if sample_weight:
            weights = sample_weight.value().contiguous().astype[DType.float64]()
        return self.model.score(X.contiguous().astype[DType.float32](), y, weights)

    def classes_(self) raises -> Self.Classes:
        self._single_output()
        return self.model.classes_[0].copy()

    def n_features(self) raises -> Int:
        self._single_output()
        return self.model.n_features_in_

    def feature_importances(self) raises -> Self.Input:
        self._single_output()
        return self.model.feature_importances_.astype[Self.dtype]()

    def uses_l1_threshold(self) -> Bool:
        return False
