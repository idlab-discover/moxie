# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense public multilayer perceptron estimators."""

from numojo import NDArray, Shape, zeros
from moxie._feature_metadata import InputFeatureCount
from moxie._labels import encode_column
from moxie._random import RandomState
from moxie.base import (
    ClassifierMixin, LogProbabilityPredictor, RegressorMixin, WeightedSupervisedEstimator,
)
from moxie.warnings import FitWarning
from ._mlp_engine import _log, _weight_sum
from ._mlp_model import (
    MLPParams, _MLPState, _regression_output, _regression_score, _sample_weight, _target_matrix,
)


def _validate_labels[dtype: DType](y: NDArray[dtype], rows: Int) raises:
    if y.ndim != 1 and (y.ndim != 2 or y.shape[1] != 1):
        raise Error("ValueError: expected a vector or one column of integer labels")
    if y.shape[0] != rows or rows == 0:
        raise Error("ValueError: expected one label per sample")


struct MLPClassifier[dtype: DType = DType.float64, label_dtype: DType = DType.int64](
    ClassifierMixin, LogProbabilityPredictor, WeightedSupervisedEstimator,
    InputFeatureCount, Copyable, Movable,
):
    """Multilayer perceptron for dense single-label classification."""

    comptime Params = MLPParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label_dtype]
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.Input
    comptime Probabilities = Self.Input

    var _params: Self.Params
    var _state: Optional[_MLPState[Self.dtype]]
    var fit_warnings: List[FitWarning]
    var _classes: Optional[Self.Target]

    def __init__(
        out self,
        hidden_layer_sizes: List[Int] = [100],
        activation: String = "relu",
        *,
        solver: String = "adam",
        alpha: Float64 = 0.0001,
        batch_size: Optional[Int] = None,
        learning_rate: String = "constant",
        learning_rate_init: Float64 = 0.001,
        power_t: Float64 = 0.5,
        max_iter: Int = 200,
        shuffle: Bool = True,
        random_state: Optional[Int] = None,
        tol: Float64 = 0.0001,
        verbose: Bool = False,
        warm_start: Bool = False,
        momentum: Float64 = 0.9,
        nesterovs_momentum: Bool = True,
        early_stopping: Bool = False,
        validation_fraction: Float64 = 0.1,
        beta_1: Float64 = 0.9,
        beta_2: Float64 = 0.999,
        epsilon: Float64 = 1e-8,
        n_iter_no_change: Optional[Int] = 10,
        max_fun: Int = 15000,
    ) raises:
        self = Self(MLPParams(
            hidden_layer_sizes=hidden_layer_sizes.copy(),
            activation=activation.copy(),
            solver=solver.copy(),
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate.copy(),
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
            loss="log_loss",
        ))

    def __init__(out self, params: Self.Params) raises:
        comptime assert Self.dtype in (DType.float32, DType.float64)
        comptime assert Self.label_dtype.is_integral(), "MLPClassifier requires integer labels"
        self._params = params.copy()
        self._state = None
        self.fit_warnings = []
        self._classes = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self._params)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._state)

    def _fitted(ref self) raises -> ref[self._state._value] _MLPState[Self.dtype]:
        if not self._state:
            raise Error("NotFittedError: MLP estimator has not been fitted")
        return self._state.value()

    def n_features_in_(self) raises -> Int:
        return self._fitted().n_features

    def get_n_features_in(self) raises -> Int:
        return self.n_features_in_()

    def parameter_dtype(self) raises -> DType:
        return self._fitted().parameter_dtype()

    def coefs_[parameter_dtype: DType = Self.dtype](self) raises -> List[NDArray[parameter_dtype]]:
        return self._fitted().coefs[parameter_dtype]()

    def intercepts_[parameter_dtype: DType = Self.dtype](self) raises -> List[NDArray[parameter_dtype]]:
        return self._fitted().intercepts[parameter_dtype]()

    def loss_(self) raises -> Float64:
        return self._fitted().loss

    def n_iter_(self) raises -> Int:
        return self._fitted().n_iter

    def t_(self) raises -> Int:
        return self._fitted().t

    def n_outputs_(self) raises -> Int:
        return self._fitted().n_outputs

    def n_layers_(self) raises -> Int:
        return self._fitted().n_layers

    def out_activation_(self) raises -> String:
        return self._fitted().out_activation.copy()

    def loss_curve_(self) raises -> Optional[List[Float64]]:
        return self._fitted().loss_curve.copy()

    def best_loss_(self) raises -> Optional[Float64]:
        return self._fitted().best_loss

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        var rng = RandomState(self._params._seed())
        return self.fit(X, y, rng)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        var rng = RandomState(self._params._seed())
        return self.fit(X, y, rng, sample_weight=sample_weight)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        mut rng: RandomState,
    ) raises -> ref[self] Self:
        var weights: Optional[ImmPointer[Self.SampleWeight, ImmStaticOrigin]] = None
        self._fit(X, y, rng, weights)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        mut rng: RandomState, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        var weights: Optional[ImmPointer[Self.SampleWeight, sample_weight_origin]] = Pointer(to=sample_weight)
        self._fit(X, y, rng, weights)
        return self

    def _fit[weight_origin: ImmOrigin](
        mut self, X: Self.Input, y: Self.Target, mut rng: RandomState,
        weights: Optional[ImmPointer[Self.SampleWeight, weight_origin]],
    ) raises:
        if self._params.warm_start and self._state:
            raise Error("NotImplementedError: MLP warm-start refitting is unfinished")
        if X.ndim != 2:
            raise Error("ValueError: expected a dense rank-2 matrix")
        if self._params.loss != "log_loss":
            raise Error("ValueError: MLPClassifier requires log_loss")
        _validate_labels(y, X.shape[0])
        comptime if Self.label_dtype == DType.uint64:
            for row in range(y.shape[0]):
                if y.unsafe_get(row * Int(y.strides[0])) > Scalar[Self.label_dtype](9223372036854775807):
                    raise Error("ValueError: labels must fit in Int64")
        var stride = Int(y.strides[0])
        var labels64 = zeros[DType.int64](Shape(y.shape[0]))
        for row in range(labels64.size):
            labels64.unsafe_set(row, y.unsafe_get(row * stride).cast[DType.int64]())
        var encoding = encode_column(labels64, classification=False)
        var classes = encoding.classes.size
        var width = classes if classes > 2 else 1
        var encoded = zeros[Self.dtype](Shape(X.shape[0], width))
        for row in range(X.shape[0]):
            var code = Int(encoding.index.inverse.unsafe_get(row))
            if classes > 2:
                encoded.unsafe_set(row * width + code, 1)
            else:
                encoded.unsafe_set(row, code.cast[Self.dtype]())
        var state = _MLPState(
            X, encoded, self._params, "softmax" if classes > 2 else "logistic",
            "log_loss" if classes > 2 else "binary_log_loss", rng, weights,
        )
        var labels = encoding.classes.astype[Self.label_dtype]()
        self.fit_warnings = state.warnings.copy()
        self._state = state^
        self._classes = labels^

    def classes_(self) raises -> Self.Target:
        _ = self._fitted()
        return self._classes.value().copy()

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var activation = self._fitted().predict_matrix(X, self._params.activation)
        ref classes = self._classes.value()
        var width = activation.shape[1]
        var result = zeros[Self.label_dtype](Shape(activation.shape[0]))
        for row in range(result.size):
            var code = 0
            if width > 1:
                var start = row * width
                for column in range(1, width):
                    if activation.unsafe_get(start + column) > activation.unsafe_get(start + code):
                        code = column
            elif classes.size > 1 and activation.unsafe_get(row) > 0.5:
                code = 1
            result.unsafe_set(row, classes.unsafe_get(code))
        return result^

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict_proba[Self.dtype](X)

    def predict_proba[output_dtype: DType](
        self, X: Self.Input,
    ) raises -> NDArray[output_dtype]:
        comptime assert output_dtype in (DType.float32, DType.float64)
        var activation = self._fitted().predict_matrix(X, self._params.activation)
        if activation.shape[1] != 1:
            return activation.astype[output_dtype]()
        var rows = activation.shape[0]
        var result = zeros[output_dtype](Shape(rows, 2))
        for row in range(rows):
            var positive = activation.unsafe_get(row)
            result.unsafe_set(2 * row, (1 - positive).cast[output_dtype]())
            result.unsafe_set(2 * row + 1, positive.cast[output_dtype]())
        return result^

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict_log_proba[Self.dtype](X)

    def predict_log_proba[output_dtype: DType](
        self, X: Self.Input,
    ) raises -> NDArray[output_dtype]:
        var probabilities = self.predict_proba[output_dtype](X)
        for i in range(probabilities.size):
            probabilities.unsafe_set(i, _log(probabilities.unsafe_get(i)))
        return probabilities^

    def score(
        self, X: Self.Input, y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var prediction = self.predict(X)
        _validate_labels(y, prediction.size)
        var weights: Optional[ImmPointer[Self.SampleWeight, origin_of(sample_weight._value)]] = None
        if sample_weight:
            weights = Pointer(to=sample_weight.value())
        var mass = _weight_sum(weights, prediction.size)
        var correct: Float64 = 0
        for row in range(prediction.size):
            if prediction.unsafe_get(row) == y.unsafe_get(row * Int(y.strides[0])):
                correct += _sample_weight(sample_weight, row)
        return correct / mass


struct MLPRegressor[dtype: DType = DType.float64](
    RegressorMixin, WeightedSupervisedEstimator, InputFeatureCount, Copyable, Movable,
):
    """Multilayer perceptron for dense single- and multi-output regression."""

    comptime Params = MLPParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.Input

    var _params: Self.Params
    var _state: Optional[_MLPState[Self.dtype]]
    var fit_warnings: List[FitWarning]

    def __init__(
        out self,
        loss: String = "squared_error",
        hidden_layer_sizes: List[Int] = [100],
        activation: String = "relu",
        *,
        solver: String = "adam",
        alpha: Float64 = 0.0001,
        batch_size: Optional[Int] = None,
        learning_rate: String = "constant",
        learning_rate_init: Float64 = 0.001,
        power_t: Float64 = 0.5,
        max_iter: Int = 200,
        shuffle: Bool = True,
        random_state: Optional[Int] = None,
        tol: Float64 = 0.0001,
        verbose: Bool = False,
        warm_start: Bool = False,
        momentum: Float64 = 0.9,
        nesterovs_momentum: Bool = True,
        early_stopping: Bool = False,
        validation_fraction: Float64 = 0.1,
        beta_1: Float64 = 0.9,
        beta_2: Float64 = 0.999,
        epsilon: Float64 = 1e-8,
        n_iter_no_change: Optional[Int] = 10,
        max_fun: Int = 15000,
    ) raises:
        self = Self(MLPParams(
            hidden_layer_sizes=hidden_layer_sizes.copy(),
            activation=activation.copy(),
            solver=solver.copy(),
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate.copy(),
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
            loss=loss.copy(),
        ))

    def __init__(out self, params: Self.Params) raises:
        comptime assert Self.dtype in (DType.float32, DType.float64)
        self._params = params.copy()
        self._state = None
        self.fit_warnings = []

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self._params)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._state)

    def _fitted(ref self) raises -> ref[self._state._value] _MLPState[Self.dtype]:
        if not self._state:
            raise Error("NotFittedError: MLP estimator has not been fitted")
        return self._state.value()

    def n_features_in_(self) raises -> Int:
        return self._fitted().n_features

    def get_n_features_in(self) raises -> Int:
        return self.n_features_in_()

    def parameter_dtype(self) raises -> DType:
        return self._fitted().parameter_dtype()

    def coefs_[parameter_dtype: DType = Self.dtype](self) raises -> List[NDArray[parameter_dtype]]:
        return self._fitted().coefs[parameter_dtype]()

    def intercepts_[parameter_dtype: DType = Self.dtype](self) raises -> List[NDArray[parameter_dtype]]:
        return self._fitted().intercepts[parameter_dtype]()

    def loss_(self) raises -> Float64:
        return self._fitted().loss

    def n_iter_(self) raises -> Int:
        return self._fitted().n_iter

    def t_(self) raises -> Int:
        return self._fitted().t

    def n_outputs_(self) raises -> Int:
        return self._fitted().n_outputs

    def n_layers_(self) raises -> Int:
        return self._fitted().n_layers

    def out_activation_(self) raises -> String:
        return self._fitted().out_activation.copy()

    def loss_curve_(self) raises -> Optional[List[Float64]]:
        return self._fitted().loss_curve.copy()

    def best_loss_(self) raises -> Optional[Float64]:
        return self._fitted().best_loss

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        var rng = RandomState(self._params._seed())
        return self.fit(X, y, rng)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        var rng = RandomState(self._params._seed())
        return self.fit(X, y, rng, sample_weight=sample_weight)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        mut rng: RandomState,
    ) raises -> ref[self] Self:
        var weights: Optional[ImmPointer[Self.SampleWeight, ImmStaticOrigin]] = None
        self._fit(X, y, rng, weights)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        mut rng: RandomState, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        var weights: Optional[ImmPointer[Self.SampleWeight, sample_weight_origin]] = Pointer(to=sample_weight)
        self._fit(X, y, rng, weights)
        return self

    def _fit[weight_origin: ImmOrigin](
        mut self, X: Self.Input, y: Self.Target, mut rng: RandomState,
        weights: Optional[ImmPointer[Self.SampleWeight, weight_origin]],
    ) raises:
        if self._params.warm_start and self._state:
            raise Error("NotImplementedError: MLP warm-start refitting is unfinished")
        if X.ndim != 2:
            raise Error("ValueError: expected a dense rank-2 matrix")
        var target = _target_matrix(y, X.shape[0])
        var activation = "identity"
        if self._params.loss == "poisson":
            activation = "exp"
            for row in range(target.shape[0]):
                for column in range(target.shape[1]):
                    var value = target.unsafe_get(row * Int(target.strides[0]) + column * Int(target.strides[1]))
                    if value < 0:
                        raise Error("ValueError: Poisson targets must be nonnegative")
        elif self._params.loss != "squared_error":
            raise Error("ValueError: loss must be squared_error or poisson")
        var state = _MLPState(X, target, self._params, activation, self._params.loss, rng, weights)
        self.fit_warnings = state.warnings.copy()
        self._state = state^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var prediction = self._fitted().predict_feature_matrix(X, self._params.activation)
        return _regression_output(prediction^)

    def predict[output_dtype: DType](self, X: Self.Input) raises -> NDArray[output_dtype]:
        comptime assert output_dtype in (DType.float32, DType.float64)
        var prediction = self._fitted().predict_matrix(X, self._params.activation)
        return _regression_output(prediction.astype[output_dtype]())

    def score(
        self, X: Self.Input, y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var prediction = self._fitted().predict_matrix(X, self._params.activation)
        var weights: Optional[ImmPointer[Self.SampleWeight, origin_of(sample_weight._value)]] = None
        if sample_weight:
            weights = Pointer(to=sample_weight.value())
        return _regression_score(y, prediction, sample_weight, weights)
