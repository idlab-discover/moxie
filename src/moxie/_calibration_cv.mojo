# SPDX-License-Identifier: BSD-3-Clause
"""Typed native classifier calibration with deterministic stratified folds."""
import numojo as nm
from std.utils import Variant
from ._calibration_splits import (
    CalibrationCV,
    CalibrationSplit,
    validate_splits,
    stratified_splits,
)
from ._calibration_display import CalibrationDisplayEstimator
from .base import BaseEstimator, ClassifierMixin, ProbabilityPredictor
from ._calibration_algorithms import (
    Array,
    vector,
    matrix,
    weights,
    order,
    _SigmoidCalibration,
    _IsotonicCalibration,
    _TemperatureScaling,
)


trait CalibratedClassifier(ClassifierMixin, ProbabilityPredictor):
    pass


trait CalibrationEstimator(BaseEstimator, Copyable, Deinitable):
    """Adapter for a classifier's preferred response and class ordering.

    response returns a matrix: one decision column for binary classification,
    otherwise one column per class. Probability responses always have all classes.
    fit accepts weights; adapters without weighted fitting must reject supplied
    weights. class_labels must be sorted numeric labels after fitting.
    """

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        ...

    def response(self, X: Array) raises -> Array:
        ...

    def class_labels(self) raises -> Array:
        ...

    def uses_probabilities(self) -> Bool:
        ...

    def is_frozen(self) -> Bool:
        return False


@fieldwise_init
struct CalibratedClassifierParams[E: CalibrationEstimator](Copyable, Movable):
    var estimator: Self.E
    var method: String
    var cv: CalibrationCV
    var n_jobs: Optional[Int]
    var ensemble: Variant[Bool, String]


def rows(x: Array, indices: List[Int]) raises -> Array:
    if x.ndim == 1:
        var out = nm.zeros[DType.float64](nm.Shape(len(indices)))
        for i in range(len(indices)):
            out[nm.Item(i)] = x.item(indices[i])
        return out^
    var out = nm.zeros[DType.float64](nm.Shape(len(indices), x.shape[1]))
    for i in range(len(indices)):
        for j in range(x.shape[1]):
            out[nm.Item(i, j)] = x.item(indices[i], j)
    return out^


def column(x: Array, j: Int) raises -> Array:
    var out = nm.zeros[DType.float64](nm.Shape(x.shape[0]))
    for i in range(x.shape[0]):
        out[nm.Item(i)] = x.item(i, j)
    return out^


def unique(y: Array) raises -> Array:
    var indices = order(y)
    var count = 1
    for i in range(1, y.size):
        if y.item(indices[i]) != y.item(indices[i - 1]):
            count += 1
    var result = nm.zeros[DType.float64](nm.Shape(count))
    var j = 0
    result[nm.Item(0)] = y.item(indices[0])
    for i in range(1, y.size):
        if y.item(indices[i]) != y.item(indices[i - 1]):
            j += 1
            result[nm.Item(j)] = y.item(indices[i])
    return result^


def encoded(y: Array, classes: Array) raises -> Array:
    var out = nm.zeros[DType.float64](nm.Shape(y.size))
    for i in range(y.size):
        var found = False
        for j in range(classes.size):
            if y.item(i) == classes.item(j):
                out[nm.Item(i)] = Float64(j)
                found = True
                break
        if not found:
            raise Error(
                "ValueError: target class absent from estimator classes"
            )
    return out^


def response[
    E: CalibrationEstimator
](
    estimator: E,
    X: Array,
    classes: Array,
    method: String,
    *,
    pooled: Bool = False,
) raises -> Array:
    var labels = estimator.class_labels()
    vector(labels)
    var indices = encoded(labels, classes)
    for j in range(1, labels.size):
        if labels.item(j) <= labels.item(j - 1):
            raise Error(
                "ValueError: estimator classes must be sorted and unique"
            )
    var values = estimator.response(X)
    matrix(values)
    var columns = labels.size
    if labels.size == 2 and not estimator.uses_probabilities():
        columns = 1
    if values.shape[0] != X.shape[0] or values.shape[1] != columns:
        raise Error("ValueError: estimator response has incorrect shape")
    if estimator.uses_probabilities():
        for i in range(values.shape[0]):
            var total = Float64(0)
            for j in range(columns):
                var p = values.item(i, j)
                if p < 0 or p > 1:
                    raise Error(
                        "ValueError: estimator probabilities must be in [0, 1]"
                    )
                total += p
            if abs(total - 1) > 1e-8:
                raise Error(
                    "ValueError: estimator probabilities must sum to one"
                )
    if labels.size != classes.size:
        if method == "temperature" and not pooled:
            raise Error(
                "ValueError: temperature ensemble folds must contain all"
                " classes"
            )
        if pooled:
            if not estimator.uses_probabilities() and labels.size <= 2:
                raise Error(
                    "ValueError: pooled decision responses require at least"
                    " three classes in an incomplete fold"
                )
            var aligned = nm.zeros[DType.float64](
                nm.Shape(X.shape[0], classes.size)
            )
            for i in range(X.shape[0]):
                if not estimator.uses_probabilities():
                    for j in range(classes.size):
                        aligned[nm.Item(i, j)] = -1.7976931348623157e308
                for j in range(labels.size):
                    aligned[nm.Item(i, Int(indices.item(j)))] = values.item(
                        i, j
                    )
            values = aligned^
    if estimator.uses_probabilities():
        var response_classes = classes.size if pooled else labels.size
        if response_classes < 2:
            raise Error(
                "ValueError: a binary classifier needs two probability columns"
            )
        if response_classes == 2 and method != "temperature":
            var selected = nm.zeros[DType.float64](nm.Shape(X.shape[0], 1))
            for i in range(X.shape[0]):
                selected[nm.Item(i, 0)] = values.item(i, 1)
            return selected^
    return values^


struct _CalibratedClassifier[E: CalibrationEstimator](Copyable, Movable):
    var estimator: Self.E
    var classes: Array
    var method: String
    var sigmoid: List[_SigmoidCalibration]
    var isotonic: List[_IsotonicCalibration]
    var temperature: _TemperatureScaling
    var class_indices: List[Int]

    def __init__(
        out self, var estimator: Self.E, var classes: Array, var method: String
    ):
        self.estimator = estimator^
        self.classes = classes^
        self.method = method^
        self.sigmoid = List[_SigmoidCalibration]()
        self.isotonic = List[_IsotonicCalibration]()
        self.temperature = _TemperatureScaling()
        self.class_indices = List[Int]()

    def fit_calibrators(
        mut self, predictions: Array, y: Array, w: Array
    ) raises:
        var labels = encoded(y, self.classes)
        if self.method == "temperature":
            self.temperature.fit(predictions, labels, w.copy())
            return
        var estimator_classes = self.estimator.class_labels()
        var positions = encoded(estimator_classes, self.classes)
        for j in range(predictions.shape[1]):
            var target = nm.zeros[DType.float64](nm.Shape(y.size))
            var positive = 1 if self.classes.size == 2 else Int(
                positions.item(j)
            )
            self.class_indices.append(positive)
            for i in range(y.size):
                target[nm.Item(i)] = (
                    1.0 if labels.item(i) == Float64(positive) else 0.0
                )
            var values = column(predictions, j)
            if self.method == "sigmoid":
                var calibrator = _SigmoidCalibration()
                calibrator.fit(values, target, w.copy())
                self.sigmoid.append(calibrator^)
            else:
                var calibrator = _IsotonicCalibration()
                calibrator.fit(values, target, w.copy())
                self.isotonic.append(calibrator^)

    def predict_proba(self, X: Array) raises -> Array:
        var values = response(self.estimator, X, self.classes, self.method)
        if self.method == "temperature":
            return self.temperature.predict(values)
        var n = X.shape[0]
        var c = self.classes.size
        var out = nm.zeros[DType.float64](nm.Shape(n, c))
        for j in range(len(self.class_indices)):
            var input = column(values, j)
            var predicted = self.sigmoid[j].predict(
                input
            ) if self.method == "sigmoid" else self.isotonic[j].predict(input)
            for i in range(n):
                out[nm.Item(i, self.class_indices[j])] = predicted.item(i)
        for i in range(n):
            if c == 2:
                out[nm.Item(i, 0)] = 1 - out.item(i, 1)
            else:
                var total = Float64(0)
                for j in range(c):
                    total += out.item(i, j)
                for j in range(c):
                    out[nm.Item(i, j)] = out.item(
                        i, j
                    ) / total if total > 0 else 1 / Float64(c)
        return out^


struct CalibratedClassifierCV[E: CalibrationEstimator](
    CalibratedClassifier, CalibrationDisplayEstimator, Copyable, Movable
):
    comptime Input = Array
    comptime Target = Array
    comptime Prediction = Array
    comptime Probabilities = Array
    comptime SampleWeight = Array
    comptime Params = CalibratedClassifierParams[Self.E]
    var estimator: Self.E
    var method: String
    var cv: CalibrationCV
    var n_jobs: Optional[Int]
    var ensemble: Variant[Bool, String]
    var classes_: Array
    var n_features_in_: Int
    var calibrated_classifiers_: List[_CalibratedClassifier[Self.E]]

    def __init__(
        out self,
        var estimator: Self.E,
        *,
        var method: String = "sigmoid",
        var cv: CalibrationCV = None,
        n_jobs: Optional[Int] = None,
        var ensemble: Variant[Bool, String] = "auto",
    ) raises:
        self.estimator = estimator^
        self.method = method^
        self.cv = cv^
        self.n_jobs = n_jobs
        self.ensemble = ensemble^
        self.classes_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_features_in_ = 0
        self.calibrated_classifiers_ = List[_CalibratedClassifier[Self.E]]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.estimator.copy(),
            self.method.copy(),
            self.cv.copy(),
            self.n_jobs,
            self.ensemble.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.method = params.method.copy()
        self.cv = params.cv.copy()
        self.n_jobs = params.n_jobs
        self.ensemble = params.ensemble.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            self.estimator.__sklearn_clone__(),
            method=self.method.copy(),
            cv=self.cv.copy(),
            n_jobs=self.n_jobs,
            ensemble=self.ensemble.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return len(self.calibrated_classifiers_) > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        return self.fit(X, y, None)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[Self.SampleWeight],
    ) raises -> ref[self] Self:
        matrix(X)
        vector(y)
        if X.shape[0] != y.size:
            raise Error("ValueError: inconsistent sample count")
        for i in range(y.size):
            if abs(y.item(i)) > 9007199254740991.0 or y.item(i) != Float64(
                Int(y.item(i))
            ):
                raise Error(
                    "ValueError: classification requires integral numeric"
                    " labels"
                )
        if (
            self.method != "sigmoid"
            and self.method != "isotonic"
            and self.method != "temperature"
        ):
            raise Error(
                "ValueError: method must be sigmoid, isotonic or temperature"
            )
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error(
                "NotImplementedError: only serial n_jobs=None or 1 is supported"
            )
        var use_ensemble = not self.estimator.is_frozen()
        if self.ensemble.isa[Bool]():
            use_ensemble = self.ensemble[Bool]
        elif self.ensemble[String] != "auto":
            raise Error("ValueError: ensemble must be auto or a Boolean")
        var w = weights(y.size, sample_weight)
        var classes = unique(y)
        if classes.size < 2:
            raise Error("ValueError: at least two classes are required")
        var labels = encoded(y, classes)
        var splits: List[CalibrationSplit]
        if self.cv.isa[List[CalibrationSplit]]():
            splits = self.cv[List[CalibrationSplit]].copy()
        else:
            var explicit_count = self.cv.isa[Int]()
            var folds = self.cv[Int] if explicit_count else 5
            splits = stratified_splits(labels, folds, explicit_count)
        validate_splits(splits, y.size, not use_ensemble)
        var fitted = List[_CalibratedClassifier[Self.E]]()
        var columns = classes.size
        if classes.size == 2 and (
            self.method != "temperature"
            or not self.estimator.uses_probabilities()
        ):
            columns = 1
        var oof = nm.zeros[DType.float64](nm.Shape(y.size, columns))
        for split in splits:
            var train = split.train.copy()
            var test = split.test.copy()
            var estimator = self.estimator.__sklearn_clone__()
            var train_x = rows(X, train)
            var train_y = rows(y, train)
            var train_w: Optional[Array] = None
            if sample_weight:
                train_w = rows(w, train)
            estimator.fit(train_x, train_y, train_w)
            var test_x = rows(X, test)
            var predictions = response(
                estimator, test_x, classes, self.method, pooled=not use_ensemble
            )
            if use_ensemble:
                var pair = _CalibratedClassifier[Self.E](
                    estimator^, classes.copy(), self.method.copy()
                )
                var test_y = rows(y, test)
                var test_w = rows(w, test)
                pair.fit_calibrators(predictions, test_y, test_w)
                fitted.append(pair^)
            else:
                for i in range(len(test)):
                    for j in range(columns):
                        oof[nm.Item(test[i], j)] = predictions.item(i, j)
        if not use_ensemble:
            var estimator = self.estimator.__sklearn_clone__()
            estimator.fit(X, y, sample_weight)
            var pair = _CalibratedClassifier[Self.E](
                estimator^, classes.copy(), self.method.copy()
            )
            pair.fit_calibrators(oof, y, w)
            fitted.append(pair^)
        self.classes_ = classes^
        self.n_features_in_ = X.shape[1]
        self.calibrated_classifiers_ = fitted^
        return self

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: CalibratedClassifierCV is unfitted")
        matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: inconsistent feature count")
        var out = nm.zeros[DType.float64](
            nm.Shape(X.shape[0], self.classes_.size)
        )
        for pair in self.calibrated_classifiers_:
            var values = pair.predict_proba(X)
            for i in range(X.shape[0]):
                for j in range(self.classes_.size):
                    out[nm.Item(i, j)] = out.item(i, j) + (
                        values.item(i, j)
                        / Float64(len(self.calibrated_classifiers_))
                    )
        return out^

    def class_labels(self) raises -> Array:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: CalibratedClassifierCV is unfitted")
        return self.classes_.copy()

    def calibration_probabilities(self, X: Array) raises -> Array:
        return self.predict_proba(X)

    def display_name(self) -> String:
        return "CalibratedClassifierCV"

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var probabilities = self.predict_proba(X)
        var out = nm.zeros[DType.float64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var best = 0
            for j in range(1, self.classes_.size):
                if probabilities.item(i, j) > probabilities.item(i, best):
                    best = j
            out[nm.Item(i)] = self.classes_.item(best)
        return out^

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var predictions = self.predict(X)
        vector(y)
        if y.size != predictions.size:
            raise Error("ValueError: inconsistent sample count")
        var w = weights(y.size, sample_weight)
        var correct = Float64(0)
        var total = Float64(0)
        for i in range(y.size):
            total += w.item(i)
            if y.item(i) == predictions.item(i):
                correct += w.item(i)
        return correct / total
