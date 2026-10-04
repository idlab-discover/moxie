# Adapted from scikit-learn Dummy estimators and weighted statistics.
# Original authors: The scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSES/scikit-learn.txt for the retained notice.
"""Dense baseline estimators. See docs/dummy/api.md for native type adaptations."""
from moxie._dummy_contracts import BaselineClassifier, BaselineRegressor
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie._random import RandomState, check_random_state
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import isfinite, nan, floor
from std.sys import llvm_intrinsic

comptime Dense = NDArray[DType.float64]


def _target[dtype: DType](y: NDArray[dtype]) raises -> Int:
    if y.ndim < 1 or y.ndim > 2 or y.shape[0] == 0:
        raise Error("ValueError: y must be nonempty and rank 1 or 2")
    var outputs = y.shape[1] if y.ndim == 2 else 1
    if outputs == 0:
        raise Error("ValueError: y must have at least one output")
    for i in range(y.shape[0]):
        for j in range(outputs):
            if not isfinite(_value(y, i, j)):
                raise Error("ValueError: y must be finite")
    return outputs


def _value[dtype: DType](y: NDArray[dtype], i: Int, j: Int) raises -> Scalar[dtype]:
    if y.ndim == 1:
        return y[Item(i)]
    return y.item(i, j)


def _weights[dtype: DType](n: Int, weights: Optional[NDArray[dtype]]) raises -> NDArray[dtype]:
    var result = nm.ones[dtype](Shape(n))
    if weights:
        if weights.value().ndim != 1 or weights.value().shape[0] != n:
            raise Error("ValueError: sample_weight must have shape (n_samples,)")
        for i in range(n):
            var w = weights.value()[Item(i)]
            if not isfinite(w):
                raise Error("ValueError: sample_weight must be finite")
            result[Item(i)] = w
        var any_nonzero = False
        for i in range(n):
            any_nonzero = any_nonzero or result.item(i) != 0
        if not any_nonzero:
            raise Error("ValueError: sample weights must contain a non-zero value")
    return result^


def _length[dtype: DType](X: NDArray[dtype], n: Int) raises:
    if X.ndim < 1 or X.shape[0] != n:
        raise Error("ValueError: inconsistent numbers of samples")


@fieldwise_init
struct DummyRegressorParams[dtype: DType = DType.float64](Copyable, Movable):
    comptime Array = NDArray[Self.dtype]
    var strategy: String
    var constant: Optional[Self.Array]
    var quantile: Optional[Float64]


@fieldwise_init
struct DummyRegressorPrediction[dtype: DType = DType.float64](Copyable, Movable):
    comptime Array = NDArray[Self.dtype]
    var prediction: Self.Array
    var std: Optional[Dense]


struct DummyRegressor[dtype: DType = DType.float64](BaselineRegressor, WeightedSupervisedEstimator):
    comptime Array = NDArray[Self.dtype]
    comptime Number = Scalar[Self.dtype]
    comptime Params = DummyRegressorParams[Self.dtype]
    comptime Input = Self.Array
    comptime Target = Self.Array
    comptime Prediction = Self.Array
    comptime SampleWeight = Self.Array
    var strategy: String
    var constant: Optional[Self.Array]
    var quantile: Optional[Float64]
    var constant_: Optional[Self.Array]
    var n_features_in_: Optional[Int]
    var n_outputs_: Optional[Int]

    def __init__(out self, *, strategy: String = "mean", constant: Optional[Self.Array] = None, quantile: Optional[Float64] = None):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "dummy estimators support Float32 and Float64"
        self.strategy = strategy
        self.constant = constant.copy()
        self.quantile = quantile
        self.constant_ = None
        self.n_features_in_ = None
        self.n_outputs_ = None

    def __init__(out self, *, constant: Self.Number, strategy: String = "mean", quantile: Optional[Float64] = None) raises:
        var values = nm.zeros[Self.dtype](Shape(1))
        values[Item(0)] = constant
        self = Self(strategy=strategy, constant=Optional[Self.Array](values^), quantile=quantile)

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        constant: Self.Number,
        strategy: Optional[String] = None,
        quantile: Optional[Optional[Float64]] = None,
    ) raises -> ref[self] Self:
        var values = nm.zeros[Self.dtype](Shape(1))
        values[Item(0)] = constant
        _ = self.set_params(strategy=strategy, constant=Optional[Optional[Self.Array]](Optional[Self.Array](values^)), quantile=quantile)
        return self

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.strategy, self.constant.copy(), self.quantile)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.strategy = params.strategy
        self.constant = params.constant.copy()
        self.quantile = params.quantile
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        strategy: Optional[String] = None,
        constant: Optional[Optional[Self.Array]] = None,
        quantile: Optional[Optional[Float64]] = None,
    ) raises -> ref[self] Self:
        if strategy:
            self.strategy = strategy.value()
        if constant:
            self.constant = constant.value().copy()
        if quantile:
            self.quantile = quantile.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(strategy=self.strategy, constant=self.constant.copy(), quantile=self.quantile)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.constant_)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Array,
        ref[y_origin] y: Self.Array,
    ) raises -> ref[self] Self:
        _ = _target(y)
        _length(X, y.shape[0])
        self._fit(y, None)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else Optional[Int]()
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Array,
        ref[y_origin] y: Self.Array,
        *,
        ref[sample_weight_origin] sample_weight: Self.Array,
    ) raises -> ref[self] Self:
        _ = _target(y)
        _length(X, y.shape[0])
        self._fit(y, sample_weight.copy())
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else Optional[Int]()
        return self

    def fit[
        origin: MutOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        X: Optional[Self.Array],
        ref[y_origin] y: Self.Array,
        sample_weight: Optional[Self.Array] = None,
    ) raises -> ref[self] Self:
        _ = _target(y)
        if X:
            _length(X.value(), y.shape[0])
        self._fit(y, sample_weight)
        self.n_features_in_ = None
        if X and X.value().ndim > 1:
            self.n_features_in_ = X.value().shape[1]
        return self

    def _fit(mut self, y: Self.Array, weights: Optional[Self.Array]) raises:
        var outputs = _target(y)
        var n = y.shape[0]
        var w = _weights(n, weights)
        if self.strategy != "mean" and self.strategy != "median" and self.strategy != "quantile" and self.strategy != "constant":
            raise Error("ValueError: invalid strategy")
        if self.quantile:
            if not isfinite(self.quantile.value()) or self.quantile.value() < 0 or self.quantile.value() > 1:
                raise Error("ValueError: quantile must be in [0, 1]")
        if self.strategy == "quantile" and not self.quantile:
            raise Error("ValueError: quantile is required")
        var result = nm.zeros[Self.dtype](Shape(1, outputs))
        var total = Self.Number(0)
        for i in range(n):
            total += w.item(i)
        if self.strategy == "constant":
            if not self.constant:
                raise Error("TypeError: constant is required")
            if self.constant.value().size != outputs:
                raise Error("ValueError: constant must have one value per output")
            if self.constant.value().ndim != 1:
                raise Error("ValueError: constant must be a vector")
            for j in range(outputs):
                var v = self.constant.value()[Item(j)]
                if not isfinite(v):
                    raise Error("ValueError: constant must be finite")
                result[Item(0, j)] = v
        else:
            for j in range(outputs):
                var value = Self.Number(0)
                if self.strategy == "mean":
                    if total == 0:
                        raise Error("ValueError: weights sum to zero")
                    for i in range(n):
                        value += w.item(i) * _value(y, i, j)
                    value /= total
                else:
                    # Stable index insertion sort; values remain in NuMojo storage.
                    var order = List[Int]()
                    for i in range(n):
                        order.append(i)
                        var k = i
                        while k > 0 and _value(y, order[k], j) < _value(y, order[k - 1], j):
                            var temp = order[k]
                            order[k] = order[k - 1]
                            order[k - 1] = temp
                            k -= 1
                    var q = self.quantile.value() if self.strategy == "quantile" else Float64(0.5)
                    if weights:
                        # Match searchsorted(side='left') even for signed weights.
                        var cdf = nm.zeros[Self.dtype](Shape(n))
                        var running = Self.Number(0)
                        for i in range(n):
                            running += w.item(order[i])
                            cdf[Item(i)] = running
                        var threshold = Self.Number(q * 100.0) / Self.Number(100) * running
                        # Searching above zero is equivalent to sklearn's
                        # nextafter(0, +inf), without a subnormal constant.
                        var lo = 0
                        var hi = n
                        while lo < hi:
                            var mid = (lo + hi) // 2
                            if cdf.item(mid) < threshold or (threshold == 0 and cdf.item(mid) == 0):
                                lo = mid + 1
                            else:
                                hi = mid
                        value = _value(y, order[min(lo, n - 1)], j)
                    else:
                        var pos = (q * 100.0) / 100.0 * Float64(n - 1)
                        var lower = Int(pos)
                        var upper = min(lower + 1, n - 1)
                        var frac = pos - Float64(lower)
                        var a = _value(y, order[lower], j)
                        var b = _value(y, order[upper], j)
                        value = a + (b - a) * Self.Number(frac)
                        if frac >= 0.5:
                            value = b - (b - a) * Self.Number(1.0 - frac)
                result[Item(0, j)] = value
        self.constant_ = result^
        self.n_outputs_ = outputs

    def predict(self, X: Self.Array) raises -> Self.Array:
        check_is_fitted(self)
        if X.ndim < 1:
            raise Error("ValueError: X must have a sample dimension")
        var n = X.shape[0]
        var outputs = self.n_outputs_.value()
        var shape = Shape(n, outputs) if outputs > 1 else Shape(n)
        var result = nm.zeros[Self.dtype](shape)
        for i in range(n):
            for j in range(outputs):
                result.unsafe_set(i * outputs + j, self.constant_.value().item(0, j))
        return result^

    def predict(self, X: Self.Array, *, return_std: Bool) raises -> DummyRegressorPrediction[Self.dtype]:
        var pred = self.predict(X)
        var deviation = Optional[Dense]()
        if return_std:
            deviation = nm.zeros[DType.float64](pred.shape)
        return DummyRegressorPrediction[Self.dtype](pred^, deviation^)

    def score(self, X: Self.Array, y: Self.Array, sample_weight: Optional[Self.Array] = None) raises -> Float64:
        check_is_fitted(self)
        _ = _target(y)
        _length(X, y.shape[0])
        return self.score(Optional[Self.Array](), y, sample_weight)

    def score(self, X: Optional[Self.Array], y: Self.Array, sample_weight: Optional[Self.Array] = None) raises -> Float64:
        check_is_fitted(self)
        var outputs = _target(y)
        if outputs != self.n_outputs_.value():
            raise Error("ValueError: output count mismatch")
        for j in range(outputs):
            if not isfinite(self.constant_.value().item(0, j)):
                raise Error("ValueError: predictions must be finite for scoring")
        if X:
            _length(X.value(), y.shape[0])
        var n = y.shape[0]
        var w = _weights(n, sample_weight)
        if n < 2:
            return nan[DType.float64]()
        var total = Self.Number(0)
        for i in range(n):
            total += w.item(i)
        if total == 0:
            raise Error("ValueError: weights sum to zero")
        var score = Float64(0)
        for j in range(outputs):
            var mean = Self.Number(0)
            for i in range(n):
                mean += w.item(i) * _value(y, i, j)
            mean /= total
            var numerator = Self.Number(0)
            var denominator = Self.Number(0)
            for i in range(n):
                var residual = _value(y, i, j) - self.constant_.value().item(0, j)
                var centered = _value(y, i, j) - mean
                numerator += w.item(i) * residual * residual
                denominator += w.item(i) * centered * centered
            if numerator == 0:
                score += 1
            elif denominator != 0:
                score += Float64(Self.Number(1) - numerator / denominator)
        return score / Float64(outputs)


struct Label(Copyable, Movable, Comparable):
    """A homogeneous column uses integer, floating-point, or UTF-8 labels."""
    var kind: Int
    var integer: Int
    var number: Float64
    var text: String

    @implicit
    def __init__(out self, value: Int):
        self.kind = 0
        self.integer = value
        self.number = 0
        self.text = ""

    @implicit
    def __init__(out self, value: Float64):
        self.kind = 1
        self.integer = 0
        self.number = value
        self.text = ""

    @implicit
    def __init__(out self, value: String):
        self.kind = 2
        self.integer = 0
        self.number = 0
        self.text = value

    def __eq__(self, other: Self) -> Bool:
        if self.kind != other.kind:
            return False
        if self.kind == 0:
            return self.integer == other.integer
        if self.kind == 1:
            return self.number == other.number or (self.number != self.number and other.number != other.number)
        return self.text == other.text

    def __lt__(self, other: Self) -> Bool:
        if self.kind != other.kind:
            return self.kind < other.kind
        if self.kind == 0:
            return self.integer < other.integer
        if self.kind == 1:
            return self.number < other.number or (self.number == self.number and other.number != other.number)
        return self.text < other.text


@fieldwise_init
struct DummyClassifierParams(Copyable, Movable):
    var strategy: String
    var random_state: Optional[Int]
    comptime Constants = List[Label]
    var constant: Optional[Self.Constants]


struct DummyClassifier[multioutput: Bool = False, dtype: DType = DType.float64](BaselineClassifier, WeightedSupervisedEstimator):
    """Statically shaped single-output (default) or multioutput classification."""
    comptime Params = DummyClassifierParams
    comptime Input = NDArray[Self.dtype]
    comptime Rows = List[List[Label]]
    comptime Target = Self.Rows if Self.multioutput else List[Label]
    comptime OtherTarget = List[Label] if Self.multioutput else Self.Rows
    comptime Prediction = Self.Target
    comptime SampleWeight = NDArray[Self.dtype]
    # Mojo 1.0 requires rebind inside comptime branches to refine conditional types.
    comptime Probabilities = List[Dense] if Self.multioutput else Dense
    comptime Classes = Self.Rows if Self.multioutput else List[Label]
    comptime ClassCounts = List[Int] if Self.multioutput else Int
    var strategy: String
    var random_state: Optional[Int]
    comptime Constants = List[Label]
    var constant: Optional[Self.Constants]
    var classes_: Self.Classes
    var n_classes_: Self.ClassCounts
    var class_prior_: Self.Probabilities
    var n_outputs_: Optional[Int]
    var n_features_in_: Optional[Int]
    var sparse_output_: Optional[Bool]
    var _strategy: String

    def __init__(out self, *, strategy: String = "prior", random_state: Optional[Int] = None, constant: Optional[List[Label]] = None) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "dummy estimators support Float32 and Float64"
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant.copy()
        self.classes_ = Self.Classes()
        self.n_classes_ = Self.ClassCounts()
        comptime if Self.multioutput:
            self.class_prior_ = rebind[Self.Probabilities](List[Dense]()).copy()
        else:
            self.class_prior_ = rebind[Self.Probabilities](nm.zeros[DType.float64](Shape(0))).copy()
        self.n_outputs_ = None
        self.n_features_in_ = None
        self.sparse_output_ = None
        self._strategy = ""

    def __init__(out self, *, constant: Label, strategy: String = "prior", random_state: Optional[Int] = None) raises:
        var constants: List[Label] = [constant.copy()]
        self = Self(strategy=strategy, random_state=random_state, constant=Optional[List[Label]](constants^))

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        constant: Label,
        strategy: Optional[String] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        var constants: List[Label] = [constant.copy()]
        _ = self.set_params(strategy=strategy, random_state=random_state, constant=Optional[Optional[List[Label]]](Optional[List[Label]](constants^)))
        return self

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.strategy, self.random_state, self.constant.copy())

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.strategy = params.strategy
        self.random_state = params.random_state
        self.constant = params.constant.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        strategy: Optional[String] = None,
        random_state: Optional[Optional[Int]] = None,
        constant: Optional[Optional[List[Label]]] = None,
    ) raises -> ref[self] Self:
        if strategy:
            self.strategy = strategy.value()
        if random_state:
            self.random_state = random_state.value()
        if constant:
            self.constant = constant.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(strategy=self.strategy, random_state=self.random_state, constant=self.constant.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_outputs_)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._fit(X, self._target_rows(y), None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, self._target_rows(y), sample_weight.copy())
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[Self.SampleWeight],
    ) raises -> ref[self] Self:
        self._fit(X, self._target_rows(y), sample_weight)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.OtherTarget,
    ) raises -> ref[self] Self:
        self._fit(X, self._other_rows(y), None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.OtherTarget,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, self._other_rows(y), sample_weight.copy())
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.OtherTarget,
        sample_weight: Optional[Self.SampleWeight],
    ) raises -> ref[self] Self:
        self._fit(X, self._other_rows(y), sample_weight)
        return self

    def _target_rows(self, y: Self.Target) -> Self.Rows:
        comptime if Self.multioutput:
            return rebind[Self.Rows](y).copy()
        else:
            return _label_rows(rebind[List[Label]](y))

    def _other_rows(self, y: Self.OtherTarget) -> Self.Rows:
        comptime if Self.multioutput:
            return _label_rows(rebind[List[Label]](y))
        else:
            return rebind[Self.Rows](y).copy()

    def _fit(mut self, X: Self.Input, y: Self.Rows, weights: Optional[Self.SampleWeight]) raises:
        _length(X, len(y))
        if len(y) == 0 or len(y[0]) == 0:
            raise Error("ValueError: y must have samples and outputs")
        var outputs = len(y[0])
        comptime if not Self.multioutput:
            if outputs != 1:
                raise Error("ValueError: multiple outputs require DummyClassifier[multioutput=True]")
        for row in y:
            if len(row) != outputs:
                raise Error("ValueError: ragged targets")
        if self.strategy != "prior" and self.strategy != "most_frequent" and self.strategy != "uniform" and self.strategy != "stratified" and self.strategy != "constant":
            raise Error("ValueError: invalid strategy")
        if self.random_state:
            var validated = RandomState(self.random_state.value())
        if self.strategy == "constant":
            if not self.constant or len(self.constant.value()) != outputs:
                raise Error("ValueError: constant must contain one label per output")
        var w = _weights(len(y), weights)
        var classes = List[List[Label]]()
        var counts = List[Int]()
        var priors = List[Dense]()
        for j in range(outputs):
            for i in range(len(y)):
                if y[i][j].kind != y[0][j].kind:
                    raise Error("ValueError: label columns must have a homogeneous type")
            var labels = List[Label]()
            for i in range(len(y)):
                var found = False
                for label in labels:
                    if label == y[i][j]:
                        found = True
                if not found:
                    labels.append(y[i][j].copy())
                    var k = len(labels) - 1
                    while k > 0 and labels[k] < labels[k - 1]:
                        var temp = labels[k].copy()
                        labels[k] = labels[k - 1].copy()
                        labels[k - 1] = temp^
                        k -= 1
            if self.strategy == "constant":
                var constant_label = self.constant.value()[j].copy()
                if constant_label.kind == 1 and constant_label.number != constant_label.number:
                    raise Error("ValueError: NaN constant is not equal to any training label")
                var found = False
                for label in labels:
                    found = found or label == self.constant.value()[j]
                if not found:
                    raise Error("ValueError: constant label is absent from y")
            var prior = nm.zeros[DType.float64](Shape(len(labels)))
            for i in range(len(y)):
                for k in range(len(labels)):
                    if y[i][j] == labels[k]:
                        prior[Item(k)] = prior.item(k) + Float64(w.item(i))
            var total = Float64(0)
            for k in range(len(labels)):
                total += prior.item(k)
            for k in range(len(labels)):
                prior[Item(k)] = prior.item(k) / total
            counts.append(len(labels))
            classes.append(labels^)
            priors.append(prior^)
        comptime if Self.multioutput:
            self.classes_ = rebind[Self.Classes](classes).copy()
            self.n_classes_ = rebind[Self.ClassCounts](counts).copy()
            self.class_prior_ = rebind[Self.Probabilities](priors).copy()
        else:
            self.classes_ = rebind[Self.Classes](classes[0]).copy()
            self.n_classes_ = rebind[Self.ClassCounts](counts[0]).copy()
            self.class_prior_ = rebind[Self.Probabilities](priors[0]).copy()
        self.n_outputs_ = outputs
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else Optional[Int]()
        self.sparse_output_ = False
        self._strategy = self.strategy

    def _class(self, j: Int, k: Int) -> Label:
        comptime if Self.multioutput:
            return rebind[Self.Rows](self.classes_)[j][k].copy()
        else:
            return rebind[List[Label]](self.classes_)[k].copy()

    def _prior_value(self, j: Int, k: Int) raises -> Float64:
        comptime if Self.multioutput:
            return rebind[List[Dense]](self.class_prior_)[j].item(k)
        else:
            return rebind[Dense](self.class_prior_).item(k)

    def _class_count(self, j: Int) -> Int:
        comptime if Self.multioutput:
            return rebind[List[Int]](self.n_classes_)[j]
        else:
            return rebind[Int](self.n_classes_)

    def _index(self, j: Int) raises -> Int:
        var best = 0
        if self._strategy == "constant":
            if not self.constant or len(self.constant.value()) != self.n_outputs_.value():
                raise Error("ValueError: constant shape changed")
            for k in range(self._class_count(j)):
                if self._class(j, k) == self.constant.value()[j]:
                    return k
            return -1
        for k in range(1, self._class_count(j)):
            var current = self._prior_value(j, k)
            var previous = self._prior_value(j, best)
            if (current != current and previous == previous) or current > previous:
                best = k
        return best

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        check_is_fitted(self)
        with check_random_state(self.random_state) as rng:
            return self.predict_proba(X, rng[])

    def predict_proba(self, X: Self.Input, mut rng: RandomState) raises -> Self.Probabilities:
        var outputs = self._probability_outputs(X, rng)
        comptime if Self.multioutput:
            return rebind[Self.Probabilities](outputs).copy()
        else:
            return rebind[Self.Probabilities](outputs[0]).copy()

    def _probability_outputs(self, X: Self.Input, mut rng: RandomState) raises -> List[Dense]:
        check_is_fitted(self)
        if X.ndim < 1:
            raise Error("ValueError: X must have a sample dimension")
        var result = List[Dense]()
        for j in range(self.n_outputs_.value()):
            var n = X.shape[0]
            var c = self._class_count(j)
            var p = nm.zeros[DType.float64](Shape(n, c))
            if self._strategy == "stratified":
                var sampled = nm.zeros[DType.int64](Shape(0, 0))
                comptime if Self.multioutput:
                    sampled = rng.multinomial(1, rebind[List[Dense]](self.class_prior_)[j], size=n)
                else:
                    sampled = rng.multinomial(1, rebind[Dense](self.class_prior_), size=n)
                for i in range(n):
                    for k in range(c):
                        p[Item(i, k)] = Float64(sampled.item(i, k))
            else:
                var selected = self._index(j)
                for i in range(n):
                    for k in range(c):
                        var value = Float64(k == selected)
                        if self._strategy == "prior":
                            value = self._prior_value(j, k)
                        elif self._strategy == "uniform":
                            value = 1.0 / Float64(c)
                        p[Item(i, k)] = value
            result.append(p^)
        return result^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        with check_random_state(self.random_state) as rng:
            return self.predict(X, rng[])

    def predict(self, X: Self.Input, mut rng: RandomState) raises -> Self.Prediction:
        var rows = self._predict_rows(X, rng)
        comptime if Self.multioutput:
            return rebind[Self.Prediction](rows).copy()
        else:
            var flat = List[Label]()
            for row in rows:
                flat.append(row[0].copy())
            return rebind[Self.Prediction](flat).copy()

    def _predict_rows(self, X: Self.Input, mut rng: RandomState) raises -> Self.Rows:
        check_is_fitted(self)
        if X.ndim < 1:
            raise Error("ValueError: X must have a sample dimension")
        var n = X.shape[0]
        var result = Self.Rows()
        for _ in range(n):
            result.append(List[Label]())
        var proba = List[Dense]()
        if self._strategy == "stratified":
            proba = self._probability_outputs(X, rng)
        for j in range(self.n_outputs_.value()):
            var selected = self._index(j)
            for i in range(n):
                var k = selected
                if self._strategy == "uniform":
                    k = rng.randint(self._class_count(j))
                elif self._strategy == "stratified":
                    k = 0
                    for c in range(1, self._class_count(j)):
                        if proba[j].item(i, c) > proba[j].item(i, k):
                            k = c
                if self._strategy == "constant":
                    result[i].append(self.constant.value()[j].copy())
                else:
                    result[i].append(self._class(j, k))
        return result^

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        check_is_fitted(self)
        with check_random_state(self.random_state) as rng:
            return self.predict_log_proba(X, rng[])

    def predict_log_proba(self, X: Self.Input, mut rng: RandomState) raises -> Self.Probabilities:
        var result = self._probability_outputs(X, rng)
        for j in range(len(result)):
            for i in range(result[j].shape[0]):
                for k in range(result[j].shape[1]):
                    result[j][Item(i, k)] = llvm_intrinsic["llvm.log", Float64](result[j].item(i, k))
        comptime if Self.multioutput:
            return rebind[Self.Probabilities](result).copy()
        else:
            return rebind[Self.Probabilities](result[0]).copy()

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        check_is_fitted(self)
        with check_random_state(self.random_state) as rng:
            return self._score_rows(X, self._target_rows(y), rng[], sample_weight)

    def score(self, X: Optional[Self.Input], y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        if X:
            return self.score(X.value(), y, sample_weight)
        var features = nm.zeros[Self.dtype](Shape(len(y), 1))
        return self.score(features, y, sample_weight)

    def score(self, X: Self.Input, y: Self.Target, mut rng: RandomState, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self._score_rows(X, self._target_rows(y), rng, sample_weight)

    def score(self, X: Self.Input, y: Self.OtherTarget, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        check_is_fitted(self)
        with check_random_state(self.random_state) as rng:
            return self._score_rows(X, self._other_rows(y), rng[], sample_weight)

    def score(self, X: Optional[Self.Input], y: Self.OtherTarget, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        if X:
            return self.score(X.value(), y, sample_weight)
        var features = nm.zeros[Self.dtype](Shape(len(y), 1))
        return self.score(features, y, sample_weight)

    def score(self, X: Self.Input, y: Self.OtherTarget, mut rng: RandomState, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self._score_rows(X, self._other_rows(y), rng, sample_weight)

    def _score_rows(self, X: Self.Input, y: Self.Rows, mut rng: RandomState, sample_weight: Optional[Self.SampleWeight]) raises -> Float64:
        _length(X, len(y))
        var predicted = self._predict_rows(X, rng)
        # Accuracy accepts single-output discrete classes or multilabel indicators;
        # multiclass-multioutput and continuous targets have no accuracy score.
        for collection in [y.copy(), predicted.copy()]:
            var unique = List[Label]()
            for row in collection:
                for value in row:
                    if value.kind == 1 and (not isfinite(value.number) or floor(value.number) != value.number):
                        raise Error("ValueError: continuous targets are not supported by accuracy")
                    if self.n_outputs_.value() > 1:
                        if value.kind == 2:
                            raise Error("ValueError: multiclass-multioutput accuracy is not supported")
                        var found = False
                        for seen in unique:
                            found = found or value == seen
                        if not found:
                            unique.append(value.copy())
            if self.n_outputs_.value() > 1 and len(unique) > 2:
                raise Error("ValueError: multiclass-multioutput accuracy is not supported")
        var w = _weights(len(y), sample_weight)
        var correct = Float64(0)
        var total = Float64(0)
        for i in range(len(y)):
            if len(y[i]) != self.n_outputs_.value():
                raise Error("ValueError: output count mismatch")
            var matches = True
            for j in range(len(y[i])):
                matches = matches and predicted[i][j] == y[i][j]
            if matches:
                correct += Float64(w.item(i))
            total += Float64(w.item(i))
        if total == 0:
            raise Error("ValueError: weights sum to zero")
        return correct / total


def _label_rows(y: List[Label]) -> List[List[Label]]:
    var rows = List[List[Label]]()
    for label in y:
        rows.append([label.copy()])
    return rows^
