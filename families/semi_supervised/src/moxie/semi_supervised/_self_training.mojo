# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Self-training over a native probabilistic estimator adapter."""
from std.math import isfinite
from numojo import NDArray, Shape, zeros
from moxie.base import BaseEstimator
from ._label_propagation import GraphClassifier
from ._affinity import RBFAffinity
from ._contracts import ProbabilisticEstimatorAdapter, ScoringEstimatorAdapter, ClassEncoding, integer_targets, validate_features, validate_probability_shape

trait PredictingEstimatorAdapter(ProbabilisticEstimatorAdapter):
    def predict(self, X: Self.Input) raises -> List[Self.Label]:
        ...

trait LogProbabilityEstimatorAdapter(ProbabilisticEstimatorAdapter):
    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        ...

trait DecisionEstimatorAdapter(ProbabilisticEstimatorAdapter):
    def decision_function(self, X: Self.Input) raises -> NDArray[DType.float64]:
        ...

struct GraphClassifierAdapter[spreading: Bool = False, dtype: DType = DType.float64, L: Copyable & Comparable & Deinitable = Int](ScoringEstimatorAdapter, PredictingEstimatorAdapter):
    """Use a graph classifier as the supervised estimator in self-training."""
    comptime Input = NDArray[Self.dtype]
    comptime Label = Self.L
    comptime Estimator = GraphClassifier[Self.spreading, Self.dtype, RBFAffinity[Self.dtype], Self.L]
    var estimator: Self.Estimator
    def __init__(out self, var estimator: Self.Estimator):
        self.estimator = estimator^
    def clone_unfitted(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())
    def fit(mut self, X: Self.Input, y: List[Self.L]) raises:
        self.estimator.fit(X, y)
    def classes(self) raises -> List[Self.L]:
        return self.estimator.classes_.copy()
    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_proba(X)
    def predict(self, X: Self.Input) raises -> List[Self.L]:
        return self.estimator.predict(X)
    def score(self, X: Self.Input, y: List[Self.L]) raises -> Float64:
        return self.estimator.score(X, y)

@fieldwise_init
struct MissingEstimator(ProbabilisticEstimatorAdapter):
    comptime Input = NDArray[DType.float64]
    comptime Label = Int
    def clone_unfitted(self) raises -> Self:
        raise Error("ValueError: an explicit probabilistic estimator is required")
    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        raise Error("ValueError: an explicit probabilistic estimator is required")
    def classes(self) raises -> List[Int]:
        raise Error("NotFittedError: no estimator")
    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        raise Error("NotFittedError: no estimator")

@fieldwise_init
struct SelfTrainingParams[E: ProbabilisticEstimatorAdapter & Deinitable](Copyable, Movable):
    var estimator: Optional[Self.E]
    var threshold: Float64
    var criterion: String
    var k_best: Int
    var max_iter: Optional[Int]
    var verbose: Bool

struct SelfTrainingClassifier[E: ProbabilisticEstimatorAdapter & Deinitable = MissingEstimator, dtype: DType = DType.float64](BaseEstimator, Copyable, Movable):
    comptime Input = NDArray[Self.dtype]
    comptime Label = Self.E.Label
    comptime Probabilities = NDArray[DType.float64]
    comptime Params = SelfTrainingParams[Self.E]
    var estimator: Optional[Self.E]
    var threshold: Float64
    var criterion: String
    var k_best: Int
    var max_iter: Optional[Int]
    var verbose: Bool
    var estimator_: Optional[Self.E]
    var classes_: List[Self.Label]
    var transduction_: List[Optional[Self.Label]]
    var labeled_iter_: List[Int]
    var n_iter_: Int
    var n_features_in_: Int
    var termination_condition_: String
    var _fitted: Bool

    def __init__(out self, var estimator: Optional[Self.E] = None, threshold: Float64 = 0.75, criterion: String = "threshold", k_best: Int = 10, max_iter: Optional[Int] = Optional(10), verbose: Bool = False):
        comptime assert Self.E.Input == Self.Input, "adapter input must match the selected dense dtype"
        self.estimator = estimator^
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.estimator_ = None
        self.classes_ = List[Self.Label]()
        self.transduction_ = List[Optional[Self.Label]]()
        self.labeled_iter_ = List[Int]()
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self.termination_condition_ = ""
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        var estimator: Optional[Self.E] = None
        if self.estimator:
            estimator = self.estimator.value().clone_unfitted()
        return Self.Params(estimator^, self.threshold, self.criterion, self.k_best, self.max_iter, self.verbose)

    def set_params[origin: MutOrigin, p_origin: ImmOrigin](ref[origin] self, ref[p_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.threshold = params.threshold
        self.criterion = params.criterion
        self.k_best = params.k_best
        self.max_iter = params.max_iter
        self.verbose = params.verbose
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, estimator: Optional[Optional[Self.E]] = None, threshold: Optional[Float64] = None, criterion: Optional[String] = None, k_best: Optional[Int] = None, max_iter: Optional[Optional[Int]] = None, verbose: Optional[Bool] = None) raises -> ref[self] Self:
        if estimator:
            self.estimator = estimator.value().copy()
        if threshold:
            self.threshold = threshold.value()
        if criterion:
            self.criterion = criterion.value()
        if k_best:
            self.k_best = k_best.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if verbose:
            self.verbose = verbose.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var params = self.get_params()
        return Self(params.estimator.copy(), self.threshold, self.criterion, self.k_best, self.max_iter, self.verbose)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _rows(self, X: Self.Input, indices: List[Int]) raises -> Self.Input:
        var result = zeros[Self.dtype](Shape(len(indices), X.shape[1]))
        for i in range(len(indices)):
            for j in range(X.shape[1]):
                result.unsafe_set(i * X.shape[1] + j, X.unsafe_get(indices[i] * Int(X.strides[0]) + j * Int(X.strides[1])))
        return result^

    def _fit_labeled(mut self, X: Self.Input) raises:
        var indices = List[Int]()
        var labels = List[Self.Label]()
        for i in range(len(self.transduction_)):
            if self.transduction_[i]:
                indices.append(i)
                labels.append(self.transduction_[i].value().copy())
        var selected = self._rows(X, indices)
        self.estimator_.value().fit(rebind[Self.E.Input](selected), labels)
        self.classes_ = self.estimator_.value().classes()
        var encoding = ClassEncoding(self.classes_)
        _ = encoding.encode(self.transduction_)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: List[Int]) raises -> ref[self] Self where Self.Label == Int:
        self.fit(X, rebind[List[Optional[Self.Label]]](integer_targets(y)))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: List[Optional[Self.Label]]) raises -> ref[self] Self:
        self._fitted = False
        validate_features(X, len(y))
        if not self.estimator:
            raise Error("ValueError: an explicit probabilistic estimator is required")
        if not isfinite(self.threshold) or self.threshold < 0 or self.threshold >= 1 or self.k_best < 1:
            raise Error("ValueError: invalid threshold or k_best")
        if self.criterion != "threshold" and self.criterion != "k_best":
            raise Error("ValueError: criterion must be threshold or k_best")
        if self.max_iter and self.max_iter.value() < 0:
            raise Error("ValueError: max_iter must be nonnegative or None")
        self.estimator_ = self.estimator.value().clone_unfitted()
        self.transduction_ = y.copy()
        self.labeled_iter_ = List[Int]()
        self.n_iter_ = 0
        self.n_features_in_ = X.shape[1]
        self.termination_condition_ = ""
        var remaining = 0
        for label in y:
            comptime if Self.Label == Int:
                if label:
                    if rebind[Int](label.value()) == -1:
                        raise Error("ValueError: use None for an unlabeled tagged integer target")
            self.labeled_iter_.append(0 if label else -1)
            if not label:
                remaining += 1
        if remaining == len(y):
            raise Error("ValueError: at least one labeled sample is required")
        if remaining == 0:
            print("UserWarning: y contains no unlabeled samples")
        if self.criterion == "k_best" and self.k_best > remaining:
            print("UserWarning: k_best is larger than the number of unlabeled samples")
        while remaining > 0 and (not self.max_iter or self.n_iter_ < self.max_iter.value()):
            self.n_iter_ += 1
            self._fit_labeled(X)
            var indices = List[Int]()
            for i in range(len(y)):
                if not self.transduction_[i]:
                    indices.append(i)
            var query = self._rows(X, indices)
            var probabilities = self.estimator_.value().predict_proba(rebind[Self.E.Input](query))
            validate_probability_shape(probabilities, remaining, len(self.classes_))
            var confidence = List[Float64]()
            var predicted = List[Int]()
            var order = List[Int]()
            for i in range(remaining):
                var best = 0
                for j in range(len(self.classes_)):
                    var value = probabilities.unsafe_get(i * Int(probabilities.strides[0]) + j * Int(probabilities.strides[1]))
                    if not isfinite(value) or value < 0 or value > 1:
                        raise Error("ValueError: estimator probabilities must be finite and in [0, 1]")
                    if value > probabilities.unsafe_get(i * Int(probabilities.strides[0]) + best * Int(probabilities.strides[1])):
                        best = j
                predicted.append(best)
                confidence.append(probabilities.unsafe_get(i * Int(probabilities.strides[0]) + best * Int(probabilities.strides[1])))
                order.append(i)
            if self.criterion == "k_best" and self.k_best < remaining:
                for i in range(self.k_best):
                    for j in range(i + 1, remaining):
                        if confidence[order[j]] > confidence[order[i]] or (confidence[order[j]] == confidence[order[i]] and order[j] < order[i]):
                            var temp = order[i]
                            order[i] = order[j]
                            order[j] = temp
            var added = 0
            for rank in range(remaining):
                var index = order[rank]
                var selected = confidence[index] > self.threshold if self.criterion == "threshold" else rank < self.k_best
                if selected:
                    self.transduction_[indices[index]] = self.classes_[predicted[index]].copy()
                    self.labeled_iter_[indices[index]] = self.n_iter_
                    added += 1
            remaining -= added
            if added == 0:
                self.termination_condition_ = "no_change"
                break
            if self.verbose:
                print(String("End of iteration {}, added {} new labels.").format(self.n_iter_, added))
        if self.max_iter and self.n_iter_ == self.max_iter.value():
            self.termination_condition_ = "max_iter"
        if remaining == 0:
            self.termination_condition_ = "all_labeled"
        self._fit_labeled(X)
        self._fitted = True
        return self

    def _check(self, X: Self.Input) raises:
        if not self._fitted:
            raise Error("NotFittedError: fit SelfTrainingClassifier first")
        validate_features(X, X.shape[0] if X.ndim == 2 else 0)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: feature count differs from fit")

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        self._check(X)
        var probabilities = self.estimator_.value().predict_proba(rebind[Self.E.Input](X))
        validate_probability_shape(probabilities, X.shape[0], len(self.classes_))
        return probabilities^

    def predict(self, X: Self.Input) raises -> List[Self.Label] where conforms_to(Self.E, PredictingEstimatorAdapter):
        self._check(X)
        return self.estimator_.value().predict(rebind[Self.E.Input](X))

    def score(self, X: Self.Input, y: List[Self.Label]) raises -> Float64 where conforms_to(Self.E, ScoringEstimatorAdapter):
        self._check(X)
        return self.estimator_.value().score(rebind[Self.E.Input](X), y)

    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64] where conforms_to(Self.E, LogProbabilityEstimatorAdapter):
        self._check(X)
        return self.estimator_.value().predict_log_proba(rebind[Self.E.Input](X))

    def decision_function(self, X: Self.Input) raises -> NDArray[DType.float64] where conforms_to(Self.E, DecisionEstimatorAdapter):
        self._check(X)
        return self.estimator_.value().decision_function(rebind[Self.E.Input](X))
