"""Dense GaussianNB; sklearn 4358542a1bd3 (BSD-3-Clause), Chan moment merge."""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.math import isfinite
from moxie._naive_bayes_common import _scalar_weights
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie._naive_bayes_common import NaiveBayesClassifier, _Matrix, _check_prior, _log
from moxie._naive_bayes_dense import (
    ClassDomain, IntListDomain, DenseMatrix, _prepare, validate_matrix,
    dense_from_lists, probability_lists, predict_labels, log_normalize,
    exp_normalize, weighted_accuracy, GaussianStatistics, TypedGaussianStatistics,
    GaussianUpdater, BatchContract,
)

@fieldwise_init
struct GaussianMoments(GaussianUpdater):
    """Weighted population moments, using the family statistics ABI."""
    var var_smoothing: Float64
    var first: Bool
    var weighted: Bool

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: GaussianStatistics) raises:
        self.accumulate_typed(X, batch, statistics)

    def accumulate_typed[dtype: DType, state_dtype: DType](
        self, X: NDArray[dtype], batch: BatchContract,
        mut statistics: TypedGaussianStatistics[state_dtype],
    ) raises:
        comptime T = Scalar[state_dtype]
        _ = validate_matrix(X, batch.n_features)
        var n_classes = batch.n_classes
        if batch.targets.ndim != 1 or batch.targets.size != X.shape[0] or len(batch.weights) != X.shape[0]:
            raise Error("ValueError: Gaussian batch dimensions differ")
        if statistics.class_count.ndim != 1 or statistics.mean.ndim != 2 or statistics.variance.ndim != 2:
            raise Error("ValueError: Gaussian statistics ranks differ")
        if statistics.class_count.size != n_classes or statistics.mean.shape[0] != n_classes or statistics.variance.shape[0] != n_classes or statistics.mean.shape[1] != batch.n_features or statistics.variance.shape[1] != batch.n_features:
            raise Error("ValueError: Gaussian statistics dimensions differ")
        for i in range(X.shape[0]):
            if batch.targets.item(i) < 0 or Int(batch.targets.item(i)) >= n_classes:
                raise Error("ValueError: Gaussian class code out of range")
        # Two-pass centered moments avoid cancellation in E[X²] - E[X]².
        # Accumulate in the selected state dtype; NuMojo variance/ddof is excluded.
        var largest = T(0)
        for f in range(batch.n_features):
            var mean = T(0)
            for i in range(X.shape[0]):
                mean += T(X.item(i, f))
            mean /= T(X.shape[0])
            var variance = T(0)
            for i in range(X.shape[0]):
                var delta = T(X.item(i, f)) - mean
                variance += delta * delta
            variance /= T(X.shape[0])
            largest = max(largest, variance)
        statistics.epsilon = T(self.var_smoothing) * largest

        for c in range(n_classes):
            var n_new = T(0)
            for i in range(X.shape[0]):
                if Int(batch.targets.item(i)) == c:
                    n_new += T(batch.weights[i])
            var n_old = statistics.class_count.item(c)
            for f in range(batch.n_features):
                var old_variance = statistics.variance.item(c, f)
                if not self.first:
                    # The reference subtracts the NEW batch epsilon, not the old.
                    old_variance -= statistics.epsilon
                if n_new > 1e-8 or (not self.weighted and n_new > 0):
                    var mean = T(0)
                    for i in range(X.shape[0]):
                        if Int(batch.targets.item(i)) == c:
                            mean += T(batch.weights[i]) * T(X.item(i, f))
                    mean /= n_new
                    var variance = T(0)
                    for i in range(X.shape[0]):
                        if Int(batch.targets.item(i)) == c:
                            var delta = T(X.item(i, f)) - mean
                            variance += (delta * delta) * T(batch.weights[i])
                    variance /= n_new
                    if n_old == 0:
                        statistics.mean[Item(c, f)] = mean
                        old_variance = variance
                    else:
                        var total = n_old + n_new
                        var old_mean = statistics.mean.item(c, f)
                        var delta = old_mean - mean
                        old_variance = (n_old * old_variance + n_new * variance
                            + (n_old * n_new / total) * (delta * delta)) / total
                        statistics.mean[Item(c, f)] = (n_old * old_mean + n_new * mean) / total
                statistics.variance[Item(c, f)] = old_variance + statistics.epsilon
            statistics.class_count[Item(c)] = n_old + n_new


@fieldwise_init
struct GaussianNBParams(Copyable, Movable):
    var priors: Optional[List[Float64]]
    var var_smoothing: Float64


struct GaussianNB[D: ClassDomain = IntListDomain, dtype: DType = DType.float64](
    Copyable, WeightedSupervisedEstimator, NaiveBayesClassifier
):
    """Gaussian classifier; the compile-time domain selects the target type."""
    comptime Array = NDArray[Self.dtype]
    comptime ScalarType = Scalar[Self.dtype]
    comptime Params = GaussianNBParams
    comptime Input = _Matrix
    comptime Target = Self.D.Target
    comptime Prediction = Self.Target
    comptime Classes = Self.Target
    comptime Probabilities = _Matrix
    comptime SampleWeight = List[Float64]

    var priors: Optional[List[Float64]]
    var var_smoothing: Float64
    var classes_: Self.Target
    var class_count_: Self.Array
    var class_prior_: Self.Array
    var theta_: Self.Array
    var var_: Self.Array
    var epsilon_: Self.ScalarType
    var n_features_in_: Int
    var _domain: Optional[Self.D]

    def __init__(out self, *, var priors: Optional[List[Float64]] = None,
                 var_smoothing: Float64 = 1e-9) raises:
        comptime assert Self.dtype in (DType.float32, DType.float64), "Gaussian dtype must be Float32 or Float64"
        self.priors = priors^
        self.var_smoothing = var_smoothing
        self.classes_ = Self.D.empty_target()
        self.class_count_ = nm.zeros[Self.dtype](Shape(0))
        self.class_prior_ = nm.zeros[Self.dtype](Shape(0))
        self.theta_ = nm.zeros[Self.dtype](Shape(0, 0))
        self.var_ = nm.zeros[Self.dtype](Shape(0, 0))
        self.epsilon_ = 0
        self.n_features_in_ = 0
        self._domain = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.priors.copy(), self.var_smoothing)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.priors = params.priors.copy()
        self.var_smoothing = params.var_smoothing
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        priors: Optional[Optional[List[Float64]]] = None,
        var_smoothing: Optional[Float64] = None,
    ) raises -> ref[self] Self:
        if priors:
            self.priors = priors.value().copy()
        if var_smoothing:
            self.var_smoothing = var_smoothing.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(priors=self.priors.copy(), var_smoothing=self.var_smoothing)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

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
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = self.fit(X, y, sample_weight=_scalar_weights(len(X), sample_weight))
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = self.partial_fit(X, y, classes, sample_weight=_scalar_weights(len(X), sample_weight))
        return self

    def fit[
        input_dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[input_dtype],
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = validate_matrix(X)
        _ = self.fit(X, y, sample_weight=_scalar_weights(X.shape[0], sample_weight))
        return self

    def partial_fit[
        input_dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[input_dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = validate_matrix(X)
        _ = self.partial_fit(X, y, classes, sample_weight=_scalar_weights(X.shape[0], sample_weight))
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        _ = self.fit(dense_from_lists(X), y)
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
        _ = self.fit(dense_from_lists(X), y, sample_weight=sample_weight.copy())
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises -> ref[self] Self:
        _ = self.partial_fit(dense_from_lists(X), y, classes)
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        _ = self.partial_fit(dense_from_lists(X), y, classes, sample_weight.copy())
        return self

    def fit[
        input_dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[input_dtype],
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), sample_weight)
        self = fresh^
        return self

    def partial_fit[
        input_dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[input_dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var updated = self.copy()
        updated._update(X, y, classes, sample_weight)
        self = updated^
        return self

    def _check_X[input_dtype: DType](self, X: NDArray[input_dtype]) raises:
        _ = validate_matrix(X, self.n_features_in_)
        comptime if Self.dtype == DType.float32 and input_dtype != DType.float32:
            for i in range(X.shape[0]):
                for f in range(X.shape[1]):
                    if not isfinite(Self.ScalarType(X.item(i, f))):
                        raise Error("ValueError: X is not finite in the selected Gaussian dtype")

    def _update[input_dtype: DType](
        mut self, X: NDArray[input_dtype], y: Self.Target,
        classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]],
    ) raises:
        self._check_X(X)
        var prepared = _prepare[input_dtype, Self.D](
            X, y, self._domain, classes, sample_weight, self.n_features_in_, False
        )
        ref batch = prepared.batch
        comptime if Self.dtype == DType.float32:
            var weight_total = Float32(0)
            for weight in batch.weights:
                if not isfinite(Float32(weight)):
                    raise Error("ValueError: sample_weight is not finite in Float32")
                weight_total += Float32(weight)
            if not isfinite(weight_total) or weight_total <= 0:
                raise Error("ValueError: sample_weight must have positive finite Float32 total")
        var n_classes = batch.n_classes
        _check_prior(self.priors, n_classes, normalized=True)
        if not isfinite(self.var_smoothing) or self.var_smoothing < 0:
            raise Error("ValueError: var_smoothing must be finite and nonnegative")
        var first = not self.__sklearn_is_fitted__()
        if first:
            self.classes_ = prepared.domain.class_values()
            self._domain = prepared.domain.copy()
            self.n_features_in_ = batch.n_features
            self.class_count_ = nm.zeros[Self.dtype](Shape(n_classes))
            self.class_prior_ = nm.zeros[Self.dtype](Shape(n_classes))
            self.theta_ = nm.zeros[Self.dtype](Shape(n_classes, batch.n_features))
            self.var_ = nm.zeros[Self.dtype](Shape(n_classes, batch.n_features))
            if self.priors:
                for c in range(n_classes):
                    self.class_prior_[Item(c)] = Self.ScalarType(self.priors.value()[c])

        # Local views mutate only this transaction's independently owned state.
        var statistics = TypedGaussianStatistics[Self.dtype](
            self.class_count_.view(), self.theta_.view(), self.var_.view(), self.epsilon_
        )
        GaussianMoments(self.var_smoothing, first, Bool(sample_weight)).accumulate_typed(
            X, batch, statistics
        )
        self.epsilon_ = statistics.epsilon
        if not self.priors:
            var total = nm.sum(self.class_count_)
            for c in range(n_classes):
                self.class_prior_[Item(c)] = self.class_count_.item(c) / total

    def predict_joint_log_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> Self.Array:
        check_is_fitted(self)
        self._check_X(X)
        var result = nm.zeros[Self.dtype](Shape(X.shape[0], self.class_count_.size))
        for c in range(self.class_count_.size):
            var normalization = Self.ScalarType(0)
            for f in range(self.n_features_in_):
                var scaled_variance = Self.ScalarType(6.283185307179586) * self.var_.item(c, f)
                normalization += Self.ScalarType(_log(Float64(scaled_variance)))
            var log_prior = Self.ScalarType(_log(Float64(self.class_prior_.item(c))))
            for i in range(X.shape[0]):
                var distance = Self.ScalarType(0)
                for f in range(self.n_features_in_):
                    var delta = Self.ScalarType(X.item(i, f)) - self.theta_.item(c, f)
                    distance += delta * delta / self.var_.item(c, f)
                result[Item(i, c)] = log_prior + (-0.5 * normalization - 0.5 * distance)
        return result^

    def predict[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> Self.Target:
        var joint = self.predict_joint_log_proba(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> Self.Array:
        return log_normalize(self.predict_joint_log_proba(X))

    def predict_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> Self.Array:
        return exp_normalize(self.predict_joint_log_proba(X))

    def score[input_dtype: DType](self, X: NDArray[input_dtype], y: Self.Target,
                           sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return weighted_accuracy[Self.D](y, self.predict(X), sample_weight)

    def predict_joint_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(self.predict_joint_log_proba(dense_from_lists(X)))

    def predict(self, X: Self.Input) raises -> Self.Target:
        return self.predict(dense_from_lists(X))

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(self.predict_log_proba(dense_from_lists(X)))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(self.predict_proba(dense_from_lists(X)))

    def score(self, X: Self.Input, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.score(dense_from_lists(X), y, sample_weight)
