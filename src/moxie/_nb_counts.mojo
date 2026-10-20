"""Native count-model kernels; formulas adapted from sklearn (BSD-3-Clause)."""
import numojo as nm
from moxie._naive_bayes_dense import ClassDomain, _prepare
from moxie._dense import sum_axis2d
from moxie._naive_bayes_common import _alpha, _log
from moxie._naive_bayes_dense import (
    DenseMatrix, BatchContract, CountStatistics, CountUpdater,
    FeatureAlpha, DiscreteFinalizer, validate_matrix,
)

struct CountsUpdater(CountUpdater):
    def __init__(out self):
        pass

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: CountStatistics) raises:
        _ = validate_matrix(X, batch.n_features, nonnegative=True)
        var classes = batch.n_classes
        if batch.targets.size != X.shape[0] or len(batch.weights) != X.shape[0]:
            raise Error("ValueError: batch sample dimensions differ")
        if statistics.feature_count.ndim != 2 or statistics.class_count.ndim != 1:
            raise Error("ValueError: count ranks differ")
        if statistics.feature_count.shape[0] != classes or statistics.feature_count.shape[1] != X.shape[1] or statistics.class_count.shape[0] != classes:
            raise Error("ValueError: count dimensions differ")
        var indicators = nm.zeros[DType.float64](nm.Shape(classes, X.shape[0]))
        for i in range(X.shape[0]):
            var c = Int(batch.targets.item(i))
            indicators[nm.Item(c, i)] = batch.weights[i]
        var addition = indicators @ X
        var totals = sum_axis2d(indicators, 1)
        for c in range(classes):
            statistics.class_count[nm.Item(c)] = statistics.class_count.item(c) + totals.item(c)
            for f in range(X.shape[1]):
                statistics.feature_count[nm.Item(c, f)] = statistics.feature_count.item(c, f) + addition.item(c, f)

@fieldwise_init
struct CountsFinalizer[complement: Bool](DiscreteFinalizer):
    var norm: Bool

    def finalize(self, statistics: CountStatistics, alpha: FeatureAlpha,
                 force_alpha: Bool) raises -> DenseMatrix:
        if statistics.feature_count.ndim != 2:
            raise Error("ValueError: feature counts must be a matrix")
        var classes = statistics.feature_count.shape[0]
        var features = statistics.feature_count.shape[1]
        if classes == 0 or features == 0:
            raise Error("ValueError: feature counts must not be empty")
        alpha.validate(features)
        var result = nm.zeros[DType.float64](nm.Shape(classes, features))
        var global_counts = sum_axis2d(statistics.feature_count, 0)
        for c in range(classes):
            for f in range(features):
                var smoothing = alpha.scalar
                if alpha.values:
                    smoothing = alpha.values.value()[nm.Item(f)]
                smoothing = _alpha(smoothing, force_alpha)
                var count = statistics.feature_count.item(c, f) + smoothing
                comptime if Self.complement:
                    count = global_counts.item(f) + smoothing - statistics.feature_count.item(c, f)
                result[nm.Item(c, f)] = count
        var totals = sum_axis2d(result, 1)
        for c in range(classes):
            for f in range(features):
                comptime if Self.complement:
                    result[nm.Item(c, f)] = _log(result.item(c, f) / totals.item(c))
                else:
                    result[nm.Item(c, f)] = _log(result.item(c, f)) - _log(totals.item(c))
        comptime if Self.complement:
            var log_totals = sum_axis2d(result, 1)
            for c in range(classes):
                for f in range(features):
                    if self.norm:
                        result[nm.Item(c, f)] = result.item(c, f) / log_totals.item(c)
                    else:
                        result[nm.Item(c, f)] = -result.item(c, f)
        return result^

from moxie._naive_bayes_common import _scalar_weights
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie._naive_bayes_common import (
    NaiveBayesClassifier, _Matrix, _check_X, _check_prior, _log_priors,
)
from moxie._naive_bayes_dense import IntListDomain, predict_labels, log_normalize, exp_normalize, weighted_accuracy, probability_lists


def _array(X: _Matrix) raises -> DenseMatrix:
    var features = _check_X(X)
    var result = nm.zeros[DType.float64](nm.Shape(len(X), features))
    for i in range(len(X)):
        for f in range(features):
            result[nm.Item(i, f)] = X[i][f]
    return result^


def _lists[dtype: DType](X: nm.NDArray[dtype]) raises -> _Matrix:
    _ = validate_matrix(X)
    var result = _Matrix()
    for i in range(X.shape[0]):
        var row = List[Float64]()
        for f in range(X.shape[1]):
            row.append(Float64(X.item(i, f)))
        result.append(row^)
    return result^


def _vector(X: DenseMatrix) raises -> List[Float64]:
    var result = List[Float64]()
    for i in range(X.shape[0]):
        result.append(X.item(i))
    return result^

@fieldwise_init
struct MultinomialNBParams(Copyable, Movable):
    var alpha: Float64
    var alpha_values: Optional[DenseMatrix]
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]


struct MultinomialNB[D: ClassDomain = IntListDomain](Copyable, WeightedSupervisedEstimator, NaiveBayesClassifier):
    """Dense MultinomialNB with weighted counts and incremental fitting."""

    comptime Params = MultinomialNBParams
    comptime Input = _Matrix
    comptime Target = Self.D.Target
    comptime Prediction = Self.Target
    comptime Probabilities = _Matrix
    comptime SampleWeight = List[Float64]
    comptime Classes = Self.Target

    var alpha: Float64
    var alpha_values: Optional[DenseMatrix]
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var classes_: Self.Target
    var _domain: Optional[Self.D]
    var class_count_: DenseMatrix
    var class_log_prior_: List[Float64]
    var n_features_in_: Int
    var feature_count_: DenseMatrix
    var feature_log_prob_: DenseMatrix

    def __init__(
        out self,
        *,
        alpha: Float64 = 1.0,
        var alpha_values: Optional[DenseMatrix] = None,
        force_alpha: Bool = True,
        fit_prior: Bool = True,
        var class_prior: Optional[List[Float64]] = None,
    ) raises:
        self.alpha_values = alpha_values^
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.classes_ = Self.D.empty_target()
        self._domain = None
        self.class_count_ = nm.zeros[DType.float64](nm.Shape(0))
        self.class_log_prior_ = List[Float64]()
        self.n_features_in_ = 0
        self.feature_count_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.feature_log_prob_ = nm.zeros[DType.float64](nm.Shape(0, 0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.alpha,
            self.alpha_values.copy(),
            self.force_alpha,
            self.fit_prior,
            self.class_prior.copy(),
        )

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.alpha_values = params.alpha_values.copy()
        self.alpha = params.alpha
        self.force_alpha = params.force_alpha
        self.fit_prior = params.fit_prior
        self.class_prior = params.class_prior.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        alpha: Optional[Float64] = None,
        alpha_values: Optional[Optional[DenseMatrix]] = None,
        force_alpha: Optional[Bool] = None,
        fit_prior: Optional[Bool] = None,
        class_prior: Optional[Optional[List[Float64]]] = None,
    ) raises -> ref[self] Self:
        if alpha:
            self.alpha = alpha.value()
            self.alpha_values = None
        if alpha_values:
            self.alpha_values = alpha_values.value().copy()
        if force_alpha:
            self.force_alpha = force_alpha.value()
        if fit_prior:
            self.fit_prior = fit_prior.value()
        if class_prior:
            self.class_prior = class_prior.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            alpha=self.alpha,
            alpha_values=self.alpha_values.copy(),
            force_alpha=self.force_alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def _labels(self) raises -> Self.Target:
        return self.classes_.copy()

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
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = validate_matrix(X)
        _ = self.fit(X, y, sample_weight=_scalar_weights(X.shape[0], sample_weight))
        return self

    def partial_fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
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
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), None)
        self = fresh^
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
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), sample_weight.copy())
        self = fresh^
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
        var updated = self.copy()
        updated._update(X, y, classes, None)
        self = updated^
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
        var updated = self.copy()
        updated._update(X, y, classes, sample_weight.copy())
        self = updated^
        return self

    def _update(mut self, X: _Matrix, y: Self.Target,
                classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]]) raises:
        var array = _array(X)
        self._update_array(array, y, classes, sample_weight)

    def _update_array[dtype: DType](mut self, X: nm.NDArray[dtype], y: Self.Target,
                classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]]) raises:
        var labeled = _prepare[dtype, Self.D](X, y, self._domain, classes, sample_weight,
                                  self.n_features_in_, True)
        var batch = labeled.batch.copy()
        var alpha = FeatureAlpha(self.alpha, self.alpha_values.copy())
        alpha.validate(batch.n_features)
        _check_prior(self.class_prior, batch.n_classes)
        if not self.__sklearn_is_fitted__():
            self.classes_ = labeled.domain.class_values()
            self._domain = labeled.domain.copy()
            self.n_features_in_ = batch.n_features
            self.class_count_ = nm.zeros[DType.float64](nm.Shape(batch.n_classes))
            self.feature_count_ = nm.zeros[DType.float64](nm.Shape(batch.n_classes, batch.n_features))
        var stats = CountStatistics(self.class_count_.copy(), self.feature_count_.copy())
        var widened = nm.zeros[DType.float64](nm.Shape(X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for f in range(X.shape[1]):
                widened[nm.Item(i, f)] = Float64(X.item(i, f))
        CountsUpdater().accumulate(widened, batch, stats)
        self.feature_log_prob_ = CountsFinalizer[False](False).finalize(stats, alpha, self.force_alpha)
        self.class_count_ = stats.class_count.copy()
        self.feature_count_ = stats.feature_count.copy()
        self.class_log_prior_ = _log_priors(_vector(self.class_count_), self.class_prior, self.fit_prior)

    def fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var fresh = self.__sklearn_clone__()
        fresh._update_array(X, y, y.copy(), sample_weight)
        self = fresh^
        return self

    def partial_fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var updated = self.copy()
        updated._update_array(X, y, classes, sample_weight)
        self = updated^
        return self

    def _joint_array(self, X: _Matrix) raises -> DenseMatrix:
        check_is_fitted(self)
        _ = _check_X(X, self.n_features_in_)
        var result = nm.zeros[DType.float64](nm.Shape(len(X), self.class_count_.size))
        for i in range(len(X)):
            for c in range(self.class_count_.size):
                var score = Float64(0)
                for f in range(self.n_features_in_):
                    score += X[i][f] * self.feature_log_prob_.item(c, f)
                score += self.class_log_prior_[c]
                result[nm.Item(i, c)] = score
        return result^

    def predict_joint_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(self._joint_array(X))

    def predict(self, X: Self.Input) raises -> Self.Target:
        var joint = self._joint_array(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(log_normalize(self._joint_array(X)))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(exp_normalize(self._joint_array(X)))

    def score(self, X: Self.Input, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return weighted_accuracy[Self.D](y, self.predict(X), sample_weight)

    def predict[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Self.Target:
        return self.predict(_lists(X))

    def predict_joint_log_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_joint_log_proba(_lists(X))

    def predict_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_proba(_lists(X))

    def predict_log_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_log_proba(_lists(X))

    def score[dtype: DType](self, X: nm.NDArray[dtype], y: Self.Target,
            sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.score(_lists(X), y, sample_weight)


@fieldwise_init
struct ComplementNBParams(Copyable, Movable):
    var alpha: Float64
    var alpha_values: Optional[DenseMatrix]
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var norm: Bool


struct ComplementNB[D: ClassDomain = IntListDomain](Copyable, WeightedSupervisedEstimator, NaiveBayesClassifier):
    """Dense ComplementNB with weighted counts and incremental fitting."""

    comptime Params = ComplementNBParams
    comptime Input = _Matrix
    comptime Target = Self.D.Target
    comptime Prediction = Self.Target
    comptime Probabilities = _Matrix
    comptime SampleWeight = List[Float64]
    comptime Classes = Self.Target

    var alpha: Float64
    var alpha_values: Optional[DenseMatrix]
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var norm: Bool
    var classes_: Self.Target
    var _domain: Optional[Self.D]
    var class_count_: DenseMatrix
    var class_log_prior_: List[Float64]
    var n_features_in_: Int
    var feature_count_: DenseMatrix
    var feature_log_prob_: DenseMatrix
    var feature_all_: DenseMatrix

    def __init__(
        out self,
        *,
        alpha: Float64 = 1.0,
        var alpha_values: Optional[DenseMatrix] = None,
        force_alpha: Bool = True,
        fit_prior: Bool = True,
        var class_prior: Optional[List[Float64]] = None,
        norm: Bool = False,
    ) raises:
        self.alpha_values = alpha_values^
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.norm = norm
        self.classes_ = Self.D.empty_target()
        self._domain = None
        self.class_count_ = nm.zeros[DType.float64](nm.Shape(0))
        self.class_log_prior_ = List[Float64]()
        self.n_features_in_ = 0
        self.feature_count_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.feature_log_prob_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.feature_all_ = nm.zeros[DType.float64](nm.Shape(0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.alpha,
            self.alpha_values.copy(),
            self.force_alpha,
            self.fit_prior,
            self.class_prior.copy(),
            self.norm,
        )

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.alpha_values = params.alpha_values.copy()
        self.alpha = params.alpha
        self.force_alpha = params.force_alpha
        self.fit_prior = params.fit_prior
        self.class_prior = params.class_prior.copy()
        self.norm = params.norm
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        alpha: Optional[Float64] = None,
        alpha_values: Optional[Optional[DenseMatrix]] = None,
        force_alpha: Optional[Bool] = None,
        fit_prior: Optional[Bool] = None,
        class_prior: Optional[Optional[List[Float64]]] = None,
        norm: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if alpha:
            self.alpha = alpha.value()
            self.alpha_values = None
        if alpha_values:
            self.alpha_values = alpha_values.value().copy()
        if force_alpha:
            self.force_alpha = force_alpha.value()
        if fit_prior:
            self.fit_prior = fit_prior.value()
        if class_prior:
            self.class_prior = class_prior.value().copy()
        if norm:
            self.norm = norm.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            alpha=self.alpha,
            alpha_values=self.alpha_values.copy(),
            force_alpha=self.force_alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior.copy(),
            norm=self.norm,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def _labels(self) raises -> Self.Target:
        return self.classes_.copy()

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
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        _ = validate_matrix(X)
        _ = self.fit(X, y, sample_weight=_scalar_weights(X.shape[0], sample_weight))
        return self

    def partial_fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
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
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), None)
        self = fresh^
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
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), sample_weight.copy())
        self = fresh^
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
        var updated = self.copy()
        updated._update(X, y, classes, None)
        self = updated^
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
        var updated = self.copy()
        updated._update(X, y, classes, sample_weight.copy())
        self = updated^
        return self

    def _update(mut self, X: _Matrix, y: Self.Target,
                classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]]) raises:
        var array = _array(X)
        self._update_array(array, y, classes, sample_weight)

    def _update_array[dtype: DType](mut self, X: nm.NDArray[dtype], y: Self.Target,
                classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]]) raises:
        var labeled = _prepare[dtype, Self.D](X, y, self._domain, classes, sample_weight,
                                  self.n_features_in_, True)
        var batch = labeled.batch.copy()
        var alpha = FeatureAlpha(self.alpha, self.alpha_values.copy())
        alpha.validate(batch.n_features)
        _check_prior(self.class_prior, batch.n_classes)
        if not self.__sklearn_is_fitted__():
            self.classes_ = labeled.domain.class_values()
            self._domain = labeled.domain.copy()
            self.n_features_in_ = batch.n_features
            self.class_count_ = nm.zeros[DType.float64](nm.Shape(batch.n_classes))
            self.feature_count_ = nm.zeros[DType.float64](nm.Shape(batch.n_classes, batch.n_features))
        var stats = CountStatistics(self.class_count_.copy(), self.feature_count_.copy())
        var widened = nm.zeros[DType.float64](nm.Shape(X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for f in range(X.shape[1]):
                widened[nm.Item(i, f)] = Float64(X.item(i, f))
        CountsUpdater().accumulate(widened, batch, stats)
        self.feature_log_prob_ = CountsFinalizer[True](self.norm).finalize(stats, alpha, self.force_alpha)
        self.class_count_ = stats.class_count.copy()
        self.feature_count_ = stats.feature_count.copy()
        self.class_log_prior_ = _log_priors(_vector(self.class_count_), self.class_prior, self.fit_prior)
        self.feature_all_ = sum_axis2d(self.feature_count_, 0)

    def fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var fresh = self.__sklearn_clone__()
        fresh._update_array(X, y, y.copy(), sample_weight)
        self = fresh^
        return self

    def partial_fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: nm.NDArray[dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var updated = self.copy()
        updated._update_array(X, y, classes, sample_weight)
        self = updated^
        return self

    def _joint_array(self, X: _Matrix) raises -> DenseMatrix:
        check_is_fitted(self)
        _ = _check_X(X, self.n_features_in_)
        var result = nm.zeros[DType.float64](nm.Shape(len(X), self.class_count_.size))
        for i in range(len(X)):
            for c in range(self.class_count_.size):
                var score = Float64(0)
                for f in range(self.n_features_in_):
                    score += X[i][f] * self.feature_log_prob_.item(c, f)
                if self.class_count_.size == 1:
                    score += self.class_log_prior_[c]
                result[nm.Item(i, c)] = score
        return result^

    def predict_joint_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(self._joint_array(X))

    def predict(self, X: Self.Input) raises -> Self.Target:
        var joint = self._joint_array(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(log_normalize(self._joint_array(X)))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return probability_lists(exp_normalize(self._joint_array(X)))

    def score(self, X: Self.Input, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return weighted_accuracy[Self.D](y, self.predict(X), sample_weight)

    def predict[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Self.Target:
        return self.predict(_lists(X))

    def predict_joint_log_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_joint_log_proba(_lists(X))

    def predict_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_proba(_lists(X))

    def predict_log_proba[dtype: DType](self, X: nm.NDArray[dtype]) raises -> _Matrix:
        return self.predict_log_proba(_lists(X))

    def score[dtype: DType](self, X: nm.NDArray[dtype], y: Self.Target,
            sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.score(_lists(X), y, sample_weight)

