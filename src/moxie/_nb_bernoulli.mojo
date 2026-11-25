"""Native dense BernoulliNB; sklearn 4358542a1bd3 (BSD-3-Clause)."""
import numojo as nm
from moxie.compose._table import ColumnTable
from moxie.warnings import FitWarning
from numojo import NDArray, Shape, Item
from moxie._dense import sum_axis2d
from moxie.metadata_routing import FitScoreMetadataRequest, SampleWeightRequest, SampleWeightUpdate
from moxie._naive_bayes_common import (
    NaiveBayesMetadataRequestProvider, _NaiveBayesMetadata, _check_feature_names,
)
from moxie._naive_bayes_common import _scalar_weights
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie._naive_bayes_common import (
    NaiveBayesClassifier, _Matrix, _alpha, _exp, _log,
    _check_prior, _log_priors, _check_binarize,
)
from moxie._naive_bayes_dense import (
    DenseMatrix, BatchContract, CountStatistics, CountUpdater, FeatureAlpha, validate_matrix,
    ClassDomain, _prepare, IntListDomain, dense_from_lists, probability_lists,
    predict_labels, log_normalize, exp_normalize, weighted_accuracy,
)


struct BernoulliNBParams(Copyable, Movable):
    var alpha: Float64
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var binarize: Optional[Float64]
    var alpha_values: Optional[DenseMatrix]

    def __init__(out self, alpha: Float64, force_alpha: Bool, fit_prior: Bool,
                 var class_prior: Optional[List[Float64]], binarize: Optional[Float64],
                 var alpha_values: Optional[DenseMatrix] = None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.binarize = binarize
        self.alpha_values = alpha_values.copy()


def _binary_value[dtype: DType](value: SIMD[dtype, 1], threshold: Optional[Float64]) -> Float64:
    if threshold:
        comptime if dtype == DType.float32 or dtype == DType.float64:
            # NumPy compares a weak Python scalar in the floating input dtype.
            return Float64(value > SIMD[dtype, 1](threshold.value()))
        else:
            # Integer/bool arrays promote for comparison with a float threshold.
            return Float64(Float64(value) > threshold.value())
    return Float64(value)


@fieldwise_init
struct BernoulliCounts(CountUpdater):
    var binarize: Optional[Float64]

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: CountStatistics) raises:
        self.accumulate_typed(X, batch, statistics)

    def accumulate_typed[dtype: DType](self, X: NDArray[dtype], batch: BatchContract,
                                     mut statistics: CountStatistics) raises:
        _ = validate_matrix(X, batch.n_features)
        _check_binarize(self.binarize)
        if batch.targets.ndim != 1 or batch.targets.size != X.shape[0] or len(batch.weights) != X.shape[0]:
            raise Error("ValueError: Bernoulli batch dimensions differ")
        if statistics.class_count.ndim != 1 or statistics.feature_count.ndim != 2:
            raise Error("ValueError: Bernoulli statistics ranks differ")
        if statistics.class_count.size != batch.n_classes or statistics.feature_count.shape[0] != batch.n_classes or statistics.feature_count.shape[1] != batch.n_features:
            raise Error("ValueError: Bernoulli statistics dimensions differ")
        var indicators = nm.zeros[DType.float64](Shape(batch.n_classes, X.shape[0]))
        var values = nm.zeros[DType.float64](Shape(X.shape[0], batch.n_features))
        for i in range(X.shape[0]):
            var c = Int(batch.targets.item(i))
            if c < 0 or c >= batch.n_classes:
                raise Error("ValueError: Bernoulli class code out of range")
            indicators[Item(c, i)] = batch.weights[i]
            for f in range(batch.n_features):
                values[Item(i, f)] = _binary_value(X.item(i, f), self.binarize)
        var addition = indicators @ values
        var totals = sum_axis2d(indicators, 1)
        for c in range(batch.n_classes):
            statistics.class_count[Item(c)] = statistics.class_count.item(c) + totals.item(c)
            for f in range(batch.n_features):
                statistics.feature_count[Item(c, f)] = statistics.feature_count.item(c, f) + addition.item(c, f)


struct BernoulliNB[D: ClassDomain = IntListDomain, dtype: DType = DType.float64](
    Copyable, WeightedSupervisedEstimator, NaiveBayesClassifier, NaiveBayesMetadataRequestProvider
):
    """Dense BernoulliNB with weighted counts and incremental fitting."""

    comptime Params = BernoulliNBParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.D.Target
    comptime Prediction = Self.Target
    comptime Probabilities = DenseMatrix
    comptime SampleWeight = List[Float64]
    comptime Classes = Self.Target

    var alpha: Float64
    var alpha_values: Optional[DenseMatrix]
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var binarize: Optional[Float64]
    var classes_: Self.Target
    var _domain: Optional[Self.D]
    var _metadata: _NaiveBayesMetadata
    var class_count_: DenseMatrix
    var class_log_prior_: DenseMatrix
    var fit_warnings: List[FitWarning]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
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
        binarize: Optional[Float64] = 0.0,
    ) raises:
        self.alpha = alpha
        self.alpha_values = alpha_values.copy()
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.binarize = binarize
        self.classes_ = Self.D.empty_target()
        self._domain = None
        self._metadata = _NaiveBayesMetadata()
        self.class_count_ = nm.zeros[DType.float64](Shape(0))
        self.fit_warnings = []
        self.class_log_prior_ = nm.zeros[DType.float64](Shape(0))
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.feature_count_ = nm.zeros[DType.float64](Shape(0, 0))
        self.feature_log_prob_ = nm.zeros[DType.float64](Shape(0, 0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior.copy(),
            self.binarize,
            self.alpha_values.copy(),
        )

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.alpha = params.alpha
        self.alpha_values = params.alpha_values.copy()
        self.force_alpha = params.force_alpha
        self.fit_prior = params.fit_prior
        self.class_prior = params.class_prior.copy()
        self.binarize = params.binarize
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
        binarize: Optional[Optional[Float64]] = None,
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
        if binarize:
            self.binarize = binarize.value()
        return self


    def set_fit_request[origin: MutOrigin](
        ref[origin] self,
        *,
        sample_weight: SampleWeightUpdate = SampleWeightUpdate(),
    ) raises -> ref[self] Self:
        self._metadata.fit_score.fit.set(sample_weight)
        return self

    def set_partial_fit_request[origin: MutOrigin](
        ref[origin] self,
        *,
        sample_weight: SampleWeightUpdate = SampleWeightUpdate(),
    ) raises -> ref[self] Self:
        self._metadata.partial_fit.set(sample_weight)
        return self

    def set_score_request[origin: MutOrigin](
        ref[origin] self,
        *,
        sample_weight: SampleWeightUpdate = SampleWeightUpdate(),
    ) raises -> ref[self] Self:
        self._metadata.fit_score.score.set(sample_weight)
        return self

    def get_metadata_routing(self) -> FitScoreMetadataRequest:
        return self._metadata.fit_score.copy()

    def get_partial_fit_request(self) -> SampleWeightRequest:
        return self._metadata.partial_fit.copy()

    def __sklearn_clone__(self) raises -> Self:
        var cloned = Self(
            alpha=self.alpha,
            alpha_values=self.alpha_values.copy(),
            force_alpha=self.force_alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior.copy(),
            binarize=self.binarize,
        )
        cloned._metadata = self._metadata.copy()
        return cloned^

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
        ref[x_origin] X: _Matrix,
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
        ref[x_origin] X: _Matrix,
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
        ref[x_origin] X: _Matrix,
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
        ref[x_origin] X: _Matrix,
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
        ref[x_origin] X: _Matrix,
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
        ref[x_origin] X: _Matrix,
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
        self._update_array(dense_from_lists(X), y, classes, sample_weight)

    def fit[
        input_dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[input_dtype],
        ref[y_origin] y: Self.Target,
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var fresh = self.__sklearn_clone__()
        fresh._update_array(X, y, y.copy(), sample_weight)
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
        *,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var updated = self.copy()
        updated._update_array(X, y, classes, sample_weight)
        self = updated^
        return self

    def _update_array[input_dtype: DType](mut self, X: NDArray[input_dtype], y: Self.Target,
            classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]]) raises:
        var prepared = _prepare[input_dtype, Self.D](X, y, self._domain, classes, sample_weight,
                                     self.n_features_in_, False)
        ref batch = prepared.batch
        var alpha = FeatureAlpha(self.alpha, self.alpha_values.copy())
        alpha.validate(batch.n_features)
        _check_binarize(self.binarize)
        _check_prior(self.class_prior, batch.n_classes)
        if not self.__sklearn_is_fitted__():
            self.classes_ = prepared.domain.class_values()
            self._domain = prepared.domain.copy()
            self.n_features_in_ = batch.n_features
            self.class_count_ = nm.zeros[DType.float64](Shape(batch.n_classes))
            self.feature_count_ = nm.zeros[DType.float64](Shape(batch.n_classes, batch.n_features))
        var statistics = CountStatistics(self.class_count_.view(), self.feature_count_.view())
        BernoulliCounts(self.binarize).accumulate_typed(X, batch, statistics)
        self.feature_log_prob_ = nm.zeros[DType.float64](Shape(batch.n_classes, batch.n_features))
        for c in range(batch.n_classes):
            for f in range(batch.n_features):
                var smoothing = alpha.scalar
                if alpha.values:
                    smoothing = alpha.values.value()[Item(f)]
                smoothing = _alpha(smoothing, self.force_alpha)
                # Apply each feature's alpha to both outcomes. The pinned sklearn
                # vector path incorrectly broadcasts alpha against class counts.
                var denominator = _log(self.class_count_.item(c) + 2.0 * smoothing)
                self.feature_log_prob_[Item(c, f)] = _log(
                    self.feature_count_.item(c, f) + smoothing
                ) - denominator
        self.class_log_prior_ = _log_priors(self.class_count_, self.class_prior, self.fit_prior)
        self.fit_warnings = alpha._fit_warnings(self.force_alpha)

    def fit[input_dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: ColumnTable[input_dtype],
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        var names = X.schema().names.copy()
        _ = self.fit[input_dtype](X.values(), y, sample_weight=sample_weight)
        self.feature_names_in_ = names^
        return self

    def fit[input_dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: ColumnTable[input_dtype],
        ref[y_origin] y: Self.Target,
        *,
        sample_weight: Float64,
    ) raises -> ref[self] Self:
        return self.fit(X, y, sample_weight=_scalar_weights(X.rows(), sample_weight))

    def partial_fit[input_dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: ColumnTable[input_dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        var names = X.schema().names.copy()
        _check_feature_names(self.feature_names_in_, names)
        var first = not self.__sklearn_is_fitted__()
        _ = self.partial_fit[input_dtype](X.values(), y, classes, sample_weight=sample_weight)
        if first:
            self.feature_names_in_ = names^
        return self

    def partial_fit[input_dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: ColumnTable[input_dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        *,
        sample_weight: Float64,
    ) raises -> ref[self] Self:
        return self.partial_fit(X, y, classes, sample_weight=_scalar_weights(X.rows(), sample_weight))

    def predict_joint_log_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> DenseMatrix:
        check_is_fitted(self)
        _ = validate_matrix(X, self.n_features_in_)
        _check_binarize(self.binarize)
        var result = nm.zeros[DType.float64](Shape(X.shape[0], self.class_count_.size))
        for c in range(self.class_count_.size):
            var negative = nm.zeros[DType.float64](Shape(self.n_features_in_))
            var negative_sum = Float64(0)
            for f in range(self.n_features_in_):
                negative[Item(f)] = _log(1.0 - _exp(self.feature_log_prob_.item(c, f)))
                negative_sum += negative.item(f)
            for i in range(X.shape[0]):
                var joint = Float64(0)
                for f in range(self.n_features_in_):
                    # Do not skip zero products: 0 * infinity must remain NaN.
                    joint += _binary_value(X.item(i, f), self.binarize) * (
                        self.feature_log_prob_.item(c, f) - negative.item(f)
                    )
                result[Item(i, c)] = joint + (self.class_log_prior_.item(c) + negative_sum)
        return result^

    def predict[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> Self.Target:
        var joint = self.predict_joint_log_proba(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> DenseMatrix:
        return log_normalize(self.predict_joint_log_proba(X))

    def predict_proba[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> DenseMatrix:
        return exp_normalize(self.predict_joint_log_proba(X))

    def score[input_dtype: DType](self, X: NDArray[input_dtype], y: Self.Target,
                           sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return weighted_accuracy[Self.D](y, self.predict(X), sample_weight)

    def predict_joint_log_proba[input_dtype: DType](
        self, X: ColumnTable[input_dtype],
    ) raises -> DenseMatrix:
        check_is_fitted(self)
        _check_feature_names(self.feature_names_in_, X.schema().names)
        return self.predict_joint_log_proba(X.values())

    def predict[input_dtype: DType](self, X: ColumnTable[input_dtype]) raises -> Self.Target:
        var joint = self.predict_joint_log_proba(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba[input_dtype: DType](self, X: ColumnTable[input_dtype]) raises -> DenseMatrix:
        return log_normalize(self.predict_joint_log_proba(X))

    def predict_proba[input_dtype: DType](self, X: ColumnTable[input_dtype]) raises -> DenseMatrix:
        return exp_normalize(self.predict_joint_log_proba(X))

    def score[input_dtype: DType](
        self, X: ColumnTable[input_dtype], y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        return weighted_accuracy[Self.D](y, self.predict(X), sample_weight)

    # Trait-typed NDArray[dtype] overloads; they delegate to the generic paths.
    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        _ = self.fit[input_dtype=Self.dtype](X, y, sample_weight=Optional[List[Float64]](None))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        _ = self.fit[input_dtype=Self.dtype](X, y, sample_weight=Optional[List[Float64]](sample_weight.copy()))
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises -> ref[self] Self:
        _ = self.partial_fit[input_dtype=Self.dtype](X, y, classes, sample_weight=Optional[List[Float64]](None))
        return self

    def predict(self, X: Self.Input) raises -> Self.Target:
        return self.predict[input_dtype=Self.dtype](X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict_log_proba[input_dtype=Self.dtype](X)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict_proba[input_dtype=Self.dtype](X)

    def score(self, X: Self.Input, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.score[input_dtype=Self.dtype](X, y, sample_weight)

    def predict_joint_log_proba(self, X: _Matrix) raises -> _Matrix:
        return probability_lists(self.predict_joint_log_proba(dense_from_lists(X)))

    def predict(self, X: _Matrix) raises -> Self.Target:
        return self.predict(dense_from_lists(X))

    def predict_log_proba(self, X: _Matrix) raises -> _Matrix:
        return probability_lists(self.predict_log_proba(dense_from_lists(X)))

    def predict_proba(self, X: _Matrix) raises -> _Matrix:
        return probability_lists(self.predict_proba(dense_from_lists(X)))

    def score(self, X: _Matrix, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.score(dense_from_lists(X), y, sample_weight)
