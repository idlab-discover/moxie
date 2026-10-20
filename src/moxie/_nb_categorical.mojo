"""Native dense CategoricalNB; sklearn 4358542a1bd3 (BSD-3-Clause)."""
import numojo as nm
from numojo import NDArray, Shape, Item
from moxie._dense import sum_axis2d
from moxie._naive_bayes_common import _scalar_weights
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie._naive_bayes_common import NaiveBayesClassifier, _Matrix, _alpha, _log, _check_prior, _log_priors
from moxie._naive_bayes_dense import (
    ClassDomain, IntListDomain, DenseMatrix, _prepare, validate_matrix,
    dense_from_lists, probability_lists, predict_labels, log_normalize,
    exp_normalize, weighted_accuracy, CategoryMinimum,
)
struct CategoricalNBParams(Copyable, Movable):
    var alpha: Float64
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var min_categories: Optional[Int]
    var min_categories_values: Optional[List[Int]]


    def __init__(
        out self,
        alpha: Float64,
        force_alpha: Bool,
        fit_prior: Bool,
        var class_prior: Optional[List[Float64]],
        min_categories: Optional[Int],
        var min_categories_values: Optional[List[Int]] = None,
    ):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.min_categories = min_categories
        self.min_categories_values = min_categories_values^





def _validate_categories[dtype: DType](X: NDArray[dtype], expected: Int = 0) raises:
    _ = validate_matrix(X, expected)
    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            var value = Float64(X.item(i, f))
            # sklearn converts to signed integers before checking nonnegativity.
            # Int truncates toward zero, so (-1, 0) maps to category zero.
            # Check the representable index range before converting to Int.
            if value >= 9223372036854775808.0 or value <= -1.0:
                raise Error("ValueError: converted categories must be nonnegative representable integers")


struct CategoricalNB[D: ClassDomain = IntListDomain](Copyable, WeightedSupervisedEstimator, NaiveBayesClassifier):
    """Dense CategoricalNB with weighted counts and incremental fitting."""

    comptime Params = CategoricalNBParams
    comptime Input = _Matrix
    comptime Target = Self.D.Target
    comptime Prediction = Self.Target
    comptime Probabilities = _Matrix
    comptime SampleWeight = List[Float64]
    comptime Classes = Self.Target

    var alpha: Float64
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var min_categories: Optional[Int]
    var min_categories_values: Optional[List[Int]]
    var classes_: Self.Target
    var _domain: Optional[Self.D]
    var class_count_: DenseMatrix
    var class_log_prior_: List[Float64]
    var n_features_in_: Int
    var category_count_: List[DenseMatrix]
    var feature_log_prob_: List[DenseMatrix]
    var n_categories_: List[Int]

    def __init__(
        out self,
        *,
        alpha: Float64 = 1.0,
        force_alpha: Bool = True,
        fit_prior: Bool = True,
        var class_prior: Optional[List[Float64]] = None,
        min_categories: Optional[Int] = None,
        var min_categories_values: Optional[List[Int]] = None,
    ) raises:
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior^
        self.min_categories = min_categories
        self.min_categories_values = min_categories_values^
        self.classes_ = Self.D.empty_target()
        self._domain = None
        self.class_count_ = nm.zeros[DType.float64](Shape(0))
        self.class_log_prior_ = List[Float64]()
        self.n_features_in_ = 0
        self.category_count_ = List[DenseMatrix]()
        self.feature_log_prob_ = List[DenseMatrix]()
        self.n_categories_ = List[Int]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior.copy(),
            self.min_categories,
            self.min_categories_values.copy(),
        )

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.alpha = params.alpha
        self.force_alpha = params.force_alpha
        self.fit_prior = params.fit_prior
        self.class_prior = params.class_prior.copy()
        self.min_categories = params.min_categories
        self.min_categories_values = params.min_categories_values.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        alpha: Optional[Float64] = None,
        force_alpha: Optional[Bool] = None,
        fit_prior: Optional[Bool] = None,
        class_prior: Optional[Optional[List[Float64]]] = None,
        min_categories: Optional[Optional[Int]] = None,
        min_categories_values: Optional[Optional[List[Int]]] = None,
    ) raises -> ref[self] Self:
        if alpha:
            self.alpha = alpha.value()
        if force_alpha:
            self.force_alpha = force_alpha.value()
        if fit_prior:
            self.fit_prior = fit_prior.value()
        if class_prior:
            self.class_prior = class_prior.value().copy()
        if min_categories:
            self.min_categories = min_categories.value()
            self.min_categories_values = None
        if min_categories_values:
            self.min_categories_values = min_categories_values.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            alpha=self.alpha,
            force_alpha=self.force_alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior.copy(),
            min_categories=self.min_categories,
            min_categories_values=self.min_categories_values.copy(),
        )

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
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[dtype],
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
        ref[x_origin] X: NDArray[dtype],
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
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[dtype],
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var fresh = self.__sklearn_clone__()
        fresh._update(X, y, y.copy(), sample_weight)
        self = fresh^
        return self

    def partial_fit[
        dtype: DType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[dtype],
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> ref[self] Self:
        var updated = self.copy()
        updated._update(X, y, classes, sample_weight)
        self = updated^
        return self

    def _update[dtype: DType](
        mut self, X: NDArray[dtype], y: Self.Target,
        classes: Optional[Self.Classes], sample_weight: Optional[List[Float64]],
    ) raises:
        _validate_categories(X, self.n_features_in_)
        var prepared = _prepare[dtype, Self.D](
            X, y, self._domain, classes, sample_weight, self.n_features_in_, False
        )
        ref batch = prepared.batch
        var alpha = _alpha(self.alpha, self.force_alpha)
        _check_prior(self.class_prior, batch.n_classes)
        var minimum = CategoryMinimum(0, self.min_categories_values.copy())
        if self.min_categories:
            minimum.scalar = self.min_categories.value()
        minimum.validate(batch.n_features)
        if not self.min_categories_values and self.min_categories:
            if self.min_categories.value() < 1:
                raise Error("ValueError: min_categories must be positive")
        if not self.__sklearn_is_fitted__():
            self.classes_ = prepared.domain.class_values()
            self._domain = prepared.domain.copy()
            self.n_features_in_ = batch.n_features
            self.class_count_ = nm.zeros[DType.float64](Shape(batch.n_classes))
            for _ in range(batch.n_features):
                self.category_count_.append(nm.zeros[DType.float64](Shape(batch.n_classes, 0)))
                self.n_categories_.append(0)
        for f in range(batch.n_features):
            var categories = minimum.scalar
            if minimum.values:
                var index = 0 if len(minimum.values.value()) == 1 else f
                categories = minimum.values.value()[index]
            for i in range(X.shape[0]):
                categories = max(categories, Int(X.item(i, f)) + 1)
            # Per-batch metadata can shrink, but learned histogram columns cannot.
            self.n_categories_[f] = categories
            var previous = self.category_count_[f].shape[1]
            if categories > previous:
                var grown = nm.zeros[DType.float64](Shape(batch.n_classes, categories))
                for c in range(batch.n_classes):
                    for k in range(previous):
                        grown[Item(c, k)] = self.category_count_[f].item(c, k)
                self.category_count_[f] = grown^
        for i in range(X.shape[0]):
            var c = Int(batch.targets.item(i))
            self.class_count_[Item(c)] = self.class_count_.item(c) + batch.weights[i]
            for f in range(batch.n_features):
                var k = Int(X.item(i, f))
                self.category_count_[f][Item(c, k)] = self.category_count_[f].item(c, k) + batch.weights[i]
        self.feature_log_prob_ = List[DenseMatrix]()
        for f in range(batch.n_features):
            var smoothed = self.category_count_[f].copy()
            for c in range(batch.n_classes):
                for k in range(smoothed.shape[1]):
                    smoothed[Item(c, k)] = smoothed.item(c, k) + alpha
            var totals = sum_axis2d(smoothed, 1)
            for c in range(batch.n_classes):
                for k in range(smoothed.shape[1]):
                    smoothed[Item(c, k)] = _log(smoothed.item(c, k)) - _log(totals.item(c))
            self.feature_log_prob_.append(smoothed^)
        var counts = List[Float64]()
        for c in range(batch.n_classes):
            counts.append(self.class_count_.item(c))
        self.class_log_prior_ = _log_priors(counts, self.class_prior, self.fit_prior)

    def predict_joint_log_proba[dtype: DType](self, X: NDArray[dtype]) raises -> DenseMatrix:
        check_is_fitted(self)
        _validate_categories(X, self.n_features_in_)
        var result = nm.zeros[DType.float64](Shape(X.shape[0], self.class_count_.size))
        for i in range(X.shape[0]):
            for c in range(self.class_count_.size):
                var joint = Float64(0)
                for f in range(self.n_features_in_):
                    var k = Int(X.item(i, f))
                    if k >= self.feature_log_prob_[f].shape[1]:
                        raise Error("ValueError: category was not allocated during fit")
                    joint += self.feature_log_prob_[f].item(c, k)
                result[Item(i, c)] = joint + self.class_log_prior_[c]
        return result^

    def predict[dtype: DType](self, X: NDArray[dtype]) raises -> Self.Target:
        var joint = self.predict_joint_log_proba(X)
        return predict_labels(self._domain.value(), joint)

    def predict_log_proba[dtype: DType](self, X: NDArray[dtype]) raises -> DenseMatrix:
        return log_normalize(self.predict_joint_log_proba(X))

    def predict_proba[dtype: DType](self, X: NDArray[dtype]) raises -> DenseMatrix:
        return exp_normalize(self.predict_joint_log_proba(X))

    def score[dtype: DType](self, X: NDArray[dtype], y: Self.Target,
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
