# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Dense selection by fitted model importances; no model optimization lives here."""
import numojo as nm
from std.utils import Variant
from std.math import isnan, isfinite, inf, nan, floor
from moxie.base import FitTransformer, IncrementalTransformer
from ._base import UnsupervisedSelector
from ._selection import SelectionState, SupportResult
from ._model_importance import SelectionEstimator, IncrementalSelectionEstimator, aggregate_importances

comptime ModelThreshold = Variant[Float64, String]

trait FeatureLimitPolicy(Copyable, Movable, Deinitable):
    """Typed replacement for None, integer or callable max_features."""
    def __init__(out self):
        ...
    def enabled(self) -> Bool:
        return True
    def without_fit(self) raises -> Int:
        raise Error("NotFittedError: callable max_features requires fit")
    def resolve[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Int:
        ...

struct MaxFeatures(FeatureLimitPolicy):
    var count: Optional[Int]
    def __init__(out self):
        self.count = None
    def __init__(out self, count: Int):
        self.count = count
    def enabled(self) -> Bool:
        return Bool(self.count)
    def without_fit(self) raises -> Int:
        if not self.count or self.count.value() < 0:
            raise Error("ValueError: max_features must be nonnegative")
        return self.count.value()
    def resolve[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Int:
        return self.without_fit()


def importance_threshold[dtype: DType](scores: nm.NDArray[dtype], threshold: Optional[ModelThreshold], l1_default: Bool) raises -> Float64:
    if not threshold and l1_default:
        return 1e-5
    var reference = String("mean")
    var scale = Float64(1)
    if threshold:
        ref value = threshold.value()
        if value.isa[Float64]():
            return value[Float64]
        reference = value[String].copy()
        var parts = reference.split("*")
        if len(parts) == 2:
            scale = Float64(String(parts[0]).strip())
            reference = String(String(parts[1]).strip())
        elif len(parts) != 1:
            raise Error("ValueError: invalid threshold expression")
    if reference == "mean":
        return scale * Float64(nm.sum(scores) / SIMD[dtype, 1](scores.shape[0]))
    if reference != "median":
        raise Error("ValueError: threshold must be numeric, mean, median or a scaled mean/median")
    for j in range(scores.shape[0]):
        if isnan(scores.unsafe_get(j * Int(scores.strides[0]))):
            return nan[DType.float64]()
    var ordered = nm.sort(scores)
    var n = ordered.shape[0]
    if n % 2 == 1:
        return scale * Float64(ordered.unsafe_get(n // 2))
    return scale * Float64((ordered.unsafe_get(n // 2 - 1) + ordered.unsafe_get(n // 2)) / 2)


def importance_mask[dtype: DType](scores: nm.NDArray[dtype], threshold: Float64, max_features: Optional[Int]) raises -> List[Bool]:
    var n = scores.shape[0]
    var mask = List[Bool]()
    for _ in range(n):
        mask.append(not max_features)
    if max_features:
        var k = min(max_features.value(), n)
        if k < 0:
            raise Error("ValueError: max_features must be nonnegative")
        if k > 0:
            var clean = nm.zeros[dtype](nm.Shape(n))
            for j in range(n):
                var value = scores.unsafe_get(j * Int(scores.strides[0]))
                clean.unsafe_set(j, -inf[dtype]() if isnan(value) else value)
            var ordered = nm.sort(clean)
            var cutoff = ordered.unsafe_get(n - k)
            var remaining = k
            for j in range(n):
                if clean.unsafe_get(j) > cutoff:
                    mask[j] = True
                    remaining -= 1
            # Stable ties: original columns, with NaNs after all numeric values.
            for pass_id in range(2):
                for j in range(n):
                    var value = scores.unsafe_get(j * Int(scores.strides[0]))
                    if remaining > 0 and clean.unsafe_get(j) == cutoff and isnan(value) == (pass_id == 1):
                        mask[j] = True
                        remaining -= 1
    for j in range(n):
        if scores.unsafe_get(j * Int(scores.strides[0])) < SIMD[dtype, 1](threshold):
            mask[j] = False
    return mask^

@fieldwise_init
struct SelectFromModelParams[E: SelectionEstimator, L: FeatureLimitPolicy](Copyable, Movable):
    var estimator: Self.E
    var threshold: Optional[ModelThreshold]
    var prefit: Bool
    var norm_order: Float64
    var max_features: Self.L
    var importance_getter: String

struct SelectFromModel[E: SelectionEstimator, L: FeatureLimitPolicy = MaxFeatures](UnsupervisedSelector, FitTransformer, IncrementalTransformer where conforms_to(E, IncrementalSelectionEstimator)):
    comptime Input = nm.NDArray[Self.E.dtype]
    comptime Target = Self.E.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectFromModelParams[Self.E, Self.L]
    comptime PartialFitParams where conforms_to(Self.E, IncrementalSelectionEstimator) = Self.E.PartialFitParams
    var estimator: Self.E
    var threshold: Optional[ModelThreshold]
    var prefit: Bool
    var norm_order: Float64
    var max_features: Self.L
    var importance_getter: String
    var _estimator: Optional[Self.E]
    var _max_features: Optional[Int]
    var _names: List[String]

    def __init__(out self, var estimator: Self.E, *, var threshold: Optional[ModelThreshold] = None, prefit: Bool = False, norm_order: Float64 = 1, var max_features: Self.L = Self.L(), var importance_getter: String = "auto"):
        self.estimator = estimator^
        self.threshold = threshold^
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features^
        self.importance_getter = importance_getter^
        self._estimator = None
        self._max_features = None
        self._names = List[String]()

    def _validate_params(self) raises:
        if self.threshold:
            if self.threshold.value().isa[Float64]() and isnan(self.threshold.value()[Float64]):
                raise Error("ValueError: threshold cannot be NaN")
        if isnan(self.norm_order) or self.norm_order == 0 or (isfinite(self.norm_order) and self.norm_order != floor(self.norm_order)):
            raise Error("ValueError: norm_order must be a nonzero integer or infinity")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target], *, ref[sample_weight_origin] sample_weight: Self.Input) raises -> ref[self] Self:
        self.fit_named(X, y, List[String](), sample_weight=sample_weight.copy())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String], sample_weight: Optional[Self.Input] = None) raises:
        self._validate_params()
        var limit: Optional[Int] = None
        if self.max_features.enabled():
            var count = self.max_features.resolve(X)
            if count < 0:
                raise Error("ValueError: max_features must be nonnegative")
            limit = count
        var fitted: Self.E
        if self.prefit:
            if not self.estimator.__sklearn_is_fitted__():
                raise Error("NotFittedError: prefit=True requires a fitted estimator")
            fitted = self.estimator.copy()
        else:
            fitted = self.estimator.__sklearn_clone__()
            fitted.fit_selection(X, y, sample_weight)
        if len(names) != 0 and len(names) != fitted.n_features():
            raise Error("ValueError: feature names length mismatch")
        self._estimator = fitted^
        self._max_features = limit
        self._names = names.copy()

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self where conforms_to(Self.E, IncrementalSelectionEstimator):
        self.partial_fit_named(X, y, List[String]())
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, partial_fit_params: Self.PartialFitParams[]) raises -> ref[self] Self where conforms_to(Self.E, IncrementalSelectionEstimator):
        self.partial_fit_named(X, y, List[String](), partial_fit_params=partial_fit_params.copy())
        return self

    def partial_fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String], *, partial_fit_params: Optional[Self.PartialFitParams[]] = None) raises where conforms_to(Self.E, IncrementalSelectionEstimator):
        self._validate_params()
        var first_call = not self._estimator
        var limit = self._max_features
        if first_call and self.max_features.enabled():
            var count = self.max_features.resolve(X)
            if count < 0:
                raise Error("ValueError: max_features must be nonnegative")
            limit = count
        if self.prefit:
            if first_call:
                if not self.estimator.__sklearn_is_fitted__():
                    raise Error("NotFittedError: prefit=True requires a fitted estimator")
                self._estimator = self.estimator.copy()
                self._max_features = limit
            # The reference does not train or update feature names in this branch.
            return
        # Copy learned state for transactional updates; only the first call clones.
        var updated = self.estimator.__sklearn_clone__() if first_call else self._estimator.value().copy()
        updated.partial_fit_selection(X, y, partial_fit_params)
        if len(names) != 0 and len(names) != updated.n_features():
            raise Error("ValueError: feature names length mismatch")
        if not first_call and len(names) != 0 and len(self._names) != 0:
            if names != self._names:
                raise Error("ValueError: feature names must match the first fit in order")
        self._estimator = updated^
        self._max_features = limit
        if first_call:
            self._names = names.copy()

    def _check(self) raises:
        if self.prefit and not self.estimator.__sklearn_is_fitted__():
            raise Error("NotFittedError: prefit=True requires a fitted estimator")
        if not self._estimator and not self.prefit:
            raise Error("NotFittedError: SelectFromModel requires fit")

    def _scores(self) raises -> nm.NDArray[Self.E.dtype]:
        self._check()
        if self._estimator:
            return aggregate_importances(self._estimator.value().importances(self.importance_getter), self.norm_order)
        return aggregate_importances(self.estimator.importances(self.importance_getter), self.norm_order)

    def _selected(self) raises -> SelectionState:
        self._validate_params()
        var scores = self._scores()
        var features = self.estimator.n_features() if not self._estimator else self._estimator.value().n_features()
        if scores.shape[0] != features:
            raise Error("ValueError: importance count does not match fitted feature count")
        var l1 = self.estimator.uses_l1_threshold() if not self._estimator else self._estimator.value().uses_l1_threshold()
        var threshold = importance_threshold(scores, self.threshold, l1)
        var limit: Optional[Int] = None
        if self.max_features.enabled():
            limit = self._max_features if self._max_features else Optional(self.max_features.without_fit())
        var selected = SelectionState()
        selected.commit(importance_mask(scores, threshold, limit), self._names)
        return selected^

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._selected().get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        # Like SelectorMixin, allow_nan also disables infinity checking.
        var allow_nan = self.estimator.allows_nan()
        if not allow_nan:
            if X.ndim != 2:
                raise Error("ValueError: expected rank-2 input")
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if not isfinite(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))):
                        raise Error("ValueError: X contains NaN or infinity")
        return self._selected().transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        if X.ndim != 2:
            raise Error("ValueError: expected rank-2 input")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not isfinite(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))):
                    raise Error("ValueError: inverse input must be finite")
        return self._selected().inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._selected().get_feature_names_out(input_features)

    def estimator_(self) raises -> Self.E:
        if not self._estimator:
            raise Error("NotFittedError: estimator_ requires fit even with prefit=True")
        return self._estimator.value().copy()

    def n_features_in_(self) raises -> Int:
        if not self._estimator:
            raise Error("NotFittedError: n_features_in_ requires fit")
        return self._estimator.value().n_features()

    def feature_names_in_(self) raises -> List[String]:
        _ = self.n_features_in_()
        return self._names.copy()

    def max_features_(self) raises -> Int:
        if not self._estimator or not self._max_features:
            raise Error("NotFittedError: max_features_ requires fit with a limit")
        return self._max_features.value()

    def threshold_(self) raises -> Float64:
        if not self._estimator:
            raise Error("NotFittedError: threshold_ requires fit")
        var scores = aggregate_importances(self._estimator.value().importances(self.importance_getter), self.norm_order)
        return importance_threshold(scores, self.threshold, self.estimator.uses_l1_threshold())

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self.estimator.copy(), self.threshold.copy(), self.prefit, self.norm_order, self.max_features.copy(), self.importance_getter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator = params.estimator.copy()
        self.threshold = params.threshold.copy()
        self.prefit = params.prefit
        self.norm_order = params.norm_order
        self.max_features = params.max_features.copy()
        self.importance_getter = params.importance_getter.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, threshold: Optional[ModelThreshold]) raises -> ref[self] Self:
        self.threshold = threshold.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__(), threshold=self.threshold.copy(), prefit=self.prefit, norm_order=self.norm_order, max_features=self.max_features.copy(), importance_getter=self.importance_getter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._estimator)
