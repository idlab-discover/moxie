# Adapted from scikit-learn sklearn/impute/_iterative.py.
# Authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Dense round-robin imputation with independently owned fitted regressors."""

import numojo as nm
from std.math import abs, inf, isfinite, sqrt
from std.builtin.sort import sort
from std.utils import Variant
from moxie.base import Predictor, SupervisedEstimator
from moxie._feature_metadata import FeatureNamesOut
from moxie._random import RandomState, check_random_state
from moxie.linear_model._bayes import BayesianRidge
from moxie.impute._base import Imputer
from moxie.impute._simple import SimpleImputer
from moxie.impute._indicator import MissingIndicator
from moxie.impute._plumbing import (
    ImputationStep,
    MissingValue,
    append_indicator,
    missing_mask,
    promote_integer_matrix,
    select_features,
)

comptime IterativeRegressor = SupervisedEstimator & Predictor & Copyable & Deinitable
comptime IterativeBound = Variant[Float64, List[Float64]]


@fieldwise_init
struct IterativeParams[dtype: DType, R: IterativeRegressor](Copyable, Movable):
    var estimator: Optional[Self.R]
    var missing_values: MissingValue[Self.dtype]
    var sample_posterior: Bool
    var max_iter: Int
    var tol: Float64
    var n_nearest_features: Optional[Int]
    var initial_strategy: String
    var fill_value: Optional[SIMD[Self.dtype, 1]]
    var imputation_order: String
    var skip_complete: Bool
    var min_value: IterativeBound
    var max_value: IterativeBound
    var verbose: Int
    var random_state: Optional[Int]
    var add_indicator: Bool
    var keep_empty_features: Bool


@fieldwise_init
struct _Feature(Comparable, Copyable, Movable):
    var missing: Int
    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.missing == other.missing and self.index == other.index

    def __lt__(self, other: Self) -> Bool:
        return self.missing < other.missing or (
            self.missing == other.missing and self.index < other.index
        )


def _order(
    mask: nm.NDArray[DType.bool], order: String, skip_complete: Bool,
    mut rng: Optional[RandomState],
) raises -> List[Int]:
    var features = List[_Feature]()
    for j in range(mask.shape[1]):
        var count = 0
        for i in range(mask.shape[0]):
            count += Int(mask.unsafe_get(i * mask.shape[1] + j))
        if count > 0 or not skip_complete:
            features.append(_Feature(count, j))
    if order == "ascending" or order == "descending":
        sort(Span(features))
    var result = List[Int](capacity=len(features))
    if order == "random":
        var permutation: nm.NDArray[DType.int64]
        if rng:
            permutation = rng.value().permutation(len(features))
        else:
            var stream = check_random_state()
            permutation = stream[].permutation(len(features))
        for i in range(len(features)):
            result.append(features[Int(permutation.unsafe_get(i))].index)
    elif order == "descending" or order == "arabic":
        for i in range(len(features) - 1, -1, -1):
            result.append(features[i].index)
    else:
        for feature in features:
            result.append(feature.index)
    return result^


def _correlations[dtype: DType](
    X: nm.NDArray[dtype],
) raises -> nm.NDArray[DType.float64]:
    # Form a centered Gram matrix with the approved NuMojo rank-two product.
    # Rows of this workspace are features. Undefined correlations get the
    # reference's 1e-6 exploration weight, including constant features.
    var rows = X.shape[0]
    var width = X.shape[1]
    var centered = nm.zeros[DType.float64](nm.Shape(width, rows))
    for j in range(width):
        var mean = Float64(0)
        for i in range(rows):
            mean += Float64(X.unsafe_get(i * width + j))
        mean /= Float64(rows)
        for i in range(rows):
            centered.unsafe_set(j * rows + i, Float64(X.unsafe_get(i * width + j)) - mean)
    var gram = centered @ centered.T()
    var column_norms = List[Float64](capacity=width)
    for j in range(width):
        column_norms.append(sqrt(gram.unsafe_get(j * width + j)))
    for j in range(width):
        var total = Float64(0)
        for k in range(width):
            var offset = j * width + k
            if k == j:
                gram.unsafe_set(offset, 0)
                continue
            var scale = column_norms[j] * column_norms[k]
            var weight = Float64(1e-6)
            if scale > 0:
                var correlation = gram.unsafe_get(offset) / scale
                if isfinite(correlation):
                    weight = max(weight, min(1.0, abs(correlation)))
            gram.unsafe_set(offset, weight)
            total += weight
        for k in range(width):
            var offset = j * width + k
            gram.unsafe_set(offset, gram.unsafe_get(offset) / total)
    return gram^


def _neighbors(
    width: Int, feature: Int, count: Optional[Int],
    weights: nm.NDArray[DType.float64], mut rng: Optional[RandomState],
) raises -> List[Int]:
    var result = List[Int]()
    if count and count.value() < width:
        var p = nm.zeros[DType.float64](nm.Shape(width))
        for j in range(width):
            p.unsafe_set(j, weights.unsafe_get(feature * width + j))
        var draws: nm.NDArray[DType.int64]
        if rng:
            draws = rng.value().choice(width, p, size=count.value(), replace=False)
        else:
            var stream = check_random_state()
            draws = stream[].choice(width, p, size=count.value(), replace=False)
        for j in range(draws.size):
            result.append(Int(draws.unsafe_get(j)))
    else:
        for j in range(width):
            if j != feature:
                result.append(j)
    return result^


def _limit(
    bound: IterativeBound, n_features: Int, retained: List[Int],
) raises -> List[Float64]:
    var result = List[Float64](capacity=len(retained))
    if bound.isa[Float64]():
        for _ in retained:
            result.append(bound[Float64])
    else:
        if len(bound[List[Float64]]) != n_features:
            raise Error("ValueError: imputation bounds must match the input feature count")
        for j in retained:
            result.append(bound[List[Float64]][j])
    return result^


struct IterativeImputer[
    dtype: DType = DType.float64,
    R: IterativeRegressor = BayesianRidge[dtype],
](Imputer, FeatureNamesOut, Copyable, Movable):
    """Impute missing cells from regressions on the other features."""

    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = IterativeParams[Self.dtype, Self.R]
    comptime Step = ImputationStep[Self.R]

    var _params: Self.Params
    var initial_imputer_: SimpleImputer[Self.dtype]
    var imputation_sequence_: List[Self.Step]
    var n_iter_: Int
    var n_features_with_missing_: Int
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var indicator_: Optional[MissingIndicator[Self.dtype]]
    var random_state_: Optional[RandomState]
    var _min: List[Float64]
    var _max: List[Float64]

    def __init__(
        out self, estimator: Optional[Self.R] = None, *,
        missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](),
        sample_posterior: Bool = False,
        max_iter: Int = 10,
        tol: Float64 = 1e-3,
        n_nearest_features: Optional[Int] = None,
        initial_strategy: String = "mean",
        fill_value: Optional[SIMD[Self.dtype, 1]] = None,
        imputation_order: String = "ascending",
        skip_complete: Bool = False,
        min_value: IterativeBound = IterativeBound(-inf[DType.float64]()),
        max_value: IterativeBound = IterativeBound(inf[DType.float64]()),
        verbose: Int = 0,
        random_state: Optional[Int] = None,
        add_indicator: Bool = False,
        keep_empty_features: Bool = False,
    ) raises:
        self = Self(Self.Params(
            estimator.copy(), missing_values.copy(), sample_posterior, max_iter, tol,
            n_nearest_features, initial_strategy, fill_value, imputation_order,
            skip_complete, min_value.copy(), max_value.copy(), verbose, random_state,
            add_indicator, keep_empty_features,
        ))

    def __init__(out self, var params: Self.Params) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime assert (
            Self.R.Input == Self.Input
            and Self.R.Target == Self.Target
            and Self.R.Prediction == Self.Input
        ), "regressor must use matching dense arrays"
        self.initial_imputer_ = SimpleImputer[Self.dtype](
            missing_values=params.missing_values, strategy=params.initial_strategy,
            fill_value=params.fill_value,
            keep_empty_features=params.keep_empty_features,
        )
        if params.estimator:
            var estimator = params.estimator.value().__sklearn_clone__()
            params.estimator = estimator^
        self._params = params^
        self.imputation_sequence_ = List[Self.Step]()
        self.n_iter_ = 0
        self.n_features_with_missing_ = 0
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.indicator_ = None
        self.random_state_ = None
        self._min = List[Float64]()
        self._max = List[Float64]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self = Self(params.copy())
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        estimator: Optional[Optional[Self.R]] = None,
        missing_values: Optional[MissingValue[Self.dtype]] = None,
        sample_posterior: Optional[Bool] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        n_nearest_features: Optional[Optional[Int]] = None,
        initial_strategy: Optional[String] = None,
        fill_value: Optional[Optional[SIMD[Self.dtype, 1]]] = None,
        imputation_order: Optional[String] = None,
        skip_complete: Optional[Bool] = None,
        min_value: Optional[IterativeBound] = None,
        max_value: Optional[IterativeBound] = None,
        verbose: Optional[Int] = None,
        random_state: Optional[Optional[Int]] = None,
        add_indicator: Optional[Bool] = None,
        keep_empty_features: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        var params = self.get_params()
        if estimator:
            params.estimator = estimator.value().copy()
        if missing_values:
            params.missing_values = missing_values.value().copy()
        if sample_posterior:
            params.sample_posterior = sample_posterior.value()
        if max_iter:
            params.max_iter = max_iter.value()
        if tol:
            params.tol = tol.value()
        if n_nearest_features:
            params.n_nearest_features = n_nearest_features.value()
        if initial_strategy:
            params.initial_strategy = initial_strategy.value()
        if fill_value:
            params.fill_value = fill_value.value()
        if imputation_order:
            params.imputation_order = imputation_order.value()
        if skip_complete:
            params.skip_complete = skip_complete.value()
        if min_value:
            params.min_value = min_value.value().copy()
        if max_value:
            params.max_value = max_value.value().copy()
        if verbose:
            params.verbose = verbose.value()
        if random_state:
            params.random_state = random_state.value()
        if add_indicator:
            params.add_indicator = add_indicator.value()
        if keep_empty_features:
            params.keep_empty_features = keep_empty_features.value()
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        return Self(self._params.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ != 0

    def _validate_params(self) raises:
        if self._params.sample_posterior:
            raise Error("NotImplementedError: IterativeImputer sample_posterior=True")
        if self._params.max_iter < 0 or not isfinite(self._params.tol) or self._params.tol < 0:
            raise Error("ValueError: max_iter and finite tol must be nonnegative")
        if self._params.n_nearest_features and self._params.n_nearest_features.value() < 1:
            raise Error("ValueError: n_nearest_features must be positive")
        if self._params.verbose < 0:
            raise Error("ValueError: verbose must be nonnegative")
        var order = self._params.imputation_order
        if (
            order != "ascending" and order != "descending" and order != "roman"
            and order != "arabic" and order != "random"
        ):
            raise Error("ValueError: unknown imputation_order")
        var strategy = self._params.initial_strategy
        if (
            strategy != "mean" and strategy != "median"
            and strategy != "most_frequent" and strategy != "constant"
        ):
            raise Error("ValueError: unknown initial_strategy")
        comptime if Self.R != BayesianRidge[Self.dtype]:
            if not self._params.estimator:
                raise Error("ValueError: a custom regressor type requires an estimator")

    def _new_regressor(self) raises -> Self.R:
        if self._params.estimator:
            return self._params.estimator.value().__sklearn_clone__()
        comptime if Self.R == BayesianRidge[Self.dtype]:
            return rebind_var[Self.R](BayesianRidge[Self.dtype]())
        else:
            raise Error("ValueError: a custom regressor type requires an estimator")

    def _initial_mask(
        self, original_mask: nm.NDArray[DType.bool],
    ) raises -> nm.NDArray[DType.bool]:
        ref layout = self.initial_imputer_._layout
        var mask = select_features(original_mask, layout.retained_features)
        if self._params.keep_empty_features:
            # The initial imputer supplies fills for empty fitted features.
            # Exclude their missing cells from the iterative updates.
            for j in layout.empty_features:
                for i in range(mask.shape[0]):
                    mask.unsafe_set(i * mask.shape[1] + j, False)
        return mask^

    def _finish(
        self, var filled: Self.Input, original_mask: nm.NDArray[DType.bool],
    ) raises -> Self.Transformed:
        if self._params.add_indicator:
            return append_indicator(filled, original_mask, self.initial_imputer_._layout)
        return filled^

    def _fit_step(
        self, X: Self.Input, mask: nm.NDArray[DType.bool],
        feature: Int, neighbors: List[Int],
    ) raises -> Self.Step:
        var rows = List[Int]()
        for i in range(X.shape[0]):
            if not mask.unsafe_get(i * X.shape[1] + feature):
                rows.append(i)
        var train = Self._gather(X, rows, neighbors)
        var target = nm.zeros[Self.dtype](nm.Shape(len(rows)))
        for i in range(len(rows)):
            target.unsafe_set(i, X.unsafe_get(rows[i] * X.shape[1] + feature))
        var estimator = self._new_regressor()
        _ = estimator.fit(rebind[Self.R.Input](train), rebind[Self.R.Target](target))
        return Self.Step(feature, neighbors, estimator^, X.shape[1])

    @staticmethod
    def _gather(X: Self.Input, rows: List[Int], columns: List[Int]) raises -> Self.Input:
        var result = nm.zeros[Self.dtype](nm.Shape(len(rows), len(columns)))
        for i in range(len(rows)):
            for j in range(len(columns)):
                result.unsafe_set(
                    i * len(columns) + j, X.unsafe_get(rows[i] * X.shape[1] + columns[j]),
                )
        return result^

    def _apply_step(
        self, mut X: Self.Input, mask: nm.NDArray[DType.bool], step: Self.Step,
    ) raises:
        var feature = step.feat_idx
        var rows = List[Int]()
        for i in range(X.shape[0]):
            if mask.unsafe_get(i * X.shape[1] + feature):
                rows.append(i)
        if len(rows) == 0:
            return
        var query = Self._gather(X, rows, step.neighbor_feat_idx)
        var prediction = rebind_var[Self.Input](
            step.estimator.predict(rebind[Self.R.Input](query)),
        )
        if prediction.ndim != 1 or prediction.shape[0] != len(rows):
            raise Error("ValueError: iterative regressor must predict one value per row")
        for i in range(len(rows)):
            var value = Float64(prediction.unsafe_get(i * prediction.strides[0]))
            if not isfinite(value):
                raise Error("ValueError: iterative regressor predicted a nonfinite value")
            value = min(max(value, self._min[feature]), self._max[feature])
            X.unsafe_set(rows[i] * X.shape[1] + feature, value.cast[Self.dtype]())

    def _iterate(
        mut self, mut filled: Self.Input, mask: nm.NDArray[DType.bool], mut rng: Optional[RandomState],
    ) raises:
        var width = filled.shape[1]
        if self._params.max_iter == 0 or width <= 1:
            return
        ref retained = self.initial_imputer_._layout.retained_features
        self._min = _limit(self._params.min_value, self.n_features_in_, retained)
        self._max = _limit(self._params.max_value, self.n_features_in_, retained)
        for j in range(width):
            if not (self._max[j] > self._min[j]):
                raise Error("ValueError: min_value must be strictly below max_value")

        var order = _order(mask, self._params.imputation_order, self._params.skip_complete, rng)
        self.n_features_with_missing_ = len(order)
        var weights = nm.zeros[DType.float64](nm.Shape(0, 0))
        if self._params.n_nearest_features and self._params.n_nearest_features.value() < width:
            weights = _correlations(filled)
        var scale = Float64(0)
        for i in range(filled.size):
            if not mask.unsafe_get(i):
                scale = max(scale, abs(Float64(filled.unsafe_get(i))))
        var tolerance = self._params.tol * scale
        var previous = filled.copy()
        if self._params.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape", filled.shape)
        for iteration in range(self._params.max_iter):
            if self._params.imputation_order == "random":
                order = _order(mask, "random", self._params.skip_complete, rng)
            for feature in order:
                var neighbors = _neighbors(
                    width, feature, self._params.n_nearest_features, weights, rng,
                )
                var step = self._fit_step(filled, mask, feature, neighbors)
                self._apply_step(filled, mask, step)
                self.imputation_sequence_.append(step^)
            self.n_iter_ = iteration + 1
            # Matrix infinity norm: the largest absolute row sum.
            var change = Float64(0)
            for i in range(filled.shape[0]):
                var row_change = Float64(0)
                for j in range(width):
                    var offset = i * width + j
                    var updated = filled.unsafe_get(offset)
                    row_change += Float64(abs(updated - previous.unsafe_get(offset)))
                    previous.unsafe_set(offset, updated)
                change = max(change, row_change)
            if self._params.verbose > 1:
                print("[IterativeImputer] Ending imputation round", self.n_iter_, "/", self._params.max_iter)
            if self._params.verbose > 0:
                print("[IterativeImputer] Change:", change, "scaled tolerance:", tolerance)
            if change < tolerance:
                if self._params.verbose > 0:
                    print("[IterativeImputer] Early stopping criterion reached.")
                break

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self.fit_transform(X, y)
        return self

    def fit_named(
        mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None,
    ) raises:
        _ = self._fit_transform(X, Optional(feature_names.copy()))

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        return self._fit_transform(X, None)

    def _fit_transform(
        mut self, X: Self.Input, names: Optional[List[String]],
    ) raises -> Self.Transformed:
        self._validate_params()
        var next = Self(self._params.copy())
        if names:
            next.initial_imputer_.fit_named(X, names.value())
        else:
            _ = next.initial_imputer_.fit(X)
        next.feature_names_in_ = names.copy()
        next.n_features_in_ = X.shape[1]
        var original_mask = missing_mask(X, self._params.missing_values)
        if self._params.add_indicator:
            var indicator = MissingIndicator[Self.dtype](
                missing_values=self._params.missing_values, error_on_new=False,
            )
            indicator.fit_mask(original_mask)
            next.indicator_ = indicator^
        var filled = next.initial_imputer_.transform(X)
        var mask = next._initial_mask(original_mask)
        var rng: Optional[RandomState] = None
        if self._params.random_state:
            rng = (
                self.random_state_.value().copy() if self.random_state_
                else RandomState(self._params.random_state.value())
            )
        # Lease the default stream only around draws, so an inner estimator
        # can draw from that stream during its own fit or predict.
        next._iterate(filled, mask, rng)
        if not rng:
            var stream = check_random_state()
            rng = stream[].copy()
        next.random_state_ = rng^
        var result = next._finish(filled^, original_mask)
        self = next^
        return result^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: IterativeImputer")
        var original_mask = missing_mask(X, self._params.missing_values)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: input feature count differs from fitted data")
        var filled = self.initial_imputer_.transform(X)
        var mask = self._initial_mask(original_mask)
        if len(self.imputation_sequence_) > 0 and not nm.all(mask):
            for step in self.imputation_sequence_:
                self._apply_step(filled, mask, step)
        return self._finish(filled^, original_mask)

    @staticmethod
    def _promote_integer_input[input_dtype: DType](
        X: nm.NDArray[input_dtype],
    ) raises -> Self.Input:
        comptime assert Self.dtype == DType.float64, (
            "integer input requires IterativeImputer[DType.float64]"
        )
        return rebind_var[Self.Input](promote_integer_matrix(X))

    def fit[input_dtype: DType, origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: nm.NDArray[input_dtype],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self where input_dtype.is_integral():
        _ = self.fit_transform(X, y)
        return self

    def fit_named[input_dtype: DType](
        mut self, X: nm.NDArray[input_dtype], feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises where input_dtype.is_integral():
        self.fit_named(Self._promote_integer_input(X), feature_names, y)

    def fit_transform[input_dtype: DType](
        mut self, X: nm.NDArray[input_dtype], y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed where input_dtype.is_integral():
        return self.fit_transform(Self._promote_integer_input(X), y)

    def transform[input_dtype: DType](
        self, X: nm.NDArray[input_dtype],
    ) raises -> Self.Transformed where input_dtype.is_integral():
        return self.transform(Self._promote_integer_input(X))

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None,
    ) raises -> List[String]:
        var names = self.initial_imputer_.get_feature_names_out(input_features)
        if self.indicator_:
            var source_names = input_features.copy()
            if not source_names:
                source_names = self.feature_names_in_.copy()
            var indicators = self.indicator_.value().get_feature_names_out(source_names)
            for name in indicators:
                names.append(name)
        return names^
