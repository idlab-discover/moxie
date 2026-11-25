# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Public dense histogram gradient boosting regression."""

from std.math import ceil, isfinite, sqrt

from numojo import NDArray, Shape, full

from moxie._random import RandomState
from moxie.base import WeightedSupervisedEstimator
from moxie.tree._classes import DecisionTreeRegressor

from .._base import EnsembleRegressor, StagedPredictor
from .._sampling import _at
from .binning import _BinMapper
from .common import (
    HistGradientBoostingCategoricalFeatures, _CategoryEncoder, _categorical_mask,
    _EarlyStopping, _FitRandomState, _Interactions, _ValidationSize, _Values,
    _check_settings, _check_warm_start, _resume_history,
    _history, _interaction_groups, _should_stop, _take_rows, _uses_early_stopping,
)
from .grower import grow_tree
from .losses import _RegressionLoss
from .predictor import _RawTree, make_raw_tree
from .splitting import _SplitOptions


@fieldwise_init
struct HistGradientBoostingRegressorParams(Copyable, Movable):
    var loss: String
    var quantile: Optional[Float64]
    var learning_rate: Float64
    var max_iter: Int
    var max_leaf_nodes: Optional[Int]
    var max_depth: Optional[Int]
    var min_samples_leaf: Int
    var l2_regularization: Float64
    var max_features: Float64
    var max_bins: Int
    var categorical_features: HistGradientBoostingCategoricalFeatures
    var monotonic_cst: Optional[List[Int]]
    var interaction_cst: Optional[_Interactions]
    var warm_start: Bool
    var early_stopping: _EarlyStopping
    var scoring: Optional[String]
    var validation_fraction: Optional[_ValidationSize]
    var n_iter_no_change: Int
    var tol: Float64
    var verbose: Int
    var random_state: Optional[Int]


def _add_prediction(mut raw: _Values, increment: _Values):
    for i in range(raw.size):
        raw.unsafe_set(i, raw.unsafe_get(i) + increment.unsafe_get(i))


def _validate_target(
    loss: _RegressionLoss, y: _Values, rows: Int, weights: Optional[_Values]
) raises:
    if y.ndim != 1 or y.size != rows:
        raise Error("ValueError: y must be a vector with one entry per sample")
    loss.validate_targets(y, weights)


def _linked(loss: _RegressionLoss, raw: _Values) raises -> _Values:
    var result = _Values(Shape(raw.size))
    for i in range(raw.size):
        result.unsafe_set(i, loss.inverse_link(raw.unsafe_get(i)))
    return result^


def _regression_score(
    scoring: String, loss: _RegressionLoss, y: _Values, raw: _Values,
    weights: Optional[_Values],
) raises -> Float64:
    if scoring == "loss":
        return -loss.mean_loss(y, raw, weights)
    if scoring == "r2":
        var prediction = _linked(loss, raw)
        for i in range(prediction.size):
            if not isfinite(prediction.unsafe_get(i)):
                raise Error("ValueError: nonfinite score prediction")
        var scorer = DecisionTreeRegressor()
        scorer.n_outputs_ = 1
        return scorer._score(prediction, y, weights)

    var total_weight = Float64(0)
    var total_error = Float64(0)
    var absolute = scoring == "neg_mean_absolute_error"
    for i in range(y.size):
        var prediction = loss.inverse_link(raw.unsafe_get(i))
        if not isfinite(prediction):
            raise Error("ValueError: nonfinite score prediction")
        var difference = prediction - _at(y, i)
        var error = abs(difference) if absolute else difference * difference
        var weight = _at(weights.value(), i) if weights else 1.0
        total_error += weight * error
        total_weight += weight
    if total_weight == 0.0:
        raise Error("ValueError: sample weights sum to zero")
    var mean_error = total_error / total_weight
    return -sqrt(mean_error) if scoring == "neg_root_mean_squared_error" else -mean_error


@fieldwise_init
struct _ScoringSubset(Movable):
    var indices: NDArray[DType.int64]
    var y: _Values
    var weights: Optional[_Values]

    @staticmethod
    def sample(y: _Values, weights: Optional[_Values], seed: Int) raises -> Self:
        var rng = RandomState(seed)
        var indices = rng.permutation(y.size)
        var subset_weights: Optional[_Values] = None
        if weights:
            subset_weights = _take_rows(weights.value(), indices, 0, 10_000)
        var subset_targets = _take_rows(y, indices, 0, 10_000)
        return Self(indices^, subset_targets^, subset_weights^)


def _training_score(
    scoring: String, loss: _RegressionLoss, y: _Values, raw: _Values,
    weights: Optional[_Values], subset: Optional[_ScoringSubset],
) raises -> Float64:
    if subset:
        ref sample = subset.value()
        var subset_raw = _take_rows(raw, sample.indices, 0, sample.y.size)
        return _regression_score(scoring, loss, sample.y, subset_raw, sample.weights)
    return _regression_score(scoring, loss, y, raw, weights)


@fieldwise_init
struct _Validation[dtype: DType](Copyable, Movable):
    var X: NDArray[Self.dtype]
    var y: _Values
    var weights: Optional[_Values]


@fieldwise_init
struct _FittedRegressor(Copyable, Movable):
    var loss: _RegressionLoss
    var baseline: Float64
    var trees: List[_RawTree]
    var n_features: Int
    var random: _FitRandomState
    var encoder: _CategoryEncoder


struct HistGradientBoostingRegressor[feature_dtype: DType = DType.float64](
    EnsembleRegressor, WeightedSupervisedEstimator, StagedPredictor,
):
    """Histogram gradient boosting for dense numerical features."""

    comptime Input = NDArray[Self.feature_dtype]
    comptime Target = _Values
    comptime SampleWeight = _Values
    comptime Prediction = _Values
    comptime Stages = List[Self.Prediction]
    comptime Params = HistGradientBoostingRegressorParams

    var _params: Self.Params
    var _model: Optional[_FittedRegressor]
    var n_features_in_: Int
    var n_iter_: Int
    var n_trees_per_iteration_: Int
    var do_early_stopping_: Bool
    var train_score_: _Values
    var validation_score_: _Values
    var is_categorical_: Optional[List[Bool]]

    def __init__(
        out self,
        var loss: String = "squared_error",
        *,
        quantile: Optional[Float64] = None,
        learning_rate: Float64 = 0.1,
        max_iter: Int = 100,
        max_leaf_nodes: Optional[Int] = 31,
        max_depth: Optional[Int] = None,
        min_samples_leaf: Int = 20,
        l2_regularization: Float64 = 0.0,
        max_features: Float64 = 1.0,
        max_bins: Int = 255,
        var categorical_features: HistGradientBoostingCategoricalFeatures = String("from_dtype"),
        var monotonic_cst: Optional[List[Int]] = None,
        var interaction_cst: Optional[_Interactions] = None,
        warm_start: Bool = False,
        var early_stopping: _EarlyStopping = String("auto"),
        var scoring: Optional[String] = String("loss"),
        var validation_fraction: Optional[_ValidationSize] = _ValidationSize(Float64(0.1)),
        n_iter_no_change: Int = 10,
        tol: Float64 = 1e-7,
        verbose: Int = 0,
        random_state: Optional[Int] = None,
    ) raises:
        self = Self(Self.Params(
            loss=loss^, quantile=quantile, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_features=max_features, max_bins=max_bins,
            categorical_features=categorical_features^, monotonic_cst=monotonic_cst^,
            interaction_cst=interaction_cst^, warm_start=warm_start, early_stopping=early_stopping^,
            scoring=scoring^, validation_fraction=validation_fraction^, n_iter_no_change=n_iter_no_change,
            tol=tol, verbose=verbose, random_state=random_state,
        ))

    def __init__(out self, var params: Self.Params) raises:
        comptime assert Self.feature_dtype in (DType.float32, DType.float64), "features must be Float32 or Float64"
        self._params = params^
        self._model = None
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.n_trees_per_iteration_ = 0
        self.do_early_stopping_ = False
        self.train_score_ = _Values(Shape(0))
        self.validation_score_ = _Values(Shape(0))
        self.is_categorical_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._model)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        return self.fit(X, y, None)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        return self.fit(X, y, Optional[_Values](sample_weight.copy()))

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
        *, var X_val: Optional[Self.Input] = None, var y_val: Optional[Self.Target] = None,
        var sample_weight_val: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        _BinMapper._check_matrix(X)
        ref params = self._params
        _check_settings(
            params.max_iter, params.max_depth, params.max_leaf_nodes, params.learning_rate,
            params.max_features, params.max_bins, params.n_iter_no_change, params.tol,
            params.verbose, params.validation_fraction,
        )
        if params.quantile and not (0.0 <= params.quantile.value() <= 1.0):
            raise Error("ValueError: quantile must be in [0, 1]")
        var loss = _RegressionLoss(self._params.loss, self._params.quantile)
        _validate_target(loss, y, X.shape[0], sample_weight)
        var early = _uses_early_stopping(params.early_stopping, X.shape[0])
        var scoring = params.scoring.or_else("r2")
        if early and scoring not in (
            "loss", "r2", "neg_mean_absolute_error", "neg_mean_squared_error",
            "neg_root_mean_squared_error",
        ):
            raise Error("NotImplementedError: unsupported histogram regression early-stopping scorer")
        if Bool(X_val) != Bool(y_val) or (sample_weight_val and not X_val):
            raise Error("ValueError: explicit validation requires both X_val and y_val")
        var validation: Optional[_Validation[Self.feature_dtype]] = None
        if X_val:
            if self._params.early_stopping.isa[Bool]() and not early:
                raise Error("ValueError: explicit validation requires early_stopping=True or auto")
            _BinMapper._check_matrix(X_val.value())
            if X_val.value().shape[1] != X.shape[1]:
                raise Error("ValueError: validation feature count must match training")
            _validate_target(loss, y_val.value(), X_val.value().shape[0], sample_weight_val)
            validation = _Validation[Self.feature_dtype](
                X_val.take(), y_val.take(), sample_weight_val^
            )

        var categorical = _categorical_mask(params.categorical_features, X.shape[1])
        var encoder = _CategoryEncoder(X, categorical, params.max_bins)
        var resume = params.warm_start and Bool(self._model)
        var random: _FitRandomState
        if resume:
            ref prior = self._model.value()
            _check_warm_start(params.max_iter, len(prior.trees), X.shape[1], prior.n_features)
            encoder.check_warm_start(prior.encoder)
            random = prior.random.copy()
        else:
            random = _FitRandomState.start(params.random_state)

        if early and self._params.validation_fraction and not validation:
            ref size = self._params.validation_fraction.value()
            var n = X.shape[0]
            var count = size[Int] if size.isa[Int]() else Int(ceil(Float64(n) * size[Float64]))
            if count >= n:
                raise Error("ValueError: validation split leaves no training samples")
            var split_rng = RandomState(random.split_seed)
            var indices = split_rng.permutation(n)
            var train_weights: Optional[_Values] = None
            var validation_weights: Optional[_Values] = None
            if sample_weight:
                train_weights = _take_rows(sample_weight.value(), indices, count, n)
                validation_weights = _take_rows(sample_weight.value(), indices, 0, count)
            validation = _Validation[Self.feature_dtype](
                _take_rows(X, indices, 0, count), _take_rows(y, indices, 0, count),
                validation_weights^,
            )
            self._fit(
                _take_rows(X, indices, count, n), _take_rows(y, indices, count, n),
                train_weights, validation, loss^, early, random^, resume, encoder^,
            )
        else:
            self._fit(X, y, sample_weight, validation, loss^, early, random^, resume, encoder^)
        return self

    def _fit(
        mut self, X_input: Self.Input, y: Self.Target, weights: Optional[_Values],
        validation: Optional[_Validation[Self.feature_dtype]], var loss: _RegressionLoss,
        early: Bool, var random: _FitRandomState, resume: Bool,
        var encoder: _CategoryEncoder,
    ) raises:
        ref params = self._params
        var baseline: Float64
        var trees = List[_RawTree]()
        var train_scores = List[Float64]()
        var validation_scores = List[Float64]()
        if resume:
            ref prior = self._model.value()
            baseline = prior.baseline
            trees = prior.trees.copy()
            train_scores = _resume_history(self.train_score_)
            validation_scores = _resume_history(self.validation_score_)
        else:
            baseline = loss.intercept(y, weights)
        var X = encoder.transform(X_input)
        var n_features = X.shape[1]
        var monotonic = encoder.monotonic_constraints(params.monotonic_cst, n_features)
        var interactions = _interaction_groups(params.interaction_cst, n_features)
        var n_bins = params.max_bins + 1
        var mapper = _BinMapper(n_bins=n_bins, random_state=random.split_seed)
        _ = mapper.fit(X, known_categories=encoder.known_categories(), sample_weight=weights)
        var binned = mapper.transform(X)
        var raw = Self._raw_predictions(X, baseline, trees)
        var raw_val = _Values(Shape(0))
        var binned_val: Optional[NDArray[DType.uint8]] = None
        var score_validation = early and Bool(validation)
        var scoring = params.scoring.or_else("r2")
        var scoring_subset: Optional[_ScoringSubset] = None
        if early and scoring != "loss" and y.size > 10_000:
            scoring_subset = _ScoringSubset.sample(y, weights, random.split_seed)
        if early and not resume:
            train_scores.append(_training_score(scoring, loss, y, raw, weights, scoring_subset))
        if validation:
            ref val_data = validation.value()
            var X_val = encoder.transform(val_data.X)
            if score_validation:
                binned_val = mapper.transform(X_val)
                raw_val = Self._raw_predictions(X_val, baseline, trees)
                if not resume:
                    validation_scores.append(_regression_score(scoring, loss, val_data.y, raw_val, val_data.weights))

        var gradients = List[Float32]()
        var hessians = List[Float32]()
        var options = _SplitOptions(
            min_samples_leaf=params.min_samples_leaf, l2_regularization=params.l2_regularization,
        )
        if params.verbose > 0:
            print("Fitting gradient boosted rounds:")
        for stage in range(len(trees), params.max_iter):
            loss.gradient_hessian(y, raw, weights, gradients, hessians)
            var grown = grow_tree(
                binned, n_bins, Span(gradients), Span(hessians), mapper.n_bins_non_missing_,
                mapper.is_categorical_, monotonic, feature_sampler=random.feature_sampler,
                feature_fraction_per_split=params.max_features, interaction_groups=interactions,
                max_leaf_nodes=params.max_leaf_nodes, max_depth=params.max_depth, options=options,
                shrinkage=params.learning_rate,
            )
            loss.update_leaves(grown, y, raw, weights)
            for leaf in grown.leaves:
                ref node = grown.tree.nodes[leaf]
                for position in range(node.start, node.stop):
                    var row = grown.partition[position]
                    raw.unsafe_set(row, raw.unsafe_get(row) + node.value)
            var tree = make_raw_tree(grown.tree, mapper)
            if binned_val:
                _add_prediction(raw_val, grown.tree.predict(binned_val.value()))
            trees.append(tree^)
            if params.verbose > 1:
                print("Boosting iteration", stage + 1, "of", params.max_iter)
            if early:
                train_scores.append(_training_score(scoring, loss, y, raw, weights, scoring_subset))
                if score_validation:
                    ref val_data = validation.value()
                    validation_scores.append(_regression_score(scoring, loss, val_data.y, raw_val, val_data.weights))
                ref scores = validation_scores if score_validation else train_scores
                if _should_stop(scores, params.n_iter_no_change, params.tol):
                    break

        var train_history = _history(train_scores)
        var validation_history = _history(validation_scores)
        self.n_features_in_ = n_features
        self.n_iter_ = len(trees)
        self.n_trees_per_iteration_ = 1
        self.do_early_stopping_ = early
        self.train_score_ = train_history^
        self.validation_score_ = validation_history^
        self.is_categorical_ = encoder.categorical_mask()
        self._model = _FittedRegressor(loss^, baseline, trees^, n_features, random^, encoder^)

    def _check_predict(self, X: Self.Input) raises:
        if not self._model:
            raise Error("NotFittedError: HistGradientBoostingRegressor must be fitted")
        _BinMapper._check_matrix(X)
        if X.shape[1] != self._model.value().n_features:
            raise Error("ValueError: prediction feature count must match training")

    @staticmethod
    def _raw_predictions(X: Self.Input, baseline: Float64, trees: List[_RawTree]) raises -> _Values:
        var raw = full[DType.float64](Shape(X.shape[0]), baseline)
        for tree in trees:
            _add_prediction(raw, tree.predict(X))
        return raw^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check_predict(X)
        ref model = self._model.value()
        return _linked(model.loss, Self._raw_predictions(model.encoder.transform(X), model.baseline, model.trees))

    def staged_predict(self, X: Self.Input) raises -> Self.Stages:
        self._check_predict(X)
        ref model = self._model.value()
        var raw = full[DType.float64](Shape(X.shape[0]), model.baseline)
        var stages = Self.Stages(capacity=len(model.trees))
        var encoded = model.encoder.transform(X)
        for tree in model.trees:
            _add_prediction(raw, tree.predict(encoded))
            stages.append(_linked(model.loss, raw))
        return stages^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        var scorer = DecisionTreeRegressor()
        scorer.n_outputs_ = 1
        return scorer._score(prediction, y, sample_weight)
