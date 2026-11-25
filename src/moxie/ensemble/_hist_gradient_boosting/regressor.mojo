# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Public dense histogram gradient boosting regression."""

from std.math import ceil, isfinite
from std.utils import Variant

from numojo import NDArray, Shape, full

from moxie._random import RandomState, check_random_state
from moxie.base import WeightedSupervisedEstimator
from moxie.tree._classes import DecisionTreeRegressor

from .._base import EnsembleRegressor, StagedPredictor
from .binning import _BinMapper
from .feature_sampling import _FeatureSampler
from .grower import grow_tree
from .losses import _RegressionLoss
from .predictor import _RawTree, make_raw_tree
from .splitting import _SplitOptions


comptime _Values = NDArray[DType.float64]
comptime _EarlyStopping = Variant[Bool, String]
comptime _ValidationSize = Variant[Int, Float64]
comptime _Interactions = Variant[String, List[List[Int]]]


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
    var categorical_features: Optional[String]
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

    def _uses_early_stopping(self, n_samples: Int) raises -> Bool:
        if self.early_stopping.isa[Bool]():
            return self.early_stopping[Bool]
        if self.early_stopping[String] != "auto":
            raise Error("ValueError: early_stopping must be true, false, or auto")
        return n_samples > 10_000

    def _check(self) raises:
        if self.max_iter < 1 or self.max_depth.or_else(1) < 1 or self.max_leaf_nodes.or_else(2) < 2:
            raise Error("ValueError: invalid iteration, depth, or leaf limit")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise Error("ValueError: learning_rate must be positive and finite")
        if not (0.0 < self.max_features <= 1.0) or not (2 <= self.max_bins <= 255):
            raise Error("ValueError: max_features must be in (0, 1] and max_bins in [2, 255]")
        if self.quantile and not (0.0 <= self.quantile.value() <= 1.0):
            raise Error("ValueError: quantile must be in [0, 1]")
        if self.n_iter_no_change < 1 or not isfinite(self.tol) or self.tol < 0 or self.verbose < 0:
            raise Error("ValueError: invalid early stopping tolerance, patience, or verbosity")
        if self.validation_fraction:
            ref size = self.validation_fraction.value()
            if size.isa[Int]():
                if size[Int] < 1:
                    raise Error("ValueError: validation row count must be positive")
            elif not (0.0 < size[Float64] < 1.0):
                raise Error("ValueError: validation fraction must be in (0, 1)")
        if self.categorical_features and self.categorical_features.value() != "from_dtype":
            raise Error("ValueError: categorical_features must be from_dtype or None on numeric arrays")

    def _interaction_groups(self, n_features: Int) raises -> List[List[Int]]:
        var groups = List[List[Int]]()
        if not self.interaction_cst:
            return groups^
        ref constraints = self.interaction_cst.value()
        if constraints.isa[String]():
            if constraints[String] == "no_interactions":
                for i in range(n_features):
                    groups.append([i])
            elif constraints[String] == "pairwise":
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        groups.append([i, j])
            else:
                raise Error("ValueError: unknown interaction constraint")
            return groups^
        groups = constraints[List[List[Int]]].copy()
        var listed = List[Bool](length=n_features, fill=False)
        for group in groups:
            for feature in group:
                if not (0 <= feature < n_features):
                    raise Error("ValueError: interaction feature index is out of range")
                listed[feature] = True
        var remaining = List[Int]()
        for i in range(n_features):
            if not listed[i]:
                remaining.append(i)
        if remaining:
            groups.append(remaining^)
        return groups^


def _add_prediction(mut raw: _Values, increment: _Values):
    for i in range(raw.size):
        raw.unsafe_set(i, raw.unsafe_get(i) + increment.unsafe_get(i))


def _take_rows[dtype: DType](
    data: NDArray[dtype], indices: NDArray[DType.int64], start: Int, stop: Int
) raises -> NDArray[dtype]:
    var count = stop - start
    if data.ndim == 1:
        var vector = NDArray[dtype](Shape(count))
        for i in range(count):
            var row = Int(indices.unsafe_get(start + i))
            vector.unsafe_set(i, data.unsafe_get(row * data.strides[0]))
        return vector^

    var columns = data.shape[1]
    var matrix = NDArray[dtype](Shape(count, columns))
    for i in range(count):
        var row = Int(indices.unsafe_get(start + i))
        for j in range(columns):
            matrix.unsafe_set(
                i * columns + j, data.unsafe_get(row * data.strides[0] + j * data.strides[1])
            )
    return matrix^


def _validate_target(
    loss: _RegressionLoss, y: _Values, rows: Int, weights: Optional[_Values]
) raises:
    if y.ndim != 1 or y.size != rows:
        raise Error("ValueError: y must be a vector with one entry per sample")
    loss.validate_targets(y, weights)


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
        var categorical_features: Optional[String] = String("from_dtype"),
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
        self._params._check()
        var loss = _RegressionLoss(self._params.loss, self._params.quantile)
        _validate_target(loss, y, X.shape[0], sample_weight)
        if self._params.warm_start and self._model:
            raise Error("NotImplementedError: histogram regression warm starts are pending")
        var early = self._params._uses_early_stopping(X.shape[0])
        if early and self._params.scoring.or_else("") != "loss":
            raise Error("NotImplementedError: histogram regression early stopping supports scoring=loss")
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

        var seed: Int
        var feature_seed: UInt32
        with check_random_state(self._params.random_state) as rng:
            seed = rng[].randint(4294967295)
            feature_seed = UInt32(rng[].randint(4294967295))

        if early and self._params.validation_fraction and not validation:
            ref size = self._params.validation_fraction.value()
            var n = X.shape[0]
            var count = size[Int] if size.isa[Int]() else Int(ceil(Float64(n) * size[Float64]))
            if count >= n:
                raise Error("ValueError: validation split leaves no training samples")
            var split_rng = RandomState(seed)
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
                train_weights, validation, loss^, early, seed, feature_seed,
            )
        else:
            self._fit(X, y, sample_weight, validation, loss^, early, seed, feature_seed)
        return self

    def _fit(
        mut self, X: Self.Input, y: Self.Target, weights: Optional[_Values],
        validation: Optional[_Validation[Self.feature_dtype]], var loss: _RegressionLoss,
        early: Bool, seed: Int, feature_seed: UInt32,
    ) raises:
        ref params = self._params
        var baseline = loss.intercept(y, weights)
        var n_features = X.shape[1]
        var monotonic = params.monotonic_cst.value().copy() if params.monotonic_cst else List[Int](length=n_features, fill=0)
        var interactions = params._interaction_groups(n_features)
        var n_bins = params.max_bins + 1
        var mapper = _BinMapper(n_bins=n_bins, random_state=seed)
        _ = mapper.fit(X, sample_weight=weights)
        var binned = mapper.transform(X)
        var raw = full[DType.float64](Shape(y.size), baseline)
        var raw_val = _Values(Shape(0))
        var binned_val: Optional[NDArray[DType.uint8]] = None
        var train_scores = List[Float64]()
        var validation_scores = List[Float64]()
        var score_validation = early and Bool(validation)
        if early:
            train_scores.append(-loss.mean_loss(y, raw, weights))
            if score_validation:
                ref val_data = validation.value()
                binned_val = mapper.transform(val_data.X)
                raw_val = full[DType.float64](Shape(val_data.y.size), baseline)
                validation_scores.append(-loss.mean_loss(val_data.y, raw_val, val_data.weights))

        var feature_sampler = _FeatureSampler(feature_seed)
        var gradients = List[Float32]()
        var hessians = List[Float32]()
        var trees = List[_RawTree]()
        var options = _SplitOptions(
            min_samples_leaf=params.min_samples_leaf, l2_regularization=params.l2_regularization,
        )
        if params.verbose > 0:
            print("Fitting gradient boosted rounds:")
        for stage in range(params.max_iter):
            loss.gradient_hessian(y, raw, weights, gradients, hessians)
            var grown = grow_tree(
                binned, n_bins, Span(gradients), Span(hessians), mapper.n_bins_non_missing_,
                mapper.is_categorical_, monotonic, feature_sampler=feature_sampler,
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
                train_scores.append(-loss.mean_loss(y, raw, weights))
                if score_validation:
                    ref val_data = validation.value()
                    validation_scores.append(-loss.mean_loss(val_data.y, raw_val, val_data.weights))
                ref scores = validation_scores if score_validation else train_scores
                if Self._should_stop(scores, params.n_iter_no_change, params.tol):
                    break

        var train_history = Self._history(train_scores)
        var validation_history = Self._history(validation_scores)
        self.n_features_in_ = n_features
        self.n_iter_ = len(trees)
        self.n_trees_per_iteration_ = 1
        self.do_early_stopping_ = early
        self.train_score_ = train_history^
        self.validation_score_ = validation_history^
        self._model = _FittedRegressor(loss^, baseline, trees^, n_features)

    @staticmethod
    def _should_stop(scores: List[Float64], patience: Int, tol: Float64) -> Bool:
        if len(scores) <= patience:
            return False
        var start = len(scores) - patience
        var threshold = scores[start - 1] + tol
        for i in range(start, len(scores)):
            if scores[i] > threshold:
                return False
        return True

    @staticmethod
    def _history(scores: List[Float64]) raises -> _Values:
        var result = _Values(Shape(len(scores)))
        for i in range(len(scores)):
            result.unsafe_set(i, scores[i])
        return result^

    def _check_predict(self, X: Self.Input) raises:
        if not self._model:
            raise Error("NotFittedError: HistGradientBoostingRegressor must be fitted")
        _BinMapper._check_matrix(X)
        if X.shape[1] != self._model.value().n_features:
            raise Error("ValueError: prediction feature count must match training")

    @staticmethod
    def _linked(loss: _RegressionLoss, raw: _Values) raises -> _Values:
        var result = _Values(Shape(raw.size))
        for i in range(raw.size):
            result.unsafe_set(i, loss.inverse_link(raw.unsafe_get(i)))
        return result^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check_predict(X)
        ref model = self._model.value()
        var raw = full[DType.float64](Shape(X.shape[0]), model.baseline)
        for tree in model.trees:
            _add_prediction(raw, tree.predict(X))
        return Self._linked(model.loss, raw)

    def staged_predict(self, X: Self.Input) raises -> Self.Stages:
        self._check_predict(X)
        ref model = self._model.value()
        var raw = full[DType.float64](Shape(X.shape[0]), model.baseline)
        var stages = Self.Stages(capacity=len(model.trees))
        for tree in model.trees:
            _add_prediction(raw, tree.predict(X))
            stages.append(Self._linked(model.loss, raw))
        return stages^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        var scorer = DecisionTreeRegressor()
        scorer.n_outputs_ = 1
        return scorer._score(prediction, y, sample_weight)
