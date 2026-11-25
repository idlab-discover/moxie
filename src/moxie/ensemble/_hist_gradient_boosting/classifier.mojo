# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Public dense histogram gradient boosting classification."""

from std.collections import Set
from std.math import isfinite
from std.utils import Variant

from numax.core.libm import log
from numojo import NDArray, Shape, arange

from moxie._labels import LabelIndex, StringTarget, encode_column
from moxie._random import RandomState
from moxie.base import NumericClassifierClasses, WeightedSupervisedEstimator
from moxie.metrics import accuracy_score, balanced_accuracy_score
from moxie.model_selection import SplitSize, StratifiedShuffleSplit
from moxie.model_selection._split import _allocate_classes

from .._base import DecisionEnsemble, ProbabilisticEnsemble, StagedPredictor
from .._sampling import check_sample_weight, class_weight_factors
from .binning import _BinMapper
from .common import (
    HistGradientBoostingCategoricalFeatures,
    _CategoryEncoder,
    _EarlyStopping,
    _FitRandomState,
    _Interactions,
    _ValidationSize,
    _Values,
    _categorical_mask,
    _check_settings,
    _check_warm_start,
    _history,
    _interaction_groups,
    _resume_history,
    _should_stop,
    _take_rows,
    _uses_early_stopping,
)
from .grower import grow_tree
from .losses import _ClassificationLoss, _weighted_mean
from .predictor import _RawTree, make_raw_tree
from .splitting import _SplitOptions


comptime _Codes = NDArray[DType.int64]
comptime _ClassWeight = Variant[NoneType, String, Dict[Int64, Float64]]
comptime _ScoringSampleSize = 10_000


def _stratified_score_rows(
    y: _Values, n_classes: Int, seed: Int
) raises -> _Codes:
    # Group rows in encounter order, matching sklearn's stable class sort.
    var groups = List[List[Int]](capacity=n_classes)
    for _ in range(n_classes):
        groups.append(List[Int]())
    for i in range(y.size):
        groups[Int(y.unsafe_get(i * y.strides[0]))].append(i)
    var counts = _Codes(Shape(n_classes))
    for k in range(n_classes):
        counts.unsafe_set(k, Int64(len(groups[k])))

    var rng = RandomState(seed)
    var quotas = _allocate_classes(counts, _ScoringSampleSize, rng)
    var rows = _Codes(Shape(_ScoringSampleSize))
    var offset = 0
    for k in range(n_classes):
        ref group = groups[k]
        var order = rng.permutation(len(group))
        for j in range(Int(quotas.unsafe_get(k))):
            rows.unsafe_set(offset, Int64(group[Int(order.unsafe_get(j))]))
            offset += 1
    rng.shuffle(rows)
    return rows^


@fieldwise_init
struct _ClassScoringData(Movable):
    var targets: _Codes
    var weights: Optional[_Values]
    var rows: Optional[_Codes]

    @staticmethod
    def prepare(
        y: _Values, weights: Optional[_Values], var rows: Optional[_Codes]
    ) raises -> Self:
        var targets: _Codes
        var score_weights: Optional[_Values] = None
        if rows:
            ref indices = rows.value()
            targets = _take_rows(y, indices, 0, indices.size).astype[DType.int64]()
            if weights:
                ref w = weights.value()
                score_weights = _take_rows(w, indices, 0, indices.size)
        else:
            targets = y.astype[DType.int64]()
            if weights:
                ref w = weights.value()
                score_weights = w.view_with_layout(w.shape, w.strides, w.offset)
        return Self(targets^, score_weights^, rows^)


def _probability_score(
    scoring: String,
    loss: _ClassificationLoss,
    raw: _Values,
    sample: _ClassScoringData,
    brier_target: Optional[Int],
) raises -> Float64:
    var log_loss = scoring == "neg_log_loss"
    # Binomial predictions have two columns even for a one-class fit.
    var n_classes = max(2, loss._n_classes)
    if log_loss or n_classes > 2:
        var seen = Set[Int64]()
        for i in range(sample.targets.size):
            seen.add(sample.targets.unsafe_get(i * sample.targets.strides[0]))
        if len(seen) != n_classes:
            raise Error(
                "ValueError: probability scoring requires all fitted classes"
                " and at least two classes in the scoring targets"
            )

    var probabilities: _Values
    if sample.rows:
        ref rows = sample.rows.value()
        probabilities = loss.predict_proba(_take_rows(raw, rows, 0, rows.size))
    else:
        probabilities = loss.predict_proba(raw)

    comptime epsilon: Float64 = 2.220446049250313e-16
    var row_scores = _Values(Shape(sample.targets.size))
    for i in range(row_scores.size):
        var target = Int(sample.targets.unsafe_get(i * sample.targets.strides[0]))
        if log_loss:
            var probability = probabilities.unsafe_get(i * n_classes + target)
            if not isfinite(probability):
                raise Error("ValueError: probability scoring requires finite probabilities")
            row_scores.unsafe_set(i, log(probability.clamp(epsilon, 1.0 - epsilon)))
        else:
            target = brier_target.or_else(target)
            var row_loss: Float64 = 0.0
            for k in range(n_classes):
                var probability = probabilities.unsafe_get(i * n_classes + k)
                if not isfinite(probability):
                    raise Error("ValueError: probability scoring requires finite probabilities")
                var residual = Float64(k == target) - probability
                row_loss += residual * residual
            row_scores.unsafe_set(i, -row_loss)
    var score = _weighted_mean(row_scores, sample.weights)
    return score * 0.5 if not log_loss and n_classes == 2 else score


@fieldwise_init
struct HistGradientBoostingClassifierParams(Copyable, Movable):
    var loss: String
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
    var class_weight: _ClassWeight


@fieldwise_init
struct _Validation[dtype: DType](Movable):
    var X: NDArray[Self.dtype]
    var y: _Values
    var weights: Optional[_Values]


@fieldwise_init
struct _FittedClassifier(Copyable, Movable):
    var loss: _ClassificationLoss
    var baseline: _Values
    var stages: List[List[_RawTree]]
    var n_features: Int
    var random: _FitRandomState
    var encoder: _CategoryEncoder


struct HistGradientBoostingClassifier[
    feature_dtype: DType = DType.float64,
    label_dtype: DType = DType.float64,
    string_labels: Bool = False,
](
    DecisionEnsemble,
    NumericClassifierClasses where not string_labels,
    ProbabilisticEnsemble,
    StagedPredictor,
    WeightedSupervisedEstimator,
):
    """Histogram gradient boosting for binary and multiclass classification."""

    comptime Input = NDArray[Self.feature_dtype]
    comptime ClassLabelDType = Self.label_dtype
    comptime Target = StringTarget if Self.string_labels else NDArray[
        Self.label_dtype
    ]
    comptime Prediction = Self.Target
    comptime Probabilities = _Values
    comptime Decision = _Values
    comptime SampleWeight = _Values
    comptime Classes = List[String] if Self.string_labels else NDArray[
        Self.label_dtype
    ]
    comptime Weights = _ClassWeight
    comptime Params = HistGradientBoostingClassifierParams
    comptime Stages = List[Self.Prediction]

    var _params: Self.Params
    var _model: Optional[_FittedClassifier]
    var classes_: Self.Classes
    var n_features_in_: Int
    var n_iter_: Int
    var n_trees_per_iteration_: Int
    var do_early_stopping_: Bool
    var train_score_: _Values
    var validation_score_: _Values
    var is_categorical_: Optional[List[Bool]]

    def __init__(
        out self,
        var loss: String = "log_loss",
        *,
        learning_rate: Float64 = 0.1,
        max_iter: Int = 100,
        max_leaf_nodes: Optional[Int] = 31,
        max_depth: Optional[Int] = None,
        min_samples_leaf: Int = 20,
        l2_regularization: Float64 = 0.0,
        max_features: Float64 = 1.0,
        max_bins: Int = 255,
        var categorical_features: HistGradientBoostingCategoricalFeatures = String(
            "from_dtype"
        ),
        var monotonic_cst: Optional[List[Int]] = None,
        var interaction_cst: Optional[_Interactions] = None,
        warm_start: Bool = False,
        var early_stopping: _EarlyStopping = String("auto"),
        var scoring: Optional[String] = String("loss"),
        var validation_fraction: Optional[_ValidationSize] = _ValidationSize(
            Float64(0.1)
        ),
        n_iter_no_change: Int = 10,
        tol: Float64 = 1e-7,
        verbose: Int = 0,
        random_state: Optional[Int] = None,
        var class_weight: Self.Weights = NoneType(),
    ) raises:
        self = Self(
            Self.Params(
                loss=loss^,
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                max_features=max_features,
                max_bins=max_bins,
                categorical_features=categorical_features^,
                monotonic_cst=monotonic_cst^,
                interaction_cst=interaction_cst^,
                warm_start=warm_start,
                early_stopping=early_stopping^,
                scoring=scoring^,
                validation_fraction=validation_fraction^,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                verbose=verbose,
                random_state=random_state,
                class_weight=class_weight^,
            )
        )

    def __init__(out self, var params: Self.Params) raises:
        comptime assert Self.feature_dtype in (
            DType.float32,
            DType.float64,
        ), "features must be Float32 or Float64"
        self._params = params^
        comptime if Self.string_labels:
            self.classes_ = rebind_var[Self.Classes](List[String]())
        else:
            self.classes_ = rebind_var[Self.Classes](
                NDArray[Self.label_dtype](Shape(0))
            )
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

    def set_params[
        origin: MutOrigin, params_origin: ImmOrigin
    ](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[
        self
    ] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._model)

    def get_classes(self) raises -> Self.Target:
        if not self._model:
            raise Error(
                "NotFittedError: HistGradientBoostingClassifier must be fitted"
            )
        comptime if Self.string_labels:
            return rebind_var[Self.Target](
                StringTarget(rebind[List[String]](self.classes_))
            )
        else:
            return rebind_var[Self.Target](self.classes_.copy())

    @staticmethod
    def _check_target(y: Self.Target, rows: Int) raises:
        var valid: Bool
        comptime if Self.string_labels:
            ref labels = rebind[StringTarget](y)
            valid = labels.columns() == 1 and labels.rows() == rows
        else:
            ref labels = rebind[NDArray[Self.label_dtype]](y)
            valid = (
                labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1)
            ) and labels.size == rows
        if not valid:
            raise Error("ValueError: y must have one label per sample")

    @staticmethod
    def _check_weights(weights: Optional[_Values], rows: Int) raises:
        if weights:
            ref w = weights.value()
            check_sample_weight(w, rows)
            for i in range(rows):
                if w.unsafe_get(i * w.strides[0]) < 0.0:
                    raise Error(
                        "ValueError: sample weights must be nonnegative"
                    )

    def _training_weights(
        self,
        index: LabelIndex,
        sample_weight: Optional[_Values],
    ) raises -> Optional[_Values]:
        ref weights = self._params.class_weight
        ref codes = index.inverse
        Self._check_weights(sample_weight, codes.size)
        if weights.isa[NoneType]():
            return None

        var factors: _Values
        if weights.isa[String]():
            if weights[String] != "balanced":
                raise Error(
                    "ValueError: class_weight must be balanced, a mapping, or"
                    " None"
                )
            ref counts = index.counts
            factors = _Values(Shape(counts.size))
            for k in range(counts.size):
                factors.unsafe_set(
                    k,
                    Float64(codes.size)
                    / (Float64(counts.size) * Float64(counts.unsafe_get(k))),
                )
        else:
            # The pinned classifier computes weights after encoding the targets.
            factors = class_weight_factors(
                arange[DType.int64](Int64(index.counts.size)),
                weights[Dict[Int64, Float64]],
            )
        var result = _Values(Shape(codes.size))
        for i in range(codes.size):
            var factor = factors.unsafe_get(Int(codes.unsafe_get(i)))
            if sample_weight:
                ref w = sample_weight.value()
                factor *= w.unsafe_get(i * w.strides[0])
            if not isfinite(factor) or factor < 0.0:
                raise Error(
                    "ValueError: combined sample and class weights must be"
                    " finite and nonnegative"
                )
            result.unsafe_set(i, factor)
        return result^

    def fit[
        origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        return self.fit(X, y, None)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        return self.fit(X, y, Optional[_Values](sample_weight.copy()))

    def fit[
        origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
        *,
        var X_val: Optional[Self.Input] = None,
        y_val: Optional[Self.Target] = None,
        var sample_weight_val: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        _BinMapper._check_matrix(X)
        ref params = self._params
        _check_settings(
            params.max_iter,
            params.max_depth,
            params.max_leaf_nodes,
            params.learning_rate,
            params.max_features,
            params.max_bins,
            params.n_iter_no_change,
            params.tol,
            params.verbose,
            params.validation_fraction,
        )
        if params.loss != "log_loss":
            raise Error("ValueError: classifier loss must be log_loss")
        Self._check_target(y, X.shape[0])
        var early = _uses_early_stopping(params.early_stopping, X.shape[0])
        var scoring = params.scoring.or_else("accuracy")
        if early and scoring not in (
            "loss", "accuracy", "balanced_accuracy", "neg_log_loss", "neg_brier_score",
        ):
            raise Error(
                "NotImplementedError: unsupported histogram classification scorer"
            )
        if Bool(X_val) != Bool(y_val) or (sample_weight_val and not X_val):
            raise Error(
                "ValueError: explicit validation requires both X_val and y_val"
            )
        if X_val:
            if params.early_stopping.isa[Bool]() and not early:
                raise Error(
                    "ValueError: explicit validation requires"
                    " early_stopping=True or auto"
                )
            ref val_features = X_val.value()
            _BinMapper._check_matrix(val_features)
            if val_features.shape[1] != X.shape[1]:
                raise Error(
                    "ValueError: validation feature count must match training"
                )
            Self._check_target(y_val.value(), val_features.shape[0])
            Self._check_weights(sample_weight_val, val_features.shape[0])

        comptime if Self.string_labels:
            var encoding = encode_column(rebind[StringTarget](y))
            var validation_targets: Optional[_Values] = None
            if y_val:
                validation_targets = encoding.encode(
                    rebind[StringTarget](y_val.value()),
                ).astype[DType.float64]()
            self._fit_encoded(
                X,
                rebind_var[Self.Classes](encoding.classes.copy()),
                encoding.index,
                sample_weight,
                X_val^,
                validation_targets^,
                sample_weight_val^,
                early,
            )
        else:
            var encoding = encode_column(rebind[NDArray[Self.label_dtype]](y))
            var validation_targets: Optional[_Values] = None
            if y_val:
                validation_targets = encoding.encode(
                    rebind[NDArray[Self.label_dtype]](y_val.value()),
                ).astype[DType.float64]()
            self._fit_encoded(
                X,
                rebind_var[Self.Classes](encoding.classes.copy()),
                encoding.index,
                sample_weight,
                X_val^,
                validation_targets^,
                sample_weight_val^,
                early,
            )
        return self

    def _fit_encoded(
        mut self,
        X: Self.Input,
        var classes: Self.Classes,
        index: LabelIndex,
        sample_weight: Optional[_Values],
        var X_val: Optional[Self.Input],
        var validation_targets: Optional[_Values],
        var sample_weight_val: Optional[_Values],
        early: Bool,
    ) raises:
        ref params = self._params
        var validation: Optional[_Validation[Self.feature_dtype]] = None
        if X_val:
            validation = _Validation[Self.feature_dtype](
                X_val.take(),
                validation_targets.take(),
                sample_weight_val^,
            )

        var loss = _ClassificationLoss(index.counts.size)
        if loss.raw_columns() > 1 and params.monotonic_cst:
            raise Error(
                "ValueError: monotonic constraints are not supported for"
                " multiclass classification"
            )
        var adjusted_weights = self._training_weights(index, sample_weight)
        ref weights = adjusted_weights if adjusted_weights else sample_weight
        var targets = index.inverse.astype[DType.float64]()
        var categorical = _categorical_mask(
            params.categorical_features, X.shape[1]
        )
        var encoder = _CategoryEncoder(X, categorical, params.max_bins)
        var resume = params.warm_start and Bool(self._model)
        var random: _FitRandomState
        if resume:
            ref prior = self._model.value()
            _check_warm_start(
                params.max_iter, len(prior.stages), X.shape[1], prior.n_features
            )
            self._check_warm_start_classes(classes)
            encoder.check_warm_start(prior.encoder)
            random = prior.random.copy()
        else:
            random = _FitRandomState.start(params.random_state)

        if early and params.validation_fraction and not validation:
            ref size = params.validation_fraction.value()
            var split_size = SplitSize(size[Int]) if size.isa[
                Int
            ]() else SplitSize(size[Float64])
            var splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=split_size, random_state=random.split_seed
            )
            var splits = splitter.split(
                X.shape[0],
                Optional[_Codes](
                    index.inverse.view_with_layout(
                        index.inverse.shape,
                        index.inverse.strides,
                        index.inverse.offset,
                    )
                ),
            )
            var first = splits.next()
            var split = first.take()
            var train_weights: Optional[_Values] = None
            var validation_weights: Optional[_Values] = None
            if weights:
                ref w = weights.value()
                train_weights = _take_rows(w, split.train, 0, split.train.size)
                validation_weights = _take_rows(
                    w, split.test, 0, split.test.size
                )
            validation = _Validation[Self.feature_dtype](
                _take_rows(X, split.test, 0, split.test.size),
                _take_rows(targets, split.test, 0, split.test.size),
                validation_weights^,
            )
            self._fit(
                _take_rows(X, split.train, 0, split.train.size),
                _take_rows(targets, split.train, 0, split.train.size),
                train_weights,
                validation,
                loss^,
                early,
                random^,
                resume,
                encoder^,
                classes,
            )
        else:
            self._fit(
                X,
                targets,
                weights,
                validation,
                loss^,
                early,
                random^,
                resume,
                encoder^,
                classes,
            )
        self.classes_ = classes^

    def _check_warm_start_classes(self, classes: Self.Classes) raises:
        var same: Bool
        comptime if Self.string_labels:
            ref fitted = rebind[List[String]](self.classes_)
            ref current = rebind[List[String]](classes)
            same = fitted == current
        else:
            ref fitted = rebind[NDArray[Self.label_dtype]](self.classes_)
            ref current = rebind[NDArray[Self.label_dtype]](classes)
            same = fitted.ndim == 1 and fitted.size == current.size
            if same:
                for i in range(fitted.size):
                    if fitted.unsafe_get(i * fitted.strides[0]) != current.unsafe_get(i):
                        same = False
                        break
        if not same:
            raise Error("ValueError: warm start requires the fitted classes")

    @staticmethod
    def _baseline_rows(baseline: _Values, rows: Int) raises -> _Values:
        var raw = _Values(Shape(rows, baseline.size))
        for i in range(rows):
            for k in range(baseline.size):
                raw.unsafe_set(i * baseline.size + k, baseline.unsafe_get(k))
        return raw^

    @staticmethod
    def _add_column(mut raw: _Values, column: Int, values: _Values) raises:
        var columns = raw.shape[1]
        for i in range(values.size):
            var at = i * columns + column
            raw.unsafe_set(at, raw.unsafe_get(at) + values.unsafe_get(i))

    def _fit(
        mut self,
        X_input: Self.Input,
        y: _Values,
        weights: Optional[_Values],
        validation: Optional[_Validation[Self.feature_dtype]],
        var loss: _ClassificationLoss,
        early: Bool,
        var random: _FitRandomState,
        resume: Bool,
        var encoder: _CategoryEncoder,
        classes: Self.Classes,
    ) raises:
        ref params = self._params
        var scoring = params.scoring.or_else("accuracy")
        var baseline: _Values
        var stages = List[List[_RawTree]]()
        var train_scores = List[Float64]()
        var validation_scores = List[Float64]()
        if resume:
            ref prior = self._model.value()
            baseline = prior.baseline.copy()
            stages = prior.stages.copy()
            train_scores = _resume_history(self.train_score_)
            validation_scores = _resume_history(self.validation_score_)
        else:
            baseline = loss.fit_intercept(y, weights)
        var X = encoder.transform(X_input)
        var n_features = X.shape[1]
        var columns = loss.raw_columns()
        var rows = y.size
        var monotonic = encoder.monotonic_constraints(
            params.monotonic_cst, n_features
        )
        var interactions = _interaction_groups(
            params.interaction_cst, n_features
        )
        var n_bins = params.max_bins + 1
        var mapper = _BinMapper(n_bins=n_bins, random_state=random.split_seed)
        _ = mapper.fit(
            X, known_categories=encoder.known_categories(), sample_weight=weights
        )
        var binned = mapper.transform(X)
        var raw = Self._raw_predictions(X, baseline, stages)
        var raw_val = _Values(Shape(0))
        var binned_val: Optional[NDArray[DType.uint8]] = None
        var train_scoring: Optional[_ClassScoringData] = None
        var validation_scoring: Optional[_ClassScoringData] = None
        if early and scoring != "loss":
            var score_rows: Optional[_Codes] = None
            if rows > _ScoringSampleSize:
                score_rows = _stratified_score_rows(
                    y, loss._n_classes, random.split_seed
                )
            train_scoring = _ClassScoringData.prepare(y, weights, score_rows^)
            if validation:
                ref val_data = validation.value()
                validation_scoring = _ClassScoringData.prepare(
                    val_data.y, val_data.weights, None
                )
        if early and not resume:
            train_scores.append(
                Self._stopping_score(
                    scoring, loss, y, raw, weights, train_scoring, classes
                )
            )
        if validation:
            ref val_data = validation.value()
            var X_val = encoder.transform(val_data.X)
            if early:
                binned_val = mapper.transform(X_val)
                raw_val = Self._raw_predictions(X_val, baseline, stages)
                if not resume:
                    validation_scores.append(
                        Self._stopping_score(
                            scoring, loss, val_data.y, raw_val,
                            val_data.weights, validation_scoring, classes,
                        )
                    )

        var gradients = List[Float32]()
        var hessians = List[Float32]()
        var options = _SplitOptions(
            min_samples_leaf=params.min_samples_leaf,
            l2_regularization=params.l2_regularization,
        )
        if params.verbose > 0:
            print("Fitting gradient boosted rounds:")
        for iteration in range(len(stages), params.max_iter):
            loss.gradient_hessian(y, raw, weights, gradients, hessians)
            var stage = List[_RawTree](capacity=columns)
            for k in range(columns):
                var start = k * rows
                var stop = start + rows
                var grown = grow_tree(
                    binned,
                    n_bins,
                    Span(gradients)[start:stop],
                    Span(hessians)[start:stop],
                    mapper.n_bins_non_missing_,
                    mapper.is_categorical_,
                    monotonic,
                    feature_sampler=random.feature_sampler,
                    feature_fraction_per_split=params.max_features,
                    interaction_groups=interactions,
                    max_leaf_nodes=params.max_leaf_nodes,
                    max_depth=params.max_depth,
                    options=options,
                    shrinkage=params.learning_rate,
                )
                for leaf in grown.leaves:
                    ref node = grown.tree.nodes[leaf]
                    for position in range(node.start, node.stop):
                        var at = grown.partition[position] * columns + k
                        raw.unsafe_set(at, raw.unsafe_get(at) + node.value)
                if binned_val:
                    Self._add_column(
                        raw_val, k, grown.tree.predict(binned_val.value())
                    )
                stage.append(make_raw_tree(grown.tree, mapper))
            stages.append(stage^)
            if params.verbose > 1:
                print(
                    "Boosting iteration", iteration + 1, "of", params.max_iter
                )
            if early:
                train_scores.append(
                    Self._stopping_score(
                        scoring, loss, y, raw, weights, train_scoring, classes
                    )
                )
                if validation:
                    ref val_data = validation.value()
                    validation_scores.append(
                        Self._stopping_score(
                            scoring, loss, val_data.y, raw_val,
                            val_data.weights, validation_scoring, classes,
                        )
                    )
                ref scores = validation_scores if validation else train_scores
                if _should_stop(scores, params.n_iter_no_change, params.tol):
                    break

        var train_history = _history(train_scores)
        var validation_history = _history(validation_scores)
        self.n_features_in_ = n_features
        self.n_iter_ = len(stages)
        self.n_trees_per_iteration_ = columns
        self.do_early_stopping_ = early
        self.train_score_ = train_history^
        self.validation_score_ = validation_history^
        self.is_categorical_ = encoder.categorical_mask()
        self._model = _FittedClassifier(
            loss^, baseline^, stages^, n_features, random^, encoder^
        )

    def _check_predict(self, X: Self.Input) raises:
        if not self._model:
            raise Error(
                "NotFittedError: HistGradientBoostingClassifier must be fitted"
            )
        _BinMapper._check_matrix(X)
        if X.shape[1] != self._model.value().n_features:
            raise Error(
                "ValueError: prediction feature count must match training"
            )

    @staticmethod
    def _apply_stage(
        mut raw: _Values, stage: List[_RawTree], X: Self.Input
    ) raises:
        for k in range(len(stage)):
            Self._add_column(raw, k, stage[k].predict(X))

    @staticmethod
    def _raw_predictions(
        X: Self.Input, baseline: _Values, stages: List[List[_RawTree]]
    ) raises -> _Values:
        var raw = Self._baseline_rows(baseline, X.shape[0])
        for stage in stages:
            Self._apply_stage(raw, stage, X)
        return raw^

    def _raw_predict(self, X: Self.Input) raises -> _Values:
        self._check_predict(X)
        ref model = self._model.value()
        return Self._raw_predictions(
            model.encoder.transform(X), model.baseline, model.stages
        )

    @staticmethod
    def _winning_class(raw: _Values, row: Int, columns: Int) -> Int:
        if columns == 1:
            return Int(raw.unsafe_get(row) > 0.0)
        var best = 0
        for k in range(1, columns):
            if raw.unsafe_get(row * columns + k) > raw.unsafe_get(
                row * columns + best
            ):
                best = k
        return best

    @staticmethod
    def _binary_brier_target(
        targets: _Codes, classes: Self.Classes, n_classes: Int,
    ) raises -> Optional[Int]:
        comptime if Self.string_labels:
            raise Error("ValueError: binary Brier scoring cannot infer a positive string label")
        else:
            var first = targets.unsafe_get(0)
            for i in range(1, targets.size):
                if targets.unsafe_get(i * targets.strides[0]) != first:
                    return None
            ref labels = rebind[NDArray[Self.label_dtype]](classes)
            var observed = labels.unsafe_get(Int(first))
            # The scorer uses label 1 for conventional binary labels and the
            # observed label for other numeric singletons.
            var positive = observed != 0 and observed != -1
            # The probability response selects column zero for a one-class fit.
            return Int(positive) if n_classes == 2 else Int(not positive)

    @staticmethod
    def _stopping_score(
        scoring: String, loss: _ClassificationLoss, y: _Values, raw: _Values,
        weights: Optional[_Values], data: Optional[_ClassScoringData],
        classes: Self.Classes,
    ) raises -> Float64:
        if not data:
            return -loss.mean_loss(y, raw, weights)
        ref sample = data.value()
        if scoring in ("neg_log_loss", "neg_brier_score"):
            var brier_target: Optional[Int] = None
            if scoring == "neg_brier_score" and loss._n_classes <= 2:
                brier_target = Self._binary_brier_target(
                    sample.targets, classes, loss._n_classes
                )
            return _probability_score(scoring, loss, raw, sample, brier_target)
        var prediction = _Codes(Shape(sample.targets.size))
        var columns = raw.shape[1]
        for i in range(prediction.size):
            var row = i
            if sample.rows:
                row = Int(sample.rows.value().unsafe_get(i))
            var label = 0 if loss._n_classes == 1 else Self._winning_class(
                raw, row, columns
            )
            prediction.unsafe_set(i, Int64(label))
        if scoring == "balanced_accuracy":
            return balanced_accuracy_score(
                sample.targets, prediction, sample_weight=sample.weights
            )
        return accuracy_score(
            sample.targets, prediction, sample_weight=sample.weights
        )

    def _labels(self, raw: _Values) raises -> Self.Prediction:
        var rows = raw.shape[0]
        var columns = raw.shape[1]
        comptime if Self.string_labels:
            ref classes = rebind[List[String]](self.classes_)
            var result = List[String](capacity=rows)
            for i in range(rows):
                result.append(
                    classes[
                        0 if len(classes)
                        == 1 else Self._winning_class(raw, i, columns)
                    ].copy()
                )
            return rebind_var[Self.Prediction](StringTarget(result))
        else:
            ref classes = rebind[NDArray[Self.label_dtype]](self.classes_)
            var result = NDArray[Self.label_dtype](Shape(rows))
            for i in range(rows):
                result.unsafe_set(
                    i,
                    classes.unsafe_get(
                        0 if classes.size
                        == 1 else Self._winning_class(raw, i, columns)
                    ),
                )
            return rebind_var[Self.Prediction](result^)

    @staticmethod
    def _decision(var raw: _Values) raises -> _Values:
        if raw.shape[1] == 1:
            raw.resize(Shape(raw.size))
        return raw^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self._labels(self._raw_predict(X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var raw = self._raw_predict(X)
        return self._model.value().loss.predict_proba(raw)

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        return Self._decision(self._raw_predict(X))

    def _staged[
        Output: Movable & Deinitable, F: def(_Values) raises -> Output
    ](self, X: Self.Input, convert: F,) raises -> List[Output]:
        self._check_predict(X)
        ref model = self._model.value()
        var raw = Self._baseline_rows(model.baseline, X.shape[0])
        var results = List[Output](capacity=len(model.stages))
        var encoded = model.encoder.transform(X)
        for stage in model.stages:
            Self._apply_stage(raw, stage, encoded)
            results.append(convert(raw))
        return results^

    def staged_predict(self, X: Self.Input) raises -> Self.Stages:
        def labels(raw: _Values) raises {imm self} -> Self.Prediction:
            return self._labels(raw)

        return self._staged(X, labels)

    def staged_predict_proba(
        self, X: Self.Input
    ) raises -> List[Self.Probabilities]:
        def probabilities(raw: _Values) raises {imm self} -> Self.Probabilities:
            return self._model.value().loss.predict_proba(raw)

        return self._staged(X, probabilities)

    def staged_decision_function(
        self, X: Self.Input
    ) raises -> List[Self.Decision]:
        def decisions(raw: _Values) raises -> Self.Decision:
            return Self._decision(raw.copy())

        return self._staged(X, decisions)

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var prediction = self.predict(X)
        comptime if Self.string_labels:
            return accuracy_score(
                rebind[StringTarget](y),
                rebind[StringTarget](prediction),
                sample_weight=sample_weight,
            )
        else:
            return accuracy_score(
                rebind[NDArray[Self.label_dtype]](y),
                rebind[NDArray[Self.label_dtype]](prediction),
                sample_weight=sample_weight,
            )
