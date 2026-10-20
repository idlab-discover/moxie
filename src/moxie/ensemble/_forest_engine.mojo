# scikit-learn developers, BSD-3-Clause; see LICENSES/scikit-learn.txt.
"""BaseEnsemble and the serial forest engine over prepared-input decision trees.

`ForestEngine` ports sklearn `BaseForest.fit` with `_parallel_build_trees` as a
serial loop, plus `_compute_oob_predictions`, `feature_importances_`,
`estimators_samples_`, `apply` and the member averaging of `predict` /
`predict_proba`. Training input is validated and encoded once
(`prepare_fit`), then each member is built from it with its own seed and
bootstrap counts as sample weights, the native form of `tree._fit(check_input=False)`.

Classification members train on Float64 class codes 0..K-1 per output, as
sklearn forests train on `np.unique(..., return_inverse=True)` codes; the public
estimator owns the label vocabulary and maps codes back to labels.
"""
import numojo as nm
from std.utils import Variant
from moxie.base import BaseEstimator, clone
from moxie._random import RandomState
from moxie.tree._classes import BaseDecisionTree, TreeInputEncoder
from moxie.tree._records import Features
from moxie.ensemble._records import Values, Indices, OOBAccumulator
from moxie.ensemble._plumbing import additional_estimators, next_estimator_seed, warn
from moxie.ensemble._sampling import (
    MaxSamples, check_sample_weight, get_n_samples_bootstrap, generate_sample_indices,
    generate_unsampled_indices, sample_counts, count_weights, expand_class_weight,
    balanced_class_weight, balanced_subsample_weight,
)
from moxie.ensemble._aggregation import (
    ForestAggregator, identity_alignment, new_oob, forest_oob_denominators,
    constant_denominators, mean_feature_importances, oob_accuracy, r2_uniform,
)

comptime ForestClassWeight = Variant[NoneType, String, List[Values]]
"""None, "balanced", "balanced_subsample", or per-output factors indexed by class code."""


struct BaseEnsemble[E: BaseEstimator & Deinitable](Movable, Sized):
    """sklearn `BaseEnsemble`: a template estimator and the fitted members cloned from it.

    `estimator` is constructor configuration, `estimator_` the validated template
    that `make_estimator` clones, and `estimators_` the fitted members in build
    order. Typed configuration replaces sklearn's `estimator_params` reflection:
    owners configure the template directly before cloning.
    """
    var estimator: Self.E
    var n_estimators: Int
    var estimator_: Self.E
    var estimators_: List[Self.E]

    def __init__(out self, var estimator: Self.E, n_estimators: Int = 10) raises:
        self.estimator_ = clone(estimator)
        self.estimator = estimator^
        self.n_estimators = n_estimators
        self.estimators_ = List[Self.E]()

    def validate_estimator(mut self) raises:
        self.estimator_ = clone(self.estimator)

    def make_estimator(self) raises -> Self.E:
        """An unfitted clone of `estimator_`; the caller seeds and appends it."""
        return clone(self.estimator_)

    def __len__(self) -> Int:
        return len(self.estimators_)

    def __getitem__(ref self, index: Int) raises -> ref[self.estimators_[0]] Self.E:
        var at = index + len(self.estimators_) if index < 0 else index
        if at < 0 or at >= len(self.estimators_):
            raise Error("IndexError: ensemble member index out of range")
        return self.estimators_[at]


def set_tree_random_state[classifier: Bool, extra: Bool](mut tree: BaseDecisionTree[classifier, extra], mut rng: RandomState) raises -> Int:
    """sklearn `_set_random_states` for a tree: its only random_state draws randint(MAX_INT)."""
    var seed = next_estimator_seed(rng)
    var params = tree.get_params()
    params.random_state = seed
    _ = tree.set_params(params)
    return seed


def take_rows(X: Features, rows: Indices) raises -> Features:
    """Copy the selected rows of a matrix in index order."""
    var p = X.shape[1]
    var result = nm.zeros[DType.float32](nm.Shape(rows.size, p))
    for r in range(rows.size):
        var row = Int(rows.unsafe_get(r * Int(rows.strides[0])))
        for j in range(p):
            result.unsafe_set(r * p + j, X.unsafe_get(row * Int(X.strides[0]) + j * Int(X.strides[1])))
    return result^


def _as_matrix(x: Values) raises -> Values:
    if x.ndim == 2:
        return x.view_with_layout(x.shape, x.strides, x.offset)
    return x.reshape(nm.Shape(x.shape[0], 1))


struct ForestEngine[classifier: Bool, extra: Bool = False](Movable, Sized):
    """sklearn `BaseForest` over `DecisionTree*`/`ExtraTree*` members, built serially.

    Configuration fields mirror the forest constructor; the member template in
    `base.estimator` carries every tree parameter. `fit` takes the random
    context explicitly so integer seeds and shared `RandomState` objects keep
    their reference stream semantics. Prediction matrices are `(samples, outputs)`;
    probabilities and OOB decisions are one `(samples, classes)` matrix per output.
    """
    comptime Tree = BaseDecisionTree[Self.classifier, Self.extra]
    var base: BaseEnsemble[Self.Tree]
    var bootstrap: Bool
    var oob_score: Bool
    var warm_start: Bool
    var class_weight: ForestClassWeight
    var max_samples: MaxSamples
    var n_outputs_: Int
    var n_features_in_: Int
    var n_classes_: List[Int]
    var oob_score_: Float64
    var oob_prediction_: Values
    var oob_decision_function_: List[Values]
    var warnings_: List[String]
    var _n_samples: Int
    var _n_samples_bootstrap: Int
    var _sample_weight: Optional[Values]
    var _encoder: TreeInputEncoder
    var _has_oob: Bool

    def __init__(out self, var estimator: Self.Tree, n_estimators: Int = 100, *, bootstrap: Bool = True,
            oob_score: Bool = False, warm_start: Bool = False, class_weight: ForestClassWeight = NoneType(),
            max_samples: MaxSamples = NoneType()) raises:
        self.base = BaseEnsemble[Self.Tree](estimator^, n_estimators)
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.class_weight = class_weight.copy()
        self.max_samples = max_samples.copy()
        self.n_outputs_ = 0
        self.n_features_in_ = 0
        self.n_classes_ = List[Int]()
        self.oob_score_ = Float64("nan")
        self.oob_prediction_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.oob_decision_function_ = List[Values]()
        self.warnings_ = List[String]()
        self._n_samples = 0
        self._n_samples_bootstrap = 0
        self._sample_weight = None
        self._encoder = TreeInputEncoder()
        self._has_oob = False

    def __len__(self) -> Int:
        return len(self.base)

    def is_fitted(self) -> Bool:
        return len(self.base.estimators_) > 0

    def has_oob(self) -> Bool:
        return self._has_oob

    def _ready(self) raises:
        if not self.is_fitted():
            raise Error("NotFittedError: call fit before prediction or inspection")

    def _expanded_class_weight(mut self, y: Values, sample_weight: Optional[Values]) raises -> Optional[Values]:
        """`ForestClassifier._validate_y_class_weight` on encoded targets."""
        if self.class_weight.isa[NoneType]():
            return None
        comptime if not Self.classifier:
            raise Error("ValueError: class_weight is only supported by forest classifiers")
        if self.class_weight.isa[List[Values]]():
            return expand_class_weight(y, self.class_weight[List[Values]])
        var preset = self.class_weight[String]
        if preset != "balanced" and preset != "balanced_subsample":
            raise Error("ValueError: Valid presets for class_weight include \"balanced\" and \"balanced_subsample\".Given \"" + preset + "\".")
        if self.warm_start:
            warn(self.warnings_, "UserWarning: class_weight presets \"balanced\" or \"balanced_subsample\" are not recommended for warm_start if the fitted data differs from the full dataset. In order to use \"balanced\" weights, use compute_class_weight (\"balanced\", classes, y). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions. Pass the resulting weights as the class_weight parameter.")
        if preset == "balanced_subsample" and self.bootstrap:
            return None  # computed per member on its bootstrap draw
        return balanced_class_weight(y, sample_weight)

    def fit(mut self, X: Features, y: Values, sample_weight: Optional[Values], mut rng: RandomState) raises:
        """Build `n_estimators` members (or add members under warm_start) from `rng`.

        `y` holds Float64 regression targets or class codes, `(samples,)` or
        `(samples, outputs)`. Bootstrap rows are drawn with the combined sample and
        class weights; members then receive the draw counts as sample weights.
        """
        self.warnings_ = List[String]()
        self.base.validate_estimator()
        var prepared = self.base.estimator_.prepare_fit(X, y)
        var n = prepared.n_samples()
        if sample_weight:
            check_sample_weight(sample_weight.value(), n)
        var expanded = self._expanded_class_weight(prepared.y, sample_weight)
        var combined = Optional[Values]()
        if Bool(sample_weight) and Bool(expanded):
            var product = nm.zeros[DType.float64](nm.Shape(n))
            for i in range(n):
                var w = sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))
                product.unsafe_set(i, w * expanded.value().unsafe_get(i))
            combined = product^
        elif sample_weight:
            combined = sample_weight.value().copy()
        elif expanded:
            combined = expanded.value().copy()
        var n_bootstrap = 0
        if not self.bootstrap and not self.max_samples.isa[NoneType]():
            raise Error("ValueError: `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`.")
        elif self.bootstrap:
            n_bootstrap = get_n_samples_bootstrap(n, self.max_samples, combined, self.warnings_)
        if not self.bootstrap and self.oob_score:
            raise Error("ValueError: Out of bag estimation only available if bootstrap=True")
        var subsample = self.bootstrap and self.class_weight.isa[String]() and self.class_weight[String] == "balanced_subsample"
        if not self.warm_start or len(self.base.estimators_) == 0:
            self.base.estimators_ = List[Self.Tree]()
            self._has_oob = False
        var fitted = len(self.base.estimators_)
        var more = additional_estimators(self.base.n_estimators, fitted, self.warm_start)
        self._n_samples = n
        self._n_samples_bootstrap = n_bootstrap
        self._sample_weight = combined.copy()
        self.n_outputs_ = prepared.y.shape[1]
        self.n_features_in_ = prepared.X.shape[1]
        self.n_classes_ = prepared.n_classes.copy() if Self.classifier else List[Int]()
        self._encoder = prepared.encoder.copy()
        if more == 0:
            warn(self.warnings_, "UserWarning: Warm-start fitting without increasing n_estimators does not fit new trees.")
        else:
            if self.warm_start and fitted > 0:
                for _ in range(fitted):  # replay the seeds a cold fit would have drawn
                    _ = next_estimator_seed(rng)
            var trees = List[Self.Tree]()
            for _ in range(more):  # every seed is drawn before any member is built
                var tree = self.base.make_estimator()
                _ = set_tree_random_state(tree, rng)
                trees.append(tree^)
            for i in range(more):
                if self.bootstrap:
                    var seed = trees[i].params.random_state.value()
                    var indices = generate_sample_indices(seed, n, n_bootstrap, combined)
                    var weights = count_weights(sample_counts(indices, n))
                    if subsample:
                        var factors = balanced_subsample_weight(prepared.y, indices)
                        for r in range(n):
                            weights.unsafe_set(r, weights.unsafe_get(r) * factors.unsafe_get(r))
                    _ = trees[i].fit_prepared(prepared, sample_weight=weights)
                else:
                    _ = trees[i].fit_prepared(prepared, combined)
            self.base.estimators_.extend(trees^)  # members join only after all are built
        if self.oob_score and (more > 0 or not self._has_oob):
            self._set_oob(prepared.X, prepared.y)

    def _set_oob(mut self, X: Features, y: Values) raises:
        """`_compute_oob_predictions` plus the forest's default accuracy or R2 score."""
        var n = self._n_samples
        var aggregator = ForestAggregator()
        var accumulators = List[OOBAccumulator]()
        comptime if Self.classifier:
            for k in range(self.n_outputs_):
                if self.n_outputs_ > 1 and self.n_classes_[k] > 2:
                    raise Error("ValueError: The type of target cannot be used to compute OOB estimates. Got multiclass-multioutput while only the following are supported: continuous, continuous-multioutput, binary, multiclass, multilabel-indicator.")
                if self.n_classes_[k] != self.n_classes_[0]:
                    raise Error("ValueError: OOB estimates need the same number of classes in every output")
                accumulators.append(new_oob(n, self.n_classes_[k]))
        else:
            accumulators.append(new_oob(n, self.n_outputs_))
        for m in range(len(self.base.estimators_)):
            ref tree = self.base.estimators_[m]
            var rows = generate_unsampled_indices(tree.params.random_state.value(), n, self._n_samples_bootstrap, self._sample_weight)
            if rows.size == 0:
                continue
            var subset = take_rows(X, rows)
            comptime if Self.classifier:
                var probabilities = tree.predict_proba_encoded(subset)
                for k in range(self.n_outputs_):
                    aggregator.add_oob(rows, probabilities[k], accumulators[k])
            else:
                aggregator.add_oob(rows, _as_matrix(tree.predict_encoded(subset)), accumulators[0])
        var denominators = forest_oob_denominators(accumulators[0].counts, self.warnings_)
        comptime if Self.classifier:
            self.oob_decision_function_ = List[Values]()
            for k in range(self.n_outputs_):
                self.oob_decision_function_.append(aggregator.normalize(accumulators[k].sums, denominators))
            self.oob_score_ = oob_accuracy(y, self.oob_decision_function_)
        else:
            self.oob_prediction_ = aggregator.normalize(accumulators[0].sums, denominators)
            self.oob_score_ = r2_uniform(y, self.oob_prediction_, self.warnings_)
        self._has_oob = True

    def encode(self, X: Features) raises -> Features:
        """Validate prediction input once for every member (`_validate_X_predict`)."""
        self._ready()
        ref params = self.base.estimator_.params
        return self._encoder.encode(X, params.criterion == "absolute_error", Bool(params.monotonic_cst))

    def apply(self, X: Features) raises -> Indices:
        """Leaf index of every sample in every member, `(samples, n_estimators)`."""
        var encoded = self.encode(X)
        var count = len(self.base.estimators_)
        var result = nm.zeros[DType.int64](nm.Shape(encoded.shape[0], count))
        for m in range(count):
            var leaves = self.base.estimators_[m].apply_encoded(encoded)
            for i in range(encoded.shape[0]):
                result.unsafe_set(i * count + m, leaves.unsafe_get(i))
        return result^

    def predict(self, X: Features) raises -> Values where not Self.classifier:
        """Mean member prediction, `(samples, outputs)`."""
        var encoded = self.encode(X)
        var aggregator = ForestAggregator()
        var sums = nm.zeros[DType.float64](nm.Shape(encoded.shape[0], self.n_outputs_))
        for m in range(len(self.base.estimators_)):
            aggregator.add_regression(self.base.estimators_[m].predict_encoded(encoded), 1, sums)
        return aggregator.normalize(sums, constant_denominators(encoded.shape[0], len(self.base.estimators_)))

    def predict_proba(self, X: Features) raises -> List[Values] where Self.classifier:
        """Mean member class probabilities, one `(samples, classes)` matrix per output."""
        var encoded = self.encode(X)
        var aggregator = ForestAggregator()
        var n = encoded.shape[0]
        var sums = List[Values]()
        for k in range(self.n_outputs_):
            sums.append(nm.zeros[DType.float64](nm.Shape(n, self.n_classes_[k])))
        for m in range(len(self.base.estimators_)):
            var probabilities = self.base.estimators_[m].predict_proba_encoded(encoded)
            for k in range(self.n_outputs_):
                aggregator.add_probability(probabilities[k], identity_alignment(self.n_classes_[k]), 1, sums[k])
        var result = List[Values]()
        var denominators = constant_denominators(n, len(self.base.estimators_))
        for k in range(self.n_outputs_):
            result.append(aggregator.normalize(sums[k], denominators))
        return result^

    def feature_importances_(self) raises -> Values:
        """Mean impurity importances of members with at least one split, normalized to sum to one."""
        self._ready()
        var importances = List[Values]()
        for m in range(len(self.base.estimators_)):
            if self.base.estimators_[m].tree_.node_count > 1:
                importances.append(self.base.estimators_[m].feature_importances_.copy())
        return mean_feature_importances(importances, self.n_features_in_)

    def estimators_samples_(self) raises -> List[Indices]:
        """In-bag rows per member, regenerated from member seeds (all rows without bootstrap)."""
        self._ready()
        var result = List[Indices]()
        for m in range(len(self.base.estimators_)):
            if self.bootstrap:
                result.append(generate_sample_indices(self.base.estimators_[m].params.random_state.value(),
                    self._n_samples, self._n_samples_bootstrap, self._sample_weight))
            else:
                var rows = nm.zeros[DType.int64](nm.Shape(self._n_samples))
                for i in range(self._n_samples):
                    rows.unsafe_set(i, Int64(i))
                result.append(rows^)
        return result^
