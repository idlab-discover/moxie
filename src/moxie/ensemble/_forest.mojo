# scikit-learn developers, BSD-3-Clause; see LICENSES/scikit-learn.txt.
"""Dense public forests over the shared forest engine."""
import numojo as nm
from moxie.tree._criterion import log
from moxie.base import BaseEstimator, Predictor, SupervisedEstimator, SupervisedScorer, WeightedSupervisedEstimator, LogProbabilityPredictor
from moxie._random import check_random_state
from moxie._labels import encode_column
from moxie.tree._classes import BaseDecisionTree, SampleCount, FeatureCount, ClassWeight, ClassWeightMap
from moxie.ensemble._base import ProbabilisticEnsemble, EnsembleRegressor
from moxie.ensemble._engine_api import DenseForestInspection
from moxie.ensemble._records import Features, Values, Indices
from moxie.ensemble._forest_engine import ForestEngine, ForestClassWeight
from moxie.ensemble._sampling import MaxSamples, class_weight_factors

struct _NullableForestParam[T: Copyable & Deinitable](Copyable, Movable):
    """Distinguish an omitted keyword from an explicit None reset."""
    var supplied: Bool
    var value: Optional[Self.T]

    def __init__(out self):
        self.supplied = False
        self.value = None

    @implicit
    def __init__(out self, var value: Self.T):
        self.supplied = True
        self.value = value^

    @implicit
    def __init__(out self, var value: Optional[Self.T]):
        self.supplied = True
        self.value = value^

    @implicit
    def __init__(out self, none: NoneType._mlir_type):
        self.supplied = True
        self.value = None


@fieldwise_init
struct ForestParams(Copyable, Movable):
    var n_estimators: Int
    var criterion: String
    var max_depth: Optional[Int]
    var min_samples_split: SampleCount
    var min_samples_leaf: SampleCount
    var min_weight_fraction_leaf: Float64
    var max_features: FeatureCount
    var max_leaf_nodes: Optional[Int]
    var min_impurity_decrease: Float64
    var bootstrap: Bool
    var oob_score: Bool
    var n_jobs: Optional[Int]
    var random_state: Optional[Int]
    var verbose: Int
    var warm_start: Bool
    var class_weight: ClassWeight
    var ccp_alpha: Float64
    var max_samples: MaxSamples
    var monotonic_cst: Optional[List[Int]]

struct Forest[classifier: Bool, extra: Bool = False, multi_output: Bool = False](BaseEstimator, Predictor, SupervisedEstimator, SupervisedScorer, WeightedSupervisedEstimator, DenseForestInspection, ProbabilisticEnsemble where classifier, LogProbabilityPredictor where classifier, EnsembleRegressor where not classifier, Sized):
    """Float32 features and Float64 targets; multi_output selects classifier result types."""
    comptime Params = ForestParams
    comptime Input = Features
    comptime Target = Values
    comptime Prediction = Values
    comptime Leaves = Indices
    comptime SampleWeight = Values
    comptime Tree = BaseDecisionTree[Self.classifier, Self.extra]
    comptime Classes = List[Values] if Self.multi_output else Values
    comptime ClassCounts = List[Int] if Self.multi_output else Int
    comptime Probabilities = List[Values] if Self.multi_output else Values
    var n_estimators: Int
    var criterion: String
    var max_depth: Optional[Int]
    var min_samples_split: SampleCount
    var min_samples_leaf: SampleCount
    var min_weight_fraction_leaf: Float64
    var max_features: FeatureCount
    var max_leaf_nodes: Optional[Int]
    var min_impurity_decrease: Float64
    var bootstrap: Bool
    var oob_score: Bool
    var n_jobs: Optional[Int]
    var random_state: Optional[Int]
    var verbose: Int
    var warm_start: Bool
    var class_weight: ClassWeight
    var ccp_alpha: Float64
    var max_samples: MaxSamples
    var monotonic_cst: Optional[List[Int]]
    var classes_: Self.Classes
    var n_classes_: Self.ClassCounts
    var n_features_in_: Int
    var n_outputs_: Int
    var feature_importances_: Values
    var oob_score_: Float64
    var oob_prediction_: Values
    var oob_decision_function_: Self.Probabilities
    var _engine: ForestEngine[Self.classifier, Self.extra]

    def __init__(out self, n_estimators: Int = 100, *,
            criterion: String = "gini" if Self.classifier else "squared_error",
            max_depth: Optional[Int] = None,
            min_samples_split: SampleCount = 2,
            min_samples_leaf: SampleCount = 1,
            min_weight_fraction_leaf: Float64 = 0,
            max_features: FeatureCount = FeatureCount("sqrt") if Self.classifier else FeatureCount(Float64(1)),
            max_leaf_nodes: Optional[Int] = None,
            min_impurity_decrease: Float64 = 0,
            bootstrap: Bool = not Self.extra,
            oob_score: Bool = False,
            n_jobs: Optional[Int] = None,
            random_state: Optional[Int] = None,
            verbose: Int = 0,
            warm_start: Bool = False,
            class_weight: ClassWeight = NoneType(),
            ccp_alpha: Float64 = 0,
            max_samples: MaxSamples = NoneType(),
            monotonic_cst: Optional[List[Int]] = None) raises:
        self.n_estimators = n_estimators.copy()
        self.criterion = criterion.copy()
        self.max_depth = max_depth.copy()
        self.min_samples_split = min_samples_split.copy()
        self.min_samples_leaf = min_samples_leaf.copy()
        self.min_weight_fraction_leaf = min_weight_fraction_leaf.copy()
        self.max_features = max_features.copy()
        self.max_leaf_nodes = max_leaf_nodes.copy()
        self.min_impurity_decrease = min_impurity_decrease.copy()
        self.bootstrap = bootstrap.copy()
        self.oob_score = oob_score.copy()
        self.n_jobs = n_jobs.copy()
        self.random_state = random_state.copy()
        self.verbose = verbose.copy()
        self.warm_start = warm_start.copy()
        self.class_weight = class_weight.copy()
        self.ccp_alpha = ccp_alpha.copy()
        self.max_samples = max_samples.copy()
        self.monotonic_cst = monotonic_cst.copy()
        self._engine = ForestEngine[Self.classifier, Self.extra](Self.Tree())
        self.n_features_in_ = 0
        self.n_outputs_ = 0
        self.feature_importances_ = nm.zeros[DType.float64](nm.Shape(0))
        self.oob_prediction_ = nm.zeros[DType.float64](nm.Shape(0))
        self.oob_score_ = Float64("nan")
        comptime if Self.multi_output:
            self.classes_ = rebind[Self.Classes](List[Values]()).copy()
            self.n_classes_ = rebind[Self.ClassCounts](List[Int]()).copy()
            self.oob_decision_function_ = rebind[Self.Probabilities](List[Values]()).copy()
        else:
            self.classes_ = rebind[Self.Classes](nm.zeros[DType.float64](nm.Shape(0))).copy()
            self.n_classes_ = rebind[Self.ClassCounts](Int(0)).copy()
            self.oob_decision_function_ = rebind[Self.Probabilities](nm.zeros[DType.float64](nm.Shape(0, 0))).copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_estimators.copy(), self.criterion.copy(), self.max_depth.copy(), self.min_samples_split.copy(), self.min_samples_leaf.copy(), self.min_weight_fraction_leaf.copy(), self.max_features.copy(), self.max_leaf_nodes.copy(), self.min_impurity_decrease.copy(), self.bootstrap.copy(), self.oob_score.copy(), self.n_jobs.copy(), self.random_state.copy(), self.verbose.copy(), self.warm_start.copy(), self.class_weight.copy(), self.ccp_alpha.copy(), self.max_samples.copy(), self.monotonic_cst.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_estimators = params.n_estimators.copy()
        self.criterion = params.criterion.copy()
        self.max_depth = params.max_depth.copy()
        self.min_samples_split = params.min_samples_split.copy()
        self.min_samples_leaf = params.min_samples_leaf.copy()
        self.min_weight_fraction_leaf = params.min_weight_fraction_leaf.copy()
        self.max_features = params.max_features.copy()
        self.max_leaf_nodes = params.max_leaf_nodes.copy()
        self.min_impurity_decrease = params.min_impurity_decrease.copy()
        self.bootstrap = params.bootstrap.copy()
        self.oob_score = params.oob_score.copy()
        self.n_jobs = params.n_jobs.copy()
        self.random_state = params.random_state.copy()
        self.verbose = params.verbose.copy()
        self.warm_start = params.warm_start.copy()
        self.class_weight = params.class_weight.copy()
        self.ccp_alpha = params.ccp_alpha.copy()
        self.max_samples = params.max_samples.copy()
        self.monotonic_cst = params.monotonic_cst.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        n_estimators: Optional[Int] = None,
        criterion: Optional[String] = None,
        max_depth: _NullableForestParam[Int] = _NullableForestParam[Int](),
        min_samples_split: Optional[SampleCount] = None,
        min_samples_leaf: Optional[SampleCount] = None,
        min_weight_fraction_leaf: Optional[Float64] = None,
        max_features: Optional[FeatureCount] = None,
        max_leaf_nodes: _NullableForestParam[Int] = _NullableForestParam[Int](),
        min_impurity_decrease: Optional[Float64] = None,
        bootstrap: Optional[Bool] = None,
        oob_score: Optional[Bool] = None,
        n_jobs: _NullableForestParam[Int] = _NullableForestParam[Int](),
        random_state: _NullableForestParam[Int] = _NullableForestParam[Int](),
        verbose: Optional[Int] = None,
        warm_start: Optional[Bool] = None,
        class_weight: Optional[ClassWeight] = None,
        ccp_alpha: Optional[Float64] = None,
        max_samples: Optional[MaxSamples] = None,
        monotonic_cst: _NullableForestParam[List[Int]] = _NullableForestParam[List[Int]](),
    ) raises -> ref[self] Self:
        """Update supplied configuration; fitted members remain until the next fit."""
        if n_estimators:
            self.n_estimators = n_estimators.value().copy()
        if criterion:
            self.criterion = criterion.value().copy()
        if max_depth.supplied:
            self.max_depth = max_depth.value.copy()
        if min_samples_split:
            self.min_samples_split = min_samples_split.value().copy()
        if min_samples_leaf:
            self.min_samples_leaf = min_samples_leaf.value().copy()
        if min_weight_fraction_leaf:
            self.min_weight_fraction_leaf = min_weight_fraction_leaf.value().copy()
        if max_features:
            self.max_features = max_features.value().copy()
        if max_leaf_nodes.supplied:
            self.max_leaf_nodes = max_leaf_nodes.value.copy()
        if min_impurity_decrease:
            self.min_impurity_decrease = min_impurity_decrease.value().copy()
        if bootstrap:
            self.bootstrap = bootstrap.value().copy()
        if oob_score:
            self.oob_score = oob_score.value().copy()
        if n_jobs.supplied:
            self.n_jobs = n_jobs.value.copy()
        if random_state.supplied:
            self.random_state = random_state.value.copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if warm_start:
            self.warm_start = warm_start.value().copy()
        if class_weight:
            self.class_weight = class_weight.value().copy()
        if ccp_alpha:
            self.ccp_alpha = ccp_alpha.value().copy()
        if max_samples:
            self.max_samples = max_samples.value().copy()
        if monotonic_cst.supplied:
            self.monotonic_cst = monotonic_cst.value.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        _ = result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._engine.is_fitted()

    def __len__(self) -> Int:
        return len(self._engine)

    def __getitem__(ref self, index: Int) raises -> ref[self._engine.base.estimators_[0]] Self.Tree:
        return self._engine.base[index]

    def estimators_(ref self) -> ref[self._engine.base.estimators_] List[Self.Tree]:
        return self._engine.base.estimators_

    def estimator_(ref self) -> ref[self._engine.base.estimator_] Self.Tree:
        return self._engine.base.estimator_

    def estimators_samples_(self) raises -> List[Indices]:
        return self._engine.estimators_samples_()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        return self.fit(X, y, None)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Values) raises -> ref[self] Self:
        return self.fit(X, y, Optional[Values](sample_weight.copy()))

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target, sample_weight: Optional[Values]) raises -> ref[self] Self:
        if self.verbose != 0:
            raise Error("NotImplementedError: verbose progress is not implemented")
        # n_jobs only distributes member building in sklearn; members are built
        # serially here, with the same seeds and therefore the same results.
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
        if y.ndim != 1 and y.ndim != 2:
            raise Error("ValueError: y must have rank one or two")
        var outputs = 1 if y.ndim == 1 else y.shape[1]
        if outputs == 0:
            raise Error("ValueError: y must have at least one output")
        comptime if Self.classifier and not Self.multi_output:
            if outputs != 1:
                raise Error("ValueError: use multi_output=True for multiple classification outputs")
        if self.criterion == "":
            raise Error("ValueError: criterion must name a supported impurity measure")
        var template = Self.Tree(criterion=self.criterion, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha, monotonic_cst=self.monotonic_cst)
        var target = y.copy()
        var domains = List[Values]()
        comptime if Self.classifier:
            target = nm.zeros[DType.float64](nm.Shape(y.shape[0], outputs))
            for k in range(outputs):
                var encoded = encode_column(y, k)
                domains.append(encoded.classes.copy())
                for i in range(y.shape[0]):
                    target.unsafe_set(i * outputs + k, Float64(encoded.index.inverse.unsafe_get(i)))
        var weights = ForestClassWeight(NoneType())
        if self.class_weight.isa[String]():
            weights = ForestClassWeight(self.class_weight[String])
        elif self.class_weight.isa[ClassWeightMap]() or self.class_weight.isa[List[ClassWeightMap]]():
            if self.class_weight.isa[ClassWeightMap]() and outputs != 1:
                raise Error("ValueError: multiple outputs require a list of class weight dictionaries")
            if self.class_weight.isa[List[ClassWeightMap]]() and len(self.class_weight[List[ClassWeightMap]]) != outputs:
                raise Error("ValueError: class weight count must match outputs")
            var factors = List[Values]()
            for k in range(len(domains)):
                if self.class_weight.isa[ClassWeightMap]():
                    factors.append(class_weight_factors(domains[k], self.class_weight[ClassWeightMap]))
                else:
                    factors.append(class_weight_factors(domains[k], self.class_weight[List[ClassWeightMap]][k]))
            weights = ForestClassWeight(factors^)
        comptime if not Self.classifier:
            if not self.class_weight.isa[NoneType]():
                raise Error("ValueError: regression does not support class_weight")
        self._engine.base.estimator = template^
        self._engine.base.n_estimators = self.n_estimators
        self._engine.bootstrap = self.bootstrap
        self._engine.oob_score = self.oob_score
        self._engine.warm_start = self.warm_start
        self._engine.max_samples = self.max_samples.copy()
        self._engine.class_weight = weights^
        var rng = check_random_state(self.random_state)
        self._engine.fit(X, target, sample_weight, rng[])
        self.n_features_in_ = self._engine.n_features_in_
        self.n_outputs_ = self._engine.n_outputs_
        self.feature_importances_ = self._engine.feature_importances_()
        comptime if Self.classifier:
            comptime if Self.multi_output:
                self.classes_ = rebind[Self.Classes](domains).copy()
                self.n_classes_ = rebind[Self.ClassCounts](self._engine.n_classes_.copy()).copy()
            else:
                self.classes_ = rebind[Self.Classes](domains[0].copy()).copy()
                self.n_classes_ = rebind[Self.ClassCounts](self._engine.n_classes_[0]).copy()
        if self._engine.has_oob():
            self.oob_score_ = self._engine.oob_score_
            comptime if Self.classifier:
                comptime if Self.multi_output:
                    self.oob_decision_function_ = rebind[Self.Probabilities](self._engine.oob_decision_function_.copy()).copy()
                else:
                    self.oob_decision_function_ = rebind[Self.Probabilities](self._engine.oob_decision_function_[0].copy()).copy()
            else:
                self.oob_prediction_ = self._engine.oob_prediction_.reshape(nm.Shape(y.shape[0])) if outputs == 1 else self._engine.oob_prediction_.copy()
        return self

    def apply(self, X: Features) raises -> Indices:
        return self._engine.apply(X)

    def predict_proba(self, X: Features) raises -> Self.Probabilities where Self.classifier:
        var result = self._engine.predict_proba(X)
        comptime if Self.multi_output:
            return rebind[Self.Probabilities](result).copy()
        else:
            return rebind[Self.Probabilities](result[0]).copy()

    def predict_log_proba(self, X: Features) raises -> Self.Probabilities where Self.classifier:
        var result = self.predict_proba(X)
        comptime if Self.multi_output:
            ref matrices = rebind[List[Values]](result)
            for k in range(len(matrices)):
                for i in range(matrices[k].size):
                    matrices[k].unsafe_set(i, log(matrices[k].unsafe_get(i)))
        else:
            ref matrix = rebind[Values](result)
            for i in range(matrix.size):
                matrix.unsafe_set(i, log(matrix.unsafe_get(i)))
        return result^

    def predict(self, X: Features) raises -> Values:
        comptime if not Self.classifier:
            var result = self._engine.predict(X)
            return result.reshape(nm.Shape(X.shape[0])) if self.n_outputs_ == 1 else result^
        else:
            var probabilities = self._engine.predict_proba(X)
            var result = nm.zeros[DType.float64](nm.Shape(X.shape[0]) if self.n_outputs_ == 1 else nm.Shape(X.shape[0], self.n_outputs_))
            for k in range(self.n_outputs_):
                ref probability = probabilities[k]
                for i in range(X.shape[0]):
                    var chosen = 0
                    for c in range(1, probability.shape[1]):
                        if probability.unsafe_get(i * probability.shape[1] + c) > probability.unsafe_get(i * probability.shape[1] + chosen):
                            chosen = c
                    comptime if Self.multi_output:
                        result.unsafe_set(i * self.n_outputs_ + k, rebind[List[Values]](self.classes_)[k].unsafe_get(chosen))
                    else:
                        result.unsafe_set(i, rebind[Values](self.classes_).unsafe_get(chosen))
            return result^

    def score(self, X: Features, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Float64:
        var prediction = self.predict(X)
        # The tree's scorer implements the same sklearn accuracy/R2 contract.
        var scorer = Self.Tree()
        scorer.n_outputs_ = self.n_outputs_
        return scorer._score(prediction, y, sample_weight)

comptime RandomForestClassifier[multi_output: Bool = False] = Forest[True, False, multi_output]
comptime RandomForestRegressor = Forest[False]
comptime ExtraTreesClassifier[multi_output: Bool = False] = Forest[True, True, multi_output]
comptime ExtraTreesRegressor = Forest[False, True]
