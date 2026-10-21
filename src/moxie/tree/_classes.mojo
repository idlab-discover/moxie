# scikit-learn developers, BSD-3-Clause; see family LICENSES/scikit-learn.txt.
"""Native dense decision trees. Typed labels, feature metadata and parameter records."""
import numojo as nm
from std.builtin.sort import sort
from std.math import isfinite, sqrt, ceil
from std.utils import Variant
from moxie.base import BaseEstimator, InputEstimator, Predictor, SupervisedEstimator, SupervisedScorer, WeightedSupervisedEstimator
from moxie.tree._dense_api import DenseDecisionTreeClassifierAPI, DenseDecisionTreeRegressorAPI, DenseTreeInspection
from moxie._labels import StringTarget, LabelIndex, encode_column, type_of_target
from moxie._categorical import NumericCategories, StringCategories, categorical_column
from moxie._random import RandomState, entropy_state
from moxie.tree._records import (Features, Values, TreeData, Indices, TrainingData,
    Mask, Constraints, CriterionConfig, SplitterConfig, BuilderConfig, PruningPath, TreeTable, StringColumn)
from moxie.tree._criterion import log
from moxie.tree._tree import Tree, new_tree
from moxie.tree._splitter import DenseSplitter
from moxie.tree._builders import DepthFirstTreeBuilder, prune, cost_complexity_pruning_path

comptime ClassWeightMap = Dict[Float64, Float64]
comptime ClassWeight = Variant[NoneType, String, ClassWeightMap, List[ClassWeightMap]]
comptime TypedClassWeight[Label: Copyable & Hashable & Equatable & Deinitable] = Variant[NoneType, String, Dict[Label, Float64], List[Dict[Label, Float64]]]
comptime CategoricalFeatures = Variant[NoneType, type_of(None), Optional[List[Int]], List[Int], List[Bool], List[String], String]
comptime SampleCount = Variant[Int, Float64]
comptime FeatureCount = Variant[Int, Float64, String, NoneType]

def resolve_samples(option: SampleCount, n: Int, minimum: Int, include_one: Bool = True) raises -> Int:
    """Resolve a count or fraction. sklearn's min_samples_leaf fraction excludes 1.0."""
    if option.isa[Int]():
        var count = option[Int]
        if count < minimum or count > 4611686018427387903:
            raise Error("ValueError: invalid minimum sample count")
        return count
    var fraction = option[Float64]
    if not (fraction > 0 and (fraction <= 1 if include_one else fraction < 1)):
        raise Error("ValueError: sample fraction must be in (0,1]" if include_one else "ValueError: min_samples_leaf fraction must be in (0,1)")
    return max(minimum, Int(ceil(fraction * Float64(n))))

@fieldwise_init
struct TreeParams[label_dtype: DType = DType.float64, string_labels: Bool = False](Copyable, Movable):
    comptime Label = String if Self.string_labels else Scalar[Self.label_dtype]
    var criterion: String
    var splitter: String
    var max_depth: Optional[Int]
    var min_samples_split: SampleCount
    var min_samples_leaf: SampleCount
    var min_weight_fraction_leaf: Float64
    var max_features: FeatureCount
    var random_state: Optional[Int]
    var max_leaf_nodes: Optional[Int]
    var min_impurity_decrease: Float64
    var class_weight: TypedClassWeight[Self.Label]
    var ccp_alpha: Float64
    var monotonic_cst: Optional[List[Int]]
    var categorical_features: CategoricalFeatures

struct TreeInputEncoder(Copyable, Movable):
    """Fitted feature validation and categorical encoding.

    Trees built from one PreparedTreeInput share this encoder, so an ensemble can
    encode prediction input once and pass it to every member's `*_encoded` method.
    """
    var n_features: Int
    var feature_names: Optional[List[String]]
    var is_categorical: Optional[List[Bool]]
    var numeric_domains: List[NumericCategories[DType.float32]]
    var string_domains: List[Optional[StringCategories]]

    def __init__(out self):
        self.n_features = 0
        self.feature_names = None
        self.is_categorical = None
        self.numeric_domains = List[NumericCategories[DType.float32]]()
        self.string_domains = List[Optional[StringCategories]]()

    def check_feature_names(self, names: Optional[List[String]]) raises:
        if Bool(self.feature_names) and Bool(names):
            if len(names.value()) != len(self.feature_names.value()):
                raise Error("ValueError: feature names differ from fitted names")
            for j in range(len(names.value())):
                if names.value()[j] != self.feature_names.value()[j]:
                    raise Error("ValueError: feature names must match fitted names and order")

    def check_encoded(self, X: Features) raises:
        """Shape-only check for already encoded input, like sklearn check_input=False."""
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.n_features:
            raise Error("ValueError: prediction requires nonempty X with fitted feature count")

    def encode(self, X: Features, reject_missing: Bool, monotonic: Bool) raises -> Features:
        """Validate X and map categorical columns to fitted codes (unknown -> NaN).

        `reject_missing` (absolute_error) and `monotonic` reflect the caller's current parameters.
        """
        for domain in self.string_domains:
            if domain:
                raise Error("ValueError: model fitted with string columns requires TreeTable input")
        return self._encode_numeric(X, reject_missing, monotonic)

    def encode(self, X: TreeTable, reject_missing: Bool, monotonic: Bool) raises -> Features:
        self.check_feature_names(X.feature_names)
        if X.shape[1] != self.n_features or X.shape[0] == 0:
            raise Error("ValueError: prediction requires nonempty X with fitted feature count")
        var matrix = nm.zeros[DType.float32](X.shape)
        for j in range(X.shape[1]):
            var strings = len(self.string_domains) != 0 and Bool(self.string_domains[j])
            if strings != X._columns[j].isa[StringColumn]():
                raise Error("ValueError: table column domain differs from fitted input")
            if strings:
                for i in range(X.shape[0]):
                    var value = X._columns[j][StringColumn].get(i)
                    var code = self.string_domains[j].value().find(value.value()) if value else -1
                    matrix.unsafe_set(i * X.shape[1] + j, Float32(code) if code >= 0 else Float32(Float64("nan")))
            else:
                ref column = X._columns[j][Features]
                for i in range(X.shape[0]):
                    matrix.unsafe_set(i * X.shape[1] + j, column.unsafe_get(i * Int(column.strides[0])))
        return self._encode_numeric(matrix, reject_missing, monotonic)

    def _encode_numeric(self, X: Features, reject_missing: Bool, monotonic: Bool) raises -> Features:
        self.check_encoded(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var x = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                if x == x and not isfinite(x):
                    raise Error("ValueError: infinite features")
                if x != x and reject_missing:
                    raise Error("ValueError: absolute_error does not support missing features")
                if x != x and monotonic:
                    raise Error("ValueError: NaN with monotonic constraints")
        if not self.is_categorical:
            return X.view_with_layout(X.shape, X.strides, X.offset)
        var encoded = X.copy()
        for j in range(X.shape[1]):
            if self.is_categorical.value()[j]:
                for i in range(X.shape[0]):
                    var x = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                    var code = self.numeric_domains[j].find(x) if x == x else -1
                    if code < 0 and monotonic:
                        raise Error("ValueError: unknown categories with monotonic constraints are unsupported")
                    encoded.unsafe_set(i * Int(encoded.strides[0]) + j * Int(encoded.strides[1]), Float32(code) if code >= 0 else Float32(Float64("nan")))
        return encoded^

@fieldwise_init
struct PreparedTreeInput[label_dtype: DType = DType.float64, string_labels: Bool = False](Movable):
    """Validated, encoded training input from which several trees can be built.

    Created by `BaseDecisionTree.prepare_fit`; consumed read-only by `fit_prepared`.
    X is the encoded feature matrix (a shared view of numeric caller input), y holds
    Float64 targets or class codes, sample_weight the validated caller weights and
    class_weight the expanded per-sample class factors (ones without class_weight).
    """
    comptime ClassLabels = List[String] if Self.string_labels else nm.NDArray[Self.label_dtype]
    var X: Features
    var y: Values
    var sample_weight: Optional[Values]
    var class_weight: Values
    var missing_values_in_feature_mask: Mask
    var n_categories: Indices
    var monotonic_cst: Constraints
    var classes: List[Self.ClassLabels]
    var n_classes: List[Int]
    var encoder: TreeInputEncoder
    var key: String

    def n_samples(self) raises -> Int:
        return self.X.shape[0]

def check_sample_weight(sample_weight: Values, n: Int) raises:
    if sample_weight.ndim != 1 or sample_weight.size != n:
        raise Error("ValueError: sample_weight shape mismatch")
    var any_nonzero = False
    for i in range(n):
        var w = sample_weight.unsafe_get(i * Int(sample_weight.strides[0]))
        if not isfinite(w):
            raise Error("ValueError: nonfinite sample weight")
        any_nonzero = any_nonzero or w != 0
    if not any_nonzero:
        raise Error("ValueError: sample weights must contain at least one nonzero number")

struct BaseDecisionTree[classifier: Bool, extra: Bool = False, label_dtype: DType = DType.float64, string_labels: Bool = False](BaseEstimator, InputEstimator, Predictor, SupervisedEstimator, SupervisedScorer, WeightedSupervisedEstimator, DenseTreeInspection, DenseDecisionTreeClassifierAPI where classifier, DenseDecisionTreeRegressorAPI where not classifier):
    comptime Label = String if Self.string_labels else Scalar[Self.label_dtype]
    comptime ClassLabels = List[String] if Self.string_labels else nm.NDArray[Self.label_dtype]
    comptime WeightMap = Dict[Self.Label, Float64]
    comptime Weights = TypedClassWeight[Self.Label]
    comptime Params = TreeParams[Self.label_dtype, Self.string_labels]
    comptime Prepared = PreparedTreeInput[Self.label_dtype, Self.string_labels]
    comptime Input = Features
    comptime Target = StringTarget if Self.string_labels else nm.NDArray[Self.label_dtype]
    comptime Prediction = Self.Target
    comptime Leaves = Indices
    comptime SampleWeight = Values
    comptime Probabilities = List[Values]
    var params: Self.Params
    var tree_: TreeData
    var classes_: List[Self.ClassLabels]
    var n_classes_: List[Int]
    var n_outputs_: Int
    var feature_names_in_: Optional[List[String]]
    var n_features_in_: Int
    var max_features_: Int
    var feature_importances_: Values
    var is_categorical_: Optional[List[Bool]]
    var _encoder: TreeInputEncoder
    var _fitted: Bool

    def __init__(out self, *, criterion: String = "", splitter: String = "random" if Self.extra else "best",
        max_depth: Optional[Int] = None, min_samples_split: SampleCount = 2,
        min_samples_leaf: SampleCount = 1, min_weight_fraction_leaf: Float64 = 0,
        max_features: FeatureCount = FeatureCount("sqrt") if Self.extra and Self.classifier else (FeatureCount(Float64(1)) if Self.extra else FeatureCount(NoneType())), random_state: Optional[Int] = None,
        max_leaf_nodes: Optional[Int] = None, min_impurity_decrease: Float64 = 0,
        class_weight: Self.Weights = NoneType(), ccp_alpha: Float64 = 0,
        monotonic_cst: Optional[List[Int]] = None, categorical_features: CategoricalFeatures = NoneType()) raises:
        comptime assert Self.classifier or (Self.label_dtype == DType.float64 and not Self.string_labels), "regression targets use Float64"
        comptime assert Self.string_labels or Self.label_dtype in (DType.int64, DType.float32, DType.float64), "labels require Int64, Float32, Float64 or strings"
        var selected = criterion
        if selected == "":
            selected = "gini" if Self.classifier else "squared_error"
        elif not Self.classifier and selected == "friedman_mse":
            # sklearn's deprecated alias maps here; its FutureWarning is not emitted.
            selected = "squared_error"
        self.params = Self.Params(selected, splitter, max_depth, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
            min_impurity_decrease, class_weight.copy(), ccp_alpha, monotonic_cst.copy(), categorical_features.copy())
        self.tree_ = new_tree(1, [1], [-1])
        self.classes_ = List[Self.ClassLabels]()
        self.n_classes_ = List[Int]()
        self.n_outputs_ = 0
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.max_features_ = 0
        self.feature_importances_ = nm.zeros[DType.float64](nm.Shape(0))
        self.is_categorical_ = None
        self._encoder = TreeInputEncoder()
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self
    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^
    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted
    def _ready(self) raises:
        if not self._fitted:
            raise Error("NotFittedError: call fit before prediction or inspection")

    # Mojo 1.0 needs rebind to refine these conditional associated types.
    def _target_rows(self, y: Self.Target) raises -> Int:
        comptime if Self.string_labels:
            return rebind[StringTarget](y).rows()
        else:
            return rebind[nm.NDArray[Self.label_dtype]](y).shape[0]

    def _target_columns(self, y: Self.Target) raises -> Int:
        comptime if Self.string_labels:
            return rebind[StringTarget](y).columns()
        else:
            ref array = rebind[nm.NDArray[Self.label_dtype]](y)
            if array.ndim != 1 and array.ndim != 2:
                raise Error("ValueError: targets must be a vector or matrix")
            return 1 if array.ndim == 1 else array.shape[1]

    def _target_kind(self, y: Self.Target) raises -> String:
        comptime if Self.string_labels:
            return type_of_target(rebind[StringTarget](y))
        else:
            return type_of_target(rebind[nm.NDArray[Self.label_dtype]](y))

    def _target_value(self, y: Self.Target, row: Int, column: Int) raises -> Self.Label:
        comptime if Self.string_labels:
            var value = rebind[StringTarget](y).get(row, column)
            return rebind[Self.Label](value).copy()
        else:
            ref array = rebind[nm.NDArray[Self.label_dtype]](y)
            var offset = row * Int(array.strides[0])
            if array.ndim == 2:
                offset += column * Int(array.strides[1])
            return rebind[Self.Label](array.unsafe_get(offset)).copy()

    def _label_at(self, labels: Self.ClassLabels, index: Int) raises -> Self.Label:
        comptime if Self.string_labels:
            return rebind[Self.Label](rebind[List[String]](labels)[index]).copy()
        else:
            return rebind[Self.Label](rebind[nm.NDArray[Self.label_dtype]](labels).unsafe_get(index)).copy()

    def _class_label(self, output: Int, index: Int) raises -> Self.Label:
        return self._label_at(self.classes_[output], index)

    def _class_weight(self, weights: Self.WeightMap, label: Self.Label) raises -> Optional[Float64]:
        comptime if Self.string_labels:
            ref mapping = rebind[Dict[String, Float64]](weights)
            ref key = rebind[String](label)
            if key in mapping:
                return mapping[key]
        else:
            ref mapping = rebind[Dict[Scalar[Self.label_dtype], Float64]](weights)
            var key = rebind[Scalar[Self.label_dtype]](label)
            if key in mapping:
                return mapping[key]
        return None

    def _encode_target(self, y: Self.Target, column: Int) raises -> Tuple[Self.ClassLabels, LabelIndex]:
        comptime if Self.string_labels:
            var encoding = encode_column(rebind[StringTarget](y), column)
            return (rebind[Self.ClassLabels](encoding.classes).copy(), encoding.index.copy())
        else:
            var encoding = encode_column(rebind[nm.NDArray[Self.label_dtype]](y), column)
            return (rebind[Self.ClassLabels](encoding.classes).copy(), encoding.index.copy())

    def _resolve_categories(self, n: Int, names: Optional[List[String]] = None, dtypes: Optional[List[Bool]] = None) raises -> List[Bool]:
        if names and len(names.value()) != n:
            raise Error("ValueError: feature names must match feature count")
        if dtypes and len(dtypes.value()) != n:
            raise Error("ValueError: categorical dtype mask must match feature count")
        var selected = List[Bool]()
        for _ in range(n):
            selected.append(False)
        ref option = self.params.categorical_features
        if option.isa[NoneType]() or option.isa[type_of(None)]():
            return selected^
        if option.isa[String]():
            if option[String] != "from_dtype":
                raise Error("ValueError: categorical_features string must be from_dtype")
            return dtypes.value().copy() if dtypes else selected^
        if option.isa[List[Bool]]():
            ref mask = option[List[Bool]]
            if len(mask) == 0:
                return selected^
            if len(mask) != n:
                raise Error("ValueError: categorical mask must match feature count")
            return mask.copy()
        if option.isa[List[String]]():
            for name in option[List[String]]:
                if not names:
                    raise Error("ValueError: categorical column names require feature names")
                var found = False
                for j in range(n):
                    if names.value()[j] == name:
                        selected[j] = True
                        found = True
                        break
                if not found:
                    raise Error("ValueError: unknown categorical feature name: " + name)
            return selected^
        var indices = List[Int]()
        if option.isa[List[Int]]():
            indices = option[List[Int]].copy()
        elif option[Optional[List[Int]]]:
            indices = option[Optional[List[Int]]].value().copy()
        for j in indices:
            if j < 0 or j >= n:
                raise Error("ValueError: categorical feature index out of range")
            selected[j] = True
        return selected^

    def _label_text(self, label: Self.Label) -> String:
        comptime if Self.string_labels:
            return rebind[String](label).copy()
        else:
            return String(rebind[Scalar[Self.label_dtype]](label))

    def _weight_map_text(self, weights: Self.WeightMap) raises -> String:
        var entries = List[String]()
        comptime if Self.string_labels:
            ref mapping = rebind[Dict[String, Float64]](weights)
            for item in mapping.items():
                entries.append(repr(item.key) + "=" + String(item.value))
        else:
            ref mapping = rebind[Dict[Scalar[Self.label_dtype], Float64]](weights)
            for item in mapping.items():
                entries.append(String(item.key) + "=" + String(item.value))
        sort(entries)
        var text = String("{")
        for entry in entries:
            text += entry + ","
        return text + "}"

    def _preparation_key(self) raises -> String:
        """Parameters that shape a PreparedTreeInput; build-only parameters are excluded."""
        var key = self.params.criterion + "|" + self.params.splitter + "|"
        if self.params.monotonic_cst:
            for sign in self.params.monotonic_cst.value():
                key += String(sign) + ","
        else:
            key += "None"
        key += "|"
        ref option = self.params.categorical_features
        if option.isa[NoneType]() or option.isa[type_of(None)]():
            key += "None"
        elif option.isa[String]():
            key += repr(option[String])
        elif option.isa[List[Bool]]():
            for selected in option[List[Bool]]:
                key += "T" if selected else "F"
        elif option.isa[List[String]]():
            for name in option[List[String]]:
                key += repr(name) + ","
        else:
            var indices = List[Int]()
            if option.isa[List[Int]]():
                indices = option[List[Int]].copy()
            elif option[Optional[List[Int]]]:
                indices = option[Optional[List[Int]]].value().copy()
            key += "i"
            for j in indices:
                key += String(j) + ","
        key += "|"
        ref weights = self.params.class_weight
        if weights.isa[NoneType]():
            key += "None"
        elif weights.isa[String]():
            key += repr(weights[String])
        elif weights.isa[Self.WeightMap]():
            key += self._weight_map_text(weights[Self.WeightMap])
        else:
            for mapping in weights[List[Self.WeightMap]]:
                key += self._weight_map_text(mapping) + ";"
        return key^

    def _build_settings(self, n: Int, p: Int) raises -> Int:
        """Validate build-only parameters for n samples and p features; return max_features_."""
        _ = resolve_samples(self.params.min_samples_split, n, 2)
        _ = resolve_samples(self.params.min_samples_leaf, n, 1, False)
        if not (self.params.min_weight_fraction_leaf >= 0 and self.params.min_weight_fraction_leaf <= 0.5) or not (self.params.min_impurity_decrease >= 0) or not (self.params.ccp_alpha >= 0):
            raise Error("ValueError: invalid impurity or weight parameter")
        if self.params.max_depth and self.params.max_depth.value() < 1:
            raise Error("ValueError: max_depth must be positive")
        if self.params.max_leaf_nodes and self.params.max_leaf_nodes.value() < 2:
            raise Error("ValueError: max_leaf_nodes must be at least two")
        if self.params.splitter != "best" and self.params.splitter != "random":
            raise Error("ValueError: splitter must be best or random")
        var features = p
        if self.params.max_features.isa[Int]():
            features = self.params.max_features[Int]
        elif self.params.max_features.isa[Float64]():
            var fraction = self.params.max_features[Float64]
            if not (fraction > 0 and fraction <= 1):
                raise Error("ValueError: max_features fraction must be in (0,1]")
            features = max(1, Int(fraction * Float64(p)))
        elif self.params.max_features.isa[String]():
            if self.params.max_features[String] == "sqrt":
                features = max(1, Int(sqrt(Float64(p))))
            elif self.params.max_features[String] == "log2":
                features = max(1, Int(log(Float64(p)) / log(Float64(2))))
            else:
                raise Error("ValueError: max_features must be sqrt, log2, None, integer or fraction")
        if features < 1:
            raise Error("ValueError: max_features must be positive")
        return features
    def _fit_kind[kind: Int](self, data: TrainingData, nc: List[Int], features: Int, weight: Float64, mut rng: RandomState) raises -> TreeData:
        var min_leaf = resolve_samples(self.params.min_samples_leaf, data.X.shape[0], 1, False)
        var min_split = resolve_samples(self.params.min_samples_split, data.X.shape[0], 2)
        var splitter = DenseSplitter[kind](CriterionConfig(data.y.shape[1], nc.copy(), data.y.shape[0]),
            SplitterConfig(features, min_leaf, weight * self.params.min_weight_fraction_leaf), self.params.splitter == "random")
        var depth = 2147483647
        if self.params.max_depth:
            depth = self.params.max_depth.value()
        var leaf_limit = -1
        if self.params.max_leaf_nodes:
            leaf_limit = self.params.max_leaf_nodes.value()
        var builder = DepthFirstTreeBuilder[kind](splitter^, BuilderConfig(max(min_split, 2 * min_leaf),
            min_leaf, weight * self.params.min_weight_fraction_leaf, depth, self.params.min_impurity_decrease, leaf_limit))
        var categories = List[Int]()
        for j in range(data.X.shape[1]):
            categories.append(Int(data.n_categories.unsafe_get(j * Int(data.n_categories.strides[0]))))
        var tree = new_tree(data.X.shape[1], nc, categories)
        builder.build(tree, data, rng)
        if self.params.ccp_alpha > 0:
            tree = prune(tree, self.params.ccp_alpha)
        return tree^

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        return self.fit(X, y, Optional[Values]())

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Values) raises -> ref[self] Self:
        return self.fit(X, y, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target, sample_weight: Optional[Values]) raises -> ref[self] Self:
        var rng = entropy_state()
        if self.params.random_state:
            rng = RandomState(self.params.random_state.value())
        return self.fit_with_random_state(X, y, rng, sample_weight)

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Features, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Values] = None) raises -> ref[self] Self:
        var prepared = self._prepare_dense(X, y, sample_weight)
        self._build_prepared(prepared, rng, None)
        return self

    def prepare_fit(self, X: Features, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Self.Prepared:
        """Validate and encode training input once for several `fit_prepared` builds."""
        return self._prepare_dense(X, y, sample_weight)

    def prepare_fit(self, X: Features, y: Self.Target, *, sample_weight: Values) raises -> Self.Prepared:
        return self._prepare_dense(X, y, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))

    def fit_prepared[origin: MutOrigin, prepared_origin: ImmOrigin](ref[origin] self, ref[prepared_origin] prepared: Self.Prepared, sample_weight: Optional[Values] = None) raises -> ref[self] Self:
        """Build from prepared input, seeding from random_state like fit.

        sample_weight replaces the prepared caller weights for this build, as sklearn
        forests pass bootstrap counts to `tree._fit(check_input=False)`; class weights
        expanded at preparation still multiply it.
        """
        var rng = entropy_state()
        if self.params.random_state:
            rng = RandomState(self.params.random_state.value())
        self._build_prepared(prepared, rng, sample_weight)
        return self

    def fit_prepared[origin: MutOrigin, prepared_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[prepared_origin] prepared: Self.Prepared, *, ref[sample_weight_origin] sample_weight: Values) raises -> ref[self] Self:
        return self.fit_prepared(prepared, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))

    def fit_prepared_with_random_state[origin: MutOrigin, prepared_origin: ImmOrigin](ref[origin] self, ref[prepared_origin] prepared: Self.Prepared, mut rng: RandomState, sample_weight: Optional[Values] = None) raises -> ref[self] Self:
        self._build_prepared(prepared, rng, sample_weight)
        return self

    def fit_prepared_with_random_state[origin: MutOrigin, prepared_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[prepared_origin] prepared: Self.Prepared, mut rng: RandomState, *, ref[sample_weight_origin] sample_weight: Values) raises -> ref[self] Self:
        self._build_prepared(prepared, rng, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))
        return self

    def _prepare_dense(self, X: Features, y: Self.Target, sample_weight: Optional[Values], names: Optional[List[String]] = None, dtypes: Optional[List[Bool]] = None) raises -> Self.Prepared:
        if X.ndim != 2 or X.shape[0] <= 0 or X.shape[1] <= 0:
            raise Error("ValueError: X must be a nonempty matrix")
        var n = X.shape[0]
        var p = X.shape[1]
        var outputs = self._target_columns(y)
        if self._target_rows(y) != n:
            raise Error("ValueError: y must have one row per sample")
        if outputs <= 0:
            raise Error("ValueError: y requires at least one output")
        _ = self._build_settings(n, p)
        var categorical = self._resolve_categories(p, names, dtypes)
        var has_categorical = False
        for selected in categorical:
            has_categorical = has_categorical or selected
        if has_categorical and self.params.criterion == "absolute_error":
            raise Error("ValueError: categorical features do not support absolute_error")
        if not self.params.class_weight.isa[NoneType]():
            if not Self.classifier:
                raise Error("ValueError: class_weight is classifier-only")
            if self.params.class_weight.isa[String]():
                if self.params.class_weight[String] != "balanced":
                    raise Error("ValueError: class_weight string must be balanced")
            elif outputs > 1:
                if not self.params.class_weight.isa[List[Self.WeightMap]]():
                    raise Error("ValueError: multi-output class_weight requires one dictionary per output")
                if len(self.params.class_weight[List[Self.WeightMap]]) != outputs:
                    raise Error("ValueError: class_weight output count mismatch")
            elif not self.params.class_weight.isa[Self.WeightMap]():
                raise Error("ValueError: single-output class_weight requires a dictionary")
        comptime if Self.classifier:
            if self.params.criterion != "gini" and self.params.criterion != "entropy" and self.params.criterion != "log_loss":
                raise Error("ValueError: invalid classification criterion")
        else:
            if self.params.criterion != "squared_error" and self.params.criterion != "poisson" and self.params.criterion != "absolute_error":
                raise Error("ValueError: invalid regression criterion")
        var missing = nm.zeros[DType.uint8](nm.Shape(p))
        var signs = nm.zeros[DType.int8](nm.Shape(p))
        if self.params.monotonic_cst:
            if len(self.params.monotonic_cst.value()) != p or outputs != 1:
                raise Error("ValueError: monotonic constraints require matching features and single output")
            for j in range(p):
                var sign = self.params.monotonic_cst.value()[j]
                if sign < -1 or sign > 1:
                    raise Error("ValueError: monotonic signs must be -1, 0 or 1")
                signs.unsafe_set(j, Int8(-sign if Self.classifier else sign))
        for i in range(n):
            for j in range(p):
                var x = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                if x != x:
                    if self.params.criterion == "absolute_error":
                        raise Error("ValueError: absolute_error does not support missing features")
                    missing.unsafe_set(j, 1)
                    if self.params.monotonic_cst:
                        raise Error("ValueError: missing values with monotonic constraints are unsupported")
                elif not isfinite(x):
                    raise Error("ValueError: infinite features")
        var target = nm.zeros[DType.float64](nm.Shape(n, outputs))
        var weights = Optional[Values]()
        if sample_weight:
            check_sample_weight(sample_weight.value(), n)
            var copied = nm.zeros[DType.float64](nm.Shape(n))
            for i in range(n):
                copied.unsafe_set(i, sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0])))
            weights = copied^
        var expanded_weights = nm.full[DType.float64](nm.Shape(n), 1)
        var poisson_sum = Float64(0)
        var classes = List[Self.ClassLabels]()
        var nc = List[Int]()
        for k in range(outputs):
            comptime if Self.classifier:
                var encoded_labels = self._encode_target(y, k)
                var labels = encoded_labels[0].copy()
                var index = encoded_labels[1].copy()
                var count = index.counts.size
                var class_weights = Self.WeightMap()
                if self.params.class_weight.isa[Self.WeightMap]():
                    class_weights = self.params.class_weight[Self.WeightMap].copy()
                elif self.params.class_weight.isa[List[Self.WeightMap]]():
                    class_weights = self.params.class_weight[List[Self.WeightMap]][k].copy()
                var matched = 0
                for c in range(count):
                    var label = self._label_at(labels, c)
                    if self._class_weight(class_weights, label):
                        matched += 1
                if matched < count and matched != len(class_weights):
                    raise Error("ValueError: class_weight has unknown keys and missing classes")
                for i in range(n):
                    var code = Int(index.inverse.unsafe_get(i))
                    target.unsafe_set(i * outputs + k, Float64(code))
                    var factor = Float64(1)
                    if self.params.class_weight.isa[String]():
                        factor = Float64(n) / (Float64(count) * Float64(index.counts.unsafe_get(code)))
                    else:
                        var label = self._target_value(y, i, k)
                        var configured = self._class_weight(class_weights, label)
                        if configured:
                            factor = configured.value()
                    if not isfinite(factor):
                        raise Error("ValueError: class weights must be finite")
                    expanded_weights.unsafe_set(i, expanded_weights.unsafe_get(i) * factor)
                if self.params.monotonic_cst and count > 2:
                    raise Error("ValueError: multiclass monotonic constraints are unsupported")
                classes.append(labels^)
                nc.append(count)
            else:
                nc.append(1)
                for i in range(n):
                    var value = Float64(rebind[Scalar[Self.label_dtype]](self._target_value(y, i, k)))
                    if not isfinite(value):
                        raise Error("ValueError: nonfinite target")
                    target.unsafe_set(i * outputs + k, value)
                    if self.params.criterion == "poisson":
                        if value < 0:
                            raise Error("ValueError: Poisson requires nonnegative targets")
                        poisson_sum += value
        if self.params.criterion == "poisson" and poisson_sum <= 0:
            raise Error("ValueError: Poisson requires a positive target sum")
        if has_categorical and self.params.splitter == "best":
            if outputs != 1 or (Self.classifier and nc[0] > 2):
                raise Error("ValueError: best categorical splits require single-output regression or binary classification")
        var encoded = X.copy() if has_categorical else X.view_with_layout(X.shape, X.strides, X.offset)
        var domains = List[NumericCategories[DType.float32]]()
        var counts = nm.full[DType.int64](nm.Shape(p), -1)
        for j in range(p):
            var column = nm.zeros[DType.float32](nm.Shape(n if categorical[j] else 0))
            if categorical[j]:
                if signs.unsafe_get(j) != 0:
                    raise Error("ValueError: categorical features cannot have monotonic constraints")
                for i in range(n):
                    column.unsafe_set(i, X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
            var values = categorical_column(column)
            var domain = values.domain()
            if categorical[j]:
                var limit = 256 if self.params.splitter == "best" else 16777216
                if domain.size() > limit:
                    raise Error("ValueError: categorical feature exceeds category limit")
                counts.unsafe_set(j, Int64(domain.size()))
                var codes = values.codes()
                for i in range(n):
                    var code = codes.unsafe_get(i)
                    encoded.unsafe_set(i * Int(encoded.strides[0]) + j * Int(encoded.strides[1]), Float32(code) if code >= 0 else Float32(Float64("nan")))
            domains.append(domain^)
        var encoder = TreeInputEncoder()
        encoder.n_features = p
        encoder.feature_names = names.copy()
        encoder.is_categorical = Optional[List[Bool]](categorical^) if has_categorical else None
        encoder.numeric_domains = domains^
        return Self.Prepared(encoded^, target^, weights^, expanded_weights^, missing^, counts^, signs^,
            rebind[List[Self.Prepared.ClassLabels]](classes).copy(), nc^, encoder^, self._preparation_key())

    def _build_prepared(mut self, prepared: Self.Prepared, mut rng: RandomState, sample_weight: Optional[Values]) raises:
        var n = prepared.X.shape[0]
        var p = prepared.X.shape[1]
        var features = self._build_settings(n, p)
        if self._preparation_key() != prepared.key:
            raise Error("ValueError: prepared input was validated for different criterion, splitter, monotonic_cst, categorical_features or class_weight")
        if sample_weight:
            check_sample_weight(sample_weight.value(), n)
        var weights = nm.zeros[DType.float64](nm.Shape(n))
        var weight = Float64(0)
        for i in range(n):
            var w = Float64(1)
            if sample_weight:
                w = sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))
            elif prepared.sample_weight:
                w = prepared.sample_weight.value().unsafe_get(i)
            w *= prepared.class_weight.unsafe_get(i)
            weights.unsafe_set(i, w)
            weight += w
        if not isfinite(weight):
            raise Error("ValueError: total weight must be finite")
        ref nc = prepared.n_classes
        var data = TrainingData(prepared.X.view_with_layout(prepared.X.shape, prepared.X.strides, prepared.X.offset),
            prepared.y.view_with_layout(prepared.y.shape, prepared.y.strides, prepared.y.offset), weights^,
            prepared.missing_values_in_feature_mask.copy(), prepared.n_categories.copy(), prepared.monotonic_cst.copy())
        var categories = List[Int]()
        for _ in range(p):
            categories.append(-1)
        var fitted = new_tree(p, nc, categories)
        comptime if Self.classifier:
            if self.params.criterion == "gini":
                fitted = self._fit_kind[0](data, nc, features, weight, rng)
            else:
                fitted = self._fit_kind[1](data, nc, features, weight, rng)
        else:
            if self.params.criterion == "poisson":
                fitted = self._fit_kind[4](data, nc, features, weight, rng)
            elif self.params.criterion == "absolute_error":
                fitted = self._fit_kind[3](data, nc, features, weight, rng)
            else:
                fitted = self._fit_kind[2](data, nc, features, weight, rng)
        var importances = Tree().compute_feature_importances(fitted)
        self.is_categorical_ = prepared.encoder.is_categorical.copy()
        self._encoder = prepared.encoder.copy()
        self.tree_ = fitted^
        self.classes_ = rebind[List[Self.ClassLabels]](prepared.classes).copy()
        self.n_classes_ = nc.copy()
        self.n_outputs_ = prepared.y.shape[1]
        self.n_features_in_ = p
        self.feature_names_in_ = prepared.encoder.feature_names.copy()
        self.max_features_ = features
        self.feature_importances_ = importances^
        self._fitted = True
    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: TreeTable, ref[y_origin] y: Self.Target, sample_weight: Optional[Values] = None) raises -> ref[self] Self:
        var rng = entropy_state()
        if self.params.random_state:
            rng = RandomState(self.params.random_state.value())
        return self.fit_with_random_state(X, y, rng, sample_weight)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: TreeTable, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Values) raises -> ref[self] Self:
        return self.fit(X, y, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: TreeTable, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Values] = None) raises -> ref[self] Self:
        var prepared = self.prepare_fit(X, y, sample_weight)
        self._build_prepared(prepared, rng, None)
        return self

    def prepare_fit(self, X: TreeTable, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Self.Prepared:
        var selected = self._resolve_categories(X.shape[1], X.feature_names, Optional[List[Bool]](X.categorical_mask.copy()))
        var matrix = nm.zeros[DType.float32](X.shape)
        var domains = List[Optional[StringCategories]]()
        for j in range(X.shape[1]):
            if X._columns[j].isa[StringColumn]():
                if not selected[j]:
                    raise Error("ValueError: string columns must be selected in categorical_features")
                var values = categorical_column(X._columns[j][StringColumn].values())
                var limit = 256 if self.params.splitter == "best" else 16777216
                if values.domain().size() > limit:
                    raise Error("ValueError: categorical feature exceeds category limit")
                var codes = values.codes()
                domains.append(Optional[StringCategories](values.domain()))
                for i in range(X.shape[0]):
                    var code = codes.unsafe_get(i)
                    matrix.unsafe_set(i * X.shape[1] + j, Float32(code) if code >= 0 else Float32(Float64("nan")))
            else:
                domains.append(None)
                ref column = X._columns[j][Features]
                for i in range(X.shape[0]):
                    matrix.unsafe_set(i * X.shape[1] + j, column.unsafe_get(i * Int(column.strides[0])))
        var prepared = self._prepare_dense(matrix, y, sample_weight, X.feature_names, Optional[List[Bool]](X.categorical_mask.copy()))
        prepared.encoder.string_domains = domains^
        return prepared^

    def prepare_fit(self, X: TreeTable, y: Self.Target, *, sample_weight: Values) raises -> Self.Prepared:
        return self.prepare_fit(X, y, Optional[Values](sample_weight.view_with_layout(sample_weight.shape, sample_weight.strides, sample_weight.offset)))

    def _encode_X(self, X: TreeTable) raises -> Features:
        self._ready()
        return self._encoder.encode(X, self.params.criterion == "absolute_error", Bool(self.params.monotonic_cst))

    def _encode_X(self, X: Features) raises -> Features:
        self._ready()
        return self._encoder.encode(X, self.params.criterion == "absolute_error", Bool(self.params.monotonic_cst))

    def input_encoder(self) raises -> TreeInputEncoder:
        """The fitted encoder; shared by every tree built from the same prepared input."""
        self._ready()
        return self._encoder.copy()

    # Encoded-input entry points skip validation and encoding except the shape check,
    # matching sklearn's check_input=False; X must come from this tree's encoder.
    def apply_encoded(self, X: Features) raises -> Indices:
        self._ready()
        self._encoder.check_encoded(X)
        return Tree().apply(self.tree_, X)

    def predict_encoded(self, X: Features) raises -> Self.Prediction:
        self._ready()
        self._encoder.check_encoded(X)
        return self._predict(X)

    def predict_proba_encoded(self, X: Features) raises -> List[Values] where Self.classifier:
        self._ready()
        self._encoder.check_encoded(X)
        return self._predict_proba(X)
    def apply(self, X: Features, check_input: Bool = True) raises -> Indices:
        var encoded = self._encode_X(X)
        return Tree().apply(self.tree_, encoded)

    def predict(self, X: Features) raises -> Self.Prediction:
        return self._predict(self._encode_X(X))

    def predict_proba(self, X: Features) raises -> List[Values] where Self.classifier:
        return self._predict_proba(self._encode_X(X))

    def predict_log_proba(self, X: Features) raises -> List[Values] where Self.classifier:
        return self._predict_log_proba(self.predict_proba(X))

    def get_depth(self) raises -> Int:
        self._ready()
        return self.tree_.max_depth
    def get_n_leaves(self) raises -> Int:
        self._ready()
        var count = 0
        for node in self.tree_.nodes:
            if node.left_child == -1:
                count += 1
        return count
    def cost_complexity_pruning_path(self, X: Features, y: Self.Target, sample_weight: Optional[Values] = None) raises -> PruningPath:
        var estimator = self.__sklearn_clone__()
        estimator.params.ccp_alpha = 0
        _ = estimator.fit(X, y, sample_weight)
        return cost_complexity_pruning_path(estimator.tree_)

    def score(self, X: Features, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Float64:
        return self._score(self.predict(X), y, sample_weight)

    def apply(self, X: TreeTable, check_input: Bool = True) raises -> Indices:
        var encoded = self._encode_X(X)
        return Tree().apply(self.tree_, encoded)

    def predict(self, X: TreeTable) raises -> Self.Prediction:
        return self._predict(self._encode_X(X))

    def predict_proba(self, X: TreeTable) raises -> List[Values] where Self.classifier:
        return self._predict_proba(self._encode_X(X))

    def predict_log_proba(self, X: TreeTable) raises -> List[Values] where Self.classifier:
        return self._predict_log_proba(self.predict_proba(X))

    def cost_complexity_pruning_path(self, X: TreeTable, y: Self.Target, sample_weight: Optional[Values] = None) raises -> PruningPath:
        var estimator = self.__sklearn_clone__()
        estimator.params.ccp_alpha = 0
        _ = estimator.fit(X, y, sample_weight)
        return cost_complexity_pruning_path(estimator.tree_)

    def score(self, X: TreeTable, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Float64:
        return self._score(self.predict(X), y, sample_weight)

    def _predict(self, X: Features) raises -> Self.Prediction:
        var values = Tree().predict(self.tree_, X)
        var shape = nm.Shape(X.shape[0])
        if self.n_outputs_ > 1:
            shape = nm.Shape(X.shape[0], self.n_outputs_)
        var result: Self.Prediction
        comptime if Self.string_labels:
            var labels = List[String]()
            for _ in range(X.shape[0] * self.n_outputs_):
                labels.append("")
            var target = StringTarget(labels) if self.n_outputs_ == 1 else StringTarget(labels, X.shape[0], self.n_outputs_)
            result = rebind[Self.Prediction](target).copy()
        else:
            result = rebind[Self.Prediction](nm.zeros[Self.label_dtype](shape)).copy()
        for i in range(X.shape[0]):
            for k in range(self.n_outputs_):
                var offset = (i * self.n_outputs_ + k) * self.tree_.max_n_classes
                var value: Self.Label
                comptime if Self.classifier:
                    var chosen = 0
                    for c in range(1, self.n_classes_[k]):
                        if values.unsafe_get(offset + c) > values.unsafe_get(offset + chosen):
                            chosen = c
                    value = self._class_label(k, chosen)
                else:
                    value = rebind[Self.Label](Scalar[Self.label_dtype](values.unsafe_get(offset))).copy()
                comptime if Self.string_labels:
                    rebind[StringTarget](result).set(i, rebind[String](value), k)
                else:
                    rebind[nm.NDArray[Self.label_dtype]](result).unsafe_set(i * self.n_outputs_ + k, rebind[Scalar[Self.label_dtype]](value))
        return result^

    def _predict_proba(self, X: Features) raises -> List[Values] where Self.classifier:
        var values = Tree().predict(self.tree_, X)
        var result = List[Values]()
        for k in range(self.n_outputs_):
            var probabilities = nm.zeros[DType.float64](nm.Shape(X.shape[0], self.n_classes_[k]))
            for i in range(X.shape[0]):
                for c in range(self.n_classes_[k]):
                    probabilities.unsafe_set(i * self.n_classes_[k] + c, values.unsafe_get((i * self.n_outputs_ + k) * self.tree_.max_n_classes + c))
            result.append(probabilities^)
        return result^

    def _predict_log_proba(self, var result: List[Values]) raises -> List[Values] where Self.classifier:
        for k in range(len(result)):
            for i in range(result[k].size):
                result[k].unsafe_set(i, log(result[k].unsafe_get(i)))
        return result^

    def _score(self, prediction: Self.Prediction, y: Self.Target, sample_weight: Optional[Values] = None) raises -> Float64:
        var n = self._target_rows(prediction)
        if self._target_columns(y) != self.n_outputs_ or self._target_rows(y) != n:
            raise Error("ValueError: score target shape mismatch")
        if sample_weight and (sample_weight.value().ndim != 1 or sample_weight.value().size != n):
            raise Error("ValueError: score weight shape mismatch")
        comptime if Self.classifier:
            var kind = self._target_kind(y)
            var predicted_kind = self._target_kind(prediction)
            if kind == "continuous" or kind == "continuous-multioutput" or kind == "unknown":
                raise Error("ValueError: unsupported classification score targets")
            if kind == "multiclass-multioutput" or predicted_kind == "multiclass-multioutput" or (self.n_outputs_ > 1 and (kind != "multilabel-indicator" or predicted_kind != "multilabel-indicator")):
                raise Error("ValueError: multiclass-multioutput is not supported by accuracy_score")
        var total = Float64(0)
        var result = Float64(0)
        var means = nm.zeros[DType.float64](nm.Shape(self.n_outputs_))
        for i in range(n):
            var w = Float64(1)
            if sample_weight:
                w = sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))
            if not isfinite(w):
                raise Error("ValueError: nonfinite score weight")
            total += w
            var correct = True
            for k in range(self.n_outputs_):
                var value = self._target_value(y, i, k)
                comptime if not Self.string_labels:
                    if not isfinite(rebind[Scalar[Self.label_dtype]](value)):
                        raise Error("ValueError: nonfinite score target")
                correct = correct and value == self._target_value(prediction, i, k)
                comptime if not Self.classifier:
                    means.unsafe_set(k, means.unsafe_get(k) + w * Float64(rebind[Scalar[Self.label_dtype]](value)))
            if correct:
                result += w
        if total == 0:
            raise Error("ValueError: score weights sum to zero")
        comptime if Self.classifier:
            return result / total
        else:
            if n < 2:
                return Float64("nan")
            result = 0
            for k in range(self.n_outputs_):
                var mean = means.unsafe_get(k) / total
                var numerator = Float64(0)
                var denominator = Float64(0)
                for i in range(n):
                    var w = Float64(1)
                    if sample_weight:
                        w = sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))
                    var value = Float64(rebind[Scalar[Self.label_dtype]](self._target_value(y, i, k)))
                    var predicted = Float64(rebind[Scalar[Self.label_dtype]](self._target_value(prediction, i, k)))
                    var error = value - predicted
                    numerator += w * error * error
                    denominator += w * (value - mean) * (value - mean)
                if numerator == 0:
                    result += 1
                elif denominator != 0:
                    result += 1 - numerator / denominator
            return result / Float64(self.n_outputs_)


comptime DecisionTreeClassifier[label_dtype: DType = DType.float64, string_labels: Bool = False] = BaseDecisionTree[True, False, label_dtype, string_labels]
comptime DecisionTreeRegressor = BaseDecisionTree[False]
