# Adapted from scikit-learn sklearn/preprocessing/_encoders.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense OneHotEncoder and OrdinalEncoder over typed categorical columns.

Input is one Moxie `CategoricalColumn` per feature, all of one kind
(`StringKind` or `NumericKind[dtype]`). Category inference, explicit categories,
missing values, infrequent grouping, drop and unknown handling follow sklearn.
Output is built feature by feature: besides the result, a transform holds one
rows-long Int64 code vector. See docs/api/encoders.md.
"""
from std.math import isnan, isfinite, nan
from numojo import NDArray, Shape, Item
from moxie._categorical import CategoricalColumn
from moxie.preprocessing._base import InvertiblePreprocessor
from moxie.preprocessing._common import (
    matrix, vector, check_fitted, check_n_features, check_set_output, check_names_length,
    feature_names_in,
)
from moxie.preprocessing._validation import require_dense_output
from moxie.preprocessing._vocabulary import (
    CategoryKind, StringKind, FeatureVocabulary, encode_column, category_counts,
    infer_vocabulary, explicit_vocabulary, identify_infrequent, infrequent_mapping,
)

comptime _INFREQUENT = -2  # transformed-category marker for "infrequent_sklearn"


def _output_dtype_name[d: DType]() -> String:
    comptime if d == DType.float32:
        return "float32"
    elif d == DType.float64:
        return "float64"
    else:
        return "int64"


def _number(value: Float64) -> String:
    """Python-style scalar: integral values print as ints, NaN as nan."""
    if isnan(value):
        return "nan"
    if value == Float64(Int(value)):
        return String(Int(value))
    return String(value)


def _int_list(values: List[Int]) -> String:
    var text = String("[")
    for i in range(len(values)):
        text += (" " if i else "") + String(values[i])
    return text + "]"


def _check_frequency(min_frequency: Optional[Float64], max_categories: Optional[Int], owner: String) raises:
    if min_frequency:
        var value = min_frequency.value()
        var count = value >= 1 and value == Float64(Int(value))
        if not count and not (value > 0 and value < 1):
            raise Error("ValueError: The 'min_frequency' parameter of " + owner + " must be an int in the range [1, inf), a float in the range (0.0, 1.0) or None. Got " + String(value) + " instead.")
    if max_categories and max_categories.value() < 1:
        raise Error("ValueError: The 'max_categories' parameter of " + owner + " must be an int in the range [1, inf) or None. Got " + String(max_categories.value()) + " instead.")


struct _EncoderCore[K: CategoryKind](Copyable):
    """Learned vocabularies and infrequent grouping shared by both encoders."""
    var vocabularies: List[FeatureVocabulary[Self.K]]
    var infrequent: List[List[Int]]  # per feature; empty = no infrequent categories
    var mappings: List[List[Int]]  # per feature; empty = identity
    var infrequent_enabled: Bool
    var ignore_missing: Bool  # OrdinalEncoder keeps missing out of grouping
    var fitted: Bool

    def __init__(out self):
        self.vocabularies = List[FeatureVocabulary[Self.K]]()
        self.infrequent = List[List[Int]]()
        self.mappings = List[List[Int]]()
        self.infrequent_enabled = False
        self.ignore_missing = False
        self.fitted = False

    def n_features(self) -> Int:
        return len(self.vocabularies)

    @staticmethod
    def check_columns(columns: List[CategoricalColumn[Self.K.Domain]]) raises -> Int:
        """sklearn check_array: at least one feature and one sample, equal lengths."""
        if len(columns) == 0:
            raise Error("ValueError: Found array with 0 feature(s) (shape=(?, 0)) while a minimum of 1 is required.")
        var rows = columns[0].rows()
        for column in columns:
            if column.rows() != rows:
                raise Error("ValueError: all feature columns must have the same number of rows")
        if rows == 0:
            raise Error("ValueError: Found array with 0 sample(s) (shape=(0, " + String(len(columns)) + ")) while a minimum of 1 is required.")
        return rows

    def fit(
        mut self, columns: List[CategoricalColumn[Self.K.Domain]],
        categories: Optional[List[List[Optional[Self.K.Domain.Value]]]], handle_unknown: String,
        min_frequency: Optional[Float64], max_categories: Optional[Int], ignore_missing: Bool,
    ) raises:
        var n_samples = Self.check_columns(columns)
        var p = len(columns)
        if categories and len(categories.value()) != p:
            raise Error("ValueError: Shape mismatch: if categories is an array, it has to be of shape (n_features,).")
        var enabled = (Bool(max_categories) and max_categories.value() >= 1) or Bool(min_frequency)
        var vocabularies = List[FeatureVocabulary[Self.K]]()
        var all_counts = List[List[Int]]()
        for i in range(p):
            if not categories:
                var fitted = infer_vocabulary[Self.K](columns[i])
                vocabularies.append(fitted.vocabulary.copy())
                all_counts.append(fitted.counts.copy())
            else:
                var vocabulary = explicit_vocabulary[Self.K](categories.value()[i], i)
                if handle_unknown == "error" or enabled:
                    var encoded = encode_column(vocabulary, columns[i])
                    if handle_unknown == "error" and encoded.has_unknown():
                        raise Error("ValueError: Found unknown categories " + encoded.unknown_text() + " in column " + String(i) + " during fit")
                    all_counts.append(category_counts(vocabulary, encoded.codes))
                else:
                    all_counts.append(List[Int]())
                vocabularies.append(vocabulary^)
        var infrequent = List[List[Int]]()
        var mappings = List[List[Int]]()
        if enabled:
            for i in range(p):
                var counts = all_counts[i].copy()
                if ignore_missing and vocabularies[i].has_missing:
                    _ = counts.pop()  # missing is never infrequent
                var indices = identify_infrequent(counts, n_samples, min_frequency, max_categories)
                if len(indices):
                    mappings.append(infrequent_mapping(len(counts), indices))
                else:
                    mappings.append(List[Int]())
                infrequent.append(indices^)
        self.vocabularies = vocabularies^
        self.infrequent = infrequent^
        self.mappings = mappings^
        self.infrequent_enabled = enabled
        self.ignore_missing = ignore_missing
        self.fitted = True

    def has_infrequent(self, i: Int) -> Bool:
        return self.infrequent_enabled and len(self.infrequent[i]) > 0

    def check_transform(self, columns: List[CategoricalColumn[Self.K.Domain]], owner: String) raises:
        _ = Self.check_columns(columns)
        check_n_features(owner, self.n_features(), len(columns))

    def encode(self, i: Int, column: CategoricalColumn[Self.K.Domain], handle_unknown: String) raises -> NDArray[DType.int64]:
        """sklearn `_transform` + `_map_infrequent_categories` for one feature.

        Valid rows hold their (grouped) category code; rows outside sklearn's
        X_mask hold -1. Ignored missing rows keep the ungrouped missing index.
        """
        var encoded = encode_column(self.vocabularies[i], column)
        if handle_unknown == "error" and encoded.has_unknown():
            raise Error("ValueError: Found unknown categories " + encoded.unknown_text() + " in column " + String(i) + " during transform")
        var codes = encoded^.take()
        if self.has_infrequent(i):
            var absorb = handle_unknown == "infrequent_if_exist" or handle_unknown == "warn"
            var missing = self.vocabularies[i].missing_index() if self.ignore_missing else -1
            for r in range(codes.size):
                var code = Int(codes[Item(r)])
                if code == -1:
                    if not absorb:
                        continue
                    code = self.infrequent[i][0]
                if code == missing:
                    continue
                codes[Item(r)] = Int64(self.mappings[i][code])
        return codes^

    def categories(self) raises -> List[List[Optional[Self.K.Domain.Value]]]:
        var result = List[List[Optional[Self.K.Domain.Value]]]()
        for vocabulary in self.vocabularies:
            result.append(vocabulary.values())
        return result^

    def infrequent_categories(self) raises -> Optional[List[Optional[List[Optional[Self.K.Domain.Value]]]]]:
        """Per feature, None or the infrequent categories (None = the missing category)."""
        if not self.infrequent_enabled:
            return None
        var result = List[Optional[List[Optional[Self.K.Domain.Value]]]]()
        for i in range(self.n_features()):
            if len(self.infrequent[i]) == 0:
                result.append(None)
                continue
            var values = List[Optional[Self.K.Domain.Value]]()
            for index in self.infrequent[i]:
                values.append(self.vocabularies[i].category(index))
            result.append(values^)
        return result^


def _decoded_column[K: CategoryKind](vocabulary: FeatureVocabulary[K], decoded: List[Int], feature: Int) raises -> CategoricalColumn[K.Domain]:
    """Rows of vocabulary indices (missing index or -1 = None, _INFREQUENT) as a column."""
    var needs_infrequent = False
    for index in decoded:
        needs_infrequent = needs_infrequent or index == _INFREQUENT
    var domain = vocabulary.domain.copy()
    var infrequent_code = -1
    if needs_infrequent:
        var placeholder = K.infrequent_value()
        if not placeholder:
            raise Error("ValueError: inverse_transform cannot represent 'infrequent_sklearn' in numeric column " + String(feature) + "; use string categories")
        if domain.find(placeholder.value()) >= 0:
            raise Error("ValueError: category 'infrequent_sklearn' in column " + String(feature) + " collides with the infrequent placeholder")
        var values = List[K.Domain.Value]()
        for k in range(domain.size()):
            values.append(domain.get(k))
        infrequent_code = len(values)
        values.append(placeholder.value().copy())
        domain = K.make(values)
    var codes = vector[DType.int64](len(decoded))
    for r in range(len(decoded)):
        var index = decoded[r]
        var code = -1
        if index == _INFREQUENT:
            code = infrequent_code
        elif index >= 0 and index < vocabulary.known():
            code = index
        codes[Item(r)] = Int64(code)
    return CategoricalColumn[K.Domain](domain, codes)


def _check_encoded[d: DType](X: NDArray[d], expected: Int, allow_nan: Bool) raises:
    if X.ndim != 2:
        raise Error("ValueError: Expected 2D array, got " + String(X.ndim) + "D array instead")
    if X.shape[0] < 1:
        raise Error("ValueError: Found array with 0 sample(s) (shape=(0, " + String(X.shape[1]) + ")) while a minimum of 1 is required.")
    if X.shape[1] != expected:
        raise Error("ValueError: Shape of the passed X data is not correct. Expected " + String(expected) + " columns, got " + String(X.shape[1]) + ".")
    comptime if d == DType.float32 or d == DType.float64:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var v = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                if isnan(v):
                    if not allow_nan:
                        raise Error("ValueError: Input contains NaN.")
                elif not isfinite(v):
                    raise Error("ValueError: Input contains infinity or a value too large for dtype('" + _output_dtype_name[d]() + "').")


# ---------------------------------------------------------------------------
# OneHotEncoder


@fieldwise_init
struct OneHotEncoderParams[K: CategoryKind](Copyable):
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var drop: Optional[String]
    var drop_values: Optional[List[Optional[Self.K.Domain.Value]]]
    var sparse_output: Bool
    var handle_unknown: String
    var min_frequency: Optional[Float64]
    var max_categories: Optional[Int]
    var feature_name_combiner: String


struct OneHotEncoder[K: CategoryKind = StringKind, dtype: DType = DType.float64](InvertiblePreprocessor):
    """Encode categorical features as a dense one-hot matrix.

    `sparse_output` must be False: sparse output is deferred. `dtype` is the
    output dtype (Float32, Float64 or Int64).
    """
    comptime Params = OneHotEncoderParams[Self.K]
    comptime Input = List[CategoricalColumn[Self.K.Domain]]
    comptime Target = NDArray[DType.float64]
    comptime Transformed = NDArray[Self.dtype]
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var drop: Optional[String]
    var drop_values: Optional[List[Optional[Self.K.Domain.Value]]]
    var sparse_output: Bool
    var handle_unknown: String
    var min_frequency: Optional[Float64]
    var max_categories: Optional[Int]
    var feature_name_combiner: String
    var categories_: List[List[Optional[Self.K.Domain.Value]]]
    var drop_idx_: Optional[List[Optional[Int]]]
    var infrequent_categories_: Optional[List[Optional[List[Optional[Self.K.Domain.Value]]]]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _core: _EncoderCore[Self.K]
    var _drop_after: Optional[List[Optional[Int]]]  # sklearn _drop_idx_after_grouping
    var _n_features_outs: List[Int]

    def __init__(out self, *, categories: Optional[List[List[Optional[Self.K.Domain.Value]]]] = None,
        drop: Optional[String] = None, sparse_output: Bool = True, handle_unknown: String = "error",
        min_frequency: Optional[Float64] = None, max_categories: Optional[Int] = None,
        feature_name_combiner: String = "concat",
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64 or Self.dtype == DType.int64, "output dtype must be float32, float64 or int64"
        self.categories = categories.copy()
        self.drop = drop
        self.drop_values = None
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner
        self.categories_ = List[List[Optional[Self.K.Domain.Value]]]()
        self.drop_idx_ = None
        self.infrequent_categories_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._core = _EncoderCore[Self.K]()
        self._drop_after = None
        self._n_features_outs = List[Int]()

    def __init__(out self, *, drop: List[Optional[Self.K.Domain.Value]],
        categories: Optional[List[List[Optional[Self.K.Domain.Value]]]] = None,
        sparse_output: Bool = True, handle_unknown: String = "error",
        min_frequency: Optional[Float64] = None, max_categories: Optional[Int] = None,
        feature_name_combiner: String = "concat",
    ):
        """`drop` as one category per feature; None drops the missing category."""
        self = Self(categories=categories, sparse_output=sparse_output, handle_unknown=handle_unknown, min_frequency=min_frequency, max_categories=max_categories, feature_name_combiner=feature_name_combiner)
        self.drop_values = drop.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.categories.copy(), self.drop, self.drop_values.copy(), self.sparse_output, self.handle_unknown, self.min_frequency, self.max_categories, self.feature_name_combiner)

    def set_params(mut self, params: Self.Params) raises:
        self.categories = params.categories.copy()
        self.drop = params.drop
        self.drop_values = params.drop_values.copy()
        self.sparse_output = params.sparse_output
        self.handle_unknown = params.handle_unknown
        self.min_frequency = params.min_frequency
        self.max_categories = params.max_categories
        self.feature_name_combiner = params.feature_name_combiner

    def set_params(
        mut self, *, drop: Optional[String] = None, sparse_output: Optional[Bool] = None,
        handle_unknown: Optional[String] = None, min_frequency: Optional[Float64] = None,
        max_categories: Optional[Int] = None,
    ) raises:
        """Keyword updates; a given `drop` strategy replaces explicit drop values."""
        if drop:
            self.drop = drop
            self.drop_values = None
        if sparse_output:
            self.sparse_output = sparse_output.value()
        if handle_unknown:
            self.handle_unknown = handle_unknown.value()
        if min_frequency:
            self.min_frequency = min_frequency
        if max_categories:
            self.max_categories = max_categories

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._core.fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _validate_params(self) raises:
        var h = self.handle_unknown
        if h != "error" and h != "ignore" and h != "infrequent_if_exist" and h != "warn":
            raise Error("ValueError: The 'handle_unknown' parameter of OneHotEncoder must be a str among {'error', 'ignore', 'infrequent_if_exist', 'warn'}. Got '" + h + "' instead.")
        if self.drop and self.drop.value() != "first" and self.drop.value() != "if_binary":
            raise Error("ValueError: The 'drop' parameter of OneHotEncoder must be a str among {'first', 'if_binary'}, an array-like or None. Got '" + self.drop.value() + "' instead.")
        _check_frequency(self.min_frequency, self.max_categories, "OneHotEncoder")
        if self.feature_name_combiner != "concat":
            raise Error("ValueError: feature_name_combiner must be 'concat'; callables are not supported natively")
        require_dense_output(self.sparse_output)

    def _map_drop(self, core: _EncoderCore[Self.K], i: Int, index: Int) raises -> Int:
        """sklearn `_map_drop_idx_to_infrequent`."""
        if not core.has_infrequent(i):
            return index
        for infrequent in core.infrequent[i]:
            if infrequent == index:
                var shown = Self.K.quoted(core.vocabularies[i].domain.get(index)) if index < core.vocabularies[i].known() else Self.K.missing_text()
                raise Error("ValueError: Unable to drop category " + shown + " from feature " + String(i) + " because it is infrequent")
        return core.mappings[i][index]

    def _drop_indices(self, core: _EncoderCore[Self.K]) raises -> Optional[List[Optional[Int]]]:
        """sklearn `_set_drop_idx`: drop indices after infrequent grouping."""
        var p = core.n_features()
        if self.drop_values:
            var values = self.drop_values.value().copy()
            if len(values) != p:
                raise Error("ValueError: `drop` should have length equal to the number of features (" + String(p) + "), got " + String(len(values)))
            var result = List[Optional[Int]]()
            var missing = String("")
            for i in range(p):
                var index = core.vocabularies[i].lookup(values[i])
                if index < 0:
                    # sklearn prints (feature, value) under swapped labels.
                    var shown = Self.K.text(values[i].value()) if values[i] else Self.K.missing_text()
                    missing += "\nCategory: " + String(i) + ", Feature: " + shown
                    continue
                result.append(Optional[Int](self._map_drop(core, i, index)))
            if missing.byte_length():
                raise Error("ValueError: The following categories were supposed to be dropped, but were not found in the training data." + missing)
            return result^
        if not self.drop:
            return None
        var result = List[Optional[Int]]()
        for i in range(p):
            if self.drop.value() == "first":
                result.append(Optional[Int](0))
                continue
            var n = core.vocabularies[i].size()
            if core.has_infrequent(i):
                n -= len(core.infrequent[i]) - 1
            result.append(Optional[Int](0) if n == 2 else Optional[Int](None))
        return result^

    def _fit(mut self, X: Self.Input) raises:
        self._validate_params()
        var core = _EncoderCore[Self.K]()
        core.fit(X, self.categories, self.handle_unknown, self.min_frequency, self.max_categories, False)
        var after = self._drop_indices(core)
        var outs = List[Int]()
        for i in range(core.n_features()):
            var n = core.vocabularies[i].size()
            if after and after.value()[i]:
                n -= 1
            if core.has_infrequent(i):
                n -= len(core.infrequent[i]) - 1
            outs.append(n)
        # drop_idx_ reports original category indices (before grouping).
        var original: Optional[List[Optional[Int]]] = None
        if after:
            var mapped = List[Optional[Int]]()
            for i in range(core.n_features()):
                var index = after.value()[i]
                if not index or not core.has_infrequent(i):
                    mapped.append(index)
                    continue
                var found = -1
                for k in range(len(core.mappings[i])):
                    if core.mappings[i][k] == index.value():
                        found = k
                        break
                mapped.append(Optional[Int](found))
            original = mapped^
        self.categories_ = core.categories()
        self.infrequent_categories_ = core.infrequent_categories()
        self.n_features_in_ = core.n_features()
        self.feature_names_in_ = None
        self._core = core^
        self._drop_after = after^
        self.drop_idx_ = original^
        self._n_features_outs = outs^

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        self._fit(X)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        check_names_length(feature_names, len(X))
        self._fit(X)
        self.feature_names_in_ = feature_names.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._fit(X)
        return self.transform(X)

    # -- transformation -------------------------------------------------------

    def _dropped(self, i: Int) -> Int:
        if self._drop_after and self._drop_after.value()[i]:
            return self._drop_after.value()[i].value()
        return -1

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        check_fitted(self._core.fitted, "OneHotEncoder")
        self._core.check_transform(X, "OneHotEncoder")
        # "warn" encodes like infrequent_if_exist; Moxie has no warning channel.
        var mode = "infrequent_if_exist" if self.handle_unknown == "warn" else self.handle_unknown
        var rows = X[0].rows()
        var width = 0
        for n in self._n_features_outs:
            width += n
        var out = matrix[Self.dtype](rows, width)
        var offset = 0
        for i in range(len(X)):
            var codes = self._core.encode(i, X[i], mode)
            var dropped = self._dropped(i)
            for r in range(rows):
                var code = Int(codes[Item(r)])
                if code < 0 or code == dropped:
                    continue
                if dropped >= 0 and code > dropped:
                    code -= 1
                out.unsafe_set(r * width + offset + code, 1)
            offset += self._n_features_outs[i]
        return out^

    def _transformed_categories(self, i: Int, remove_dropped: Bool) -> List[Int]:
        """sklearn `_compute_transformed_categories` as vocabulary indices."""
        var result = List[Int]()
        var size = self._core.vocabularies[i].size()
        if self._core.has_infrequent(i):
            var mapping = self._core.mappings[i].copy()
            var top = 0
            for m in mapping:
                top = max(top, m)
            for k in range(size):
                if mapping[k] < top:
                    result.append(k)
            result.append(_INFREQUENT)
        else:
            for k in range(size):
                result.append(k)
        var dropped = self._dropped(i)
        if remove_dropped and dropped >= 0:
            _ = result.pop(dropped)
        return result^

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        check_fitted(self._core.fitted, "OneHotEncoder")
        var width = 0
        for n in self._n_features_outs:
            width += n
        _check_encoded(X, width, False)
        var rows = X.shape[0]
        var result = List[CategoricalColumn[Self.K.Domain]]()
        var j = 0
        for i in range(self._core.n_features()):
            var transformed = self._transformed_categories(i, False)
            var kept = self._transformed_categories(i, True)
            var n = len(kept)
            var dropped = self._dropped(i)
            var decoded = List[Int](length=rows, fill=-1)
            if n == 0:
                # Single-category feature: sklearn fills categories_[i][drop index].
                for r in range(rows):
                    decoded[r] = dropped
                result.append(_decoded_column(self._core.vocabularies[i], decoded, i))
                continue
            var ignore = self.handle_unknown == "ignore" or ((self.handle_unknown == "infrequent_if_exist" or self.handle_unknown == "warn") and not self._core.has_infrequent(i))
            var all_zero = List[Int]()
            for r in range(rows):
                var best = 0
                var best_value = X.unsafe_get(r * Int(X.strides[0]) + j * Int(X.strides[1]))
                var total = Float64(0)
                for c in range(n):
                    var value = X.unsafe_get(r * Int(X.strides[0]) + (j + c) * Int(X.strides[1]))
                    total += Float64(value)
                    if value > best_value:
                        best = c
                        best_value = value
                decoded[r] = kept[best]
                if total != 0:
                    continue
                if ignore:
                    # Unknown rows: None, or the dropped category when one exists.
                    decoded[r] = dropped if dropped >= 0 else -1
                elif dropped < 0:
                    all_zero.append(r)
                else:
                    decoded[r] = transformed[dropped]
            if len(all_zero):
                raise Error("ValueError: Samples " + _int_list(all_zero) + " can not be inverted when drop=None and handle_unknown='error' because they contain all zeros")
            result.append(_decoded_column(self._core.vocabularies[i], decoded, i))
            j += n
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self._core.fitted, "OneHotEncoder")
        var names = feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        var result = List[String]()
        for i in range(self._core.n_features()):
            for index in self._transformed_categories(i, True):
                var text = String("infrequent_sklearn") if index == _INFREQUENT else self._core.vocabularies[i].text(index)
                result.append(names[i] + "_" + text)
        return result^


# ---------------------------------------------------------------------------
# OrdinalEncoder


@fieldwise_init
struct OrdinalEncoderParams[K: CategoryKind](Copyable):
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var handle_unknown: String
    var unknown_value: Optional[Float64]
    var encoded_missing_value: Float64
    var min_frequency: Optional[Float64]
    var max_categories: Optional[Int]


struct OrdinalEncoder[K: CategoryKind = StringKind, dtype: DType = DType.float64](InvertiblePreprocessor):
    """Encode categorical features as one integer-valued column per feature.

    `dtype` is the output dtype (Float32, Float64 or Int64). `unknown_value`
    is an integer-valued Float64 or NaN, as sklearn's int or np.nan.
    """
    comptime Params = OrdinalEncoderParams[Self.K]
    comptime Input = List[CategoricalColumn[Self.K.Domain]]
    comptime Target = NDArray[DType.float64]
    comptime Transformed = NDArray[Self.dtype]
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var handle_unknown: String
    var unknown_value: Optional[Float64]
    var encoded_missing_value: Float64
    var min_frequency: Optional[Float64]
    var max_categories: Optional[Int]
    var categories_: List[List[Optional[Self.K.Domain.Value]]]
    var infrequent_categories_: Optional[List[Optional[List[Optional[Self.K.Domain.Value]]]]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _core: _EncoderCore[Self.K]

    def __init__(out self, *, categories: Optional[List[List[Optional[Self.K.Domain.Value]]]] = None,
        handle_unknown: String = "error", unknown_value: Optional[Float64] = None,
        encoded_missing_value: Float64 = nan[DType.float64](),
        min_frequency: Optional[Float64] = None, max_categories: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64 or Self.dtype == DType.int64, "output dtype must be float32, float64 or int64"
        self.categories = categories.copy()
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.categories_ = List[List[Optional[Self.K.Domain.Value]]]()
        self.infrequent_categories_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._core = _EncoderCore[Self.K]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.categories.copy(), self.handle_unknown, self.unknown_value, self.encoded_missing_value, self.min_frequency, self.max_categories)

    def set_params(mut self, params: Self.Params) raises:
        self.categories = params.categories.copy()
        self.handle_unknown = params.handle_unknown
        self.unknown_value = params.unknown_value
        self.encoded_missing_value = params.encoded_missing_value
        self.min_frequency = params.min_frequency
        self.max_categories = params.max_categories

    def set_params(
        mut self, *, handle_unknown: Optional[String] = None, unknown_value: Optional[Float64] = None,
        encoded_missing_value: Optional[Float64] = None, min_frequency: Optional[Float64] = None,
        max_categories: Optional[Int] = None,
    ) raises:
        if handle_unknown:
            self.handle_unknown = handle_unknown.value()
        if unknown_value:
            self.unknown_value = unknown_value
        if encoded_missing_value:
            self.encoded_missing_value = encoded_missing_value.value()
        if min_frequency:
            self.min_frequency = min_frequency
        if max_categories:
            self.max_categories = max_categories

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._core.fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def _is_float(self) -> Bool:
        return Self.dtype == DType.float32 or Self.dtype == DType.float64

    def _fit(mut self, X: Self.Input) raises:
        var h = self.handle_unknown
        if h != "error" and h != "use_encoded_value":
            raise Error("ValueError: The 'handle_unknown' parameter of OrdinalEncoder must be a str among {'error', 'use_encoded_value'}. Got '" + h + "' instead.")
        _check_frequency(self.min_frequency, self.max_categories, "OrdinalEncoder")
        if h == "use_encoded_value":
            if self.unknown_value and isnan(self.unknown_value.value()):
                if not self._is_float():
                    raise Error("ValueError: When unknown_value is np.nan, the dtype parameter should be a float dtype. Got " + _output_dtype_name[Self.dtype]() + ".")
            elif not self.unknown_value or self.unknown_value.value() != Float64(Int(self.unknown_value.value())):
                var shown = _number(self.unknown_value.value()) if self.unknown_value else String("None")
                raise Error("TypeError: unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value', got " + shown + ".")
        elif self.unknown_value:
            raise Error("TypeError: unknown_value should only be set when handle_unknown is 'use_encoded_value', got " + _number(self.unknown_value.value()) + ".")
        var core = _EncoderCore[Self.K]()
        core.fit(X, self.categories, h, self.min_frequency, self.max_categories, Self.K.missing_is_nan())
        var cardinalities = List[Int]()
        var missing_features = List[Int]()
        for i in range(core.n_features()):
            var c = core.vocabularies[i].size()
            if core.has_infrequent(i):
                c -= len(core.infrequent[i])
            if core.ignore_missing and core.vocabularies[i].has_missing:
                c -= 1
                missing_features.append(i)
            cardinalities.append(c)
        if h == "use_encoded_value":
            var u = self.unknown_value.value()
            for c in cardinalities:
                if u >= 0 and u < Float64(c):
                    raise Error("ValueError: The used value for unknown_value " + _number(u) + " is one of the values already used for encoding the seen categories.")
        if len(missing_features):
            var e = self.encoded_missing_value
            if not self._is_float() and isnan(e):
                raise Error("ValueError: There are missing values in features " + _int_list(missing_features) + ". For OrdinalEncoder to encode missing values with dtype: " + _output_dtype_name[Self.dtype]() + ", set encoded_missing_value to a non-nan value, or set dtype to a float")
            if not isnan(e):
                var invalid = List[Int]()
                for i in missing_features:
                    if e >= 0 and e < Float64(cardinalities[i]):
                        invalid.append(i)
                if len(invalid):
                    raise Error("ValueError: encoded_missing_value (" + _number(e) + ") is already used to encode a known category in features: " + _int_list(invalid))
        self.categories_ = core.categories()
        self.infrequent_categories_ = core.infrequent_categories()
        self.n_features_in_ = core.n_features()
        self.feature_names_in_ = None
        self._core = core^

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        self._fit(X)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        check_names_length(feature_names, len(X))
        self._fit(X)
        self.feature_names_in_ = feature_names.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._fit(X)
        return self.transform(X)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        check_fitted(self._core.fitted, "OrdinalEncoder")
        self._core.check_transform(X, "OrdinalEncoder")
        var rows = X[0].rows()
        var p = len(X)
        var out = matrix[Self.dtype](rows, p)
        var unknown = Scalar[Self.dtype](self.unknown_value.value()) if self.unknown_value else Scalar[Self.dtype](0)
        var encoded_missing = Scalar[Self.dtype](self.encoded_missing_value)
        for i in range(p):
            var codes = self._core.encode(i, X[i], self.handle_unknown)
            # sklearn None is an ordinary category; only NaN-style missing is special.
            var missing = self._core.vocabularies[i].missing_index() if self._core.ignore_missing else -2
            for r in range(rows):
                var code = Int(codes[Item(r)])
                var value: Scalar[Self.dtype]
                if code == -1:
                    value = unknown
                elif code == missing:
                    value = encoded_missing
                else:
                    value = Scalar[Self.dtype](code)
                out.unsafe_set(r * p + i, value)
        return out^

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        check_fitted(self._core.fitted, "OrdinalEncoder")
        var p = self._core.n_features()
        _check_encoded(X, p, True)
        var rows = X.shape[0]
        var result = List[CategoricalColumn[Self.K.Domain]]()
        for i in range(p):
            var vocabulary = self._core.vocabularies[i].copy()
            var missing = vocabulary.missing_index() if self._core.ignore_missing else -1
            # Remaining categories after removing infrequent ones (sklearn's
            # frequent_categories_mask over the full categories_ entry).
            var categories = List[Int]()
            var infrequent_value = -1
            var is_infrequent = List[Bool](length=vocabulary.size(), fill=False)
            if self._core.has_infrequent(i):
                for index in self._core.infrequent[i]:
                    is_infrequent[index] = True
                infrequent_value = vocabulary.size() - len(self._core.infrequent[i])
            for k in range(vocabulary.size()):
                if not is_infrequent[k]:
                    categories.append(k)
            var decoded = List[Int](length=rows, fill=-1)
            var bad = List[Int]()
            for r in range(rows):
                var label = Float64(X.unsafe_get(r * Int(X.strides[0]) + i * Int(X.strides[1])))
                if missing >= 0 and (label == self.encoded_missing_value or (isnan(label) and isnan(self.encoded_missing_value))):
                    label = Float64(missing)
                if infrequent_value >= 0 and label == Float64(infrequent_value):
                    decoded[r] = _INFREQUENT
                    continue
                if self.handle_unknown == "use_encoded_value":
                    var u = self.unknown_value.value()
                    if label == u or (isnan(label) and isnan(u)):
                        continue  # None
                if isnan(label):
                    bad.append(r)
                    continue
                var index = Int(label)
                # NumPy indexing: negative labels count from the end.
                if index < 0:
                    index += len(categories)
                if index < 0 or index >= len(categories):
                    bad.append(r)
                    continue
                decoded[r] = categories[index]
            if len(bad):
                raise Error("ValueError: encoded values in rows " + _int_list(bad) + " of column " + String(i) + " are not valid category indices")
            result.append(_decoded_column(vocabulary, decoded, i))
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self._core.fitted, "OrdinalEncoder")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
