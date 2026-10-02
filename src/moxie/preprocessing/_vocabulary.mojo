# Adapted from scikit-learn sklearn/utils/_encode.py and sklearn/preprocessing/_encoders.py;
# authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Feature category vocabularies: the family's category engine.

Input features are Moxie `CategoricalColumn`s (one homogeneous column each,
code -1 = missing). A `FeatureVocabulary` is one entry of sklearn's
`categories_`: known values in category order, then an optional final missing
category. Each column is processed in O(rows + its dictionary) time with one
rows-long Int64 code vector; no table-sized masks or code matrices exist.
"""
from std.builtin.sort import sort
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._categorical import CategoryDomain, CategoricalColumn, StringCategories, NumericCategories
from moxie.preprocessing._contracts import CategoryVocabulary


trait CategoryKind:
    """Value operations that generic code cannot express through CategoryDomain.

    `StringKind` / `StringNaNKind` (missing as sklearn's None / NaN in object
    columns) and `NumericKind[dtype]` cover Moxie's two dictionary types.
    """
    comptime Domain: CategoryDomain

    @staticmethod
    def make(values: List[Self.Domain.Value]) raises -> Self.Domain:
        """Dictionary in the given order (numeric values must be sorted)."""
        ...

    @staticmethod
    def sort(mut values: List[Self.Domain.Value]):
        ...

    @staticmethod
    def has_duplicates(values: List[Self.Domain.Value]) -> Bool:
        ...

    @staticmethod
    def is_sorted(values: List[Self.Domain.Value]) -> Bool:
        """Strictly increasing, as numeric explicit categories require."""
        ...

    @staticmethod
    def requires_sorted() -> Bool:
        ...

    @staticmethod
    def text(value: Self.Domain.Value) -> String:
        """Python `str(category)`, used in feature names."""
        ...

    @staticmethod
    def quoted(value: Self.Domain.Value) -> String:
        """NumPy array-element form, used in error messages."""
        ...

    @staticmethod
    def missing_text() -> String:
        """`str` of the missing category: None for strings, nan for numbers."""
        ...

    @staticmethod
    def missing_is_nan() -> Bool:
        """True when missing follows sklearn's NaN rules (OrdinalEncoder's
        encoded_missing_value, never infrequent); False when it is sklearn's
        object None, an ordinary final category."""
        ...

    @staticmethod
    def infrequent_value() -> Optional[Self.Domain.Value]:
        """sklearn's "infrequent_sklearn" placeholder, if representable."""
        ...


struct StringKind(CategoryKind):
    """String features (`StringCategories`); missing cells are None."""
    comptime Domain = StringCategories

    @staticmethod
    def make(values: List[String]) raises -> StringCategories:
        return StringCategories(values)

    @staticmethod
    def sort(mut values: List[String]):
        # UTF-8 byte order equals Python's code point order.
        sort(Span(values))

    @staticmethod
    def has_duplicates(values: List[String]) -> Bool:
        var seen = Dict[String, Bool]()
        for value in values:
            if value in seen:
                return True
            seen[value.copy()] = True
        return False

    @staticmethod
    def is_sorted(values: List[String]) -> Bool:
        for i in range(1, len(values)):
            if not (values[i - 1] < values[i]):
                return False
        return True

    @staticmethod
    def requires_sorted() -> Bool:
        return False

    @staticmethod
    def text(value: String) -> String:
        return value.copy()

    @staticmethod
    def quoted(value: String) -> String:
        return "'" + value + "'"

    @staticmethod
    def missing_text() -> String:
        return "None"

    @staticmethod
    def missing_is_nan() -> Bool:
        return False

    @staticmethod
    def infrequent_value() -> Optional[String]:
        return Optional[String]("infrequent_sklearn")


struct NumericKind[dtype: DType](CategoryKind):
    """Bool/Int64/Float32/Float64 features; floating NaN and masked cells are missing."""
    comptime Domain = NumericCategories[Self.dtype]

    @staticmethod
    def make(values: List[Scalar[Self.dtype]]) raises -> NumericCategories[Self.dtype]:
        var array = nm.zeros[Self.dtype](Shape(len(values)))
        for i in range(len(values)):
            array[Item(i)] = values[i]
        return NumericCategories[Self.dtype](array)

    @staticmethod
    def sort(mut values: List[Scalar[Self.dtype]]):
        sort(Span(values))

    @staticmethod
    def has_duplicates(values: List[Scalar[Self.dtype]]) -> Bool:
        var ordered = values.copy()
        sort(Span(ordered))
        for i in range(1, len(ordered)):
            if ordered[i - 1] == ordered[i]:
                return True
        return False

    @staticmethod
    def is_sorted(values: List[Scalar[Self.dtype]]) -> Bool:
        for i in range(1, len(values)):
            if not (values[i - 1] < values[i]):
                return False
        return True

    @staticmethod
    def requires_sorted() -> Bool:
        return True

    @staticmethod
    def text(value: Scalar[Self.dtype]) -> String:
        # Mojo's shortest round-trip formatting matches Python's str(float).
        return String(value)

    @staticmethod
    def quoted(value: Scalar[Self.dtype]) -> String:
        return String(value)

    @staticmethod
    def missing_text() -> String:
        return "nan"

    @staticmethod
    def missing_is_nan() -> Bool:
        return True

    @staticmethod
    def infrequent_value() -> Optional[Scalar[Self.dtype]]:
        return None


struct StringNaNKind(CategoryKind):
    """String features whose missing cells are NaN in sklearn (pandas style)."""
    comptime Domain = StringCategories

    @staticmethod
    def make(values: List[String]) raises -> StringCategories:
        return StringKind.make(values)

    @staticmethod
    def sort(mut values: List[String]):
        StringKind.sort(values)

    @staticmethod
    def has_duplicates(values: List[String]) -> Bool:
        return StringKind.has_duplicates(values)

    @staticmethod
    def is_sorted(values: List[String]) -> Bool:
        return StringKind.is_sorted(values)

    @staticmethod
    def requires_sorted() -> Bool:
        return False

    @staticmethod
    def text(value: String) -> String:
        return value.copy()

    @staticmethod
    def quoted(value: String) -> String:
        return StringKind.quoted(value)

    @staticmethod
    def missing_text() -> String:
        return "nan"

    @staticmethod
    def missing_is_nan() -> Bool:
        return True

    @staticmethod
    def infrequent_value() -> Optional[String]:
        return StringKind.infrequent_value()


struct FeatureVocabulary[K: CategoryKind](CategoryVocabulary, Copyable):
    """One feature's categories: `domain` in order, then missing if `has_missing`."""
    comptime Value = Optional[Self.K.Domain.Value]
    var domain: Self.K.Domain
    var has_missing: Bool

    def __init__(out self, domain: Self.K.Domain, has_missing: Bool):
        self.domain = domain.copy()
        self.has_missing = has_missing

    def size(self) -> Int:
        return self.domain.size() + Int(self.has_missing)

    def known(self) -> Int:
        return self.domain.size()

    def missing_index(self) -> Int:
        return self.domain.size() if self.has_missing else -1

    def category(self, index: Int) raises -> Self.Value:
        if index < 0 or index >= self.size():
            raise Error("ValueError: category index out of bounds")
        if index == self.domain.size():
            return None
        return Self.Value(self.domain.get(index))

    def lookup(self, value: Self.Value, *, is_missing: Bool = False) raises -> Int:
        if is_missing or not value:
            return self.missing_index()
        return self.domain.find(value.value())

    def values(self) raises -> List[Self.Value]:
        var result = List[Self.Value]()
        for i in range(self.size()):
            result.append(self.category(i))
        return result^

    def text(self, index: Int) raises -> String:
        if index == self.domain.size() and self.has_missing:
            return Self.K.missing_text()
        return Self.K.text(self.domain.get(index))


def _format_values[K: CategoryKind](values: List[K.Domain.Value], missing: Bool) -> String:
    """NumPy-like `[a b]` element list, missing last, for sklearn messages."""
    var text = String("[")
    for i in range(len(values)):
        if i:
            text += " "
        text += K.quoted(values[i])
    if missing:
        text += (" " if len(values) else "") + K.missing_text()
    return text + "]"


struct EncodedColumn[K: CategoryKind](Movable):
    """Learned category index per row (-1 = unknown) and the sorted unknown values."""
    var codes: NDArray[DType.int64]
    var unknown: List[Self.K.Domain.Value]
    var unknown_missing: Bool

    def __init__(out self, var codes: NDArray[DType.int64], var unknown: List[Self.K.Domain.Value], unknown_missing: Bool):
        self.codes = codes^
        self.unknown = unknown^
        self.unknown_missing = unknown_missing

    def has_unknown(self) -> Bool:
        return len(self.unknown) > 0 or self.unknown_missing

    def unknown_text(self) -> String:
        return _format_values[Self.K](self.unknown, self.unknown_missing)

    def take(deinit self) -> NDArray[DType.int64]:
        """Release the code vector without copying it."""
        return self.codes^


def encode_column[K: CategoryKind](vocabulary: FeatureVocabulary[K], column: CategoricalColumn[K.Domain]) raises -> EncodedColumn[K]:
    """sklearn `_encode`: map each row to its learned index, -1 when unknown.

    The column's own dictionary is translated once; row codes are rewritten in
    the single compact copy returned by `codes()`.
    """
    var source = column.domain()
    var translate = List[Int](capacity=source.size())
    for k in range(source.size()):
        translate.append(vocabulary.domain.find(source.get(k)))
    var seen = List[Bool](length=source.size(), fill=False)
    var missing_seen = False
    var codes = column.codes()
    var missing = vocabulary.missing_index()
    for i in range(codes.size):
        var code = Int(codes[Item(i)])
        var learned: Int
        if code == -1:
            missing_seen = True
            learned = missing
        else:
            seen[code] = True
            learned = translate[code]
        codes[Item(i)] = Int64(learned)
    var unknown = List[K.Domain.Value]()
    for k in range(source.size()):
        if seen[k] and translate[k] < 0:
            unknown.append(source.get(k))
    K.sort(unknown)
    return EncodedColumn[K](codes^, unknown^, missing_seen and missing < 0)


def category_counts[K: CategoryKind](vocabulary: FeatureVocabulary[K], encoded: NDArray[DType.int64]) raises -> List[Int]:
    """Occurrences of each category (unknown rows ignored), sklearn `_get_counts`."""
    var counts = List[Int](length=vocabulary.size(), fill=0)
    for i in range(encoded.size):
        var code = Int(encoded[Item(i)])
        if code >= 0:
            counts[code] += 1
    return counts^


@fieldwise_init
struct FittedVocabulary[K: CategoryKind](Movable):
    var vocabulary: FeatureVocabulary[Self.K]
    var counts: List[Int]


def infer_vocabulary[K: CategoryKind](column: CategoricalColumn[K.Domain]) raises -> FittedVocabulary[K]:
    """categories='auto': observed values sorted, missing last, with counts."""
    var source = column.domain()
    var per_entry = List[Int](length=source.size(), fill=0)
    var missing = 0
    var codes = column.codes()
    for i in range(codes.size):
        var code = Int(codes[Item(i)])
        if code == -1:
            missing += 1
        else:
            per_entry[code] += 1
    var observed = List[K.Domain.Value]()
    for k in range(source.size()):
        if per_entry[k]:
            observed.append(source.get(k))
    K.sort(observed)
    var domain = K.make(observed)
    var counts = List[Int](length=domain.size(), fill=0)
    for k in range(source.size()):
        if per_entry[k]:
            counts[domain.find(source.get(k))] = per_entry[k]
    if missing:
        counts.append(missing)
    return FittedVocabulary[K](FeatureVocabulary[K](domain, missing > 0), counts^)


def explicit_vocabulary[K: CategoryKind](categories: List[Optional[K.Domain.Value]], feature: Int) raises -> FeatureVocabulary[K]:
    """Validate user categories exactly as sklearn `_fit` does (None = missing)."""
    var known = List[K.Domain.Value]()
    var has_missing = False
    for i in range(len(categories)):
        if categories[i]:
            known.append(categories[i].value().copy())
        elif i < len(categories) - 1:
            raise Error("ValueError: Nan should be the last element in user provided categories, see categories " + _format_values[K](known, True) + " in column #" + String(feature))
        else:
            has_missing = True
    if K.has_duplicates(known):
        raise Error("ValueError: In column " + String(feature) + ", the predefined categories contain duplicate elements.")
    if K.requires_sorted() and not K.is_sorted(known):
        raise Error("ValueError: Unsorted categories are not supported for numerical categories")
    return FeatureVocabulary[K](K.make(known), has_missing)


def identify_infrequent(counts: List[Int], n_samples: Int, min_frequency: Optional[Float64], max_categories: Optional[Int]) -> List[Int]:
    """sklearn `_identify_infrequent`: sorted infrequent indices (empty = none)."""
    var size = len(counts)
    var infrequent = List[Bool](length=size, fill=False)
    if min_frequency:
        var threshold = min_frequency.value()
        if threshold >= 1:
            # An absolute count (sklearn's Integral branch).
            for i in range(size):
                infrequent[i] = Float64(counts[i]) < threshold
        else:
            var absolute = Float64(n_samples) * threshold
            for i in range(size):
                infrequent[i] = Float64(counts[i]) < absolute
    var marked = 0
    for flag in infrequent:
        marked += Int(flag)
    if max_categories and max_categories.value() < size - marked + 1:
        var frequent = max_categories.value() - 1
        if frequent == 0:
            for i in range(size):
                infrequent[i] = True
        else:
            # Stable argsort by count (mergesort): ties keep category order.
            var keys = List[Int](capacity=size)
            for i in range(size):
                keys.append(counts[i] * size + i)
            sort(Span(keys))
            for k in range(size - frequent):
                infrequent[keys[k] % size] = True
    var result = List[Int]()
    for i in range(size):
        if infrequent[i]:
            result.append(i)
    return result^


def infrequent_mapping(n_categories: Int, infrequent: List[Int]) -> List[Int]:
    """Frequent categories keep their order; all infrequent map to the last code."""
    var mapping = List[Int](length=n_categories, fill=0)
    var is_infrequent = List[Bool](length=n_categories, fill=False)
    for index in infrequent:
        is_infrequent[index] = True
    var n_frequent = n_categories - len(infrequent)
    var next = 0
    for i in range(n_categories):
        if is_infrequent[i]:
            mapping[i] = n_frequent
        else:
            mapping[i] = next
            next += 1
    return mapping^
