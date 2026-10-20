"""Native dense dictionary encoding. Reference: sklearn (BSD-3-Clause)."""
from std.sys import size_of
from std.builtin.sort import sort as sort_keys
from std.math import isfinite, trunc
from numojo import NDArray, Shape, zeros
from moxie.feature_extraction._base import FeatureExtractor
from moxie.feature_extraction._mapping import OrderedVocabulary, category_key, FeatureNames
from moxie.feature_extraction._validation import require_dense_output


struct FeatureKey(Hashable, Equatable, Comparable, Copyable, Movable):
    """String or machine-integer key; display conversion never changes identity."""
    var text: String
    var integer: Int
    var is_integer: Bool

    def __init__(out self, text: String):
        self.text = text
        self.integer = 0
        self.is_integer = False

    def __init__(out self, integer: Int):
        self.text = String(integer)
        self.integer = integer
        self.is_integer = True

    def __hash__(self) -> UInt:
        return UInt(hash(self.text)) ^ UInt(self.is_integer)

    def __eq__(self, other: Self) -> Bool:
        return self.is_integer == other.is_integer and self.text == other.text

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __lt__(self, other: Self) -> Bool:
        if self.is_integer:
            return self.integer < other.integer
        return self.text < other.text

    def __le__(self, other: Self) -> Bool:
        return self == other or self < other

    def __gt__(self, other: Self) -> Bool:
        return other < self

    def __ge__(self, other: Self) -> Bool:
        return other <= self


struct FeatureValue(Copyable, Movable):
    """Tagged numeric, category, category-list or None value."""
    var kind: Int
    var number: Float64
    var integer: Int64
    var unsigned: UInt64
    var categories: List[String]

    def __init__(out self):
        self.kind = 4
        self.number = 0
        self.integer = 0
        self.unsigned = 0
        self.categories = List[String]()

    def __init__(out self, number: Float64):
        self = Self()
        self.kind = 0
        self.number = number

    def __init__(out self, number: Int64):
        self = Self()
        self.kind = 5
        self.integer = number

    def __init__(out self, number: UInt64):
        self = Self()
        self.kind = 6
        self.unsigned = number

    def __init__(out self, category: String):
        self = Self()
        self.kind = 1
        self.categories.append(category)

    def __init__(out self, categories: List[String]):
        self = Self()
        self.kind = 2
        self.categories = categories.copy()

    def scalar[dtype: DType](self) raises -> Scalar[dtype]:
        if self.kind == 1 or self.kind == 2:
            return 1
        if self.kind == 5:
            comptime if dtype.is_integral() and dtype != DType.bool:
                var converted = self.integer.cast[dtype]()
                if converted.cast[DType.int64]() != self.integer:
                    raise Error("OverflowError: integer outside dtype range")
                comptime if not dtype.is_signed():
                    if self.integer < 0:
                        raise Error("OverflowError: negative integer for unsigned dtype")
            return self.integer.cast[dtype]()
        if self.kind == 6:
            comptime if dtype.is_integral() and dtype != DType.bool:
                var converted = self.unsigned.cast[dtype]()
                if converted.cast[DType.uint64]() != self.unsigned:
                    raise Error("OverflowError: integer outside dtype range")
                comptime if dtype.is_signed():
                    if converted < 0:
                        raise Error("OverflowError: integer outside signed dtype range")
            return self.unsigned.cast[dtype]()
        if self.kind == 4:
            comptime if dtype.is_floating_point():
                return Scalar[dtype](Float64("nan"))
            elif dtype == DType.bool:
                return Scalar[dtype](0)
            else:
                raise Error("TypeError: None cannot convert to integer dtype")
        comptime if dtype.is_integral() and dtype != DType.bool:
            if not isfinite(self.number):
                raise Error("ValueError: nonfinite value cannot convert to integer dtype")
            comptime bits = Int(size_of[Scalar[dtype]]()) * 8
            var upper = Float64(2) ** (bits - Int(dtype.is_signed()))
            var lower = -upper if dtype.is_signed() else Float64(0)
            if trunc(self.number) < lower or trunc(self.number) >= upper:
                raise Error("OverflowError: numeric value outside dtype range")
        return self.number.cast[dtype]()


@fieldwise_init
struct FeatureEntry(Copyable, Movable):
    var key: FeatureKey
    var value: FeatureValue


comptime FeatureRow = List[FeatureEntry]
comptime FeatureBatch = List[FeatureRow]


@fieldwise_init
struct DictVectorizerParams[dtype: DType = DType.float64](Copyable, Movable):
    var separator: String
    var sparse: Bool
    var sort: Bool


struct _DictionaryColumns(Copyable, Movable):
    """Shared ordered keys plus sklearn restrict's possibly noncanonical indices.

    Duplicate support indices overwrite a previous index with the current unique
    key count in sklearn. Keep that observable quirk without forking key storage.
    """
    var _keys: OrderedVocabulary[FeatureKey]
    var _indices: List[Int]

    def __init__(out self, keys: List[FeatureKey]) raises:
        self._keys = OrderedVocabulary[FeatureKey](keys)
        self._indices = List[Int]()
        for i in range(len(keys)):
            self._indices.append(i)

    def size(self) -> Int:
        return self._keys.size()

    def keys(self) -> List[FeatureKey]:
        return self._keys.keys()

    def column(self, key: FeatureKey) raises -> Int:
        var position = self._keys.column(key)
        return self._indices[position] if position >= 0 else -1


struct DictVectorizer[dtype: DType = DType.float64](FeatureExtractor, FeatureNames):
    comptime Input = FeatureBatch
    comptime Target = Int
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = DictVectorizerParams[Self.dtype]
    var separator: String
    var sparse: Bool
    var sort: Bool
    var vocabulary_: _DictionaryColumns
    var feature_names_: List[FeatureKey]
    var _fitted: Bool

    def __init__(out self, *, separator: String = "=", sparse: Bool = True, sort: Bool = True) raises:
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.feature_names_ = List[FeatureKey]()
        self.vocabulary_ = _DictionaryColumns(self.feature_names_)
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.separator, self.sparse, self.sort)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.separator = params.separator
        self.sparse = params.sparse
        self.sort = params.sort
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, separator: Optional[String] = None, sparse: Optional[Bool] = None, sort: Optional[Bool] = None) raises -> ref[self] Self:
        if separator:
            self.separator = separator.value()
        if sparse:
            self.sparse = sparse.value()
        if sort:
            self.sort = sort.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(separator=self.separator, sparse=self.sparse, sort=self.sort)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _check(self) raises:
        if not self._fitted:
            raise Error("NotFittedError: DictVectorizer is not fitted")

    def _keys(self, entry: FeatureEntry) raises -> List[FeatureKey]:
        var result = List[FeatureKey]()
        if entry.value.kind == 1 or entry.value.kind == 2:
            for value in entry.value.categories:
                result.append(FeatureKey(category_key(entry.key.text, value, self.separator)))
        else:
            result.append(entry.key.copy())
        return result^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        require_dense_output(self.sparse)
        var names = List[FeatureKey]()
        var seen = Dict[FeatureKey, Int]()
        for row in X:
            for entry in row:
                var keys = self._keys(entry)
                for key in keys:
                    if key not in seen:
                        seen[key.copy()] = len(names)
                        names.append(key.copy())
        if self.sort and len(names) > 1:
            for key in names:
                if key.is_integer != names[0].is_integer:
                    raise Error("TypeError: integer and string feature keys cannot be sorted together")
            sort_keys(names)
        var vocab = _DictionaryColumns(names)
        self.feature_names_ = names^
        self.vocabulary_ = vocab^
        self._fitted = True
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check()
        require_dense_output(self.sparse)
        if len(X) == 0:
            raise Error("ValueError: Sample sequence X is empty")
        comptime if not (Self.dtype == DType.bool or Self.dtype == DType.float32 or Self.dtype == DType.float64 or (Self.dtype.is_integral() and size_of[Scalar[Self.dtype]]() <= 8)):
            raise Error("ValueError: dtype is not supported by the reference dense conversion")
        var columns = self.vocabulary_.size()
        if columns != 0 and len(X) > Int.MAX // columns // Int(size_of[Scalar[Self.dtype]]()):
            raise Error("ValueError: dense output byte size overflow")
        var result = zeros[Self.dtype](Shape(len(X), columns))
        for i in range(len(X)):
            for entry in X[i]:
                var keys = self._keys(entry)
                for key in keys:
                    var j = self.vocabulary_.column(key)
                    if j >= columns:
                        raise Error("ValueError: restricted vocabulary index exceeds output width")
                    if j >= 0:
                        var index = i * columns + j
                        comptime if Self.dtype == DType.bool:
                            result.unsafe_set(index, result.unsafe_get(index) | entry.value.scalar[Self.dtype]())
                        else:
                            result.unsafe_set(index, result.unsafe_get(index) + entry.value.scalar[Self.dtype]())
        return result^

    def transform(self, X: FeatureRow) raises -> Self.Transformed:
        var batch = FeatureBatch()
        batch.append(X.copy())
        return self.transform(batch)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        # Commit learned state only after successful conversion and allocation.
        var candidate = self.__sklearn_clone__()
        _ = candidate.fit(X, y)
        var result = candidate.transform(X)
        self = candidate^
        return result^

    def fit_transform(mut self, X: FeatureRow, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        var batch = FeatureBatch()
        batch.append(X.copy())
        return self.fit_transform(batch, y)

    def get_feature_names_out(self) raises -> List[String]:
        self._check()
        var names = List[String]()
        for key in self.feature_names_:
            names.append(key.text)
        return names^

    def get_feature_names_out(self, input_features: List[String]) raises -> List[String]:
        return self.get_feature_names_out()

    def inverse_transform[input_dtype: DType](self, X: NDArray[input_dtype]) raises -> List[Dict[FeatureKey, Scalar[input_dtype]]]:
        self._check()
        if X.ndim != 2:
            raise Error("ValueError: inverse input must be rank two")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: inverse input must have samples and features")
        var result = List[Dict[FeatureKey, Scalar[input_dtype]]]()
        for i in range(X.shape[0]):
            var row = Dict[FeatureKey, Scalar[input_dtype]]()
            for j in range(X.shape[1]):
                var value = X.unsafe_get(i * X.strides[0] + j * X.strides[1])
                comptime if input_dtype.is_floating_point():
                    if not isfinite(value):
                        raise Error("ValueError: inverse input must be finite")
                if value != 0:
                    if j >= len(self.feature_names_):
                        raise Error("IndexError: inverse feature index out of range")
                    row[self.feature_names_[j].copy()] = value
            result.append(row^)
        return result^

    def restrict[origin: MutOrigin](
        ref[origin] self, support: List[Int], indices: Bool = False
    ) raises -> ref[self] Self:
        self._check()
        var names = List[FeatureKey]()
        var seen = Dict[FeatureKey, Int]()
        for k in range(len(support)):
            if not indices and support[k] == 0:
                continue
            var i = support[k] if indices else k
            if i < 0:
                i += len(self.feature_names_)
            if i < 0 or i >= len(self.feature_names_):
                raise Error("IndexError: restrict index out of range")
            var key = self.feature_names_[i].copy()
            var next_index = len(seen)
            if key not in seen:
                names.append(key.copy())
            seen[key^] = next_index
        # Stable index ordering, including duplicate-index ties.
        var ordering = List[Tuple[Int, Int]]()
        for i in range(len(names)):
            ordering.append((seen[names[i]], i))
        sort_keys(ordering)
        var ordered = List[FeatureKey]()
        var columns = List[Int]()
        for pair in ordering:
            ordered.append(names[pair[1]].copy())
            columns.append(pair[0])
        var vocab = _DictionaryColumns(ordered)
        vocab._indices = columns^
        self.vocabulary_ = vocab^
        self.feature_names_ = ordered^
        return self


    def restrict[origin: MutOrigin](
        ref[origin] self, support: List[Bool], indices: Bool = False
    ) raises -> ref[self] Self:
        var values = List[Int]()
        for value in support:
            values.append(Int(value))
        return self.restrict(values, indices)


    def restrict[support_dtype: DType, origin: MutOrigin](
        ref[origin] self, support: NDArray[support_dtype], indices: Bool = False
    ) raises -> ref[self] Self:
        self._check()
        if support.ndim != 1:
            raise Error("ValueError: restrict support must be rank one")
        var values = List[Int]()
        for i in range(support.size):
            var value = support.unsafe_get(i * support.strides[0])
            if not indices:
                values.append(Int(not (value == 0)))
            else:
                comptime if support_dtype.is_integral() and support_dtype != DType.bool:
                    comptime if not support_dtype.is_signed():
                        if UInt64(value) >= UInt64(len(self.feature_names_)):
                            raise Error("IndexError: restrict index out of range")
                    values.append(Int(value))
                else:
                    raise Error("TypeError: restrict indices must be integers")
        return self.restrict(values, indices)
