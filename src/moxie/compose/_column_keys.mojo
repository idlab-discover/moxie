# Adapted from scikit-learn sklearn/utils/_indexing.py (_determine_key_type,
# _get_column_indices) and sklearn/compose/_column_transformer.py
# (_is_empty_column_selection, make_column_selector); authors: scikit-learn
# developers. SPDX-License-Identifier: BSD-3-Clause
"""Column selection keys for ColumnTransformer on dense arrays."""
from numojo import NDArray
from moxie.compose._table import ColumnSchema, ColumnTable


struct KeyKind:
    """Variants of a column key (sklearn's accepted `columns` values)."""

    comptime INTS = 0  # list of int positions (rank-2 selection)
    comptime INT = 1  # one int position (rank-1 selection)
    comptime SLICE = 2  # int slice
    comptime MASK = 3  # boolean mask
    comptime NAMES = 4  # list of column names (named tables only)
    comptime NAME = 5  # one column name (named tables only)
    comptime NAME_SLICE = 6  # name slice, inclusive stop (named tables only)
    comptime SELECTOR = 7  # make_column_selector (named tables only)


struct KeyType:
    """Equivalent of sklearn `_determine_key_type` results."""

    comptime NONE = 0
    comptime INT = 1
    comptime BOOL = 2
    comptime STR = 3


struct make_column_selector(Copyable, Movable, Writable):
    """Column-selector configuration with native numeric dtype filtering.

    Native ColumnTable and ColumnSchema support numeric dtype filters. Unnamed
    arrays raise the reference error. Regex patterns raise explicitly pending
    the shared native regex capability; no substring approximation is used.
    """

    var pattern: Optional[String]
    var dtype_include: Optional[List[String]]
    var dtype_exclude: Optional[List[String]]

    def __init__(
        out self,
        pattern: Optional[String] = None,
        *,
        dtype_include: Optional[List[String]] = None,
        dtype_exclude: Optional[List[String]] = None,
    ):
        self.pattern = pattern
        self.dtype_include = dtype_include.copy()
        self.dtype_exclude = dtype_exclude.copy()

    def __call__[dtype: DType](self, X: NDArray[dtype]) raises -> ColumnKey:
        raise Error("ValueError: make_column_selector can only be applied to pandas dataframes")

    def __call__[dtype: DType](self, X: ColumnTable[dtype]) raises -> ColumnKey:
        return self(X.schema())

    def __call__(self, schema: ColumnSchema) raises -> ColumnKey:
        """Select native column names by their actual numeric storage dtype."""
        if self.pattern:
            raise Error("NotImplementedError: regex column patterns require the shared native regex capability")
        var include = self.dtype_include.value().copy() if self.dtype_include else List[String]()
        var exclude = self.dtype_exclude.value().copy() if self.dtype_exclude else List[String]()
        if (Bool(self.dtype_include) or Bool(self.dtype_exclude)) and len(include) == 0 and len(exclude) == 0:
            raise Error("ValueError: at least one of include or exclude must be nonempty")
        for dtype in include:
            _validate_dtype(dtype)
            for other in exclude:
                if _canonical_dtype(dtype) == _canonical_dtype(other):
                    raise Error("ValueError: include and exclude overlap")
        for dtype in exclude:
            _validate_dtype(dtype)
        var names = List[String]()
        for i in range(len(schema.names)):
            var selected = len(include) == 0
            for dtype in include:
                selected = selected or _dtype_matches(dtype, schema.dtypes[i])
            for dtype in exclude:
                if _dtype_matches(dtype, schema.dtypes[i]):
                    selected = False
            if selected:
                names.append(schema.names[i])
        return ColumnKey(names^)


def _canonical_dtype(dtype: String) -> String:
    if dtype == "double" or dtype == "float":
        return "float64"
    if dtype == "single":
        return "float32"
    return dtype


def _validate_dtype(dtype: String) raises:
    var known = List[String]("number", "floating", "float", "float32", "float64", "single", "double",
                             "integer", "signedinteger", "unsignedinteger", "int", "int8", "int16", "int32",
                             "int64", "uint8", "uint16", "uint32", "uint64", "bool", "object", __list_literal__=None)
    for item in known:
        if item == dtype:
            return
    raise Error(String("TypeError: data type '", dtype, "' not understood"))


def _dtype_matches(filter: String, actual: String) -> Bool:
    # ColumnTable currently admits only real floating-point storage.
    return filter == "number" or filter == "floating" or _canonical_dtype(filter) == actual


struct ColumnKey(Copyable, Movable, Writable):
    """One `columns` value: int(s), slice, boolean mask, name(s) or selector.

    Built implicitly from `Int`, `List[Int]`, `Slice`, `List[Bool]`,
    `String`/`StringLiteral`, `List[String]` and `make_column_selector`.
    An `Int` selects one column as a 1-D array (sklearn's scalar selection);
    a one-element list selects a 2-D column. Name-based keys resolve against native table schemas and raise on unnamed
    arrays. Name-slice endpoints are inclusive.
    """

    var kind: Int
    var ints: List[Int]
    var mask: List[Bool]
    var names: List[String]
    var start: Optional[Int]
    var stop: Optional[Int]
    var step: Optional[Int]
    var name_start: Optional[String]
    var name_stop: Optional[String]
    var selector: Optional[make_column_selector]

    def __init__(out self, *, kind: Int):
        self.kind = kind
        self.ints = List[Int]()
        self.mask = List[Bool]()
        self.names = List[String]()
        self.start = None
        self.stop = None
        self.step = None
        self.name_start = None
        self.name_stop = None
        self.selector = None

    @implicit
    def __init__(out self, index: Int):
        self = Self(kind=KeyKind.INT)
        self.ints.append(index)

    @implicit
    def __init__(out self, var indices: List[Int]):
        self = Self(kind=KeyKind.INTS)
        self.ints = indices^

    @implicit
    def __init__[n: Int](out self, indices: Array[Int, n]):
        self = Self(kind=KeyKind.INTS)
        for i in range(n):
            self.ints.append(indices[i])

    @implicit
    def __init__[n: Int](out self, mask: Array[Bool, n]):
        self = Self(kind=KeyKind.MASK)
        for i in range(n):
            self.mask.append(mask[i])

    @implicit
    def __init__(out self, positions: Slice):
        self = Self(kind=KeyKind.SLICE)
        self.start = positions.start
        self.stop = positions.end
        self.step = positions.step

    @implicit
    def __init__(out self, var mask: List[Bool]):
        self = Self(kind=KeyKind.MASK)
        self.mask = mask^

    @implicit
    def __init__(out self, name: StringLiteral):
        self = Self(kind=KeyKind.NAME)
        self.names.append(String(name))

    @implicit
    def __init__(out self, name: String):
        self = Self(kind=KeyKind.NAME)
        self.names.append(name)

    @implicit
    def __init__(out self, var names: List[String]):
        self = Self(kind=KeyKind.NAMES)
        self.names = names^

    @implicit
    def __init__(out self, var selector: make_column_selector):
        self = Self(kind=KeyKind.SELECTOR)
        self.selector = selector^

    @staticmethod
    def name_slice(start: Optional[String], stop: Optional[String], step: Optional[Int] = None) -> Self:
        """Equivalent of sklearn's `slice("a", "c")` on DataFrame column names (inclusive stop)."""
        var key = Self(kind=KeyKind.NAME_SLICE)
        key.name_start = start
        key.name_stop = stop
        key.step = step
        return key^

    def is_scalar(self) -> Bool:
        return self.kind == KeyKind.INT or self.kind == KeyKind.NAME

    def is_callable(self) -> Bool:
        return self.kind == KeyKind.SELECTOR

    def key_type(self) raises -> Int:
        """Equivalent of sklearn `_determine_key_type`; callables raise ValueError there."""
        if self.kind == KeyKind.INT:
            return KeyType.INT
        if self.kind == KeyKind.INTS:
            return KeyType.INT if len(self.ints) > 0 else KeyType.NONE
        if self.kind == KeyKind.SLICE:
            return KeyType.INT if (self.start or self.stop) else KeyType.NONE
        if self.kind == KeyKind.MASK:
            return KeyType.BOOL if len(self.mask) > 0 else KeyType.NONE
        if self.kind == KeyKind.NAME:
            return KeyType.STR
        if self.kind == KeyKind.NAMES:
            return KeyType.STR if len(self.names) > 0 else KeyType.NONE
        if self.kind == KeyKind.NAME_SLICE:
            return KeyType.STR if (self.name_start or self.name_stop) else KeyType.NONE
        raise Error("ValueError: No valid specification of the columns. Only a scalar, list or slice of all "
                    "integers or all strings, or boolean mask is allowed")

    def is_empty(self) -> Bool:
        """Equivalent of sklearn `_is_empty_column_selection` (slices and scalars are never empty)."""
        if self.kind == KeyKind.INTS:
            return len(self.ints) == 0
        if self.kind == KeyKind.NAMES:
            return len(self.names) == 0
        if self.kind == KeyKind.MASK:
            for flag in self.mask:
                if flag:
                    return False
            return True
        return False

    def indices(self, n_features: Int) raises -> List[Int]:
        """Equivalent of sklearn `_get_column_indices` for a dense array with `n_features` columns."""
        if (self.kind == KeyKind.INTS and len(self.ints) == 0) or (self.kind == KeyKind.NAMES and len(self.names) == 0):
            return List[Int]()
        if self.kind == KeyKind.SELECTOR:
            raise Error("ValueError: make_column_selector can only be applied to pandas dataframes")
        var key_type = self.key_type()
        # A slice without start and stop has key type None, which sklearn
        # routes to the name branch: it raises on arrays even for ints.
        if key_type == KeyType.STR or key_type == KeyType.NONE:
            raise Error("ValueError: Specifying the columns using strings is only supported for dataframes.")
        var bounds = String("ValueError: all features must be in [0, ", n_features - 1, "] or [-", n_features, ", 0]")
        var out = List[Int]()
        if self.kind == KeyKind.INT or self.kind == KeyKind.INTS:
            for index in self.ints:
                if index < -n_features or index >= n_features:
                    raise Error(bounds)
                out.append(index + n_features if index < 0 else index)
        elif self.kind == KeyKind.MASK:
            if len(self.mask) != n_features:
                raise Error(bounds)
            for i in range(n_features):
                if self.mask[i]:
                    out.append(i)
        else:
            # Python slice semantics over range(n_features), as NumPy indexing.
            var step = self.step.value() if self.step else 1
            if step == 0:
                raise Error("ValueError: slice step cannot be zero")
            var start: Int
            var stop: Int
            if step > 0:
                start = _clamp_start(self.start, n_features, 0, n_features)
                stop = _clamp_start(self.stop, n_features, n_features, n_features)
                var i = start
                while i < stop:
                    out.append(i)
                    i += step
            else:
                start = _clamp_negative(self.start, n_features, n_features - 1)
                stop = _clamp_negative(self.stop, n_features, -1)
                var i = start
                while i > stop:
                    out.append(i)
                    i += step
        return out^

    def indices(self, schema: ColumnSchema) raises -> List[Int]:
        """Resolve named keys with inclusive name-slice endpoints."""
        if self.kind == KeyKind.SELECTOR:
            return self.selector.value()(schema).indices(schema)
        if self.kind == KeyKind.NAME or self.kind == KeyKind.NAMES:
            var out = List[Int]()
            for name in self.names:
                out.append(schema.index(name))
            return out^
        if self.kind == KeyKind.NAME_SLICE:
            if self.step and self.step.value() != 1:
                raise Error("NotImplementedError: key.step must be 1 or None")
            var start = schema.index(self.name_start.value()) if self.name_start else 0
            var stop = schema.index(self.name_stop.value()) + 1 if self.name_stop else len(schema.names)
            var out = List[Int]()
            for i in range(start, stop):
                out.append(i)
            return out^
        if self.kind == KeyKind.SLICE and not self.start and not self.stop:
            if self.step and self.step.value() != 1:
                raise Error("NotImplementedError: key.step must be 1 or None")
            var key = self.copy()
            key.start = 0
            return key.indices(len(schema.names))
        return self.indices(len(schema.names))

    def write_to(self, mut writer: Some[Writer]):
        """Python-like spelling used in records and messages."""
        if self.kind == KeyKind.INT:
            writer.write(self.ints[0])
        elif self.kind == KeyKind.INTS:
            writer.write("[")
            for i in range(len(self.ints)):
                writer.write(", " if i > 0 else "", self.ints[i])
            writer.write("]")
        elif self.kind == KeyKind.MASK:
            writer.write("[")
            for i in range(len(self.mask)):
                writer.write(", " if i > 0 else "", "True" if self.mask[i] else "False")
            writer.write("]")
        elif self.kind == KeyKind.SLICE:
            writer.write("slice(", _optional_text(self.start), ", ", _optional_text(self.stop), ", ",
                         _optional_text(self.step), ")")
        elif self.kind == KeyKind.SELECTOR:
            writer.write("make_column_selector(...)")
        else:
            writer.write("[")
            for i in range(len(self.names)):
                writer.write(", " if i > 0 else "", "'", self.names[i], "'")
            writer.write("]")


def _optional_text(value: Optional[Int]) -> String:
    return String(value.value()) if value else String("None")


def _clamp_start(bound: Optional[Int], n: Int, default: Int, upper: Int) -> Int:
    if not bound:
        return default
    var b = bound.value()
    if b < 0:
        b += n
        return 0 if b < 0 else b
    return upper if b > upper else b


def _clamp_negative(bound: Optional[Int], n: Int, default: Int) -> Int:
    if not bound:
        return default
    var b = bound.value()
    if b < 0:
        b += n
        return -1 if b < 0 else b
    return n - 1 if b >= n else b
