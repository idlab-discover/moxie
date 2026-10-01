"""Owned dense label domains and sklearn target semantics (BSD notice in LICENSES)."""
from std.builtin.sort import sort
from std.math import isfinite, floor
from numojo import NDArray, Shape, Item
import numojo as nm


struct StringTarget(Copyable, Movable):
    """Owned UTF-8 labels, rank one or two; matrix values are row-major."""
    var _values: List[String]
    var _rows: Int
    var _columns: Int
    var _rank: Int

    def __init__(out self, values: List[String]):
        self._values = values.copy()
        self._rows = len(values)
        self._columns = 1
        self._rank = 1

    def __init__(out self, values: List[String], rows: Int, columns: Int) raises:
        if rows < 0 or columns < 0:
            raise Error("ValueError: negative label dimension")
        if columns != 0 and rows > len(values) // columns:
            raise Error("ValueError: label shape does not match values")
        if rows * columns != len(values):
            raise Error("ValueError: label shape does not match values")
        self._values = values.copy()
        self._rows = rows
        self._columns = columns
        self._rank = 2

    def rows(self) -> Int:
        return self._rows

    def columns(self) -> Int:
        return self._columns

    def rank(self) -> Int:
        return self._rank

    def values(self) -> List[String]:
        return self._values.copy()

    def get(self, row: Int, column: Int = 0) raises -> String:
        self._check_index(row, column)
        return self._values[row * self._columns + column].copy()

    def set(mut self, row: Int, value: String, column: Int = 0) raises:
        self._check_index(row, column)
        self._values[row * self._columns + column] = value.copy()

    def _check_index(self, row: Int, column: Int) raises:
        if row < 0 or row >= self._rows or column < 0 or column >= self._columns:
            raise Error("ValueError: label index out of bounds")


@fieldwise_init
struct LabelIndex(Copyable, Movable):
    """Int64 vectors in sorted-class order; inverse follows input row order."""
    var inverse: NDArray[DType.int64]
    var counts: NDArray[DType.int64]
    var first_indices: NDArray[DType.int64]
    var appearance_order: NDArray[DType.int64]

    def first_seen_codes(self) raises -> NDArray[DType.int64]:
        var rank = nm.zeros[DType.int64](Shape(self.counts.size))
        for i in range(rank.size):
            rank[Item(Int(self.appearance_order[Item(i)]))] = Int64(i)
        var result = nm.zeros[DType.int64](Shape(self.inverse.size))
        for i in range(result.size):
            result[Item(i)] = rank[Item(Int(self.inverse[Item(i)]))]
        return result^


def _find[T: Copyable & Comparable & Deinitable](values: List[T], value: T) -> Int:
    var lo = 0
    var hi = len(values)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo == len(values) or values[lo] != value:
        return -1
    return lo


def _unique[T: Copyable & Comparable & Deinitable](values: List[T]) -> List[T]:
    var ordered = values.copy()
    sort(Span(ordered))
    var unique = List[T]()
    for value in ordered:
        if len(unique) == 0 or unique[len(unique) - 1] != value:
            unique.append(value.copy())
    return unique^


def _index[T: Copyable & Comparable & Deinitable](values: List[T], classes: List[T]) raises -> LabelIndex:
    var inverse = nm.zeros[DType.int64](Shape(len(values)))
    var counts = nm.zeros[DType.int64](Shape(len(classes)))
    var first = nm.zeros[DType.int64](Shape(len(classes)))
    var appearance = nm.zeros[DType.int64](Shape(len(classes)))
    var seen = 0
    for i in range(len(values)):
        var code = _find(classes, values[i])
        if code < 0:
            raise Error("ValueError: previously unseen label")
        inverse[Item(i)] = Int64(code)
        if counts[Item(code)] == 0:
            first[Item(code)] = Int64(i)
            appearance[Item(seen)] = Int64(code)
            seen += 1
        counts[Item(code)] = counts[Item(code)] + 1
    return LabelIndex(inverse^, counts^, first^, appearance^)


def _codes[T: Copyable & Comparable & Deinitable](values: List[T], classes: List[T]) raises -> NDArray[DType.int64]:
    var result = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        var code = _find(classes, values[i])
        if code < 0:
            raise Error("ValueError: previously unseen label")
        result[Item(i)] = Int64(code)
    return result^


def _numeric_dtype[dtype: DType]():
    comptime assert dtype == DType.int64 or dtype == DType.float32 or dtype == DType.float64, "labels support Int64, Float32 and Float64"


def _columns[dtype: DType](y: NDArray[dtype]) raises -> Int:
    _numeric_dtype[dtype]()
    if y.ndim != 1 and y.ndim != 2:
        raise Error("ValueError: labels require rank one or two")
    return 1 if y.ndim == 1 else y.shape[1]


def _numeric_column[dtype: DType](y: NDArray[dtype], column: Int) raises -> List[Scalar[dtype]]:
    var columns = _columns(y)
    if column < 0 or column >= columns:
        raise Error("ValueError: label column out of bounds")
    var values = List[Scalar[dtype]]()
    for row in range(y.shape[0]):
        var value = y[Item(row)] if y.ndim == 1 else y[Item(row, column)]
        comptime if dtype != DType.int64:
            if not isfinite(value):
                raise Error("ValueError: labels and groups must be finite")
        values.append(value)
    return values^


def _string_column(y: StringTarget, column: Int) raises -> List[String]:
    if column < 0 or column >= y.columns():
        raise Error("ValueError: label column out of bounds")
    var values = List[String]()
    for row in range(y.rows()):
        values.append(y.get(row, column))
    return values^


def _array[dtype: DType](values: List[Scalar[dtype]]) raises -> NDArray[dtype]:
    var result = nm.zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = values[i]
    return result^


def owned_target[dtype: DType](y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Compact owned snapshot preserving numeric dtype, rank and logical values."""
    var columns = _columns(y)
    var result = nm.zeros[dtype](Shape(y.shape[0])) if y.ndim == 1 else nm.zeros[dtype](Shape(y.shape[0], columns))
    for col in range(columns):
        var values = _numeric_column(y, col)
        for row in range(len(values)):
            if y.ndim == 1:
                result[Item(row)] = values[row]
            else:
                result[Item(row, col)] = values[row]
    return result^


def type_of_target[dtype: DType](y: NDArray[dtype]) raises -> String:
    _numeric_dtype[dtype]()
    if y.ndim != 1 and y.ndim != 2:
        return "unknown"
    var columns = _columns(y)
    var values = List[Scalar[dtype]]()
    var integral = True
    for col in range(columns):
        var column = _numeric_column(y, col)
        for value in column:
            comptime if dtype != DType.int64:
                # sklearn tests integrality by round-tripping through int64.
                var x = Float64(value)
                if x < -9223372036854775808.0 or x >= 9223372036854775808.0 or floor(x) != x:
                    integral = False
            values.append(value)
    var classes = _unique(values)
    if y.ndim == 2 and columns > 1 and integral and len(classes) < 3:
        return "multilabel-indicator"
    if y.ndim == 2 and (y.shape[0] == 0 or columns == 0):
        return "unknown"
    if not integral:
        return "continuous-multioutput" if columns > 1 else "continuous"
    if columns > 1:
        return "multiclass-multioutput"
    return "multiclass" if len(classes) > 2 else "binary"


def type_of_target(y: StringTarget) -> String:
    if y.rank() == 2 and (y.rows() == 0 or y.columns() == 0):
        return "unknown"
    if y.columns() > 1:
        return "multiclass-multioutput"
    var classes = _unique(y._values)
    return "multiclass" if len(classes) > 2 else "binary"


def _check_kind(kind: String) raises:
    if kind != "binary" and kind != "multiclass" and kind != "multiclass-multioutput" and kind != "multilabel-indicator":
        raise Error("ValueError: unsupported classification target: " + kind)


def check_classification_targets[dtype: DType](y: NDArray[dtype]) raises:
    _check_kind(type_of_target(y))


def check_classification_targets(y: StringTarget) raises:
    _check_kind(type_of_target(y))


@fieldwise_init
struct NumericEncoding[dtype: DType](Copyable, Movable):
    var classes: NDArray[Self.dtype]
    var index: LabelIndex

    def encode(self, y: NDArray[Self.dtype], column: Int = 0) raises -> NDArray[DType.int64]:
        var values = _numeric_column(y, column)
        var classes = _numeric_column(self.classes, 0)
        return _codes(values, classes)

    def decode(self, codes: NDArray[DType.int64]) raises -> NDArray[Self.dtype]:
        if codes.ndim != 1:
            raise Error("ValueError: decoded column requires rank-one codes")
        var result = nm.zeros[Self.dtype](Shape(codes.size))
        for row in range(codes.size):
            var code = Int(codes[Item(row)])
            if code < 0 or code >= self.classes.size:
                raise Error("ValueError: label code out of bounds")
            result[Item(row)] = self.classes[Item(code)]
        return result^


@fieldwise_init
struct StringEncoding(Copyable, Movable):
    var classes: List[String]
    var index: LabelIndex

    def encode(self, y: StringTarget, column: Int = 0) raises -> NDArray[DType.int64]:
        var values = _string_column(y, column)
        return _codes(values, self.classes)

    def decode(self, codes: NDArray[DType.int64]) raises -> StringTarget:
        if codes.ndim != 1:
            raise Error("ValueError: decoded column requires rank-one codes")
        var values = List[String]()
        for row in range(codes.size):
            var code = Int(codes[Item(row)])
            if code < 0 or code >= len(self.classes):
                raise Error("ValueError: label code out of bounds")
            values.append(self.classes[code].copy())
        return StringTarget(values)


def encode_column[dtype: DType](y: NDArray[dtype], column: Int = 0, *, classification: Bool = True) raises -> NumericEncoding[dtype]:
    """One independently sorted class domain per output; groups opt out of classification."""
    if classification:
        check_classification_targets(y)
    var values = _numeric_column(y, column)
    var classes = _unique(values)
    var mapping = _index(values, classes)
    return NumericEncoding[dtype](_array[dtype](classes), mapping^)


def encode_column(y: StringTarget, column: Int = 0, *, classification: Bool = True) raises -> StringEncoding:
    if classification:
        check_classification_targets(y)
    var values = _string_column(y, column)
    var classes = _unique(values)
    var mapping = _index(values, classes)
    return StringEncoding(classes^, mapping^)
