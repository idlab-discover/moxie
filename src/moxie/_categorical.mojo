"""Typed, owned categorical feature columns. Contract version 1."""
from std.builtin.sort import sort
from std.math import isfinite, isnan
from numojo import NDArray, Shape, Item
import numojo as nm


trait CategoryDomain(Copyable, Movable, Deinitable):
    """A unique dictionary whose category identity never depends on row codes."""
    comptime Value: Copyable & Equatable & Deinitable

    def size(self) -> Int: ...
    def get(self, index: Int) raises -> Self.Value: ...
    def find(self, value: Self.Value) raises -> Int: ...


struct StringCategories(CategoryDomain):
    comptime Value = String
    var _values: List[String]
    var _lookup: Dict[String, Int]

    def __init__(out self, values: List[String]) raises:
        # Explicit string dictionaries retain caller order, as sklearn encoders do.
        self._lookup = Dict[String, Int]()
        for i in range(len(values)):
            var value = values[i].copy()
            if value in self._lookup:
                raise Error("ValueError: duplicate category")
            self._lookup[value.copy()] = i
        self._values = values.copy()

    def size(self) -> Int:
        return len(self._values)

    def get(self, index: Int) raises -> String:
        if index < 0 or index >= self.size():
            raise Error("ValueError: category index out of bounds")
        return self._values[index].copy()

    def find(self, value: String) raises -> Int:
        if value in self._lookup:
            return self._lookup[value]
        return -1

    def values(self) -> List[String]:
        return self._values.copy()


struct NumericCategories[dtype: DType](CategoryDomain):
    comptime Value = Scalar[Self.dtype]
    var _values: NDArray[Self.dtype]

    def __init__(out self, values: NDArray[Self.dtype]) raises:
        comptime assert Self.dtype == DType.int64 or Self.dtype == DType.bool or Self.dtype == DType.float32 or Self.dtype == DType.float64, "categorical numeric dtype must be bool, int64, float32 or float64"
        if values.ndim != 1:
            raise Error("ValueError: categories require rank one")
        self._values = nm.zeros[Self.dtype](Shape(values.size))
        for i in range(values.size):
            var value = values[Item(i)]
            comptime if Self.dtype == DType.float32 or Self.dtype == DType.float64:
                if not isfinite(value):
                    raise Error("ValueError: category dictionary must be finite")
            if i > 0 and value <= values[Item(i - 1)]:
                raise Error("ValueError: numeric categories must be sorted and unique")
            self._values[Item(i)] = value

    def size(self) -> Int:
        return self._values.size

    def get(self, index: Int) raises -> Self.Value:
        if index < 0 or index >= self.size():
            raise Error("ValueError: category index out of bounds")
        return self._values[Item(index)]

    def find(self, value: Self.Value) raises -> Int:
        var lo = 0
        var hi = self.size()
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if self._values[Item(mid)] < value:
                lo = mid + 1
            else:
                hi = mid
        if lo == self.size() or self._values[Item(lo)] != value:
            return -1
        return lo

    def values(self) -> NDArray[Self.dtype]:
        return self._values.copy()


struct CategoricalColumn[Domain: CategoryDomain](Copyable, Movable):
    """One homogeneous nullable feature; -1 marks missing, never unknown."""
    var _domain: Self.Domain
    var _codes: NDArray[DType.int64]

    def __init__(out self, domain: Self.Domain, codes: NDArray[DType.int64]) raises:
        if codes.ndim != 1:
            raise Error("ValueError: categorical codes require rank one")
        self._domain = domain.copy()
        self._codes = nm.zeros[DType.int64](Shape(codes.size))
        for i in range(codes.size):
            var code = codes[Item(i)]
            if code < -1 or code >= Int64(domain.size()):
                raise Error("ValueError: invalid categorical code")
            self._codes[Item(i)] = code

    @staticmethod
    def from_values(domain: Self.Domain, values: List[Optional[Self.Domain.Value]]) raises -> Self:
        var codes = nm.zeros[DType.int64](Shape(len(values)))
        for i in range(len(values)):
            codes[Item(i)] = -1
            if values[i]:
                var code = domain.find(values[i].value())
                if code < 0:
                    raise Error("ValueError: unknown category")
                codes[Item(i)] = Int64(code)
        return Self(domain, codes)

    def rows(self) -> Int:
        return self._codes.size

    def domain(self) -> Self.Domain:
        return self._domain.copy()

    def codes(self) -> NDArray[DType.int64]:
        return self._codes.copy()

    def _check_row(self, row: Int) raises:
        if row < 0 or row >= self.rows():
            raise Error("ValueError: categorical row out of bounds")

    def get(self, row: Int) raises -> Optional[Self.Domain.Value]:
        self._check_row(row)
        var code = Int(self._codes[Item(row)])
        if code == -1:
            return None
        return Optional[Self.Domain.Value](self._domain.get(code))

    def values(self) raises -> List[Optional[Self.Domain.Value]]:
        var result = List[Optional[Self.Domain.Value]]()
        for i in range(self.rows()):
            result.append(self.get(i))
        return result^

    def set(mut self, row: Int, value: Optional[Self.Domain.Value]) raises:
        self._check_row(row)
        var code = -1
        if value:
            code = self._domain.find(value.value())
            if code < 0:
                raise Error("ValueError: unknown category")
        self._codes[Item(row)] = Int64(code)

    def missing_mask(self) raises -> NDArray[DType.bool]:
        var result = nm.zeros[DType.bool](Shape(self.rows()))
        for i in range(self.rows()):
            result[Item(i)] = self._codes[Item(i)] == -1
        return result^

    def take(self, indices: NDArray[DType.int64]) raises -> Self:
        if indices.ndim != 1:
            raise Error("ValueError: row indices require rank one")
        var codes = nm.zeros[DType.int64](Shape(indices.size))
        for i in range(indices.size):
            var row = Int(indices[Item(i)])
            self._check_row(row)
            codes[Item(i)] = self._codes[Item(row)]
        return Self(self._domain, codes)


# Inferred dictionaries follow sklearn's sorted, homogeneous category identity.
def categorical_column(values: List[Optional[String]]) raises -> CategoricalColumn[StringCategories]:
    var seen = Dict[String, Bool]()
    for value in values:
        if value:
            seen[value.value().copy()] = True
    var unique = List[String]()
    for value in seen:
        unique.append(value.copy())
    sort(Span(unique))
    var domain = StringCategories(unique)
    return CategoricalColumn[StringCategories].from_values(domain, values)


def categorical_column[dtype: DType](values: NDArray[dtype], missing: Optional[NDArray[DType.bool]] = None) raises -> CategoricalColumn[NumericCategories[dtype]]:
    if values.ndim != 1:
        raise Error("ValueError: categorical values require rank one")
    if missing:
        if missing.value().ndim != 1 or missing.value().size != values.size:
            raise Error("ValueError: missing mask must match the column")
    var cells = List[Optional[Scalar[dtype]]]()
    var ordered = List[Scalar[dtype]]()
    for i in range(values.size):
        var absent = False
        if missing:
            absent = Bool(missing.value()[Item(i)])
        var value = values[Item(i)]
        comptime if dtype == DType.float32 or dtype == DType.float64:
            absent = absent or isnan(value)
            if not absent and not isfinite(value):
                raise Error("ValueError: categorical values must be finite or missing")
        if absent:
            cells.append(None)
        else:
            cells.append(Optional[Scalar[dtype]](value))
            ordered.append(value)
    sort(Span(ordered))
    var unique = List[Scalar[dtype]]()
    for value in ordered:
        if len(unique) == 0 or unique[len(unique) - 1] != value:
            unique.append(value)
    var dictionary = nm.zeros[dtype](Shape(len(unique)))
    for i in range(len(unique)):
        dictionary[Item(i)] = unique[i]
    var domain = NumericCategories[dtype](dictionary)
    return CategoricalColumn[NumericCategories[dtype]].from_values(domain, cells)
