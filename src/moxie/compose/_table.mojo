"""Owned named numeric, categorical, and text feature columns."""
from numojo import NDArray, Shape, zeros
from std.utils import Variant
from moxie._categorical import CategoricalColumn, CategoryDomain, NumericCategories, StringCategories

comptime StringColumn = CategoricalColumn[StringCategories]
comptime NumericColumn[dtype: DType] = CategoricalColumn[NumericCategories[dtype]]
comptime _AddedColumn = Variant[
    StringColumn, NumericColumn[DType.bool], NumericColumn[DType.int64],
    NumericColumn[DType.float32], NumericColumn[DType.float64], List[String],
    NDArray[DType.float32], NDArray[DType.float64], NDArray[DType.int64],
    NDArray[DType.bool],
]


def _numeric_kind[dtype: DType]() -> String:
    comptime assert (
        dtype == DType.float32 or dtype == DType.float64
        or dtype == DType.int64 or dtype == DType.bool
    ), "add_numeric requires Float32, Float64, Int64 or Bool"
    return String(dtype)


struct ColumnSchema(Copyable, Movable):
    """Column names and actual storage dtypes, owned independently of data."""

    var names: List[String]
    var dtypes: List[String]

    def __init__(out self, names: List[String], dtypes: List[String]) raises:
        if len(names) != len(dtypes):
            raise Error("ValueError: column names and dtypes must have equal length")
        var seen = Dict[String, Bool]()
        for name in names:
            if name in seen:
                raise Error("ValueError: native column names must be unique")
            seen[name] = True
        for dtype in dtypes:
            if dtype not in ("float32", "float64", "int64", "bool", "category", "string"):
                raise Error("ValueError: native column schemas support float32, float64, int64, bool, category and string")
        self.names = names.copy()
        self.dtypes = dtypes.copy()

    def index(self, name: String) raises -> Int:
        var found = -1
        for i in range(len(self.names)):
            if self.names[i] == name:
                if found >= 0:
                    raise Error(String("ValueError: Selected columns, ['", name, "'], are not unique in dataframe"))
                found = i
        if found < 0:
            raise Error("ValueError: A given column is not a column of the dataframe")
        return found


struct ColumnTable[dtype: DType = DType.float64](Copyable, Movable):
    """An owned named table with a numeric block and typed additional columns."""

    var _data: NDArray[Self.dtype]
    var _schema: ColumnSchema
    # Nonnegative locations index the matrix. Negative locations encode added columns.
    var _locations: List[Int]
    var _added_columns: List[_AddedColumn]

    def __init__(out self, X: NDArray[Self.dtype], names: List[String]) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 tables only"
        if X.ndim != 2:
            raise Error("ValueError: ColumnTable requires a rank-2 matrix")
        if len(names) != X.shape[1]:
            raise Error("ValueError: column names must match the matrix width")
        var dtypes = List[String]()
        self._locations = List[Int]()
        for i in range(len(names)):
            dtypes.append(_numeric_kind[Self.dtype]())
            self._locations.append(i)
        self._schema = ColumnSchema(names, dtypes)
        self._data = X.copy()
        self._added_columns = List[_AddedColumn]()

    def add_numeric[column_dtype: DType](
        mut self, name: String, column: NDArray[column_dtype], *, position: Optional[Int] = None,
    ) raises:
        """Insert an owned rank-one numeric or Boolean column of its original dtype."""
        var kind = _numeric_kind[column_dtype]()
        if column.ndim != 1:
            raise Error("ValueError: add_numeric requires a rank-1 array")
        var index = self._insertion_index(name, column.shape[0], position)
        # NuMojo's strided copy assumes a nonempty last axis.
        var values = column.contiguous() if column.size > 0 else zeros[column_dtype](Shape(0))
        self._insert_column(index, name, kind, _AddedColumn(values^))

    def add_categorical[Domain: CategoryDomain](
        mut self, name: String, column: CategoricalColumn[Domain], *, position: Optional[Int] = None,
    ) raises:
        """Insert an independent categorical column, appending by default."""
        var index = self._insertion_index(name, column.rows(), position)
        self._insert_column(index, name, "category", _AddedColumn(column.copy()))

    def add_strings(
        mut self, name: String, column: List[String], *, position: Optional[Int] = None,
    ) raises:
        """Insert an owned copy of raw UTF-8 strings, appending by default."""
        var index = self._insertion_index(name, len(column), position)
        self._insert_column(index, name, "string", _AddedColumn(column.copy()))

    def string_column(self, index: Int) raises -> List[String]:
        """Copy one raw UTF-8 column in row order."""
        self._check_index(index, "string")
        return self._added_columns[-1 - self._locations[index]][List[String]].copy()

    def _insertion_index(self, name: String, rows: Int, position: Optional[Int]) raises -> Int:
        self._check_shape()
        if name in self._schema.names:
            raise Error("ValueError: native column names must be unique")
        if rows != self.rows():
            raise Error("ValueError: all table columns must have the same number of rows")
        var width = len(self._schema.names)
        var index = position.or_else(width)
        if index < 0 or index > width:
            raise Error("ValueError: column insertion position is out of bounds")
        return index

    def _insert_column(mut self, index: Int, name: String, kind: String, var column: _AddedColumn):
        self._schema.names.insert(index, name)
        self._schema.dtypes.insert(index, kind)
        self._locations.insert(index, -1 - len(self._added_columns))
        self._added_columns.append(column^)

    def values(ref self) raises -> ref[self._data] NDArray[Self.dtype]:
        """Borrow the original matrix when it represents the entire table."""
        if len(self._added_columns) > 0:
            raise Error("TypeError: mixed ColumnTable cannot be borrowed as a numeric matrix")
        return self._data

    def rows(self) raises -> Int:
        return self._data.shape[0]

    def schema(self) raises -> ColumnSchema:
        self._check_shape()
        return self._schema.copy()

    def categorical_columns[Domain: CategoryDomain = StringCategories](
        self, indices: List[Int],
    ) raises -> List[CategoricalColumn[Domain]]:
        """Copy columns of the requested domain in selection order, including repeats."""
        comptime Column = CategoricalColumn[Domain]
        var result = List[Column](capacity=len(indices))
        for index in indices:
            self._check_index(index, "category")
            ref column = self._added_columns[-1 - self._locations[index]]
            if not column.isa[Column]():
                raise Error(String("TypeError: column '", self._schema.names[index],
                                   "' has a different categorical domain than this component requires"))
            result.append(column[Column].copy())
        return result^

    def _select_numeric(self, indices: List[Int], scalar: Bool) raises -> NDArray[Self.dtype]:
        """Gather numeric columns in key order, casting at the dense component boundary."""
        var rows = self.rows()
        var width = len(indices)
        if scalar and width != 1:
            raise Error("ValueError: a scalar column key must resolve to one column")
        var shape = Shape(rows) if scalar else Shape(rows, width)
        var result = zeros[Self.dtype](shape)
        var row_stride = Int(self._data.strides[0])
        var column_stride = Int(self._data.strides[1])
        for output_index in range(width):
            var index = indices[output_index]
            var location = self._locations[index]
            if location >= 0:
                for row in range(rows):
                    result.unsafe_set(
                        row * width + output_index,
                        self._data.unsafe_get(row * row_stride + location * column_stride),
                    )
                continue
            ref column = self._added_columns[-1 - location]
            if column.isa[NDArray[DType.float32]]():
                _gather_column(column[NDArray[DType.float32]], result, output_index, width)
            elif column.isa[NDArray[DType.float64]]():
                _gather_column(column[NDArray[DType.float64]], result, output_index, width)
            elif column.isa[NDArray[DType.int64]]():
                _gather_column(column[NDArray[DType.int64]], result, output_index, width)
            elif column.isa[NDArray[DType.bool]]():
                _gather_column(column[NDArray[DType.bool]], result, output_index, width)
            else:
                raise Error(String("TypeError: column '", self._schema.names[index],
                                   "' requires numeric storage (float32, float64, int64 or bool) for this component"))
        return result^

    def _check_index(self, index: Int, expected: String) raises:
        if index < 0 or index >= len(self._locations):
            raise Error("ValueError: column index is out of bounds")
        if self._schema.dtypes[index] != expected:
            raise Error(String("TypeError: column '", self._schema.names[index], "' requires ", expected, " storage for this component"))

    def _check_shape(self) raises:
        if self._data.ndim != 2 or self._data.shape[1] != len(self._schema.names) - len(self._added_columns):
            raise Error("ValueError: table data no longer matches its column schema")


def _gather_column[source_dtype: DType, dtype: DType](
    column: NDArray[source_dtype], mut result: NDArray[dtype], index: Int, width: Int,
):
    for row in range(column.size):
        result.unsafe_set(row * width + index, column.unsafe_get(row).cast[dtype]())
