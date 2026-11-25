"""Owned named numeric and categorical feature columns."""
from numojo import NDArray
from std.utils import Variant
from moxie._categorical import CategoricalColumn, CategoryDomain, NumericCategories, StringCategories

comptime StringColumn = CategoricalColumn[StringCategories]
comptime NumericColumn[dtype: DType] = CategoricalColumn[NumericCategories[dtype]]
comptime _TableCategory = Variant[
    StringColumn, NumericColumn[DType.bool], NumericColumn[DType.int64],
    NumericColumn[DType.float32], NumericColumn[DType.float64],
]


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
            if dtype != "float32" and dtype != "float64" and dtype != "category":
                raise Error("ValueError: native column schemas support float32, float64 and category")
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
    """An owned named table with a numeric block and typed categorical columns."""

    var _data: NDArray[Self.dtype]
    var _schema: ColumnSchema
    var _locations: List[Int]
    var _categories: List[_TableCategory]

    def __init__(out self, X: NDArray[Self.dtype], names: List[String]) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 tables only"
        if X.ndim != 2:
            raise Error("ValueError: ColumnTable requires a rank-2 matrix")
        if len(names) != X.shape[1]:
            raise Error("ValueError: column names must match the matrix width")
        var dtypes = List[String]()
        self._locations = List[Int]()
        for i in range(len(names)):
            dtypes.append("float32" if Self.dtype == DType.float32 else "float64")
            self._locations.append(i)
        self._schema = ColumnSchema(names, dtypes)
        self._data = X.copy()
        self._categories = List[_TableCategory]()

    def add_categorical[Domain: CategoryDomain](
        mut self, name: String, column: CategoricalColumn[Domain], *, position: Optional[Int] = None,
    ) raises:
        """Insert an independent categorical column, appending by default."""
        self._check_shape()
        if name in self._schema.names:
            raise Error("ValueError: native column names must be unique")
        if column.rows() != self._data.shape[0]:
            raise Error("ValueError: all table columns must have the same number of rows")
        var width = len(self._schema.names)
        var index = position.or_else(width)
        if index < 0 or index > width:
            raise Error("ValueError: column insertion position is out of bounds")
        var stored = _TableCategory(column.copy())
        self._schema.names.insert(index, name)
        self._schema.dtypes.insert(index, "category")
        self._locations.insert(index, len(self._categories))
        self._categories.append(stored^)

    def values(ref self) raises -> ref[self._data] NDArray[Self.dtype]:
        """Borrow the matrix when all columns are numeric."""
        if len(self._categories) > 0:
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
            ref column = self._categories[self._locations[index]]
            if not column.isa[Column]():
                raise Error(String("TypeError: column '", self._schema.names[index],
                                   "' has a different categorical domain than this component requires"))
            result.append(column[Column].copy())
        return result^

    def _numeric_indices(self, indices: List[Int]) raises -> List[Int]:
        var result = List[Int](capacity=len(indices))
        for index in indices:
            self._check_index(index, "float32" if Self.dtype == DType.float32 else "float64")
            result.append(self._locations[index])
        return result^

    def _check_index(self, index: Int, expected: String) raises:
        if index < 0 or index >= len(self._locations):
            raise Error("ValueError: column index is out of bounds")
        if self._schema.dtypes[index] != expected:
            raise Error(String("TypeError: column '", self._schema.names[index], "' requires ", expected, " storage for this component"))

    def _check_shape(self) raises:
        if self._data.ndim != 2 or self._data.shape[1] != len(self._schema.names) - len(self._categories):
            raise Error("ValueError: table data no longer matches its column schema")
