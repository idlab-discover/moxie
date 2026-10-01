"""Named dense numeric input for native composition, backed by NuMojo."""
from numojo import NDArray


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
            if dtype != "float32" and dtype != "float64":
                raise Error("ValueError: native column schemas currently support float32 and float64 storage")
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
    """An owned named matrix; all columns retain the matrix's actual dtype.

    Construction and copy own their data. `values()` borrows the underlying array; `schema()` returns an independent metadata snapshot. This is a numeric
    table adapter, not a dataframe or a replacement array container.
    """

    var _data: NDArray[Self.dtype]
    var _schema: ColumnSchema

    def __init__(out self, X: NDArray[Self.dtype], names: List[String]) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 tables only"
        if X.ndim != 2:
            raise Error("ValueError: ColumnTable requires a rank-2 matrix")
        if len(names) != X.shape[1]:
            raise Error("ValueError: column names must match the matrix width")
        var dtypes = List[String]()
        for _ in names:
            dtypes.append("float32" if Self.dtype == DType.float32 else "float64")
        self._schema = ColumnSchema(names, dtypes)
        self._data = X.copy()

    def values(ref self) -> ref[self._data] NDArray[Self.dtype]:
        """Borrow the existing numeric array without materialization."""
        return self._data

    def schema(self) raises -> ColumnSchema:
        if self._data.ndim != 2 or self._data.shape[1] != len(self._schema.names):
            raise Error("ValueError: table data no longer matches its column schema")
        return self._schema.copy()
