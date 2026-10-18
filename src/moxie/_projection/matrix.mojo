"""Owned dense/CSR storage used by the random-projection family."""
from std.math import isfinite


struct ProjectionMatrix[dtype: DType = DType.float64](Copyable, Movable):
    """Row-major dense or CSR matrix. Constructors own their input buffers.

    CSR accepts unsorted indices and duplicate entries; duplicates add.
    Public buffers may be edited, so estimator boundaries revalidate them.
    """

    var rows: Int
    var cols: Int
    var data: List[Scalar[Self.dtype]]
    var indices: List[Int]
    var indptr: List[Int]
    var is_sparse: Bool

    def __init__(out self, rows: Int = 0, cols: Int = 0) raises:
        comptime assert (
            Self.dtype == DType.float64 or Self.dtype == DType.float32
        )
        if (
            rows < 0
            or cols < 0
            or (cols > 0 and rows > 9223372036854775807 // cols)
        ):
            raise Error("ValueError: invalid matrix dimensions")
        self.rows = rows
        self.cols = cols
        self.data = List[Scalar[Self.dtype]](length=rows * cols, fill=0)
        self.indices = List[Int]()
        self.indptr = List[Int]()
        self.is_sparse = False

    def __init__(
        out self, rows: Int, cols: Int, var data: List[Scalar[Self.dtype]]
    ) raises:
        self = Self()
        self.rows = rows
        self.cols = cols
        self.data = data^
        self.validate()

    @staticmethod
    def from_csr(
        rows: Int,
        cols: Int,
        var data: List[Scalar[Self.dtype]],
        var indices: List[Int],
        var indptr: List[Int],
    ) raises -> Self:
        var result = Self()
        result.rows = rows
        result.cols = cols
        result.data = data^
        result.indices = indices^
        result.indptr = indptr^
        result.is_sparse = True
        result.validate()
        return result^

    def validate(self, *, nonempty: Bool = False) raises:
        if (
            self.rows < 0
            or self.cols < 0
            or (self.cols > 0 and self.rows > 9223372036854775807 // self.cols)
        ):
            raise Error("ValueError: invalid matrix dimensions")
        if nonempty and (self.rows == 0 or self.cols == 0):
            raise Error("ValueError: input must contain samples and features")
        if self.is_sparse:
            if len(self.indptr) != self.rows + 1 or len(self.indices) != len(
                self.data
            ):
                raise Error("ValueError: invalid CSR buffer lengths")
            if self.indptr[0] != 0 or self.indptr[self.rows] != len(self.data):
                raise Error("ValueError: invalid CSR row pointers")
            for row in range(self.rows):
                if (
                    self.indptr[row] < 0
                    or self.indptr[row] > self.indptr[row + 1]
                ):
                    raise Error("ValueError: CSR row pointers must be monotone")
            for col in self.indices:
                if col < 0 or col >= self.cols:
                    raise Error("ValueError: CSR column index out of range")
        elif len(self.data) != self.rows * self.cols:
            raise Error("ValueError: dense storage does not match matrix shape")
        for value in self.data:
            if not isfinite(value):
                raise Error("ValueError: input contains NaN or infinity")

    def _value(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        if not self.is_sparse:
            return self.data[row * self.cols + col]
        var result = Scalar[Self.dtype](0)
        for pos in range(self.indptr[row], self.indptr[row + 1]):
            if self.indices[pos] == col:
                result += self.data[pos]
        return result

    def __getitem__(self, row: Int, col: Int) raises -> Scalar[Self.dtype]:
        self.validate()
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            raise Error("ValueError: matrix index out of range")
        return self._value(row, col)

    def to_dense(self) raises -> Self:
        self.validate()
        if not self.is_sparse:
            return self.copy()
        var result = Self(self.rows, self.cols)
        for row in range(self.rows):
            for pos in range(self.indptr[row], self.indptr[row + 1]):
                result.data[row * self.cols + self.indices[pos]] += self.data[
                    pos
                ]
        return result^

    def to_csr(self) raises -> Self:
        self.validate()
        if self.is_sparse:
            return self.copy()
        var data = List[Scalar[Self.dtype]]()
        var indices = List[Int]()
        var indptr: List[Int] = [0]
        for row in range(self.rows):
            for col in range(self.cols):
                var value = self.data[row * self.cols + col]
                if value != 0:
                    data.append(value)
                    indices.append(col)
            indptr.append(len(data))
        return Self.from_csr(self.rows, self.cols, data^, indices^, indptr^)

    def transpose(self) raises -> Self:
        var dense = self.to_dense()
        var result = Self(self.cols, self.rows)
        for row in range(self.rows):
            for col in range(self.cols):
                result.data[col * self.rows + row] = dense.data[
                    row * self.cols + col
                ]
        return result^


def project[
    dtype: DType
](
    X: ProjectionMatrix[dtype],
    components: ProjectionMatrix[dtype],
    *,
    dense_output: Bool = True,
) raises -> ProjectionMatrix[dtype]:
    """Compute X @ components.T without densifying sparse inputs."""
    X.validate(nonempty=True)
    components.validate(nonempty=True)
    if X.cols != components.cols:
        raise Error("ValueError: feature count differs from fitted projection")
    var sparse_output = (
        X.is_sparse and components.is_sparse and not dense_output
    )
    var values = List[Scalar[dtype]]()
    var indices = List[Int]()
    var indptr: List[Int] = [0]
    for row in range(X.rows):
        for comp in range(components.rows):
            var total = Scalar[dtype](0)
            if components.is_sparse:
                for pos in range(
                    components.indptr[comp], components.indptr[comp + 1]
                ):
                    total += (
                        X._value(row, components.indices[pos])
                        * components.data[pos]
                    )
            elif X.is_sparse:
                for pos in range(X.indptr[row], X.indptr[row + 1]):
                    total += (
                        X.data[pos]
                        * components.data[
                            comp * components.cols + X.indices[pos]
                        ]
                    )
            else:
                for col in range(X.cols):
                    total += (
                        X.data[row * X.cols + col]
                        * components.data[comp * components.cols + col]
                    )
            if not sparse_output or total != 0:
                values.append(total)
                if sparse_output:
                    indices.append(comp)
        if sparse_output:
            indptr.append(len(values))
    if sparse_output:
        return ProjectionMatrix[dtype].from_csr(
            X.rows, components.rows, values^, indices^, indptr^
        )
    return ProjectionMatrix[dtype](X.rows, components.rows, values^)
