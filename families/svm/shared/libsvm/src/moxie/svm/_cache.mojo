"""Bounded LRU kernel columns, with native owned Float32 storage."""
import numojo as nm
from moxie.svm._contracts import Dense, KernelSpec, KernelColumnCache
from moxie.svm._kernels import kernel_matrix, prediction_kernel_matrix

@fieldwise_init
struct _Column(Movable):
    var index: Int
    var stamp: Int
    var values: nm.NDArray[DType.float32]

struct ColumnCache(KernelColumnCache):
    var x: Dense
    var spec: KernelSpec
    var columns: List[_Column]
    var capacity: Int
    var clock: Int

    def __init__(out self, x: Dense, spec: KernelSpec, cache_mb: Float64) raises:
        self.x = x.copy()
        self.spec = spec.copy()
        self.columns = List[_Column]()
        var n = x.shape[0]
        var column_bytes = Float64(n) * 4
        self.capacity = n if cache_mb >= Float64(n) * column_bytes / 1048576 else min(n, max(2, Int(cache_mb * 1048576 / column_bytes)))
        self.clock = 0

    def raw(self, index: Int, prediction: Bool = False) raises -> Dense:
        var n = self.x.shape[0]
        var result = nm.zeros[DType.float64](nm.Shape(n))
        if self.spec.kind == "precomputed":
            for j in range(n):
                result.unsafe_set(j, self.x.unsafe_get(index * Int(self.x.strides[0]) + j * Int(self.x.strides[1])))
        else:
            var row = nm.zeros[DType.float64](nm.Shape(1, self.x.shape[1]))
            for j in range(self.x.shape[1]):
                row.unsafe_set(j, self.x.unsafe_get(index * Int(self.x.strides[0]) + j * Int(self.x.strides[1])))
            var matrix = prediction_kernel_matrix(row, self.x, self.spec) if prediction else kernel_matrix(row, self.x, self.spec)
            for j in range(n):
                result.unsafe_set(j, matrix.unsafe_get(j))
        return result^

    def diagonal(self, index: Int) raises -> Float64:
        if self.spec.kind == "precomputed":
            return self.x.unsafe_get(index * Int(self.x.strides[0]) + index * Int(self.x.strides[1]))
        var row = nm.zeros[DType.float64](nm.Shape(1, self.x.shape[1]))
        for j in range(self.x.shape[1]):
            row.unsafe_set(j, self.x.unsafe_get(index * Int(self.x.strides[0]) + j * Int(self.x.strides[1])))
        var matrix = kernel_matrix(row, row, self.spec)
        return matrix.unsafe_get(0)

    def column(mut self, index: Int) raises -> nm.NDArray[DType.float32]:
        self.clock += 1
        for i in range(len(self.columns)):
            if self.columns[i].index == index:
                self.columns[i].stamp = self.clock
                return self.columns[i].values.view()
        var values = self.raw(index).astype[DType.float32]()
        if len(self.columns) < self.capacity:
            self.columns.append(_Column(index, self.clock, values.view()))
        else:
            var oldest = 0
            for i in range(1, len(self.columns)):
                if self.columns[i].stamp < self.columns[oldest].stamp:
                    oldest = i
            self.columns[oldest] = _Column(index, self.clock, values.view())
        return values^
