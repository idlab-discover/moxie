"""Native timed region excludes data construction and process startup."""
import numojo as nm
from moxie._dense import row_squared_norms, sum_axis2d
from std.sys import argv
from std.time import perf_counter_ns


@no_inline
def expression[dtype: DType](x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    return nm.sum(x * x, axis=1)


@no_inline
def list_norms[
    dtype: DType
](x: List[SIMD[dtype, 1]], rows: Int, cols: Int) -> List[SIMD[dtype, 1]]:
    var result = List[SIMD[dtype, 1]](capacity=rows)
    for i in range(rows):
        var total = Float64(0)
        for j in range(cols):
            var v = Float64(x[i * cols + j])
            total += v * v
        result.append(total.cast[dtype]())
    return result^


def bench[
    dtype: DType
](mode: String, rows: Int, cols: Int, layout: String, reps: Int) raises:
    var owner = nm.zeros[dtype](nm.Shape(rows, cols * 2))
    for i in range(rows):
        for j in range(cols * 2):
            owner[nm.Item(i, j)] = (
                SIMD[dtype, 1]((i * 17 + j * 3) % 31 - 15) / 8
            )
    var x = owner.view_with_layout(
        nm.Shape(rows, cols), nm.NDArrayStrides(cols * 2, 2), 0
    )
    if layout == "C":
        x = x.contiguous()
    var data = List[SIMD[dtype, 1]](capacity=rows * cols)
    for i in range(rows):
        for j in range(cols):
            data.append(x.item(i, j))
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for rep in range(reps):
        # Changing input makes each repeated kernel result observable.
        var change = SIMD[dtype, 1](rep % 7) / 8
        if mode == "list":
            data[0] = change
            var result = list_norms(data, rows, cols)
            checksum += Float64(result[0]) + Float64(result[rows - 1])
        else:
            x[nm.Item(0, 0)] = change
            if mode == "expression":
                var result = expression(x)
                checksum += Float64(result.item(0)) + Float64(
                    result.item(rows - 1)
                )
            elif mode == "fused":
                var result = row_squared_norms(x)
                checksum += Float64(result.item(0)) + Float64(
                    result.item(rows - 1)
                )
            elif mode == "center":
                var result = x - nm.mean[returned_dtype=dtype](x, axis=0)
                checksum += Float64(nm.sum(result * result))
            elif mode == "sum":
                var result = sum_axis2d(x, 0)
                checksum += Float64(nm.sum(result))
            elif mode == "matmul":
                var result = x @ x.T()
                checksum += Float64(result.item(0, 0)) + Float64(
                    result.item(rows - 1, rows - 1)
                )
            else:
                raise Error("unknown mode")
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) / 1e9, checksum)


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("usage: benchmark DTYPE MODE ROWS COLS LAYOUT REPETITIONS")
    if args[1] == "f32":
        bench[DType.float32](
            args[2], Int(args[3]), Int(args[4]), args[5], Int(args[6])
        )
    else:
        bench[DType.float64](
            args[2], Int(args[3]), Int(args[4]), args[5], Int(args[6])
        )
