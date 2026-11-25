"""Exercise functional and estimator coding against the pinned oracle."""
from std.sys import argv
from std.testing import assert_equal
from moxie.decomposition import SparseCoder, sparse_encode
from moxie.decomposition_serialization import SparseCoderSchema
from moxie.serialization import dumps, loads
from numojo import NDArray, Shape
from decomposition_testing import read_matrix, emit


def evaluate[dtype: DType]() raises:
    comptime Schema = SparseCoderSchema[dtype]
    var a = argv()
    var x = read_matrix[dtype](String(a[2]))
    var d = read_matrix[dtype](String(a[3]))
    var algorithm = String(a[7])
    var alpha: Optional[Float64] = None
    if Float64(String(a[8])) >= 0:
        alpha = Float64(String(a[8]))
    var positive = Int(a[9]) != 0
    var split = Int(a[10]) != 0
    var max_iter = Int(a[11])
    var cov: Optional[NDArray[dtype]] = None
    var gram: Optional[NDArray[dtype]] = None
    var init: Optional[NDArray[dtype]] = None
    if Int(a[12]) != 0:
        cov = read_matrix[dtype](String(a[4]))
    if Int(a[13]) != 0:
        gram = read_matrix[dtype](String(a[5]))
    if Int(a[14]) != 0:
        init = read_matrix[dtype](String(a[6]))
    var code = sparse_encode(x, d, cov=cov, gram=gram, init=init, algorithm=algorithm,
        alpha=alpha, positive=positive, max_iter=max_iter)
    assert_equal(code.shape, Shape(x.shape[0], d.shape[0]))
    emit("functional", code)
    var m = SparseCoder[dtype](d, transform_algorithm=algorithm, transform_alpha=alpha,
        positive_code=positive, split_sign=split, transform_max_iter=max_iter)
    m = Schema.decode(loads(dumps(Schema.encode(m))))
    var z = m.transform(x)
    emit("code", z)
    emit("inverse", m.inverse_transform(z))
    emit("fit", m.fit_transform(x))
    m = Schema.decode(loads(dumps(Schema.encode(m))))
    emit("restored_fit", m.transform(x))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
