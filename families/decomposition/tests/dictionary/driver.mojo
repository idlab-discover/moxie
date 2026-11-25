"""Exercise batch dictionary learning against the pinned oracle."""
from std.sys import argv
from std.testing import assert_equal
from numojo import NDArray, Shape
from moxie.decomposition import dict_learning
from decomposition_testing import read_matrix, emit


def report_iteration[dtype: DType](
    iteration: Int, code: NDArray[dtype], dictionary: NDArray[dtype], cost: Float64,
) raises:
    print("callback", iteration)
    print("callback_cost", cost)


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var code_init: Optional[NDArray[dtype]] = None
    var dict_init: Optional[NDArray[dtype]] = None
    var init = Int(args[11])
    if init > 0:
        code_init = read_matrix[dtype](String(args[3]))
    if init == 2:
        dict_init = read_matrix[dtype](String(args[4]))
    var n_components: Optional[Int] = None
    if Int(args[5]) > 0:
        n_components = Int(args[5])
    var result = dict_learning(
        x, n_components, alpha=Float64(String(args[6])), method=String(args[7]),
        max_iter=Int(args[8]), positive_dict=Int(args[9]) != 0,
        positive_code=Int(args[10]) != 0, code_init=code_init, dict_init=dict_init,
        random_state=12, callback=report_iteration[dtype], tol=Float64(String(args[12])), method_max_iter=Int(args[13]),
    )
    var count = n_components.or_else(x.shape[1])
    assert_equal(result.code.shape, Shape(x.shape[0], count))
    assert_equal(result.dictionary.shape, Shape(count, x.shape[1]))
    emit("code", result.code)
    emit("dictionary", result.dictionary)
    emit("reconstruction", result.code @ result.dictionary)
    for error in result.errors:
        print("error", error)
    print("n_iter", result.n_iter)


def main() raises:
    if argv()[1] == "f32":
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
