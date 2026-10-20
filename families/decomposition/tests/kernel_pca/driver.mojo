"""KernelPCA differential driver: dtype X Y k kernel solver inverse remove seed gamma power."""
from std.sys import argv
from moxie.decomposition import KernelPCA
from moxie.decomposition._kernel_pca import KernelPower
from decomposition_testing import emit, emit_scalar, read_matrix


def evaluate[dtype: DType]() raises:
    var a = argv()
    var x = read_matrix[dtype](String(a[2]))
    var y = read_matrix[dtype](String(a[3]))
    var k: Optional[Int] = None
    if Int(a[4]) > 0: k = Int(a[4])
    var gamma: Optional[Float64] = None
    if Float64(String(a[10])) >= 0: gamma = Float64(String(a[10]))
    var power: KernelPower = "auto"
    if Int(a[11]) >= 0: power = Int(a[11])
    var model = KernelPCA[dtype](k, kernel=String(a[5]), eigen_solver=String(a[6]),
        fit_inverse_transform=Int(a[7]) != 0, remove_zero_eig=Int(a[8]) != 0,
        random_state=Int(a[9]), gamma=gamma, iterated_power=power)
    var z = model.fit_transform(x)
    emit("fit", z)
    emit("values", model.eigenvalues_.value())
    emit("vectors", model.eigenvectors_.value())
    emit("transform", model.transform(x))
    emit("test", model.transform(y))
    emit_scalar("gamma", model.gamma_)
    if model.fit_inverse_transform:
        emit("inverse", model.inverse_transform(z))
        emit("inverse_test", model.inverse_transform(model.transform(y)))
        emit("dual", model.dual_coef_.value())


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
