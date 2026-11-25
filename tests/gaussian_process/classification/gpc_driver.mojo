"""Emit shape and values for GPC comparison with pinned sklearn."""
from numojo import NDArray, Shape, zeros
from std.sys import argv

from moxie.gaussian_process import GaussianProcessClassifier
from moxie.gaussian_process.kernels import Bounds, ConstantKernel, Kernel, RBF


def read_array[dtype: DType](
    tokens: List[String], mut pos: Int, shape: Shape
) raises -> NDArray[dtype]:
    var result = zeros[dtype](shape)
    for i in range(result.size):
        result.unsafe_set(i, Float64(tokens[pos]).cast[dtype]())
        pos += 1
    return result^


def emit[dtype: DType](key: String, array: NDArray[dtype]) raises:
    var line = key + "|"
    for i in range(array.ndim):
        if i != 0:
            line += ","
        line += String(array.shape[i])
    line += "|"
    for i in range(array.size):
        if i != 0:
            line += " "
        line += String(array.unsafe_get(i))
    print(line)


def run_case[dtype: DType](path: String, config: String, strategy: String) raises:
    var tokens = List[String]()
    var contents = open(path, "r").read()
    for token in contents.split():
        tokens.append(String(token))
    var pos = 3
    var rows = Int(tokens[0])
    var features = Int(tokens[1])
    var queries = Int(tokens[2])
    var X = read_array[dtype](tokens, pos, Shape(rows, features))
    var y = read_array[DType.int64](tokens, pos, Shape(rows))
    var query = read_array[dtype](tokens, pos, Shape(queries, features))

    var model = GaussianProcessClassifier[dtype]()
    _ = model.set_params(multi_class=strategy)
    if config != "default":
        var length_scale = 0.01 if config.startswith("restart") else 0.8
        var kernel = ConstantKernel(1.0, Bounds(0.01, 100.0)) * RBF(
            length_scale, Bounds(0.001, 10.0)
        )
        _ = model.set_params(kernel=Optional[Kernel](kernel^))
    if config == "fixed" or config == "warm" or config == "cold":
        _ = model.set_params(optimizer=Optional[String]())
    if config == "warm" or config == "cold":
        _ = model.set_params(max_iter_predict=1, warm_start=config == "warm")
    if config == "restarts":
        _ = model.set_params(n_restarts_optimizer=3, random_state=Optional[Int](21))

    _ = model.fit(X, y)
    emit("classes", model.classes_())
    emit("prediction", model.predict(query))
    var binary = model.n_classes_() == 2
    if binary or strategy == "one_vs_rest":
        emit("probability", model.predict_proba(query))
    if binary:
        var latent = model.latent_mean_and_variance(query)
        emit("mean", latent.mean)
        emit("variance", latent.variance)
    print("score||", model.score(X, y))
    print("lml||", model.log_marginal_likelihood().value)
    var theta = model.kernel_().theta()
    emit("theta", theta)
    for i in range(theta.size):
        var shift = 0.1 if binary else 0.02 * Float64(i + 1)
        theta.unsafe_set(i, theta.unsafe_get(i) + shift)
    var shifted = model.log_marginal_likelihood(theta.view(), eval_gradient=binary)
    print("shifted||", shifted.value)
    if binary:
        emit("gradient", shifted.gradient.value())
    else:
        var base = model.base_estimator_()
        print("components||", len(base.estimators_))
        var shared = zeros[DType.float64](Shape(base.estimators_[0].kernel_().n_dims()))
        print("shared_lml||", model.log_marginal_likelihood(shared^).value)
    emit("cloned_theta", model.kernel_().theta())
    print("inplace_lml||", model.log_marginal_likelihood(theta^, clone_kernel=False).value)
    emit("inplace_theta", model.kernel_().theta())
    print("cached_lml||", model.log_marginal_likelihood_value_())
    print("warnings||", len(model.base_estimator_().warnings))


def main() raises:
    var args = argv()
    if args[2] == "f32":
        run_case[DType.float32](String(args[1]), String(args[3]), String(args[4]))
    else:
        run_case[DType.float64](String(args[1]), String(args[3]), String(args[4]))
