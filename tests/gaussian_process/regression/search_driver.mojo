from moxie.gaussian_process import (
    GaussianProcessSearchAdapter,
    GenericGaussianProcessRegressor,
)
from moxie.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from moxie.model_selection import GridSearchCV
from numojo import Item, Shape
from std.math import sin
from std.testing import assert_equal, assert_false, assert_raises

comptime Adapter = GaussianProcessSearchAdapter[DType.float32]
comptime Regressor = GenericGaussianProcessRegressor[DType.float32]


def main() raises:
    var X = Adapter.Dense(Shape(15, 1))
    var y = Adapter.Dense(Shape(15, 2))
    var weights = List[Float64]()
    for i in range(15):
        var x = Float64(i) / 5
        X[Item(i, 0)] = Float32(x)
        y[Item(i, 0)] = Float32(sin(x) + Float64(i % 3) / 20)
        y[Item(i, 1)] = Float32(x * x / 4)
        weights.append(Float64(i + 1))

    var model = Regressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01), optimizer=None)
    var adapter = Adapter()
    var candidates = Dict[String, List[Float64]]()
    candidates["alpha"] = [0.001, 0.05]
    candidates["kernel__k1__k2__length_scale"] = [0.2, 0.7, 1.4]
    var search = GridSearchCV[Adapter](model, candidates, adapter)
    _ = search.fit(X, y, None, weights.copy())
    assert_equal(search.n_splits_(), 5)
    var results = search.cv_results_()
    for i in range(len(results.params)):
        for fold in range(search.n_splits_()):
            print("score", i, fold, results.scores.test_scores[Item(i, fold, 0)])
    var best = search.best_params_()
    print("best", best["alpha"], best["kernel__k1__k2__length_scale"])
    var prediction = search.predict(X)
    assert_equal(prediction.ndim, 2)
    assert_equal(prediction.shape[0], 15)
    assert_equal(prediction.shape[1], 2)
    for i in range(15):
        print("prediction", prediction[Item(i, 0)], prediction[Item(i, 1)])

    assert_false(model.__sklearn_is_fitted__())
    var before = String(model.kernel.value())
    var params = Dict[String, Float64]()
    params["alpha"] = 0.05
    params["kernel__k1__k2__length_scale"] = 0.7
    params["kernel__k1__k2__invalid"] = 1.0
    with assert_raises(contains="invalid"):
        adapter.configure(model, params)
    assert_equal(model.alpha[Float64], 1e-10)
    assert_equal(String(model.kernel.value()), before)

    _ = params.pop("kernel__k1__k2__invalid")
    var implicit_kernel = Regressor()
    with assert_raises(contains="require an explicit kernel"):
        adapter.configure(implicit_kernel, params)
    params["optimizer"] = 0.0
    with assert_raises(contains="unsupported GaussianProcessRegressor search parameter"):
        adapter.configure(model, params)
    with assert_raises(contains="does not support weighted fit"):
        adapter.fit(model, X, y, weights.copy())
    with assert_raises(contains="sample_weight length"):
        _ = search.fit(X, y, None, List[Float64]())
    print("contracts passed")
