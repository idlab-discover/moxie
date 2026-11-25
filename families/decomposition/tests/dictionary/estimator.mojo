"""Compare the public DictionaryLearning estimator with the pinned oracle."""
from std.sys import argv
from std.testing import assert_equal
from numojo import NDArray, Shape
from moxie.decomposition import DictionaryLearning
from decomposition_testing import read_matrix, emit


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var held = read_matrix[dtype](String(args[3]))
    var n_components: Optional[Int] = None
    if Int(args[4]) > 0:
        n_components = Int(args[4])
    var transform_alpha: Optional[Float64] = None
    if args[8] != "none":
        transform_alpha = Float64(String(args[8]))
    var split_sign = Int(args[9]) != 0
    var max_iter = Int(args[10])
    var model = DictionaryLearning[dtype](
        n_components, alpha=0.32, fit_algorithm=String(args[5]),
        transform_algorithm=String(args[6]), transform_n_nonzero_coefs=Int(args[7]),
        transform_alpha=transform_alpha, split_sign=split_sign, max_iter=max_iter,
        random_state=12,
    )
    var train: NDArray[dtype] = model.fit_transform(x)
    var count = n_components.or_else(x.shape[1])
    assert_equal(train.shape, Shape(x.shape[0], count))
    assert_equal(model.components_.shape, Shape(count, x.shape[1]))
    emit("train_code", train)
    emit("components", model.components_)
    for error in model.error_:
        print("error", error)
    print("n_iter", model.n_iter_)
    print("n_features_in", model.get_n_features_in())
    var names = model.get_feature_names_out()
    for i in range(len(names)):
        print(String("name", i), names[i])

    var held_code: NDArray[dtype] = model.transform(held)
    assert_equal(held_code.shape, Shape(held.shape[0], count * (2 if split_sign else 1)))
    emit("held_code", held_code)
    var inverse: NDArray[dtype] = model.inverse_transform(held_code)
    assert_equal(inverse.shape, held.shape)
    emit("inverse", inverse)

    var refit: NDArray[dtype] = model.fit(x).transform(held)
    emit("after_fit", refit)


def main() raises:
    if argv()[1] == "f32":
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
