"""Drive dense LLE and its classifier pipeline for pinned sklearn parity."""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.manifold import LocallyLinearEmbedding, locally_linear_embedding
from moxie.neighbors import KNeighborsClassifier
from moxie.pipeline import make_pipeline
from trustworthiness_driver import read_matrix


def write_matrix(x: NDArray[DType.float64]) raises:
    print(x.shape[0], x.shape[1], "float64")
    for i in range(x.size):
        print(x.unsafe_get(i))


def compare[dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[dtype](String(args[2]))
    var query = read_matrix[dtype](String(args[3]))
    var k = Int(String(args[4]))
    var d = Int(String(args[5]))
    var method = String(args[6])
    var reg = Float64(String(args[7]))
    var algorithm = String(args[8])
    var solver = String(args[9])
    var operation = String(args[10])
    if operation == "function":
        var result = locally_linear_embedding(
            X,
            n_neighbors=k,
            n_components=d,
            method=method,
            reg=reg,
            eigen_solver=solver,
        )
        print(result.reconstruction_error)
        write_matrix(result.embedding)
        return

    var model = LocallyLinearEmbedding[dtype](
        n_neighbors=k,
        n_components=d,
        method=method,
        reg=reg,
        neighbors_algorithm=algorithm,
        eigen_solver=solver,
    )
    if operation == "pipeline":
        comptime if dtype == DType.float64:
            var labels = read_matrix[DType.float64](String(args[11])).reshape(
                Shape(X.shape[0])
            )
            var pipeline = make_pipeline(
                model^, KNeighborsClassifier(n_neighbors=3)
            )
            _ = pipeline.fit(X, labels)
            var predicted = pipeline.predict(query)
            print(predicted.size)
            for i in range(predicted.size):
                print(predicted.unsafe_get(i))
        else:
            raise Error("pipeline driver uses Float64")
        return

    var embedding = model.fit_transform(X)
    print(
        model.reconstruction_error_,
        model.get_n_features_in(),
        model.nbrs_.value().n_samples_fit_,
    )
    for name in model.get_feature_names_out():
        print(name)
    write_matrix(embedding)
    write_matrix(model.transform(query))


def main() raises:
    if argv()[1] == "float32":
        compare[DType.float32]()
    else:
        compare[DType.float64]()
