"""Drive Isomap and its dense FeatureUnion composition for pinned parity."""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.manifold import Isomap
from moxie.neighbors._distance import Metric
from moxie.pipeline import make_union
from moxie.preprocessing import StandardScaler
from trustworthiness_driver import read_matrix


def write_matrix[dtype: DType](x: NDArray[dtype]) raises:
    print(x.shape[0], x.shape[1], String(dtype))
    for i in range(x.size):
        print(Float64(x.unsafe_get(i)))


def compare[dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[dtype](String(args[2]))
    var query = read_matrix[dtype](String(args[3]))
    var k: Optional[Int] = None
    var radius: Optional[Float64] = None
    if args[4] != "None":
        k = Int(String(args[4]))
    if args[5] != "None":
        radius = Float64(String(args[5]))
    var metric = String(args[9])
    var p = Float64(String(args[11]))
    var metric_params: Optional[Metric[dtype]] = None
    if len(args) > 13:
        var w = read_matrix[dtype](String(args[13])).reshape(Shape(X.shape[1]))
        metric_params = Metric[dtype](metric, p, w=w^)
    var model = Isomap[dtype](
        n_neighbors=k,
        radius=radius,
        n_components=Int(String(args[10])),
        path_method=String(args[6]),
        eigen_solver=String(args[7]),
        neighbors_algorithm=String(args[8]),
        metric=metric,
        p=p,
        metric_params=metric_params,
    )
    if args[12] == "union":
        var union = make_union[dtype=dtype](model^, StandardScaler[dtype]())
        write_matrix(union.fit_transform(X))
        write_matrix(union.transform(query))
        return
    var embedding = model.fit_transform(X)
    print(
        model.n_features_in_,
        model.nbrs_.value().n_samples_fit_,
        Float64(model.reconstruction_error()),
    )
    print(len(model.fit_warnings))
    for warning in model.fit_warnings:
        print(warning.category)
        print(warning.message)
    for name in model.get_feature_names_out():
        print(name)
    write_matrix(model.dist_matrix_.value())
    write_matrix(embedding)
    write_matrix(model.transform(query))


def main() raises:
    if argv()[1] == "float32":
        compare[DType.float32]()
    else:
        compare[DType.float64]()
