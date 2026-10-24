from std.sys import argv
from moxie.manifold import ClassicalMDS
from trustworthiness_driver import read_matrix


def run[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var model = ClassicalMDS[dtype](Int(String(args[3])), metric=String(args[4]))
    if len(args) > 5:
        var params = model.get_params()
        var options = Dict[String, Float64]()
        options["p"] = Float64(String(args[5]))
        params.metric_params = options^
        _ = model.set_params(params)
    var y = model.fit_transform(x)
    for i in range(model.dissimilarity_matrix_.value().size):
        print("distance", Float64(model.dissimilarity_matrix_.value().unsafe_get(i)))
    for i in range(y.size):
        print("coordinate", y.unsafe_get(i))
    for i in range(model.eigenvalues_.value().size):
        print("eigenvalue", model.eigenvalues_.value().unsafe_get(i))


def main() raises:
    var args = argv()
    if args[1] == "float32":
        run[DType.float32]()
    else:
        run[DType.float64]()
