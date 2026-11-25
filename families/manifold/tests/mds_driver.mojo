"""Drive dense MDS and SMACOF for pinned numerical comparisons."""
from numojo import NDArray
from std.sys import argv
from moxie.manifold import (
    MDS,
    MDSMetric,
    MDSMetricParams,
    NormalizedStress,
    SMACOFResult,
    smacof,
)
from moxie.warnings import FitWarning
from trustworthiness_driver import read_matrix


def write_matrix[dtype: DType](x: NDArray[dtype]) raises:
    print(x.shape[0], x.shape[1], String(dtype))
    for i in range(x.size):
        print(Float64(x.unsafe_get(i)))


def write_warnings(warnings: List[FitWarning]):
    print(len(warnings))
    for warning in warnings:
        print(warning.category)
        print(warning.message)


def compare[dtype: DType, init_dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[dtype](String(args[3]))
    var init_file = String(args[5])
    var metric_mds = args[6] == "true"
    var components = Int(String(args[7]))
    var starts = Int(String(args[8]))
    var jobs = Int(String(args[9]))
    var iterations = Int(String(args[10]))
    var eps = Float64(String(args[11]))
    var seed = Int(String(args[12]))
    var normalized: NormalizedStress = "auto"
    if args[13] != "auto":
        normalized = args[13] == "true"
    if args[4] == "smacof":
        var result: SMACOFResult
        if init_file == "None":
            result = smacof(
                X,
                metric=metric_mds,
                n_components=components,
                n_init=starts,
                n_jobs=jobs,
                max_iter=iterations,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized,
                return_n_iter=True,
            )
        else:
            var initial = read_matrix[init_dtype](init_file)
            result = smacof(
                X,
                metric=metric_mds,
                n_components=components,
                init=initial,
                n_init=starts,
                n_jobs=jobs,
                max_iter=iterations,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized,
                return_n_iter=True,
            )
        print(result.stress, result.n_iter.value())
        write_warnings(result.fit_warnings)
        write_matrix(result.embedding)
        return

    var metric: MDSMetric = String(args[15])
    if args[15] == "true" or args[15] == "false":
        metric = args[15] == "true"
    var params: Optional[MDSMetricParams[dtype]] = None
    if args[17] != "None":
        params = MDSMetricParams[dtype](p=Float64(String(args[17])))
    var model = MDS[dtype](
        components,
        metric_mds=metric_mds,
        n_init=starts,
        n_jobs=jobs,
        max_iter=iterations,
        eps=eps,
        random_state=seed,
        normalized_stress=normalized,
        init=String(args[14]),
        metric=metric,
        dissimilarity=String(args[16]),
        metric_params=params^,
    )
    if init_file == "None":
        _ = model.fit(X)
    else:
        var initial = read_matrix[init_dtype](init_file)
        _ = model.fit(X, init=initial)
    print(model.stress_, model.n_iter_, model.n_features_in_)
    write_warnings(model.fit_warnings)
    write_matrix(model.embedding_.value())
    write_matrix(model.dissimilarity_matrix_.value())


def main() raises:
    var args = argv()
    if args[1] == "float32":
        if args[2] == "float32":
            compare[DType.float32, DType.float32]()
        else:
            compare[DType.float32, DType.float64]()
    else:
        compare[DType.float64, DType.float64]()
