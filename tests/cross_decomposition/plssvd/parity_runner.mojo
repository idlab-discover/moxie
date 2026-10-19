"""Native side of tests/parity.py for PLSSVD.

Input lines: id|dtype|n_components|scale|copy|X|y|Xt|yt (arrays `rows:cols:v,...`).
Output: `id|attribute|array`, `id|names|...` or `id|error|message`.
"""
import numojo as nm
from std.sys import argv
from moxie.cross_decomposition import PLSSVD, PLSSVDParams


def numbers(text: String) raises -> List[Float64]:
    var values = List[Float64]()
    if text.byte_length() == 0:
        return values^
    for value in text.split(","):
        values.append(Float64(String(value)))
    return values^


def array[dtype: DType](text: String) raises -> nm.NDArray[dtype]:
    var fields = text.split(":")
    var rows = Int(String(fields[0]))
    var cols = Int(String(fields[1]))
    var values = numbers(String(fields[2]))
    var a = nm.zeros[dtype](nm.Shape(rows, max(cols, 1)))
    for i in range(len(values)):
        a.unsafe_set(i, Scalar[dtype](values[i]))
    if cols == 0:
        return a.reshape(nm.Shape(rows))
    return a^


def emit[dtype: DType](case_id: String, name: String, a: nm.NDArray[dtype]) raises:
    var cols = a.shape[1] if a.ndim == 2 else 0
    var text = String(a.shape[0]) + ":" + String(cols) + ":"
    for i in range(a.shape[0] * max(cols, 1)):
        if i:
            text += ","
        var r = i // max(cols, 1)
        var c = i % max(cols, 1)
        var offset = r * Int(a.strides[0]) + (c * Int(a.strides[1]) if a.ndim == 2 else 0)
        text += String(Float64(a.unsafe_get(offset)))
    print(case_id + "|" + name + "|" + text)


def run_case[dtype: DType](f: List[String]) raises:
    var case_id = f[0]
    var est = PLSSVD[dtype](params=PLSSVDParams(Int(f[2]), f[3] == "1", f[4] == "1"))
    var X = array[dtype](f[5])
    var y = array[dtype](f[6])
    try:
        est.fit(X, y)
        emit(case_id, "x_weights_", est.x_weights_)
        emit(case_id, "y_weights_", est.y_weights_)
        var Xt = array[dtype](f[7])
        var yt = array[dtype](f[8])
        emit(case_id, "transform", est.transform(Xt))
        emit(case_id, "transform_y", est.transform(Xt, yt).y_scores)
        var names = est.get_feature_names_out()
        var joined = String("")
        for i in range(len(names)):
            if i:
                joined += ","
            joined += names[i]
        print(case_id + "|names|" + joined)
        var Xw = X.copy()
        var yw = y.copy()
        var mutable = est.__sklearn_clone__()
        mutable.fit_inplace(Xw, yw)
        emit(case_id, "fit_inplace_X", Xw)
        emit(case_id, "fit_inplace_y", yw)
    except e:
        print(case_id + "|error|" + String(e))


def main() raises:
    var text = open(String(argv()[1]), "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var parts = List[String]()
        for piece in line.split("|"):
            parts.append(String(piece))
        if parts[1] == "float32":
            run_case[DType.float32](parts)
        else:
            run_case[DType.float64](parts)
    print("native|done|mojo")
