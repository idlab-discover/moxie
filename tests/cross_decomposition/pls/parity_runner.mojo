"""Native side of tests/parity.py: fit the Mojo estimators on fixture cases.

Input: one case per line, `|`-separated:
  id|estimator|dtype|n_components|scale|algorithm|max_iter|tol|copy|X|y|Xt|yt|w
Arrays are `rows:cols:v1,v2,...` (cols=0 means a rank-1 array), `none` if absent.
Output: `id|attribute|array` lines, or `id|error|message` when fit/inference raises.
Python only writes fixtures and compares; every number below is computed here.
"""
import numojo as nm
from std.sys import argv
from moxie.cross_decomposition import CCA, PLSCanonical, PLSParams, PLSRegression
from moxie.cross_decomposition._pls import _PLS


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


def encode[dtype: DType](a: nm.NDArray[dtype]) raises -> String:
    var cols = a.shape[1] if a.ndim == 2 else 0
    var text = String(a.shape[0]) + ":" + String(cols) + ":"
    var count = a.shape[0] * max(cols, 1)
    for i in range(count):
        if i:
            text += ","
        var r = i // max(cols, 1)
        var c = i % max(cols, 1)
        var value: Float64
        if a.ndim == 2:
            value = Float64(a.unsafe_get(r * Int(a.strides[0]) + c * Int(a.strides[1])))
        else:
            value = Float64(a.unsafe_get(r * Int(a.strides[0])))
        text += String(value)
    return text


def emit[dtype: DType](case_id: String, name: String, a: nm.NDArray[dtype]) raises:
    print(case_id + "|" + name + "|" + encode(a))


def emit_list(case_id: String, name: String, values: List[Int]):
    var text = String(len(values)) + ":0:"
    for i in range(len(values)):
        if i:
            text += ","
        text += String(values[i])
    print(case_id + "|" + name + "|" + text)


def run_case[flavor: Int, dtype: DType](f: List[String]) raises:
    var case_id = f[0]
    var algorithm = f[5]
    var estimator = _PLS[flavor, dtype](
        params=PLSParams(
            Int(f[3]), f[4] == "1", algorithm, Int(f[6]), Float64(f[7]), f[8] == "1"
        )
    )
    var X = array[dtype](f[9])
    var y = array[dtype](f[10])
    try:
        estimator.fit(X, y)
    except e:
        print(case_id + "|error|" + String(e))
        return
    emit(case_id, "x_weights_", estimator.x_weights_)
    emit(case_id, "y_weights_", estimator.y_weights_)
    emit(case_id, "x_loadings_", estimator.x_loadings_)
    emit(case_id, "y_loadings_", estimator.y_loadings_)
    emit(case_id, "x_scores", estimator.x_scores_)
    emit(case_id, "y_scores", estimator.y_scores_)
    emit(case_id, "x_rotations_", estimator.x_rotations_)
    emit(case_id, "y_rotations_", estimator.y_rotations_)
    emit(case_id, "coef_", estimator.coef_)
    emit(case_id, "intercept_", estimator.intercept_)
    emit_list(case_id, "n_iter_", estimator.n_iter_)
    for message in estimator.diagnostics_:
        print(case_id + "|diagnostic|" + message)
    var Xt = array[dtype](f[11])
    var yt = array[dtype](f[12])
    try:
        emit(case_id, "transform", estimator.transform(Xt))
        var paired = estimator.transform(Xt, yt)
        emit(case_id, "transform_y", paired.y_scores)
        emit(case_id, "predict", estimator.predict(Xt))
        emit(case_id, "inverse_x", estimator.inverse_transform(paired.x_scores))
        var back = estimator.inverse_transform(paired.x_scores, paired.y_scores)
        emit(case_id, "inverse_y", back.y_scores)
        var score: Float64
        if f[13] == "none":
            score = estimator.score(Xt, yt)
        else:
            score = estimator.score(Xt, yt, array[DType.float64](f[13]))
        print(case_id + "|score|1:0:" + String(score))
        var names = estimator.get_feature_names_out()
        var joined = String("")
        for i in range(len(names)):
            if i:
                joined += ","
            joined += names[i]
        print(case_id + "|names|" + joined)
        # copy=False inference paths must agree with the copying ones and
        # leave the normalized inputs behind.
        var Xc = Xt.copy()
        emit(case_id, "transform_inplace", estimator.transform_inplace(Xc))
        emit(case_id, "transform_inplace_X", Xc)
        var Xp = Xt.copy()
        emit(case_id, "predict_inplace", estimator.predict_inplace(Xp))
        emit(case_id, "predict_inplace_X", Xp)
    except e:
        print(case_id + "|error|" + String(e))
    # fit(copy=False) end state of the caller's arrays.
    var Xw = X.copy()
    var yw = y.copy()
    var mutable = estimator.__sklearn_clone__()
    try:
        mutable.fit_inplace(Xw, yw)
        emit(case_id, "fit_inplace_X", Xw)
        emit(case_id, "fit_inplace_y", yw)
    except e:
        print(case_id + "|error|" + String(e))



def dispatch[dtype: DType](f: List[String]) raises:
    if f[1] == "PLSRegression":
        run_case[0, dtype](f)
    elif f[1] == "PLSCanonical":
        run_case[1, dtype](f)
    elif f[1] == "CCA":
        run_case[2, dtype](f)
    else:
        raise Error("unknown estimator " + f[1])


def main() raises:
    var args = argv()
    var text = open(String(args[1]), "r").read()
    # Touch the public aliases so the exported names are exercised too.
    _ = PLSRegression()
    _ = PLSCanonical()
    _ = CCA()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var parts = List[String]()
        for piece in line.split("|"):
            parts.append(String(piece))
        if parts[2] == "float32":
            dispatch[DType.float32](parts)
        else:
            dispatch[DType.float64](parts)
    print("native|done|mojo")
