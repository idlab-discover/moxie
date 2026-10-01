"""Named-column parity driver; pandas exists only in the Python oracle."""
from numojo import Shape, zeros
from moxie.compose import ColumnTransformer, ColumnEntry, ColumnTable, ColumnKey, Columns, NoTransformer, make_column_selector
from parity_io import encode


def cases[d: DType]() raises:
    var X = zeros[d](Shape(3, 3))
    var Z = zeros[d](Shape(2, 4))
    for i in range(3):
        for j in range(3):
            X.unsafe_set(i * 3 + j, Scalar[d](10 * i + j))
    for i in range(2):
        Z.unsafe_set(i * 4, Scalar[d](100 + 10 * i + 2))
        Z.unsafe_set(i * 4 + 1, Scalar[d](100 + 10 * i))
        Z.unsafe_set(i * 4 + 2, Scalar[d](100 + 10 * i + 1))
        Z.unsafe_set(i * 4 + 3, Scalar[d](999))
    var names: List[String] = ["age", "income", "unused"]
    var znames: List[String] = ["unused", "age", "income", "extra"]
    var table = ColumnTable(X, names)
    var unseen = ColumnTable(Z, znames)
    for scenario in range(7):
        var selected: List[String] = ["income", "age"]
        var key = ColumnKey(selected^)
        if scenario == 1:
            key = ColumnKey.name_slice("age", "income")
        elif scenario == 2:
            var types: List[String] = ["floating"]
            key = ColumnKey(make_column_selector(dtype_include=types^))
        elif scenario == 3:
            var types: List[String] = ["float32"]
            key = ColumnKey(make_column_selector(dtype_include=types^))
        elif scenario == 4:
            key = ColumnKey(Slice(2, None, -1))
        elif scenario == 5:
            var positions: List[Int] = [1, 0]
            key = ColumnKey(positions^)
        elif scenario == 6:
            var mask: List[Bool] = [False, True, False]
            key = ColumnKey(mask^)
        var ct = ColumnTransformer(ColumnEntry[NoTransformer[d], d]("selected", "passthrough", Columns[d](key^)), remainder="passthrough", n_jobs=2)
        var output = ct.fit_transform(table)
        var transformed = ct.transform(unseen)
        var ident = String(scenario, "_", 32 if d == DType.float32 else 64)
        print("result|", ident, "|", encode("out", output), "|", encode("tz", transformed), sep="")
        var features = ct.get_feature_names_out()
        var text = String()
        for i in range(len(features)):
            if i > 0:
                text += ","
            text += features[i]
        print("names|", ident, "|", text, sep="")


def main() raises:
    cases[DType.float32]()
    cases[DType.float64]()
