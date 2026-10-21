"""Public forest edge cases and invalid inputs; compared by public_edge_parity.py."""
import numojo as nm
from moxie.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from moxie.tree._classes import FeatureCount, ClassWeight, ClassWeightMap
from moxie.ensemble._sampling import MaxSamples
from moxie._labels import StringTarget

comptime F = nm.NDArray[DType.float32]
comptime V = nm.NDArray[DType.float64]


def data() raises -> Tuple[F, V]:
    var X = nm.zeros[DType.float32](nm.Shape(30, 3))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X.unsafe_set(3 * i, Float32((i * 7) % 11) / 3)
        X.unsafe_set(3 * i + 1, Float32((i * 5) % 13) / 4)
        X.unsafe_set(3 * i + 2, Float32(i % 4))
        y.unsafe_set(i, Float64((i * 7) % 3))
    return (X^, y^)


def line(tag: String, v: V) raises -> String:
    var s = tag
    for i in range(v.size):
        s += " " + String(v.unsafe_get(i))
    return s


def main() raises:
    var d = data()
    var X = d[0].copy()
    var y = d[1].copy()
    # 1 criterion invalid
    try:
        var m = RandomForestClassifier(5, criterion="foo", random_state=0)
        _ = m.fit(X, y)
        print("criterion_foo OK")
    except e:
        print("criterion_foo ERR", e)
    # 2 max_features invalid
    try:
        var m = RandomForestClassifier(5, max_features=FeatureCount(String("foo")), random_state=0)
        _ = m.fit(X, y)
        print("max_features_foo OK")
    except e:
        print("max_features_foo ERR", e)
    # 3 class_weight missing label
    try:
        var cw = ClassWeightMap()
        cw[0] = 2.0
        cw[99] = 1.0
        var m = RandomForestClassifier(5, class_weight=ClassWeight(cw^), random_state=0)
        _ = m.fit(X, y)
        print("cw_missing OK")
    except e:
        print("cw_missing ERR", e)
    # 4 y NaN regression
    try:
        var yn = y.copy()
        yn.unsafe_set(3, Float64("nan"))
        var m = RandomForestRegressor(5, random_state=0)
        _ = m.fit(X, yn)
        print("y_nan OK")
    except e:
        print("y_nan ERR", e)
    # 5 X inf
    try:
        var Xi = X.copy()
        Xi.unsafe_set(4, Float32(Float64("inf")))
        var m = RandomForestRegressor(5, random_state=0)
        _ = m.fit(Xi, y)
        print("x_inf OK")
    except e:
        print("x_inf ERR", e)
    # 6 NaN at predict only
    var m6 = RandomForestRegressor(5, random_state=0)
    _ = m6.fit(X, y)
    var Q = X.copy()
    for i in range(10):
        Q.unsafe_set(3 * i + (i % 3), Float32(Float64("nan")))
    print(line("nan_predict", m6.predict(Q)))
    # 7 max_samples int > n
    var m7 = RandomForestClassifier(4, max_samples=MaxSamples(100), random_state=0)
    _ = m7.fit(X, y)
    print("max_samples_100", m7.estimators_samples_()[0].size)
    print(line("max_samples_100_proba", m7.predict_proba(X)))
    # 8 log_loss criterion
    try:
        var m = RandomForestClassifier(5, criterion="log_loss", random_state=0)
        _ = m.fit(X, y)
        print(line("log_loss", m.predict_proba(X)))
    except e:
        print("log_loss ERR", e)
    # 9 friedman_mse regressor
    try:
        var m = ExtraTreesRegressor(5, criterion="friedman_mse", random_state=0)
        _ = m.fit(X, y)
        print(line("friedman", m.predict(X)))
    except e:
        print("friedman ERR", e)
    # 10 single sample
    try:
        var X1 = nm.zeros[DType.float32](nm.Shape(1, 3))
        var y1 = nm.zeros[DType.float64](nm.Shape(1))
        y1.unsafe_set(0, 4)
        var m = RandomForestClassifier(3, random_state=0)
        _ = m.fit(X1, y1)
        print(line("single", m.predict_proba(X1)), m.n_classes_)
    except e:
        print("single ERR", e)
    # 11 zero sample weights
    try:
        var w = nm.zeros[DType.float64](nm.Shape(30))
        var m = RandomForestRegressor(3, random_state=0)
        _ = m.fit(X, y, sample_weight=w)
        print(line("zero_weights", m.predict(X)))
    except e:
        print("zero_weights ERR", e)
    # 12 negative sample weight
    try:
        var w = nm.full[DType.float64](nm.Shape(30), 1)
        w.unsafe_set(0, -1)
        var m = RandomForestRegressor(3, random_state=0)
        _ = m.fit(X, y, sample_weight=w)
        print(line("neg_weight", m.predict(X)))
    except e:
        print("neg_weight ERR", e)
    # 13 unfitted
    var u = ExtraTreesClassifier(3)
    try:
        _ = u.predict_proba(X)
        print("unfitted_proba OK")
    except e:
        print("unfitted_proba ERR", e)
    try:
        _ = u.estimators_samples_()
        print("unfitted_samples OK")
    except e:
        print("unfitted_samples ERR", e)
    print("unfitted_importances size", u.feature_importances_.size)
    # 14 max_features int too large
    try:
        var m = RandomForestClassifier(3, max_features=FeatureCount(10), random_state=0)
        _ = m.fit(X, y)
        print("max_features_10 OK")
    except e:
        print("max_features_10 ERR", e)
    # 15 constant features
    var Xc = nm.zeros[DType.float32](nm.Shape(30, 3))
    var m15 = RandomForestRegressor(3, random_state=0)
    _ = m15.fit(Xc, y)
    print(line("constant_imp", m15.feature_importances_))
    print(line("constant_pred", m15.predict(Xc[0:2])))
    # 16 oob_score with max_samples float and warm start no-op
    var m16 = RandomForestClassifier(6, oob_score=True, max_samples=MaxSamples(Float64(0.5)), random_state=1)
    _ = m16.fit(X, y)
    print("oob16", m16.oob_score_)
    # 17 string label alternative: negative labels sorted
    var yneg = y.copy()
    for i in range(30):
        yneg.unsafe_set(i, -yneg.unsafe_get(i) * 2)
    var m17 = RandomForestClassifier(4, random_state=2)
    _ = m17.fit(X, yneg)
    print(line("classes17", m17.classes_))
    print(line("pred17", m17.predict(X)))
    # 18 typed labels: strings and Int64 keep their own sorted domains.
    var names: List[String] = ["pear", "apple", "fig"]
    var codes: List[Int64] = [20, -5, 7]
    var words = List[String]()
    var ints = nm.zeros[DType.int64](nm.Shape(30))
    for i in range(30):
        words.append(names[Int(y.unsafe_get(i))])
        ints.unsafe_set(i, codes[Int(y.unsafe_get(i))])
    var m18 = RandomForestClassifier[string_labels=True](4, random_state=2)
    _ = m18.fit(X, StringTarget(words))
    var text = String("strings18")
    for c in m18.classes_:
        text += " " + c
    var predicted = m18.predict(X)
    for i in range(predicted.rows()):
        text += " " + predicted.get(i)
    print(text)
    var m19 = ExtraTreesClassifier[label_dtype=DType.int64](4, random_state=2)
    _ = m19.fit(X, ints)
    var labels = String("ints19")
    for i in range(m19.classes_.size):
        labels += " " + String(m19.classes_.unsafe_get(i))
    var p19 = m19.predict(X)
    for i in range(p19.size):
        labels += " " + String(p19.unsafe_get(i))
    print(labels)
