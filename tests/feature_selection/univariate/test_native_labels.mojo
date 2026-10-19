import numojo as nm
from std.math import nan, inf
from std.testing import assert_equal, assert_raises
from moxie._labels import StringTarget
from moxie.feature_selection import f_classif, chi2, mutual_info_classif, mutual_info_regression, MutualInfoScorer, BuiltinScorer, SelectKBest, SelectionParameter, DiscreteFeatures


def main() raises:
    var x = nm.zeros[DType.float32](nm.Shape(12, 3))
    var y = nm.zeros[DType.float64](nm.Shape(12))
    var words = List[String]()
    for i in range(12):
        words.append("spam" if i % 2 else "ham")
        y.unsafe_set(i, Float64(i % 2))
        for j in range(3):
            x.unsafe_set(3 * i + j, Float32((i * i + i * j + j) % 7))
    var labels = StringTarget(words)
    var selector = SelectKBest(BuiltinScorer[DType.float32, target_type=StringTarget](), k=SelectionParameter(2))
    _ = selector.fit(x, labels.copy())
    var old = selector.scores_()
    var short = StringTarget(["spam"])
    with assert_raises():
        _ = selector.fit(x, short.copy())
    for j in range(3):
        assert_equal(selector.scores_().unsafe_get(j), old.unsafe_get(j))
    var multi = StringTarget(words, 6, 2)
    var empty = StringTarget(List[String]())
    for bad in [short^, multi^, empty^]:
        with assert_raises():
            _ = f_classif(x, bad)
        with assert_raises():
            _ = chi2(x, bad)
        with assert_raises():
            _ = mutual_info_classif(x, bad)
    var integers = nm.zeros[DType.int64](nm.Shape(12, 2))
    with assert_raises():
        _ = f_classif(x, integers)
    with assert_raises():
        _ = mutual_info_classif(x, integers)
    integers.unsafe_set(0, 3)
    with assert_raises():
        _ = chi2(x, integers)
    var scorer = BuiltinScorer[DType.float32, target_type=StringTarget]("f_regression")
    with assert_raises():
        _ = scorer.score(x, labels.copy())
    var mi_strings = MutualInfoScorer[DType.float32, target_type=StringTarget](classification=False)
    with assert_raises():
        _ = mi_strings.score(x, labels.copy())
    with assert_raises():
        _ = scorer.score(x, None)
    with assert_raises():
        _ = mi_strings.score(x, None)
    x.unsafe_set(0, -1)
    with assert_raises():
        _ = chi2(x, labels)
    x.unsafe_set(0, nan[DType.float32]())
    with assert_raises():
        _ = f_classif(x, labels)
    with assert_raises():
        _ = mutual_info_classif(x, labels)
    x.unsafe_set(0, 0)
    y.unsafe_set(0, 0.5)
    with assert_raises():
        _ = chi2(x, y)
    with assert_raises():
        _ = mutual_info_classif(x, y)
    y.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = f_classif(x, y)
    y.unsafe_set(0, 0)
    # Independent numeric target dtype on both MI adapter branches.
    var mi = MutualInfoScorer[DType.float32, DType.float64](random_state=7)
    var adapted = mi.score(x, y.copy())
    var direct = mutual_info_classif(x, y, random_state=7)
    for j in range(3):
        assert_equal(adapted.scores.unsafe_get(j), direct.unsafe_get(j))
    mi.classification = False
    adapted = mi.score(x, y.copy())
    direct = mutual_info_regression(x, y, random_state=7)
    for j in range(3):
        assert_equal(adapted.scores.unsafe_get(j), direct.unsafe_get(j))
    var x64 = x.astype[DType.float64]()
    var y32 = y.astype[DType.float32]()
    var reverse = MutualInfoScorer[DType.float64, DType.float32](random_state=7)
    adapted = reverse.score(x64, y32.copy())
    direct = mutual_info_classif(x64, y32, random_state=7)
    for j in range(3):
        assert_equal(adapted.scores.unsafe_get(j), direct.unsafe_get(j))
    reverse.classification = False
    adapted = reverse.score(x64, y32.copy())
    direct = mutual_info_regression(x64, y32, random_state=7)
    for j in range(3):
        assert_equal(adapted.scores.unsafe_get(j), direct.unsafe_get(j))
    # Refitting with a different partition replaces the scores and support.
    for i in range(12):
        labels.set(i, "a" if i < 6 else "z")
    _ = selector.fit(x, labels.copy())
    var replacement = f_classif(x, labels)
    for j in range(3):
        assert_equal(selector.scores_().unsafe_get(j), replacement.scores.unsafe_get(j))
    print("PASS native label contracts")
