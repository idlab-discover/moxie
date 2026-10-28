"""CV adapters and scorers that name concrete multiclass wrapper types.

Every wrapper is a scorer's `Estimator` without a generic helper; each fold
score equals the same wrapper cloned, fitted and scored by hand.
"""
from numojo import NDArray, Shape, Item
from std.testing import assert_equal
from moxie.base import clone
from moxie.linear_model import RidgeClassifier
from moxie.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from moxie.model_selection import IndexArray, IndexSplit, StratifiedKFold, cross_val_score
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = NDArray[DType.int64]
comptime OvR = OneVsRestClassifier[RidgeClassifier[]]
comptime OvO = OneVsOneClassifier[RidgeClassifier[]]
comptime Code = OutputCodeClassifier[RidgeClassifier[]]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(45, 3))
    var y = Labels(Shape(45))
    for i in range(45):
        var label = i % 3
        X[Item(i, 0)] = Float64(label) * 1.5 + Float64((i * 7) % 5) / 3
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4 - Float64(label)
        X[Item(i, 2)] = 100.0 + Float64((i * 3) % 7) * 10
        y[Item(i)] = Int64(label * 2 - 1)
    return (X^, y^)


def stratified(y: Labels) raises -> List[IndexSplit]:
    var cursor = StratifiedKFold(3).split(y.size, y.copy())
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


struct OvRAccuracy(TypedScorer):
    comptime Estimator = OvR
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: OvR, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct OvRRows(CVAdapter):
    comptime Scorer = OvRAccuracy

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: OvR, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


struct OvOAccuracy(TypedScorer):
    comptime Estimator = OvO
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: OvO, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct OvORows(CVAdapter):
    comptime Scorer = OvOAccuracy

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: OvO, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


struct CodeAccuracy(TypedScorer):
    comptime Estimator = Code
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: Code, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct CodeRows(CVAdapter):
    comptime Scorer = CodeAccuracy

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Code, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def assert_scores(scores: Dense, expected: List[Float64]) raises:
    assert_equal(scores.size, len(expected))
    for k in range(len(expected)):
        assert_equal(scores.item(k), expected[k])


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()
    var splits = stratified(y)

    var ovr = OvR(RidgeClassifier(2.0))
    var ovo = OvO(RidgeClassifier(2.0))
    var code = Code(RidgeClassifier(2.0), code_size=2.0, random_state=4)
    var ovr_expected = List[Float64]()
    var ovo_expected = List[Float64]()
    var code_expected = List[Float64]()
    for k in range(len(splits)):
        var train = splits[k].train.copy()
        var test = splits[k].test.copy()
        var a = clone(ovr)
        _ = a.fit(_rows(X, train), _rows(y, train))
        ovr_expected.append(a.score(_rows(X, test), _rows(y, test)))
        var b = clone(ovo)
        _ = b.fit(_rows(X, train), _rows(y, train))
        ovo_expected.append(b.score(_rows(X, test), _rows(y, test)))
        var c = clone(code)
        _ = c.fit(_rows(X, train), _rows(y, train))
        code_expected.append(c.score(_rows(X, test), _rows(y, test)))

    assert_scores(cross_val_score[OvRRows](ovr, X, y, OvRRows(), OvRAccuracy(), splits, None, None), ovr_expected)
    assert_scores(cross_val_score[OvORows](ovo, X, y, OvORows(), OvOAccuracy(), splits, None, None), ovo_expected)
    assert_scores(cross_val_score[CodeRows](code, X, y, CodeRows(), CodeAccuracy(), splits, None, None), code_expected)
    print("PASS: concrete multiclass wrapper types in cross_val_score")
