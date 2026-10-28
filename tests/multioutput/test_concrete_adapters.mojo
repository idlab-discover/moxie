"""CV adapters and scorers that name concrete multioutput wrapper types.

Every wrapper is a scorer's `Estimator` without a generic helper; each fold
score equals the same wrapper cloned, fitted and scored by hand.
"""
from numojo import NDArray, Shape, Item
from std.testing import assert_equal
from moxie.base import clone
from moxie.linear_model import Ridge, RidgeClassifier
from moxie.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain
from moxie.model_selection import IndexArray, IndexSplit, KFold, cross_val_score
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = NDArray[DType.int64]
comptime MOR = MultiOutputRegressor[Ridge[]]
comptime MOC = MultiOutputClassifier[RidgeClassifier[]]
comptime RChain = RegressorChain[Ridge[]]
comptime CChain = ClassifierChain[RidgeClassifier[]]


def dataset() raises -> Tuple[Dense, Dense, Labels]:
    var X = Dense(Shape(30, 3))
    var Y = Dense(Shape(30, 2))
    var L = Labels(Shape(30, 2))
    for i in range(30):
        X[Item(i, 0)] = Float64((i * 7) % 5) / 3 + Float64(i) / 10
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4
        X[Item(i, 2)] = 100.0 + Float64((i * 3) % 7) * 10
        Y[Item(i, 0)] = 2 * X.item(i, 0) - X.item(i, 1) + Float64(i % 3) / 2
        Y[Item(i, 1)] = X.item(i, 1) + 0.05 * X.item(i, 2) - Float64(i % 4) / 3
        L[Item(i, 0)] = Int64(1) if Y.item(i, 0) > 3 else Int64(0)
        L[Item(i, 1)] = Int64(1) if Y.item(i, 1) > 6 else Int64(0)
    return (X^, Y^, L^)


def folds(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(3).split(n)
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


struct MORScore(TypedScorer):
    comptime Estimator = MOR
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: MOR, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct MORRows(CVAdapter):
    comptime Scorer = MORScore

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: MOR, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


struct RChainScore(TypedScorer):
    comptime Estimator = RChain
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: RChain, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct RChainRows(CVAdapter):
    comptime Scorer = RChainScore

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: RChain, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


struct MOCScore(TypedScorer):
    comptime Estimator = MOC
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: MOC, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct MOCRows(CVAdapter):
    comptime Scorer = MOCScore

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: MOC, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


struct CChainScore(TypedScorer):
    comptime Estimator = CChain
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: CChain, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct CChainRows(CVAdapter):
    comptime Scorer = CChainScore

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: CChain, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def assert_scores(scores: Dense, expected: List[Float64]) raises:
    assert_equal(scores.size, len(expected))
    for k in range(len(expected)):
        assert_equal(scores.item(k), expected[k])


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var Y = data[1].copy()
    var L = data[2].copy()
    var splits = folds(30)

    var mor = MOR(Ridge(2.0))
    var mor_expected = List[Float64]()
    var rchain = RChain(Ridge(2.0), order=List[Int]([1, 0]))
    var rchain_expected = List[Float64]()
    var moc = MOC(RidgeClassifier(2.0))
    var moc_expected = List[Float64]()
    var cchain = CChain(RidgeClassifier(2.0), order=List[Int]([1, 0]))
    var cchain_expected = List[Float64]()
    for k in range(len(splits)):
        var train = splits[k].train.copy()
        var test = splits[k].test.copy()
        var a = clone(mor)
        _ = a.fit(_rows(X, train), _rows(Y, train))
        mor_expected.append(a.score(_rows(X, test), _rows(Y, test)))
        var b = clone(rchain)
        _ = b.fit(_rows(X, train), _rows(Y, train))
        rchain_expected.append(b.score(_rows(X, test), _rows(Y, test)))
        var c = clone(moc)
        _ = c.fit(_rows(X, train), _rows(L, train))
        moc_expected.append(c.score(_rows(X, test), _rows(L, test)))
        var d = clone(cchain)
        _ = d.fit(_rows(X, train), _rows(L, train))
        cchain_expected.append(d.score(_rows(X, test), _rows(L, test)))

    assert_scores(cross_val_score[MORRows](mor, X, Y, MORRows(), MORScore(), splits, None, None), mor_expected)
    assert_scores(cross_val_score[RChainRows](rchain, X, Y, RChainRows(), RChainScore(), splits, None, None), rchain_expected)
    assert_scores(cross_val_score[MOCRows](moc, X, L, MOCRows(), MOCScore(), splits, None, None), moc_expected)
    assert_scores(cross_val_score[CChainRows](cchain, X, L, CChainRows(), CChainScore(), splits, None, None), cchain_expected)
    print("PASS: concrete multioutput wrapper types in cross_val_score")
