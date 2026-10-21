"""CalibratedClassifierCV inside clone, Pipeline and cross_val_score."""
import numojo as nm
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.calibration import CalibratedClassifierCV, CalibrationCV, CalibrationSplit
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    CVAdapter, TypedScorer, IndexArray, IndexSplit, KFold, cross_val_score,
)
from calibration_centroid import Centroid

comptime Dense = nm.NDArray[DType.float64]
comptime Calibrated = CalibratedClassifierCV[Centroid]


def take_rows(X: Dense, rows: IndexArray) raises -> Dense:
    if X.ndim == 1:
        var out = Dense(nm.Shape(rows.size))
        for i in range(rows.size):
            out[nm.Item(i)] = X.item(Int(rows[nm.Item(i)]))
        return out^
    var out = Dense(nm.Shape(rows.size, X.shape[1]))
    for i in range(rows.size):
        for j in range(X.shape[1]):
            out[nm.Item(i, j)] = X.item(Int(rows[nm.Item(i)]), j)
    return out^


struct Accuracy(TypedScorer):
    comptime Estimator = Calibrated
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: Calibrated, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


@fieldwise_init
struct CalibratedCV(CVAdapter):
    comptime Scorer = Accuracy

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return take_rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return take_rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Calibrated, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def dataset() raises -> Tuple[Dense, Dense]:
    var X = Dense(nm.Shape(36, 2))
    var y = Dense(nm.Shape(36))
    for i in range(36):
        var label = Float64(i % 3)
        X[nm.Item(i, 0)] = label * 1.5 + Float64((i * 7) % 5) / 3
        X[nm.Item(i, 1)] = Float64((i * 5) % 11) / 4 - label
        y[nm.Item(i)] = label * 2 - 1
    return (X^, y^)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone keeps configuration, drops fitted state; the original is untouched.
    var model = Calibrated(Centroid(), method="isotonic", cv=3)
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.method, "isotonic")
    assert_equal(copied.cv[Int], 3)
    _ = copied.fit(X, y)
    var a = model.predict_proba(X)
    var b = copied.predict_proba(X)
    for i in range(36):
        for j in range(3):
            assert_equal(a.item(i, j), b.item(i, j))

    # Pipeline final step: scaling then calibration equals manual composition.
    var pipe = make_pipeline(StandardScaler(), Calibrated(Centroid(), cv=3))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = Calibrated(Centroid(), cv=3)
    _ = manual.fit(Xs, y)
    var piped = pipe.predict_proba(X)
    var direct = manual.predict_proba(Xs)
    for i in range(36):
        for j in range(3):
            assert_equal(piped.item(i, j), direct.item(i, j))

    # cross_val_score clones the calibrator per fold; each score equals a
    # manual fit/score on the same KFold partition.
    var cursor = KFold(n_splits=3).split(36)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var template = Calibrated(Centroid(), method="sigmoid", cv=2)
    var scores = cross_val_score[CalibratedCV](template, X, y, CalibratedCV(), Accuracy(), splits, None, None)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(take_rows(X, splits[k].train), take_rows(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(take_rows(X, splits[k].test), take_rows(y, splits[k].test)))
    print("PASS: clone, Pipeline final step and cross_val_score composition")
