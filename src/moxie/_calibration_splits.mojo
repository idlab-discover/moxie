# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation configuration and split materialization for calibration."""
from numojo import Item, Shape, zeros
from std.utils import Variant
from ._calibration_algorithms import Array
from .model_selection import IndexArray, KFold, LeaveOneOut, SplitCursor, StratifiedKFold


@fieldwise_init
struct CalibrationSplit(Copyable, Movable):
    var train: List[Int]
    var test: List[Int]


comptime CalibrationCV = Variant[
    type_of(None), Int, List[CalibrationSplit], KFold, StratifiedKFold, LeaveOneOut
]


def validate_splits(
    splits: List[CalibrationSplit], n: Int, partition: Bool
) raises:
    if len(splits) == 0:
        raise Error("ValueError: cross-validation yielded no splits")
    var tested = List[Int]()
    for _ in range(n):
        tested.append(0)
    for split in splits:
        if len(split.train) == 0 or len(split.test) == 0:
            raise Error("ValueError: train and test splits must be nonempty")
        for index in split.train:
            if index < 0 or index >= n:
                raise Error("ValueError: training index out of bounds")
        for index in split.test:
            if index < 0 or index >= n:
                raise Error("ValueError: test index out of bounds")
            tested[index] += 1
    if partition:
        for count in tested:
            if count != 1:
                raise Error(
                    "ValueError: ensemble=False requires test folds to"
                    " partition the samples"
                )


def _index_list(indices: IndexArray) raises -> List[Int]:
    var result = List[Int](capacity=indices.size)
    for i in range(indices.size):
        result.append(Int(indices[Item(i)]))
    return result^


def _materialize[C: SplitCursor & Deinitable](var cursor: C) raises -> List[CalibrationSplit]:
    var result = List[CalibrationSplit]()
    while True:
        var next = cursor.next()
        if not next:
            break
        var split = next.take()
        result.append(
            CalibrationSplit(_index_list(split.train), _index_list(split.test))
        )
    return result^


def _check_fold_count(labels: IndexArray, n_classes: Int, n_splits: Int) raises:
    if n_splits < 2:
        raise Error("ValueError: cv must have at least two splits")
    var counts = List[Int]()
    for _ in range(n_classes):
        counts.append(0)
    for i in range(labels.size):
        counts[Int(labels[Item(i)])] += 1
    for count in counts:
        if count < n_splits:
            raise Error("ValueError: fewer than cv examples for a class")


def resolve_splits(
    cv: CalibrationCV, labels: Array, n_classes: Int
) raises -> List[CalibrationSplit]:
    if cv.isa[List[CalibrationSplit]]():
        return cv[List[CalibrationSplit]].copy()
    if cv.isa[LeaveOneOut]():
        raise Error(
            "ValueError: LeaveOneOut cross-validation does not allow"
            " all classes to be present in test splits"
        )

    var codes = zeros[DType.int64](Shape(labels.size))
    for i in range(labels.size):
        codes[Item(i)] = Int64(labels.item(i))

    if cv.isa[KFold]():
        var splitter = cv[KFold].copy()
        _check_fold_count(codes, n_classes, splitter.n_splits)
        return _materialize(splitter.split(labels.size))
    if cv.isa[StratifiedKFold]():
        var splitter = cv[StratifiedKFold].copy()
        _check_fold_count(codes, n_classes, splitter.n_splits)
        return _materialize(splitter.split(labels.size, y=codes^))

    var folds = 5
    if cv.isa[Int]():
        folds = cv[Int]
        _check_fold_count(codes, n_classes, folds)
    return _materialize(StratifiedKFold(folds).split(labels.size, y=codes^))
