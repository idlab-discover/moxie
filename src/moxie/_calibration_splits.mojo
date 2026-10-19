# SPDX-License-Identifier: BSD-3-Clause
"""Materialized cross-validation splits for calibration."""
from std.utils import Variant
from ._calibration_algorithms import Array


@fieldwise_init
struct CalibrationSplit(Copyable, Movable):
    var train: List[Int]
    var test: List[Int]


comptime CalibrationCV = Variant[type_of(None), Int, List[CalibrationSplit]]


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


def stratified_splits(
    labels: Array, folds: Int, explicit_count: Bool
) raises -> List[CalibrationSplit]:
    if folds < 2 or folds > labels.size:
        raise Error(
            "ValueError: cv must be between 2 and the number of samples"
        )
    var first_order = List[Int]()
    for i in range(labels.size):
        var label = Int(labels.item(i))
        var seen = False
        for previous in first_order:
            seen = seen or previous == label
        if not seen:
            first_order.append(label)
    var fold_ids = List[Int]()
    for _ in range(labels.size):
        fold_ids.append(0)
    var offset = 0
    var largest = 0
    for label in first_order:
        var count = 0
        for i in range(labels.size):
            if Int(labels.item(i)) == label:
                count += 1
        largest = max(largest, count)
        if explicit_count and count < folds:
            raise Error("ValueError: fewer than cv examples for a class")
        var allocation = List[Int]()
        for fold in range(folds):
            var allocated = 0
            for k in range(count):
                if (offset + k) % folds == fold:
                    allocated += 1
            allocation.append(allocated)
        var fold = 0
        var used = 0
        for i in range(labels.size):
            if Int(labels.item(i)) == label:
                while used == allocation[fold]:
                    fold += 1
                    used = 0
                fold_ids[i] = fold
                used += 1
        offset += count
    if largest < folds:
        raise Error("ValueError: cv exceeds the size of every class")
    var result = List[CalibrationSplit]()
    for fold in range(folds):
        var train = List[Int]()
        var test = List[Int]()
        for i in range(labels.size):
            if fold_ids[i] == fold:
                test.append(i)
            else:
                train.append(i)
        result.append(CalibrationSplit(train^, test^))
    return result^
