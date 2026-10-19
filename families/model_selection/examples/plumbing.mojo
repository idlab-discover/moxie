"""An owned split supplied by an application, not a splitting algorithm.

Run with the assembled import root (pixi run build prepares this binary).
"""
from numojo import full, Shape, Item
from moxie.model_selection import IndexSplit

def main() raises:
    var train = full[DType.int64](Shape(2), 0)
    train[Item(1)] = 1
    var test = full[DType.int64](Shape(1), 2)
    var split = IndexSplit(train^, test^)
    split.validate(3)
    var snapshot = split.copy()
    snapshot.train[Item(0)] = 1
    print("Original training row:", split.train[Item(0)])
    print("Independent snapshot row:", snapshot.train[Item(0)])
