"""Stratify on String class labels and keep feature rows aligned with their labels."""
from moxie.model_selection import StratifiedKFold, SplitSize, train_test_split
from numojo import NDArray, Shape, Item


def main() raises:
    var y: List[String] = ["spam", "ham", "ham", "spam", "ham", "ham", "spam", "ham", "ham"]
    var X = NDArray[DType.float64](Shape(9, 1))
    for i in range(9):
        X[Item(i, 0)] = Float64(i)

    # Every fold keeps the 1:2 spam/ham ratio.
    var cursor = StratifiedKFold(3, shuffle=True, random_state=0).split(9, y=y)
    var fold = 0
    while True:
        var pair = cursor.next()
        if not pair:
            break
        var labels = String()
        for i in range(pair.value().test.size):
            labels += y[Int(pair.value().test[Item(i)])] + " "
        print("fold", fold, "test labels:", labels)
        fold += 1

    var parts = train_test_split(X, y, random_state=42, test_size=SplitSize(3), stratify=y)
    print("train labels:", len(parts[2]), "test labels:", len(parts[3]))
