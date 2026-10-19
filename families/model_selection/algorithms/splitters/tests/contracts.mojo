from moxie.model_selection import KFold, TimeSeriesSplit
from std.testing import assert_equal
from numojo import Item

def main() raises:
    var cv = KFold(3)
    var cursor = cv.split(8)
    var first_optional = cursor.next()
    var first = first_optional.value().copy()
    assert_equal(first.test.size, 3)
    assert_equal(Int(first.test[Item(2)]), 2)
    var time = TimeSeriesSplit(3, test_size=2, gap=2, max_train_size=3)
    var temporal = time.split(12)
    var fold_optional = temporal.next()
    var fold = fold_optional.value().copy()
    assert_equal(Int(fold.train[Item(0)]), 1)
    assert_equal(Int(fold.test[Item(0)]), 6)
