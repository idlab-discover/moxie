"""Three folds over eight rows; train/test indices use original row positions."""
from moxie.model_selection import KFold
from numojo import Item


def main() raises:
    var cv = KFold(3)
    var cursor = cv.split(8)
    while True:
        var result = cursor.next()
        if not result:
            break
        print("Test rows:")
        for i in range(result.value().test.size):
            print(Int(result.value().test[Item(i)]))
