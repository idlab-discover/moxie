from moxie.model_selection import StratifiedKFold, train_test_split, SplitSize, IndexArray
from numojo import Shape, Item

def main() raises:
    var labels = IndexArray(Shape(8))
    for i in range(8):
        labels[Item(i)] = Int64(i // 4)
    var cv = StratifiedKFold(2, shuffle=True, random_state=42)
    var cursor = cv.split(8, y=labels.copy())
    while True:
        var pair = cursor.next()
        if not pair:
            break
        print("training rows:", pair.value().train.size, "test rows:", pair.value().test.size)
    var pieces = train_test_split(labels, test_size=SplitSize(0.25), random_state=42, stratify=labels.copy())
    print("train/test sizes:", pieces[0].size, pieces[1].size)
