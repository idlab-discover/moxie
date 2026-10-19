"""Output codes using a deterministic test component, not a production classifier."""
from moxie.multiclass import OutputCodeClassifier
from ovr_fixtures import MarginClassifier
from numojo import Shape, Item
import numojo as nm


def main() raises:
    var X = nm.zeros[DType.float64](Shape(6, 2))
    var y = nm.zeros[DType.int64](Shape(6))
    for i in range(6):
        X[Item(i, 0)] = Float64(i)
        X[Item(i, 1)] = Float64(i % 2)
        y[Item(i)] = Int64(i // 2)
    var model = OutputCodeClassifier(MarginClassifier(), code_size=2, random_state=42)
    model.fit(X, y)
    print("classes_:", model.classes_)
    print("code_book_:", model.code_book_)
    print("predict:", model.predict(X))
