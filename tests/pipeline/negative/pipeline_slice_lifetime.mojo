from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import Shape, zeros


def main() raises:
    var pipe = make_pipeline(StandardScaler(), StandardScaler())
    var view = pipe.slice[stop=1]()
    var moved = pipe^
    _ = view.transform(zeros[DType.float64](Shape(2, 1)))
    print(len(moved))
