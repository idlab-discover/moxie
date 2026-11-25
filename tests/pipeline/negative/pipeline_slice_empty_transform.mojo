from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import Shape, zeros


def main() raises:
    var pipe = make_pipeline(StandardScaler(), StandardScaler())
    _ = pipe.slice[stop=0]().transform(zeros[DType.float64](Shape(2, 1)))
