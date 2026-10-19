import numojo as nm
from numojo import Shape
from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import ScoreRegressor


def main() raises:
    var model = OneVsRestClassifier(ScoreRegressor())
    _ = model.predict_proba(nm.zeros[DType.float64](Shape(1, 2)))
