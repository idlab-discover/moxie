from moxie.multiclass import OneVsOneClassifier
from ovr_fixtures import MarginClassifier
from test_ovr import matrix, labels


def main() raises:
    var m = OneVsOneClassifier(MarginClassifier())
    m.partial_fit(matrix([[1.0], [2.0]]), labels([0, 1]), labels([0, 1]))
