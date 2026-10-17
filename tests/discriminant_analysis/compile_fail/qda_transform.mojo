"""Must not compile: QuadraticDiscriminantAnalysis has no transform capability."""
from numojo import Shape, zeros
from moxie.discriminant_analysis import QuadraticDiscriminantAnalysis


def main() raises:
    var qda = QuadraticDiscriminantAnalysis()
    _ = qda.transform(zeros[DType.float64](Shape(4, 2)))
