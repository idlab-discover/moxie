"""Fixture only: a producer from an independent contribution."""
from numojo import NDArray, Shape, zeros
from moxie.feature_extraction._mapping import OrderedVocabulary
from moxie.feature_extraction._validation import check_feature_matrix


def produce[dtype: DType]() raises -> NDArray[dtype]:
    var columns = OrderedVocabulary[String](["left", "right"])
    var x = zeros[dtype](Shape(1, columns.size()))
    x.unsafe_set(columns.column("left"), 3)
    x.unsafe_set(columns.column("right"), 4)
    check_feature_matrix(x, 1, 2)
    return x^
