"""Fixture only: borrowed input from another contribution, no adapter copies."""
from numojo import NDArray
from moxie._dense import row_squared_norms
from moxie.feature_extraction._mapping import OrderedVocabulary
from moxie.feature_extraction._validation import check_feature_matrix
from std.testing import assert_equal


def consume[dtype: DType](mut x: NDArray[dtype]) raises:
    var columns = OrderedVocabulary[String](["left", "right"])
    check_feature_matrix(x, 1, columns.size())
    var norms = row_squared_norms(x)
    assert_equal(norms.unsafe_get(0), 25)
    var shared = x.view()
    shared.unsafe_set(columns.column("right"), 9)
