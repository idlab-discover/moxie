"""Check automatic ColumnTransformer names for aliased estimators."""
from std.testing import assert_equal
from moxie.compose import ColumnEntry, make_column_transformer
from moxie.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD


def main() raises:
    var columns = make_column_transformer(
        ColumnEntry(PLSRegression(1), [0]),
        ColumnEntry(CCA(1), [1]),
        ColumnEntry(PLSCanonical(1), [2]),
        ColumnEntry(PLSSVD(1), [3]),
        ColumnEntry(PLSRegression(1), [4]),
    )
    var names = List[String]()
    columns.transformers.append_names(names)
    var expected: List[String] = ["plsregression-1", "cca", "plscanonical", "plssvd", "plsregression-2"]
    assert_equal(len(names), len(expected))
    for i in range(len(expected)):
        assert_equal(names[i], expected[i])
    print("PASS: aliased ColumnTransformer names and duplicate suffixes")
