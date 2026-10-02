"""Validate dense data and name its columns; no estimator is implemented."""
from numojo import full, Shape
from moxie.preprocessing._validation import validate_dense, require_dense_output
from moxie.preprocessing._metadata import FeatureNames

def main() raises:
    var x = full[DType.float64](Shape(3, 2), 1)
    require_dense_output(False)
    validate_dense(x)
    var names = FeatureNames()
    names.record_fit(3, 2)
    print(names.get_feature_names_out())
