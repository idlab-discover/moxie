"""Impute a dense matrix and append the fitted missing indicators."""
import numojo as nm
from moxie.impute import SimpleImputer, MissingIndicator


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(3, 2))
    X.unsafe_set(0, 1)
    X.unsafe_set(1, 2)
    X.unsafe_set(2, Float64("nan"))
    X.unsafe_set(3, 3)
    X.unsafe_set(4, 7)
    X.unsafe_set(5, 6)
    var imputer = SimpleImputer(add_indicator=True)
    imputer.fit_named(X, [String("height"), String("width")])
    print(imputer.transform(X))
    print(imputer.get_feature_names_out())
    var indicator = MissingIndicator(sparse=False)
    print(indicator.fit_transform(X))
