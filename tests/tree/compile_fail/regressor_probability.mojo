import numojo as nm
from moxie.tree import DecisionTreeRegressor

def main() raises:
    var estimator = DecisionTreeRegressor()
    var X = nm.zeros[DType.float32](nm.Shape(1, 1))
    _ = estimator.predict_proba(X)
