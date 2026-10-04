from moxie.frozen import FrozenEstimator
from dummy_regressor import DummyRegressor, DummyParams


def main() raises:
    var frozen = FrozenEstimator(DummyRegressor())
    var X: List[Float64] = [1.0, 2.0]
    var params = frozen.get_params()
    params.estimator().set_params(DummyParams("median"))
