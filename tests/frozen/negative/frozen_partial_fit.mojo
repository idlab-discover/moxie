from moxie.frozen import FrozenEstimator
from dummy_regressor import DummyRegressor


def main() raises:
    var frozen = FrozenEstimator(DummyRegressor())
    _ = frozen.partial_fit(List[Float64](), List[Float64]())
