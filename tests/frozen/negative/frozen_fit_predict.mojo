from moxie.frozen import FrozenEstimator
from test_family import ContractProbe


def main() raises:
    var frozen = FrozenEstimator(ContractProbe())
    _ = frozen.fit_predict(1, Optional[Int](2))
