from moxie.frozen import FrozenEstimator
from moxie.pipeline import make_pipeline
from moxie.preprocessing import MinMaxScaler, StandardScaler


def main() raises:
    var frozen = FrozenEstimator(make_pipeline(StandardScaler(), MinMaxScaler()))
    _ = frozen.slice[stride=2]()
