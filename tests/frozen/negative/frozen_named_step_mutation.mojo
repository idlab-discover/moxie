from moxie.frozen import FrozenEstimator
from moxie.pipeline import make_pipeline
from moxie.preprocessing import MinMaxScaler, StandardScaler
from numojo import Shape, zeros


def main() raises:
    var frozen = FrozenEstimator(make_pipeline(StandardScaler(), MinMaxScaler()))
    _ = frozen.named_step[StandardScaler[]]("standardscaler").fit(
        zeros[DType.float64](Shape(2, 1))
    )
