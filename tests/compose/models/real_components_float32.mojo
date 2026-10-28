"""Float32 real-component contracts; assertions are shared with Float64."""
from real_components_case import run


def main() raises:
    run[DType.float32]()
