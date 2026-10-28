"""Float64 real-component contracts; each dtype compiles within the memory cap."""
from real_components_case import run


def main() raises:
    run[DType.float64]()
