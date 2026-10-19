from moxie.multiclass import OneVsOneClassifier
from ovr_fixtures import MarginClassifier


# Must fail: features are Float64 but the wrapper is declared for Float32.
def main() raises:
    _ = OneVsOneClassifier[MarginClassifier, dtype=DType.float32](MarginClassifier())
