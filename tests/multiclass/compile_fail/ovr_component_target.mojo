from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import MarginClassifier


# Must fail: features are Float64 but the wrapper is declared for Float32.
def main() raises:
    _ = OneVsRestClassifier[MarginClassifier, dtype=DType.float32](MarginClassifier())
