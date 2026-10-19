from moxie.multiclass import OutputCodeClassifier
from ovr_fixtures import MarginClassifier


# Must fail: features are Float64 but the wrapper is declared for Float32.
def main() raises:
    _ = OutputCodeClassifier[MarginClassifier, dtype=DType.float32](MarginClassifier())
