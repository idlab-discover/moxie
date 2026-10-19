from moxie.multiclass import OutputCodeClassifier
from moxie._random import RandomState
from ovr_fixtures import MarginClassifier


def main() raises:
    var rng = RandomState(42)
    _ = OutputCodeClassifier(MarginClassifier(), random_state=rng)
