from moxie.multiclass import OutputCodeClassifier
from ovr_fixtures import ScoreRegressor

def main() raises:
    _ = OutputCodeClassifier(ScoreRegressor())
