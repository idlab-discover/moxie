from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler


def main() raises:
    var pipe = make_pipeline(StandardScaler(), StandardScaler())
    _ = pipe.slice[stride=2]()
