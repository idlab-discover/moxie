from moxie.pipeline import Pipeline
from test_sequential import Fused, Shift


def main() raises:
    var pipe = Pipeline(Shift(), Fused[False]())
    _ = pipe.transform(1)
