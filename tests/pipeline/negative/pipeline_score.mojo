from moxie.pipeline import Pipeline
from test_sequential import Shift, Recorder

def main() raises:
    var pipe = Pipeline(Shift(), Recorder())
    _ = pipe.score(1, 1)
