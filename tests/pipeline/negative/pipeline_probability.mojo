from moxie.pipeline import Pipeline
from test_sequential import Shift, Recorder

def main() raises:
    var pipeline = Pipeline(Shift(), Recorder())
    _ = pipeline.predict_proba(1)
