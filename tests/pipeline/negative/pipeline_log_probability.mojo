from moxie.pipeline import Pipeline
from test_sequential import Shift
from test_pipeline_probability import ProbabilityProbe

def main() raises:
    var pipe = Pipeline(Shift(), ProbabilityProbe[False]())
    _ = pipe.predict_log_proba(1)
