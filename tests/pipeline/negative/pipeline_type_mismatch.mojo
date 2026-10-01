from moxie.pipeline import Pipeline
from test_sequential import Recorder
from test_real_pipeline import CopyStep

def main() raises:
    _ = Pipeline(CopyStep[DType.float64](), Recorder())
