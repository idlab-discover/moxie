from moxie.pipeline import FeatureUnion, union_step
from union_fixtures import TestTransformer

def main() raises:
    _ = FeatureUnion(union_step[DType.float32]("wrong", TestTransformer[DType.float64]()))
