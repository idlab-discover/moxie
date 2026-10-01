"""Pipeline family contracts and typed sequential execution. See docs/pipeline/sequential.md."""
from ._pipeline_contracts import PipelineEstimator, PredictivePipeline, TransformingPipeline

# Public scikit-learn API scaffold.
from ._pipeline_api import (
    make_pipeline,
    make_union,
)

from ._pipeline_sequential import Pipeline, PipelineParams

from ._pipeline_union import FeatureUnion, FeatureUnionParams, UnionStep, union_step, UnionPair, UnionSentinel, UnionFeatureNames, UnionInputFeatures
