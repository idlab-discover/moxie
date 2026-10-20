"""Validate a dense bootstrap plan and draw member seeds; no forest is trained by this example."""
from moxie.ensemble._records import SamplingPlan
from moxie.ensemble._plumbing import validate_sampling, additional_estimators, next_estimator_seed
from moxie._random import RandomState

def main() raises:
    var plan = SamplingPlan(100, 4, 100, 4, True, False)
    validate_sampling(plan)
    print("Additional members for a warm fit:", additional_estimators(20, 10, True))
    var rng = RandomState(42)
    print("First member seed:", next_estimator_seed(rng))
    print("See examples/forest_engine.mojo for the forest engine built on these checks.")
