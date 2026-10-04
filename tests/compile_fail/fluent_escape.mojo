"""A borrowed mutation result cannot escape a local estimator's scope."""
from moxie.base import BaseEstimator, clone

def escape[E: BaseEstimator, origin: MutOrigin](
    ref[origin] model: E, params: E.Params
) raises -> ref[model] E:
    var local = clone(model)
    return local.set_params(params)

def main():
    pass
