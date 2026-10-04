"""A fluent result cannot turn an immutable receiver into a mutable one."""
from moxie.base import BaseEstimator

def configure[E: BaseEstimator](model: E, params: E.Params) raises:
    _ = model.set_params(params)

def main():
    pass
