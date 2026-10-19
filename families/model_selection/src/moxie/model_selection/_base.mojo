"""Interfaces for the model_selection family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import BaseEstimator, Predictor


trait SearchEstimator(BaseEstimator):
    pass


trait PredictiveSearch(Predictor, SearchEstimator):
    pass


trait CrossValidator(Movable):
    comptime Input: Movable
    comptime Target: Movable
    comptime Groups: Movable
    comptime Splits: Movable

    def split(
        self,
        X: Self.Input,
        y: Optional[Self.Target] = None,
        groups: Optional[Self.Groups] = None,
    ) raises -> Self.Splits:
        ...

    def get_n_splits(
        self,
        X: Optional[Self.Input] = None,
        y: Optional[Self.Target] = None,
        groups: Optional[Self.Groups] = None,
    ) raises -> Int:
        ...
