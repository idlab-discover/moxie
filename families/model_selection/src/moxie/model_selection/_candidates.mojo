"""Typed parameter candidates, independently owned constructor configuration."""
from moxie._random import RandomState

trait ParameterCandidates(Movable):
    comptime Params: Copyable
    def next(mut self) raises -> Optional[Self.Params]:
        ...

trait ParameterDistribution(Movable):
    comptime Value: Copyable & Deinitable
    def sample(self, mut random_state: RandomState) raises -> Self.Value:
        ...

@fieldwise_init
struct Candidate[Params: Copyable & Deinitable](Copyable, Movable):
    var params: Self.Params
    var ordinal: Int

    def validate(self) raises:
        if self.ordinal < 0:
            raise Error("ValueError: candidate ordinal must be nonnegative")
