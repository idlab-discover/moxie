from moxie._random import RandomState

def evaluate(mut state: RandomState) -> Float64:
    return state.random_sample()


struct SeededProbe(Copyable, Movable):
    """Integration-only estimator-shaped consumer, not an ML implementation."""
    var seed: Int
    var learned_: Float64
    def __init__(out self, seed: Int):
        self.seed = seed
        self.learned_ = -1
    def fit(mut self) raises:
        var state = RandomState(self.seed)
        self.learned_ = state.random_sample()
    def clone(self) -> Self:
        return Self(self.seed)
