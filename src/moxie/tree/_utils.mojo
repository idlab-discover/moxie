# scikit-learn developers, BSD-3-Clause; see family LICENSES/scikit-learn.txt.
"""Tree-local rand_r protocol. Initial seed is drawn by shared RandomState."""
from moxie._random import RandomState

struct TreeRandom(Movable):
    var state: UInt32
    def __init__(out self):
        self.state = 1
    def initialize(mut self, mut rng: RandomState) raises:
        self.state = UInt32(rng.randint(2147483647))
    def draw(mut self) -> Int:
        if self.state == 0:
            self.state = 1
        self.state ^= self.state << 13
        self.state ^= self.state >> 17
        self.state ^= self.state << 5
        return Int(self.state & 0x7fffffff)
    def randint(mut self, low: Int, high: Int) -> Int:
        return low + self.draw() % (high - low)
    def uniform(mut self, low: Float64, high: Float64) -> Float64:
        return (high - low) * Float64(self.draw()) / 2147483647.0 + low
