"""Dense split index ownership; no splitting algorithm."""
from numojo import NDArray, Item

comptime IndexArray = NDArray[DType.int64]

@fieldwise_init
struct IndexSplit(Movable):
    var train: IndexArray
    var test: IndexArray

    def copy(self) raises -> Self:
        return Self(self.train.copy(), self.test.copy())

    def validate(self, n_samples: Int) raises:
        validate_indices(self.train, n_samples)
        validate_indices(self.test, n_samples)


def validate_indices(indices: IndexArray, n_samples: Int) raises:
    if n_samples < 0 or indices.ndim != 1:
        raise Error("ValueError: indices require rank one and nonnegative sample count")
    for i in range(indices.size):
        var index = Int(indices[Item(i)])
        if index < 0 or index >= n_samples:
            raise Error("ValueError: sample index out of bounds")


trait SplitCursor(Movable):
    """Streaming extension for CrossValidator.Splits; owns yielded arrays."""
    def next(mut self) raises -> Optional[IndexSplit]:
        ...
