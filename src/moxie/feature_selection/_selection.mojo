"""Dense selector plumbing. Behavioral reference: sklearn _base.py (BSD-3-Clause)."""
import numojo as nm
from std.utils import Variant

comptime SupportResult = Variant[List[Bool], List[Int]]

struct SelectionState(Copyable, Movable):
    """Owned metadata; algorithms commit a mask only after successful fitting."""
    var _mask: List[Bool]
    var _names: List[String]
    var _fitted: Bool

    def __init__(out self):
        self._mask = List[Bool]()
        self._names = List[String]()
        self._fitted = False

    def reset(mut self):
        self._mask = List[Bool]()
        self._names = List[String]()
        self._fitted = False

    def commit(mut self, mask: List[Bool], names: List[String] = List[String]()) raises:
        if len(mask) == 0:
            raise Error("ValueError: fitting requires at least one feature")
        if len(names) != 0 and len(names) != len(mask):
            raise Error("ValueError: feature names length mismatch")
        self._mask = mask.copy()
        self._names = names.copy()
        self._fitted = True

    def is_fitted(self) -> Bool:
        return self._fitted

    def check(self) raises:
        if not self._fitted:
            raise Error("NotFittedError: selector has no support mask")

    def n_features(self) raises -> Int:
        self.check()
        return len(self._mask)

    def get_support(self, indices: Bool = False) raises -> SupportResult:
        self.check()
        if indices:
            var result = List[Int]()
            for j in range(len(self._mask)):
                if self._mask[j]:
                    result.append(j)
            return SupportResult(result^)
        return SupportResult(self._mask.copy())

    def selected_count(self) raises -> Int:
        self.check()
        var count = 0
        for j in range(len(self._mask)):
            count += Int(self._mask[j])
        return count

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        self.check()
        if len(input_features) != 0 and len(input_features) != len(self._mask):
            raise Error("ValueError: input feature names length mismatch")
        var result = List[String]()
        for j in range(len(self._mask)):
            if len(input_features) != 0 and len(self._names) != 0:
                if input_features[j] != self._names[j]:
                    raise Error("ValueError: input feature names differ from fit")
            if self._mask[j]:
                if len(input_features) != 0:
                    result.append(input_features[j])
                elif len(self._names) != 0:
                    result.append(self._names[j])
                else:
                    result.append("x" + String(j))
        return result^

    def transform[dtype: DType](self, x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
        comptime assert dtype == DType.float32 or dtype == DType.float64
        self.check()
        if x.ndim != 2:
            raise Error("ValueError: expected rank-2 dense input")
        if x.shape[1] != len(self._mask) or x.shape[0] == 0:
            raise Error("ValueError: input shape mismatch or empty samples")
        var count = self.selected_count()
        var result = nm.zeros[dtype](nm.Shape(x.shape[0], count))
        for i in range(x.shape[0]):
            var k = 0
            for j in range(len(self._mask)):
                if self._mask[j]:
                    result.unsafe_set(i * count + k, x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
                    k += 1
        return result^

    def inverse_transform[dtype: DType](self, x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
        comptime assert dtype == DType.float32 or dtype == DType.float64
        self.check()
        if x.ndim != 2:
            raise Error("ValueError: expected rank-2 dense input")
        if x.shape[1] != self.selected_count() or x.shape[1] == 0 or x.shape[0] == 0:
            raise Error("ValueError: inverse input shape mismatch or empty input")
        var width = len(self._mask)
        var result = nm.zeros[dtype](nm.Shape(x.shape[0], width))
        for i in range(x.shape[0]):
            var k = 0
            for j in range(width):
                if self._mask[j]:
                    result.unsafe_set(i * width + j, x.unsafe_get(i * Int(x.strides[0]) + k * Int(x.strides[1])))
                    k += 1
        return result^
