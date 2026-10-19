"""A small weighted centroid classifier demonstrating the calibration adapter."""
import numojo as nm
from moxie.calibration import CalibrationEstimator, CalibrationDisplayEstimator
from moxie._calibration_algorithms import Array, matrix, vector, weights
from moxie._calibration_cv import unique, encoded


@fieldwise_init
struct CentroidParams(Copyable, Movable):
    var probability_response: Bool


struct Centroid(CalibrationDisplayEstimator, CalibrationEstimator):
    comptime Params = CentroidParams
    var probability_response: Bool
    var frozen: Bool
    var classes_: Array
    var centers_: Array

    def __init__(out self, probability_response: Bool = False) raises:
        self.probability_response = probability_response
        self.frozen = False
        self.classes_ = nm.zeros[DType.float64](nm.Shape(0))
        self.centers_ = nm.zeros[DType.float64](nm.Shape(0, 0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.probability_response)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.probability_response = params.probability_response
        return self

    def __sklearn_clone__(self) raises -> Self:
        if self.frozen:
            return self.copy()
        return Self(self.probability_response)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.classes_.size > 0

    def is_frozen(self) -> Bool:
        return self.frozen

    def uses_probabilities(self) -> Bool:
        return self.probability_response

    def class_labels(self) raises -> Array:
        return self.classes_.copy()

    def fit(
        mut self, X: Array, y: Array, sample_weight: Optional[Array] = None
    ) raises:
        if self.frozen:
            if not self.__sklearn_is_fitted__():
                raise Error("NotFittedError: frozen centroid is unfitted")
            return
        matrix(X)
        vector(y)
        if X.shape[0] != y.size:
            raise Error("ValueError: inconsistent sample count")
        var classes = unique(y)
        var labels = encoded(y, classes)
        var w = weights(y.size, sample_weight)
        var centers = nm.zeros[DType.float64](
            nm.Shape(classes.size, X.shape[1])
        )
        var totals = nm.zeros[DType.float64](nm.Shape(classes.size))
        for i in range(y.size):
            var j = Int(labels.item(i))
            totals[nm.Item(j)] = totals.item(j) + w.item(i)
            for k in range(X.shape[1]):
                centers[nm.Item(j, k)] = centers.item(j, k) + w.item(
                    i
                ) * X.item(i, k)
        for j in range(classes.size):
            if totals.item(j) <= 0:
                raise Error("ValueError: class has no positive weight")
            for k in range(X.shape[1]):
                centers[nm.Item(j, k)] = centers.item(j, k) / totals.item(j)
        self.classes_ = classes^
        self.centers_ = centers^

    def calibration_probabilities(self, X: Array) raises -> Array:
        var classifier = self.copy()
        classifier.probability_response = True
        return classifier.response(X)

    def display_name(self) -> String:
        return "Centroid"

    def response(self, X: Array) raises -> Array:
        from std.math import exp

        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: centroid is unfitted")
        matrix(X)
        if X.shape[1] != self.centers_.shape[1]:
            raise Error("ValueError: inconsistent feature count")
        var c = self.classes_.size
        var values = nm.zeros[DType.float64](nm.Shape(X.shape[0], c))
        for i in range(X.shape[0]):
            for j in range(c):
                var distance = Float64(0)
                for k in range(X.shape[1]):
                    var delta = X.item(i, k) - self.centers_.item(j, k)
                    distance += delta * delta
                values[nm.Item(i, j)] = -distance
        if self.probability_response:
            for i in range(X.shape[0]):
                var peak = values.item(i, 0)
                for j in range(1, c):
                    peak = max(peak, values.item(i, j))
                var total = Float64(0)
                for j in range(c):
                    values[nm.Item(i, j)] = exp(values.item(i, j) - peak)
                    total += values.item(i, j)
                for j in range(c):
                    values[nm.Item(i, j)] = values.item(i, j) / total
            return values^
        if c == 2:
            var binary = nm.zeros[DType.float64](nm.Shape(X.shape[0], 1))
            for i in range(X.shape[0]):
                binary[nm.Item(i, 0)] = values.item(i, 1) - values.item(i, 0)
            return binary^
        return values^
