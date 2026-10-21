"""Dense nearest/shrunken centroids, following sklearn's BSD-3-Clause reference.

See LICENSES/sklearn.txt and docs/estimators.md for native adaptations.
"""
import numojo as nm
from std.math import abs, sqrt, isfinite
# Numax fdlibm exp/log (within one ulp): Mojo 1.0 std.math exp/log are ~1e-10 relative in Float64.
from numax.core.libm import exp, log
from moxie.base import ClassifierMixin
from moxie._dense import sum_axis2d
from moxie.neighbors._classification_targets import ClassificationTargets, LabelOps
from moxie.neighbors._contracts import validate_matrix
from moxie.neighbors._distance import at, Metric
from moxie.neighbors._supervised import target_snapshot, target_at, class_arrays, prediction_score


@fieldwise_init
struct NearestCentroidParams(Copyable, Movable):
    var metric: String
    var shrink_threshold: Optional[Float64]
    var priors: String
    var prior_probabilities: Optional[nm.NDArray[DType.float64]]


struct NearestCentroid[dtype: DType = DType.float64, Y: Copyable & Deinitable = nm.NDArray[dtype]](ClassifierMixin):
    comptime Params = NearestCentroidParams
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Y
    comptime Ops = LabelOps[Self.Y]
    comptime Prediction = Self.Y
    comptime SampleWeight = nm.NDArray[Self.dtype]
    var _params: Self.Params
    var n_features_in_: Int
    var classes_: Self.Y
    var centroids_: nm.NDArray[DType.float64]
    var deviations_: nm.NDArray[DType.float64]
    var within_class_std_dev_: nm.NDArray[DType.float64]
    var class_prior_: nm.NDArray[DType.float64]

    def __init__(out self, metric: String = "euclidean", *,
                 shrink_threshold: Optional[Float64] = None, priors: String = "uniform") raises:
        self._params = Self.Params(metric, shrink_threshold, priors, None)
        self.n_features_in_ = 0
        self.classes_ = Self.Ops.allocate(0,1,False)
        self.centroids_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.deviations_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.within_class_std_dev_ = nm.zeros[DType.float64](nm.Shape(0))
        self.class_prior_ = nm.zeros[DType.float64](nm.Shape(0))

    def __init__(out self, metric: String = "euclidean", *,
                 shrink_threshold: Optional[Float64] = None,
                 priors: nm.NDArray[DType.float64]) raises:
        self = Self(metric, shrink_threshold=shrink_threshold)
        self._params.priors = "explicit"
        self._params.prior_probabilities = priors.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, metric: Optional[String] = None,
                                      shrink_threshold: Optional[Float64] = None,
                                      priors: Optional[String] = None) raises -> ref[self] Self:
        """Update only the supplied parameters; a string priors value clears explicit probabilities."""
        if metric: self._params.metric = metric.value()
        if shrink_threshold: self._params.shrink_threshold = shrink_threshold
        if priors:
            self._params.priors = priors.value()
            self._params.prior_probabilities = None
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, priors: nm.NDArray[DType.float64]) raises -> ref[self] Self:
        """Explicit class prior probabilities, the array form of sklearn's priors parameter."""
        self._params.priors = "explicit"
        self._params.prior_probabilities = priors.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        ref p = self._params
        if p.metric != "euclidean" and p.metric != "manhattan":
            raise Error("ValueError: metric must be euclidean or manhattan")
        if p.shrink_threshold:
            if not isfinite(p.shrink_threshold.value()) or p.shrink_threshold.value() <= 0:
                raise Error("ValueError: shrink_threshold must be positive and finite")
        validate_matrix(X, training=True)
        var target = ClassificationTargets(y,X.shape[0])
        if target.outputs_2d:
            raise Error("ValueError: NearestCentroid requires a single target column")
        var classes = target.classes[0].copy()
        var count = Self.Ops.rows(classes)
        if count < 2:
            raise Error("ValueError: the number of classes must exceed one")
        var n = X.shape[0]
        var d = X.shape[1]
        var codes = List[Int]()
        var counts = List[Int]()
        for _ in range(count): counts.append(0)
        for i in range(n):
            var c = Int(target.codes.unsafe_get(i))
            codes.append(c)
            counts[c] += 1
        var prior = nm.full[DType.float64](nm.Shape(count), Float64(1) / Float64(count))
        if p.priors == "empirical":
            for c in range(count): prior.unsafe_set(c, Float64(counts[c]) / Float64(n))
        elif p.priors == "explicit":
            if not p.prior_probabilities:
                raise Error("ValueError: explicit priors require prior_probabilities")
            ref values = p.prior_probabilities.value()
            if values.ndim != 1 or values.shape[0] != count:
                raise Error("ValueError: priors must have one entry per class")
            var total = Float64(0)
            for c in range(count):
                var value = values.unsafe_get(c * Int(values.strides[0]))
                if not isfinite(value) or value < 0:
                    raise Error("ValueError: priors must be finite and non-negative")
                prior.unsafe_set(c, value)
                total += value
            if not isfinite(total) or total <= 0:
                raise Error("ValueError: priors must have a positive finite total")
            if abs(total - 1) > 1e-8 + 1e-5:
                for c in range(count): prior.unsafe_set(c, prior.unsafe_get(c) / total)
        elif p.priors != "uniform":
            raise Error("ValueError: priors must be uniform, empirical or an explicit vector")
        var centers = nm.zeros[DType.float64](nm.Shape(count, d))
        for c in range(count):
            var members = nm.zeros[Self.dtype](nm.Shape(counts[c], d))
            var row = 0
            for i in range(n):
                if codes[i] != c: continue
                for j in range(d): members.unsafe_set(row*d+j, at(X, i, j).cast[Self.dtype]())
                row += 1
            if p.metric == "euclidean":
                var means = nm.mean[returned_dtype=Self.dtype](members, axis=0)
                for j in range(d):
                    var mean = means.unsafe_get(j)
                    centers.unsafe_set(c*d+j, Float64(mean))
            else:
                for j in range(d):
                    var column = nm.zeros[DType.float64](nm.Shape(counts[c]))
                    for i in range(counts[c]): column.unsafe_set(i, at(members, i, j))
                    centers.unsafe_set(c*d+j, nm.median(column))
        var nonconstant = False
        var residuals = nm.zeros[DType.float64](nm.Shape(n, d))
        for i in range(n):
            for j in range(d):
                var value = at(X, i, j)
                if value != at(X, 0, j): nonconstant = True
                var delta = value - at(centers, codes[i], j)
                residuals.unsafe_set(i*d+j, delta*delta)
        if not nonconstant:
            raise Error("ValueError: All features have zero variance. Division by zero.")
        var pooled = sum_axis2d(residuals, 0)
        for j in range(d): pooled.unsafe_set(j, sqrt(pooled.unsafe_get(j) / Float64(n-count)))
        # n == count produces NaN pooled deviations in sklearn; preserve them.
        var stabilizer = Float64("nan")
        if n > count: stabilizer = nm.median(pooled)
        var compact = X.contiguous()
        var overall = nm.mean[returned_dtype=Self.dtype](compact, axis=0)
        var deviations = nm.zeros[DType.float64](nm.Shape(count, d))
        for c in range(count):
            var scale = sqrt(Float64(1) / Float64(counts[c]) - Float64(1) / Float64(n))
            for j in range(d):
                var denominator = scale * (pooled.unsafe_get(j) + stabilizer)
                var center = Float64(overall.unsafe_get(j))
                var deviation = (at(centers, c, j) - center) / denominator
                if p.shrink_threshold:
                    var magnitude = abs(deviation) - p.shrink_threshold.value()
                    if magnitude < 0: magnitude = 0
                    if deviation < 0: magnitude = -magnitude
                    deviation = magnitude
                    centers.unsafe_set(c*d+j, center + denominator*deviation)
                deviations.unsafe_set(c*d+j, deviation)
        # Commit only after validation and all numerical work succeeds.
        self.classes_ = classes^
        self.centroids_ = centers^
        self.deviations_ = deviations^
        self.within_class_std_dev_ = pooled^
        self.class_prior_ = prior^
        self.n_features_in_ = d
        return self

    def _scores(self, X: Self.Input, standardized: Bool) raises -> nm.NDArray[DType.float64]:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: NearestCentroid has not been fitted")
        validate_matrix(X, expected_columns=self.n_features_in_)
        var compact = X.contiguous()
        var query = compact.astype[DType.float64]()
        var centers = self.centroids_.copy()
        if standardized:
            for j in range(self.n_features_in_):
                var scale = self.within_class_std_dev_.unsafe_get(j)
                if scale == 0: continue
                for i in range(X.shape[0]): query.unsafe_set(i*self.n_features_in_+j, at(query, i, j) / scale)
                for c in range(Self.Ops.rows(self.classes_)): centers.unsafe_set(c*self.n_features_in_+j, at(centers, c, j) / scale)
        validate_matrix(query)
        validate_matrix(centers)
        var metric = Metric[DType.float64](self._params.metric)
        var count = Self.Ops.rows(self.classes_)
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0], count))
        for i in range(X.shape[0]):
            for c in range(count):
                var distance = metric.distance(query, i, centers, c)
                var score = -distance
                if standardized: score = -distance*distance + 2*log(self.class_prior_.unsafe_get(c))
                result.unsafe_set(i*count+c, score)
        return result^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: NearestCentroid has not been fitted")
        var uniform = True
        var expected = Float64(1) / Float64(Self.Ops.rows(self.classes_))
        for c in range(Self.Ops.rows(self.classes_)):
            if abs(self.class_prior_.unsafe_get(c) - expected) > 1e-8 + 1e-5*expected:
                uniform = False
        var scores = self._scores(X, not uniform)
        var result = Self.Ops.allocate(X.shape[0],1,False)
        for i in range(X.shape[0]):
            var best = 0
            for c in range(1, Self.Ops.rows(self.classes_)):
                if at(scores, i, c) > at(scores, i, best): best = c
            Self.Ops.copy_value(result,i,0,self.classes_,best,0)
        return result^

    def _check_euclidean(self) raises:
        if self._params.metric != "euclidean":
            raise Error("AttributeError: decision/probability methods require metric=euclidean")

    def decision_function(self, X: Self.Input) raises -> nm.NDArray[DType.float64]:
        self._check_euclidean()
        var scores = self._scores(X, True)
        if Self.Ops.rows(self.classes_) != 2: return scores^
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]): result.unsafe_set(i, at(scores, i, 1) - at(scores, i, 0))
        return result^

    def predict_log_proba(self, X: Self.Input) raises -> nm.NDArray[DType.float64]:
        self._check_euclidean()
        var result = self._scores(X, True)
        var count = Self.Ops.rows(self.classes_)
        for i in range(X.shape[0]):
            var maximum = at(result, i, 0)
            for c in range(1, count):
                var value = at(result, i, c)
                if value > maximum: maximum = value
            var total = Float64(0)
            for c in range(count): total += exp(at(result, i, c) - maximum)
            var normalizer = log(total)
            for c in range(count): result.unsafe_set(i*count+c, at(result, i, c) - maximum - normalizer)
        return result^

    def predict_proba(self, X: Self.Input) raises -> nm.NDArray[DType.float64]:
        var result = self.predict_log_proba(X)
        for i in range(result.size): result.unsafe_set(i, exp(result.unsafe_get(i)))
        return result^

    def score(self, X: Self.Input, y: Self.Target,
              sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var pred = self.predict(X)
        return ClassificationTargets[Self.Y].score(pred,y,sample_weight)
