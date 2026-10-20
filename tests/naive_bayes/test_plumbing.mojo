import numojo as nm
from std.math import isnan, log
from std.testing import assert_equal, assert_true, assert_almost_equal, assert_raises
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import (
    validate_matrix, owned_snapshot, prepare_batch, CountStatistics,
    GaussianStatistics, CategoryStatistics, DenseLikelihood, CountUpdater,
    DenseMatrix, Labels, BatchContract, FeatureAlpha, CategoryMinimum,
    GaussianUpdater, CategoryUpdater, DiscreteFinalizer, DenseNBPredictor,
    ClassDomain, NumericDomain, StringDomain, int_classes,
    joint_argmax, predict_labels, log_normalize, exp_normalize,
    weighted_accuracy,
)
from moxie._dense import sum_axis2d

struct DeferredLikelihood(DenseLikelihood):
    def __init__(out self):
        pass

    def joint_log_likelihood(self, X: DenseMatrix) raises -> DenseMatrix:
        raise Error("Deferred child likelihood")

struct DeferredCountUpdater(CountUpdater):
    def __init__(out self):
        pass

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: CountStatistics) raises:
        raise Error("Deferred child count engine")


struct DeferredEngines(GaussianUpdater, CategoryUpdater, DiscreteFinalizer):
    def __init__(out self):
        pass

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: GaussianStatistics) raises:
        raise Error("Deferred Gaussian engine")

    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   minimum: CategoryMinimum, mut statistics: CategoryStatistics) raises:
        raise Error("Deferred categorical engine")

    def finalize(self, statistics: CountStatistics, alpha: FeatureAlpha,
                 force_alpha: Bool) raises -> DenseMatrix:
        raise Error("Deferred discrete finalizer")


def check_dtype[dtype: DType]() raises:
    var X = nm.ones[dtype](nm.Shape(2, 3))
    var view = X.view()
    var owned = owned_snapshot(view)
    view.unsafe_set(0, 7)
    assert_equal(X.item(0, 0), 7)
    assert_equal(owned.item(0, 0), 1)
    var backwards = X.view_with_layout(nm.Shape(2, 2), nm.NDArrayStrides(3, -1), 2)
    assert_equal(validate_matrix(backwards), 2)
    var saved_backwards = owned_snapshot(backwards)
    backwards[nm.Item(0, 0)] = 11
    assert_equal(X.item(0, 2), 11)
    assert_equal(saved_backwards.item(0, 0), 1)
    var labels: List[Int] = [9, -2]
    var current = List[Int]()
    var supplied: List[Int] = [9, -2, 9]
    var weights: List[Float64] = [2, 1]
    var labeled = prepare_batch(X, labels, current, supplied.copy(), weights.copy())
    var fitted = int_classes(labeled.domain)
    var batch = labeled.batch.copy()
    assert_equal(fitted[0], -2)
    assert_equal(fitted[1], 9)
    supplied[0] = 88
    weights[0] = 44
    assert_equal(batch.weights[0], 2)
    assert_equal(int_classes(labeled.domain)[1], 9)
    assert_equal(batch.n_features, 3)
    assert_equal(batch.n_classes, 2)
    assert_equal(batch.targets.item(0), 1)
    assert_equal(batch.targets.item(1), 0)
    var totals = sum_axis2d(X, 0)
    assert_equal(totals.item(0), 8)
    with assert_raises():
        _ = prepare_batch(X, labels, current)
    with assert_raises():
        _ = validate_matrix(X, expected=2)
    with assert_raises():
        _ = prepare_batch(X, labels, fitted, supplied.copy())
    var unknown: List[Int] = [0, 1]
    with assert_raises():
        _ = prepare_batch(X, unknown, fitted)
    var zero: List[Float64] = [0, 0]
    with assert_raises():
        _ = prepare_batch(X, labels, fitted, sample_weight=zero^)
    X.unsafe_set(0, -1)
    with assert_raises():
        _ = validate_matrix(X, nonnegative=True)
    var vector = nm.ones[dtype](nm.Shape(3))
    with assert_raises():
        _ = validate_matrix(vector)
    var empty = nm.zeros[dtype](nm.Shape(0, 3))
    with assert_raises():
        _ = validate_matrix(empty)
    X.unsafe_set(0, Scalar[dtype](Float64("nan")))
    with assert_raises():
        _ = validate_matrix(X)


struct FixedJoint[D: ClassDomain](DenseNBPredictor, Movable):
    """Test double: returns a stored joint matrix, not an estimator."""
    comptime Domain = Self.D
    var domain: Self.D
    var joint: DenseMatrix

    def __init__(out self, var domain: Self.D, var joint: DenseMatrix):
        self.domain = domain^
        self.joint = joint^

    def _class_domain(self) raises -> Self.D:
        return self.domain.copy()

    def predict_joint_log_proba[dtype: DType](
        self, X: nm.NDArray[dtype]
    ) raises -> DenseMatrix:
        _ = validate_matrix(X)
        return self.joint.copy()


def matrix(rows: List[List[Float64]]) raises -> DenseMatrix:
    var result = nm.zeros[DType.float64](nm.Shape(len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            result[nm.Item(i, j)] = rows[i][j]
    return result^


def vector[dtype: DType](values: List[Float64]) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](nm.Shape(len(values)))
    for i in range(len(values)):
        result[nm.Item(i)] = Scalar[dtype](values[i])
    return result^


def strings(values: List[String]) -> StringTarget:
    return StringTarget(values)


def check_labels() raises:
    var X = nm.ones[DType.float64](nm.Shape(3, 1))
    # String classes sort lexicographically, as np.unique does.
    var y = strings(["b", "a", "b"])
    var labeled = prepare_batch(X, y, classes=strings(["b", "a", "c"]))
    assert_equal(labeled.domain.n_classes(), 3)
    assert_equal(labeled.domain.classes()[0], "a")
    assert_equal(labeled.domain.classes()[2], "c")
    assert_equal(labeled.batch.targets.item(0), 1)
    assert_equal(labeled.batch.targets.item(1), 0)
    var later = prepare_batch(X, y, labeled.domain.copy())
    assert_equal(later.batch.targets.item(2), 1)
    with assert_raises(contains="unseen"):
        _ = prepare_batch(X, strings(["a", "z", "a"]), labeled.domain.copy())
    with assert_raises(contains="differ"):
        _ = prepare_batch(X, y, labeled.domain.copy(), strings(["a", "b"]))
    with assert_raises(contains="first call"):
        _ = prepare_batch(X, y)
    var fit = prepare_batch(X, y, classes=y.copy())
    assert_equal(fit.domain.n_classes(), 2)
    with assert_raises(contains="1d"):
        _ = prepare_batch(X, StringTarget(["a", "b", "a"], 3, 1), classes=strings(["a", "b"]))
    # Integral floating targets are discrete; others are continuous.
    var integral = vector[DType.float64]([2.0, -1.0, 2.0])
    var floats = prepare_batch(X, integral, classes=integral.copy())
    assert_equal(floats.domain.classes().item(0), -1.0)
    var continuous = vector[DType.float64]([0.5, 1.0, 0.5])
    with assert_raises(contains="continuous"):
        _ = prepare_batch(X, continuous, classes=continuous.copy())
    var missing = vector[DType.float32]([0.0, Float64("nan"), 0.0])
    with assert_raises(contains="finite"):
        _ = prepare_batch(X, missing, classes=missing.copy())
    var bad_classes = vector[DType.float64]([0.0, 1.5])
    with assert_raises(contains="continuous"):
        _ = prepare_batch(X, integral, classes=bad_classes^)
    var ints = vector[DType.int64]([7.0, -3.0, 7.0])
    var int_batch = prepare_batch(X, ints, classes=ints.copy())
    assert_equal(int_classes(int_batch.domain)[0], -3)
    with assert_raises(contains="sample counts"):
        _ = prepare_batch(X, vector[DType.int64]([1.0, 2.0]), classes=ints.copy())


def check_prediction() raises:
    var inf = Float64("inf")
    var nan = Float64("nan")
    # Expected values are scipy 1.18.1 logsumexp / numpy argmax outputs.
    var joint = matrix([[0.0, -inf, 2.0], [-inf, -inf, -inf], [inf, 0.0, 1.0],
                        [nan, 0.0, 1.0], [1.0, 1.0, nan], [3.0, 3.0, 1.0]])
    var index = joint_argmax(joint)
    var expected_index: List[Int] = [2, 0, 0, 0, 2, 0]
    for i in range(len(expected_index)):
        assert_equal(Int(index.item(i)), expected_index[i])
    var logp = log_normalize(joint)
    var lse = 2.0 + log(1.0 + Float64(0.1353352832366127))
    assert_almost_equal(logp.item(0, 0), -lse, atol=1e-15)
    assert_equal(logp.item(0, 1), -inf)
    assert_almost_equal(logp.item(0, 2), 2.0 - lse, atol=1e-15)
    for c in range(3):
        assert_true(isnan(logp.item(1, c)))
        assert_true(isnan(logp.item(3, c)))
        assert_true(isnan(logp.item(4, c)))
    assert_true(isnan(logp.item(2, 0)))
    assert_equal(logp.item(2, 1), -inf)
    assert_equal(logp.item(2, 2), -inf)
    var proba = exp_normalize(joint)
    assert_almost_equal(proba.item(5, 0) + proba.item(5, 1) + proba.item(5, 2), 1.0, atol=1e-15)
    assert_equal(proba.item(0, 1), 0.0)
    assert_true(isnan(proba.item(1, 0)))
    var domain = StringDomain.from_classes(strings(["c", "a", "b"]))
    var labels = predict_labels(domain, joint)
    assert_equal(labels.get(0), "c")
    assert_equal(labels.get(5), "a")
    var two = StringDomain.from_classes(strings(["a", "b"]))
    with assert_raises(contains="columns"):
        _ = predict_labels(two, joint)
    # "z" is outside classes_ and counts as wrong, as in accuracy_score.
    var truth = strings(["c", "a", "b", "a", "c", "z"])
    assert_almost_equal(weighted_accuracy[StringDomain](truth, labels), 4.0 / 6.0, atol=1e-15)
    var weights: List[Float64] = [1, 1, 4, 1, 1, 2]
    assert_almost_equal(weighted_accuracy[StringDomain](truth, labels, weights^), 4.0 / 10.0, atol=1e-15)
    with assert_raises(contains="sample counts"):
        _ = weighted_accuracy[StringDomain](strings(["a", "b"]), labels)
    var model = FixedJoint[StringDomain](domain.copy(), joint.copy())
    var X = nm.ones[DType.float32](nm.Shape(6, 2))
    assert_equal(model.predict(X).get(0), "c")
    assert_equal(model.predict_log_proba(X).item(0, 1), -inf)
    assert_almost_equal(model.predict_proba(X).item(0, 2), proba.item(0, 2), atol=0)
    assert_almost_equal(model.score(X, truth), 4.0 / 6.0, atol=1e-15)
    var numeric = NumericDomain[DType.int64].from_classes(vector[DType.int64]([4.0, 1.0, 9.0]))
    var numeric_model = FixedJoint[NumericDomain[DType.int64]](numeric^, joint.copy())
    assert_equal(numeric_model.predict(X).item(0), 9)
    var narrow = FixedJoint[StringDomain](two.copy(), joint.copy())
    with assert_raises(contains="columns"):
        _ = narrow.predict_proba(X)
    var flat = nm.ones[DType.float64](nm.Shape(3))
    with assert_raises(contains="matrix"):
        _ = log_normalize(flat)


def main() raises:
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
    check_labels()
    check_prediction()
    var counts = CountStatistics(nm.zeros[DType.float64](nm.Shape(2)),
                                 nm.zeros[DType.float64](nm.Shape(2, 3)))
    var gaussian = GaussianStatistics(counts.class_count.copy(),
        counts.feature_count.copy(), counts.feature_count.copy(), 0.0)
    var categories = List[DenseMatrix]()
    categories.append(nm.zeros[DType.float64](nm.Shape(2, 4)))
    var sizes: List[Int] = [4]
    var categorical = CategoryStatistics(counts.class_count.copy(), categories^, sizes^)
    assert_equal(categorical.category_count[0].shape[1], 4)
    assert_equal(gaussian.mean.shape[1], 3)
    var copied = counts.copy()
    copied.feature_count.unsafe_set(0, 10)
    assert_equal(counts.feature_count.item(0, 0), 0)
    var alpha = FeatureAlpha(1.0, None)
    alpha.validate(3)
    var alpha_values = nm.ones[DType.float64](nm.Shape(3))
    var vector_alpha = FeatureAlpha(-1.0, alpha_values.copy())
    vector_alpha.validate(3)
    alpha_values.unsafe_set(0, -1)
    vector_alpha.validate(3)
    with assert_raises():
        vector_alpha.validate(2)
    var bad_alpha = FeatureAlpha(-1.0, None)
    with assert_raises():
        bad_alpha.validate(3)
    var minimum = CategoryMinimum(0, None)
    minimum.validate(3)
    var minima: List[Int] = [2, 3, 4]
    var vector_minimum = CategoryMinimum(-1, minima.copy())
    vector_minimum.validate(3)
    with assert_raises():
        vector_minimum.validate(2)
    var bad_minimum = CategoryMinimum(-1, None)
    with assert_raises():
        bad_minimum.validate(3)
    var X = nm.ones[DType.float64](nm.Shape(2, 3))
    var y: Labels = [0, 1]
    var current = Labels()
    var batch = prepare_batch(X, y, current, y.copy()).batch.copy()
    var likelihood = DeferredLikelihood()
    with assert_raises():
        _ = likelihood.joint_log_likelihood(X)
    var updater = DeferredCountUpdater()
    with assert_raises():
        updater.accumulate(X, batch, counts)
    var engines = DeferredEngines()
    with assert_raises():
        engines.accumulate(X, batch, gaussian)
    with assert_raises():
        engines.accumulate(X, batch, minimum, categorical)
    with assert_raises():
        _ = engines.finalize(counts, alpha, True)
    print("PASS: dense NB validation, ownership and shared extension contracts")
