"""Dense extension contracts and shared NuMojo prediction (sklearn _BaseNB)."""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.math import isfinite, isnan
from moxie._labels import (
    NumericEncoding, StringEncoding, StringTarget, check_classification_targets,
    encode_column,
)
from moxie._naive_bayes_common import _exp, _log, _weights

comptime DenseMatrix = NDArray[DType.float64]
comptime Labels = List[Int]
"""Integer labels used by the preserved List estimators."""


def validate_matrix[dtype: DType](
    X: NDArray[dtype], expected: Int = 0, nonnegative: Bool = False
) raises -> Int:
    comptime assert dtype in (
        DType.float32, DType.float64, DType.bool,
        DType.int8, DType.int16, DType.int32, DType.int64,
        DType.uint8, DType.uint16, DType.uint32, DType.uint64,
    ), "unsupported dense feature dtype"
    if X.ndim != 2:
        raise Error("ValueError: X must have rank two")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise Error("ValueError: X must contain samples and features")
    if expected != 0 and X.shape[1] != expected:
        raise Error("ValueError: feature count differs from fit")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var value = X.item(i, j)
            comptime if dtype == DType.float32 or dtype == DType.float64:
                if not isfinite(value):
                    raise Error("ValueError: X must be finite")
            if nonnegative and value < 0:
                raise Error("ValueError: X must be nonnegative")
    return X.shape[1]


def owned_snapshot[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    """Validate then copy, including backing storage for strided views."""
    _ = validate_matrix(X)
    return X.copy()


trait ClassDomain(Copyable, Deinitable):
    """Sorted classes_ for one target kind, adapting `moxie._labels` encodings.

    Naive Bayes is single-output: targets are rank one. Domains come from
    `unique_labels(classes)`; `encode` rejects labels outside the domain.
    """

    comptime Target: Copyable & Deinitable

    @staticmethod
    def empty_target() raises -> Self.Target:
        ...

    def class_values(self) raises -> Self.Target:
        ...

    @staticmethod
    def from_classes(classes: Self.Target) raises -> Self:
        ...

    @staticmethod
    def check_target(y: Self.Target) raises -> Int:
        """Validate a rank-one classification target; return its length."""
        ...

    @staticmethod
    def equal_at(a: Self.Target, b: Self.Target, i: Int) raises -> Bool:
        ...

    def n_classes(self) -> Int:
        ...

    def same_classes(self, other: Self) raises -> Bool:
        ...

    def encode(self, y: Self.Target) raises -> NDArray[DType.int64]:
        ...

    def decode(self, codes: NDArray[DType.int64]) raises -> Self.Target:
        ...


def _single_output(rank: Int) raises:
    if rank != 1:
        raise Error("ValueError: y should be a 1d array")


@fieldwise_init
struct NumericDomain[label: DType](ClassDomain):
    """Int64, Float32 or Float64 labels in an NDArray target."""

    comptime Target = NDArray[Self.label]
    var encoding: NumericEncoding[Self.label]

    @staticmethod
    def empty_target() raises -> Self.Target:
        return nm.zeros[Self.label](Shape(0))

    def class_values(self) raises -> Self.Target:
        return self.classes()

    @staticmethod
    def from_classes(classes: NDArray[Self.label]) raises -> Self:
        _single_output(classes.ndim)
        return Self(encode_column(classes))

    @staticmethod
    def check_target(y: NDArray[Self.label]) raises -> Int:
        _single_output(y.ndim)
        check_classification_targets(y)
        return y.shape[0]

    @staticmethod
    def equal_at(
        a: NDArray[Self.label], b: NDArray[Self.label], i: Int
    ) raises -> Bool:
        # Item uses logical coordinates; item(i) is a flat storage index.
        return a[Item(i)] == b[Item(i)]

    def classes(self) -> NDArray[Self.label]:
        return self.encoding.classes.copy()

    def n_classes(self) -> Int:
        return self.encoding.classes.size

    def same_classes(self, other: Self) raises -> Bool:
        if self.n_classes() != other.n_classes():
            return False
        for i in range(self.n_classes()):
            if self.encoding.classes.item(i) != other.encoding.classes.item(i):
                return False
        return True

    def encode(self, y: NDArray[Self.label]) raises -> NDArray[DType.int64]:
        return self.encoding.encode(y)

    def decode(self, codes: NDArray[DType.int64]) raises -> NDArray[Self.label]:
        return self.encoding.decode(codes)


@fieldwise_init
struct StringDomain(ClassDomain):
    """String labels in a rank-one `StringTarget`."""

    comptime Target = StringTarget
    var encoding: StringEncoding

    @staticmethod
    def empty_target() raises -> Self.Target:
        return StringTarget(List[String]())

    def class_values(self) raises -> Self.Target:
        return StringTarget(self.encoding.classes)

    @staticmethod
    def from_classes(classes: StringTarget) raises -> Self:
        _single_output(classes.rank())
        return Self(encode_column(classes))

    @staticmethod
    def check_target(y: StringTarget) raises -> Int:
        _single_output(y.rank())
        check_classification_targets(y)
        return y.rows()

    @staticmethod
    def equal_at(a: StringTarget, b: StringTarget, i: Int) raises -> Bool:
        return a.get(i) == b.get(i)

    def classes(self) -> List[String]:
        return self.encoding.classes.copy()

    def n_classes(self) -> Int:
        return len(self.encoding.classes)

    def same_classes(self, other: Self) raises -> Bool:
        return self.encoding.classes == other.encoding.classes

    def encode(self, y: StringTarget) raises -> NDArray[DType.int64]:
        return self.encoding.encode(y)

    def decode(self, codes: NDArray[DType.int64]) raises -> StringTarget:
        return self.encoding.decode(codes)


@fieldwise_init
struct BatchContract(Copyable, Movable):
    """Owned validated metadata; targets[i] is sample i's sorted class code."""
    var targets: NDArray[DType.int64]
    var n_classes: Int
    var weights: List[Float64]
    var n_features: Int


@fieldwise_init
struct LabeledBatch[D: ClassDomain](Copyable, Movable):
    """The (new or unchanged) class domain and the batch encoded against it."""
    var domain: Self.D
    var batch: BatchContract


def _prepare[dtype: DType, D: ClassDomain](
    X: NDArray[dtype], y: D.Target, current: Optional[D],
    classes: Optional[D.Target], sample_weight: Optional[List[Float64]],
    expected: Int, nonnegative: Bool,
) raises -> LabeledBatch[D]:
    var features = validate_matrix(X, expected, nonnegative)
    if D.check_target(y) != X.shape[0]:
        raise Error("ValueError: X and y have different sample counts")
    var domain: D
    if not current:
        if not classes:
            raise Error("ValueError: classes must be passed on the first call to partial_fit")
        domain = D.from_classes(classes.value())
    else:
        domain = current.value().copy()
        if classes and not domain.same_classes(D.from_classes(classes.value())):
            raise Error("ValueError: classes differ from the first partial_fit")
    var targets = domain.encode(y)
    var weights = _weights(X.shape[0], sample_weight)
    var n_classes = domain.n_classes()
    return LabeledBatch(domain^, BatchContract(targets^, n_classes, weights^, features))


def prepare_batch[dtype: DType, label: DType](
    X: NDArray[dtype], y: NDArray[label],
    current: Optional[NumericDomain[label]] = None,
    classes: Optional[NDArray[label]] = None,
    sample_weight: Optional[List[Float64]] = None,
    expected: Int = 0, nonnegative: Bool = False,
) raises -> LabeledBatch[NumericDomain[label]]:
    """Fit passes `classes=y`; partial_fit passes the fitted domain as current."""
    return _prepare[dtype, NumericDomain[label]](
        X, y, current, classes, sample_weight, expected, nonnegative
    )


def prepare_batch[dtype: DType](
    X: NDArray[dtype], y: StringTarget,
    current: Optional[StringDomain] = None,
    classes: Optional[StringTarget] = None,
    sample_weight: Optional[List[Float64]] = None,
    expected: Int = 0, nonnegative: Bool = False,
) raises -> LabeledBatch[StringDomain]:
    return _prepare[dtype, StringDomain](
        X, y, current, classes, sample_weight, expected, nonnegative
    )


def _int64(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def int_classes(domain: NumericDomain[DType.int64]) raises -> List[Int]:
    var result = List[Int](capacity=domain.n_classes())
    for i in range(domain.n_classes()):
        result.append(Int(domain.encoding.classes.item(i)))
    return result^


@fieldwise_init
struct IntListDomain(ClassDomain):
    """Compatibility labels using the same shared numeric encoding as arrays."""
    comptime Target = List[Int]
    var domain: NumericDomain[DType.int64]

    @staticmethod
    def empty_target() raises -> Self.Target:
        return List[Int]()

    @staticmethod
    def from_classes(classes: List[Int]) raises -> Self:
        return Self(NumericDomain[DType.int64].from_classes(_int64(classes)))

    @staticmethod
    def check_target(y: List[Int]) raises -> Int:
        return NumericDomain[DType.int64].check_target(_int64(y))

    @staticmethod
    def equal_at(a: List[Int], b: List[Int], i: Int) raises -> Bool:
        return a[i] == b[i]

    def class_values(self) raises -> List[Int]:
        return int_classes(self.domain)

    def n_classes(self) -> Int:
        return self.domain.n_classes()

    def same_classes(self, other: Self) raises -> Bool:
        return self.domain.same_classes(other.domain)

    def encode(self, y: List[Int]) raises -> NDArray[DType.int64]:
        return self.domain.encode(_int64(y))

    def decode(self, codes: NDArray[DType.int64]) raises -> List[Int]:
        var values = self.domain.decode(codes)
        var result = List[Int]()
        for i in range(values.size):
            result.append(Int(values.item(i)))
        return result^


def dense_from_lists(X: List[List[Float64]]) raises -> DenseMatrix:
    from moxie._naive_bayes_common import _check_X
    var features = _check_X(X)
    var result = nm.zeros[DType.float64](Shape(len(X), features))
    for i in range(len(X)):
        for j in range(features):
            result[Item(i, j)] = X[i][j]
    return result^


def probability_lists[dtype: DType](X: NDArray[dtype]) raises -> List[List[Float64]]:
    """Output adapter: preserve nonfinite probabilities in degenerate models."""
    var result = List[List[Float64]]()
    for i in range(X.shape[0]):
        var row = List[Float64]()
        for j in range(X.shape[1]):
            row.append(Float64(X.item(i, j)))
        result.append(row^)
    return result^


def prepare_batch[dtype: DType](
    X: NDArray[dtype], y: List[Int], current: List[Int],
    classes: Optional[List[Int]] = None,
    sample_weight: Optional[List[Float64]] = None,
    expected: Int = 0, nonnegative: Bool = False,
) raises -> LabeledBatch[NumericDomain[DType.int64]]:
    """Compatibility path for List[Int] estimators; empty current = unfitted."""
    var domain = Optional[NumericDomain[DType.int64]](None)
    if len(current) != 0:
        domain = NumericDomain[DType.int64].from_classes(_int64(current))
    var proposed = Optional[NDArray[DType.int64]](None)
    if classes:
        proposed = _int64(classes.value())
    return _prepare[dtype, NumericDomain[DType.int64]](
        X, _int64(y), domain, proposed, sample_weight, expected, nonnegative
    )


def _check_joint[dtype: DType](joint: NDArray[dtype], n_classes: Int = 0) raises:
    if joint.ndim != 2 or joint.shape[1] == 0:
        raise Error("ValueError: joint log likelihood must be a nonempty matrix")
    if n_classes != 0 and joint.shape[1] != n_classes:
        raise Error("ValueError: joint log likelihood columns differ from classes")


def joint_argmax[dtype: DType](joint: NDArray[dtype]) raises -> NDArray[DType.int64]:
    """Row argmax as np.argmax: first maximum wins, and the first NaN wins."""
    _check_joint(joint)
    var result = nm.zeros[DType.int64](Shape(joint.shape[0]))
    for i in range(joint.shape[0]):
        var best = 0
        for c in range(1, joint.shape[1]):
            var current = joint.item(i, best)
            var value = joint.item(i, c)
            if not isnan(current) and (isnan(value) or value > current):
                best = c
        result[Item(i)] = Int64(best)
    return result^


def predict_labels[D: ClassDomain, dtype: DType](domain: D, joint: NDArray[dtype]) raises -> D.Target:
    """Return classes_[argmax(jll, axis=1)]."""
    _check_joint(joint, domain.n_classes())
    return domain.decode(joint_argmax(joint))


def log_normalize[dtype: DType](joint: NDArray[dtype]) raises -> NDArray[dtype]:
    """Same-dtype logsumexp normalization, including nonfinite rows."""
    comptime assert dtype in (DType.float32, DType.float64), "probabilities require a floating dtype"
    _check_joint(joint)
    var result = nm.zeros[dtype](Shape(joint.shape[0], joint.shape[1]))
    for i in range(joint.shape[0]):
        var largest = joint.item(i, 0)
        for c in range(1, joint.shape[1]):
            var value = joint.item(i, c)
            if isnan(value) or value > largest:
                largest = value
        var normalizer = largest
        if isfinite(largest):
            var total = Scalar[dtype](0)
            for c in range(joint.shape[1]):
                # Evaluate libm accurately, rounding each exponential before sum.
                total += Scalar[dtype](_exp(Float64(joint.item(i, c) - largest)))
            normalizer += Scalar[dtype](_log(Float64(total)))
        for c in range(joint.shape[1]):
            result[Item(i, c)] = joint.item(i, c) - normalizer
    return result^


def exp_normalize[dtype: DType](joint: NDArray[dtype]) raises -> NDArray[dtype]:
    """Return exp(log_normalize(jll)); NaN log probabilities stay NaN."""
    var result = log_normalize(joint)
    for i in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[Item(i, c)] = Scalar[dtype](_exp(Float64(result.item(i, c))))
    return result^


def weighted_accuracy[D: ClassDomain](
    y_true: D.Target, y_pred: D.Target,
    sample_weight: Optional[List[Float64]] = None,
) raises -> Float64:
    """Return accuracy_score; labels outside classes_ count as wrong."""
    var n = D.check_target(y_true)
    if n != D.check_target(y_pred):
        raise Error("ValueError: X and y have different sample counts")
    var weights = _weights(n, sample_weight)
    var correct = Float64(0)
    var total = Float64(0)
    for i in range(n):
        total += weights[i]
        if D.equal_at(y_true, y_pred, i):
            correct += weights[i]
    return correct / total


trait DenseNBPredictor:
    """Shared NDArray prediction for native dense Naive Bayes estimators.

    Implementations supply their fitted class domain (column order of the joint
    matrix) and a validated joint log likelihood of shape (n_samples, n_classes).
    """

    comptime Domain: ClassDomain

    def _class_domain(self) raises -> Self.Domain:
        ...

    def predict_joint_log_proba[dtype: DType](
        self, X: NDArray[dtype]
    ) raises -> DenseMatrix:
        ...

    def predict[dtype: DType](
        self, X: NDArray[dtype]
    ) raises -> Self.Domain.Target:
        return predict_labels(self._class_domain(), self.predict_joint_log_proba(X))

    def predict_log_proba[dtype: DType](
        self, X: NDArray[dtype]
    ) raises -> DenseMatrix:
        var joint = self.predict_joint_log_proba(X)
        _check_joint(joint, self._class_domain().n_classes())
        return log_normalize(joint)

    def predict_proba[dtype: DType](
        self, X: NDArray[dtype]
    ) raises -> DenseMatrix:
        var joint = self.predict_joint_log_proba(X)
        _check_joint(joint, self._class_domain().n_classes())
        return exp_normalize(joint)

    def score[dtype: DType](
        self, X: NDArray[dtype], y: Self.Domain.Target,
        sample_weight: Optional[List[Float64]] = None,
    ) raises -> Float64:
        return weighted_accuracy[Self.Domain](y, self.predict(X), sample_weight)


@fieldwise_init
struct CountStatistics(Copyable, Movable):
    var class_count: DenseMatrix
    var feature_count: DenseMatrix


@fieldwise_init
struct TypedGaussianStatistics[dtype: DType](Copyable, Movable):
    var class_count: NDArray[Self.dtype]
    var mean: NDArray[Self.dtype]
    var variance: NDArray[Self.dtype]
    var epsilon: Scalar[Self.dtype]


comptime GaussianStatistics = TypedGaussianStatistics[DType.float64]


@fieldwise_init
struct CategoryStatistics(Copyable, Movable):
    var class_count: DenseMatrix
    var category_count: List[DenseMatrix]
    var n_categories: List[Int]


trait DenseLikelihood:
    """Child-owned likelihood; validation/fitted checks precede evaluation."""
    def joint_log_likelihood(self, X: DenseMatrix) raises -> DenseMatrix:
        ...


trait CountUpdater:
    """Child-owned numerical update; callers provide independently owned state."""
    def accumulate(
        self, X: DenseMatrix, batch: BatchContract,
        mut statistics: CountStatistics,
    ) raises:
        ...


@fieldwise_init
struct FeatureAlpha(Copyable, Movable):
    """Per-feature values take precedence over scalar; configuration owns its data."""
    var scalar: Float64
    var values: Optional[DenseMatrix]

    def validate(self, n_features: Int) raises:
        if self.values:
            if self.values.value().ndim != 1 or self.values.value().shape[0] != n_features:
                raise Error("ValueError: alpha must have n_features entries")
            for i in range(n_features):
                if not isfinite(self.values.value()[Item(i)]) or self.values.value()[Item(i)] < 0:
                    raise Error("ValueError: alpha must be finite and nonnegative")
        elif not isfinite(self.scalar) or self.scalar < 0:
            raise Error("ValueError: alpha must be finite and nonnegative")


@fieldwise_init
struct CategoryMinimum(Copyable, Movable):
    """Zero means infer from observed values; per-feature values override scalar."""
    var scalar: Int
    var values: Optional[List[Int]]

    def validate(self, n_features: Int) raises:
        if self.values:
            if len(self.values.value()) != n_features and len(self.values.value()) != 1:
                raise Error("ValueError: min_categories must have one or n_features entries")
            for value in self.values.value():
                if value < 0:
                    raise Error("ValueError: min_categories must be nonnegative")
        elif self.scalar < 0:
            raise Error("ValueError: min_categories must be nonnegative")


trait GaussianUpdater:
    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   mut statistics: GaussianStatistics) raises:
        ...


trait CategoryUpdater:
    def accumulate(self, X: DenseMatrix, batch: BatchContract,
                   minimum: CategoryMinimum, mut statistics: CategoryStatistics) raises:
        ...


trait DiscreteFinalizer:
    def finalize(self, statistics: CountStatistics, alpha: FeatureAlpha,
                 force_alpha: Bool) raises -> DenseMatrix:
        ...
