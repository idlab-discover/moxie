"""Dense family encoding, ownership, and extension contracts."""
from numojo import NDArray

comptime UNLABELED = -1


struct ClassEncoding[T: Copyable & Equatable & Deinitable](Copyable, Movable):
    """Owned ordered vocabulary. Caller supplies sklearn class order."""
    var classes: List[Self.T]

    def __init__(out self, classes: List[Self.T]) raises:
        if len(classes) == 0:
            raise Error("ValueError: at least one labeled class is required")
        for i in range(len(classes)):
            for j in range(i):
                if classes[i] == classes[j]:
                    raise Error("ValueError: duplicate class")
        self.classes = classes.copy()

    def encode(self, targets: List[Optional[Self.T]]) raises -> List[Int]:
        var codes = List[Int]()
        for i in range(len(targets)):
            if not targets[i]:
                codes.append(UNLABELED)
                continue
            var found = UNLABELED
            for j in range(len(self.classes)):
                if targets[i].value() == self.classes[j]:
                    found = j
                    break
            if found == UNLABELED:
                raise Error("ValueError: unknown class")
            codes.append(found)
        return codes^

    def decode(self, codes: List[Int]) raises -> List[Self.T]:
        validate_codes(codes, len(self.classes), allow_unlabeled=False)
        var labels = List[Self.T]()
        for code in codes:
            labels.append(self.classes[code].copy())
        return labels^


def integer_targets(y: List[Int]) -> List[Optional[Int]]:
    """Translate sklearn's integer sentinel to the native tagged target API."""
    var result = List[Optional[Int]]()
    for value in y:
        if value == UNLABELED:
            result.append(None)
        else:
            result.append(Optional(value))
    return result^


def validate_codes(codes: List[Int], n_classes: Int, allow_unlabeled: Bool = True) raises:
    if n_classes <= 0:
        raise Error("ValueError: at least one labeled class is required")
    for code in codes:
        if code == UNLABELED and allow_unlabeled:
            continue
        if code < 0 or code >= n_classes:
            raise Error("ValueError: invalid encoded class")


def validate_features[dtype: DType](X: NDArray[dtype], n_targets: Int) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2:
        raise Error("ValueError: expected rank-2 dense features")
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[0] != n_targets:
        raise Error("ValueError: nonempty features and matching target rows required")


def snapshot_features[dtype: DType](X: NDArray[dtype], n_targets: Int) raises -> NDArray[dtype]:
    validate_features(X, n_targets)
    return X.copy()


def validate_probability_shape[dtype: DType](P: NDArray[dtype], rows: Int, classes: Int) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if P.ndim != 2:
        raise Error("ValueError: expected rank-2 class probabilities")
    if rows < 0 or classes <= 0 or P.shape[0] != rows or P.shape[1] != classes:
        raise Error("ValueError: probability rows/columns disagree with class vocabulary")


struct ZeroMassPolicy:
    comptime PRESERVE = 0
    comptime IEEE_DIVIDE = 1


trait ProbabilityNormalizer:
    """Owned Float64 output, explicit zero-mass behavior."""
    def normalize(self, values: NDArray[DType.float64], zero_mass: Int) raises -> NDArray[DType.float64]:
        ...


trait DenseAffinity:
    """Shared graph affinity: train rows by query rows, no normalization."""
    comptime Input: Copyable

    def affinity(self, train: Self.Input, query: Self.Input) raises -> NDArray[DType.float64]:
        ...


trait ProbabilisticEstimatorAdapter(Copyable, Movable):
    """Typed self-training composition, with configuration-only cloning."""
    comptime Input: Copyable
    comptime Label: Copyable & Equatable & Deinitable

    def clone_unfitted(self) raises -> Self:
        ...

    def fit(mut self, X: Self.Input, y: List[Self.Label]) raises:
        ...

    def classes(self) raises -> List[Self.Label]:
        ...

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        ...


trait ScoringEstimatorAdapter(ProbabilisticEstimatorAdapter):
    """Optional delegated score; probability-only estimators need not supply it."""
    def score(self, X: Self.Input, y: List[Self.Label]) raises -> Float64:
        ...
