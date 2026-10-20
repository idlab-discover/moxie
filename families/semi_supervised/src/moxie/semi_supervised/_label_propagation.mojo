# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense graph diffusion with hard or soft label clamping."""
from std.math import abs, sqrt, isfinite, isnan
from numojo import NDArray, Shape, zeros
from moxie._dense import sum_axis2d
from ._base import SemiSupervisedClassifier
from ._contracts import DenseAffinity, ClassEncoding, ZeroMassPolicy, integer_targets, validate_features
from ._affinity import RBFAffinity, finite_features
from ._probability import RowNormalizer

@fieldwise_init
struct GraphParams[K: DenseAffinity & Copyable & Deinitable](Copyable, Movable):
    var kernel: String
    var gamma: Float64
    var n_neighbors: Int
    var max_iter: Int
    var tol: Float64
    var n_jobs: Optional[Int]
    var alpha: Float64
    var affinity: Optional[Self.K]

struct GraphClassifier[spreading: Bool, dtype: DType = DType.float64, K: DenseAffinity & Copyable & Deinitable = RBFAffinity[dtype], L: Copyable & Comparable & Deinitable = Int](SemiSupervisedClassifier, Copyable, Movable):
    comptime Input = NDArray[Self.dtype]
    comptime Label = Self.L
    comptime Target = List[Self.L]
    comptime Prediction = List[Self.L]
    comptime Probabilities = NDArray[DType.float64]
    comptime SampleWeight = List[Float64]
    comptime Params = GraphParams[Self.K]
    var kernel: String
    var gamma: Float64
    var n_neighbors: Int
    var max_iter: Int
    var tol: Float64
    var n_jobs: Optional[Int]
    var alpha: Float64
    var affinity_provider: Optional[Self.K]
    var X_: Self.Input
    var classes_: List[Self.L]
    var label_distributions_: Self.Probabilities
    var transduction_: List[Self.L]
    var n_features_in_: Int
    var n_iter_: Int
    var converged_: Bool
    var _fitted: Bool

    def __init__(out self, kernel: String = "rbf", *, gamma: Float64 = 20, n_neighbors: Int = 7, max_iter: Int = 30 if Self.spreading else 1000, tol: Float64 = 0.001, n_jobs: Optional[Int] = None, alpha: Float64 = 0.2, var affinity: Optional[Self.K] = None) raises:
        comptime assert Self.K.Input == Self.Input, "affinity input must match the selected dense dtype"
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.affinity_provider = affinity^
        self.X_ = zeros[Self.dtype](Shape(0, 0))
        self.classes_ = List[Self.L]()
        self.label_distributions_ = zeros[DType.float64](Shape(0, 0))
        self.transduction_ = List[Self.L]()
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.converged_ = False
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.kernel, self.gamma, self.n_neighbors, self.max_iter, self.tol, self.n_jobs, self.alpha, self.affinity_provider.copy())

    def set_params[origin: MutOrigin, p_origin: ImmOrigin](ref[origin] self, ref[p_origin] params: Self.Params) raises -> ref[self] Self:
        self.kernel = params.kernel
        self.gamma = params.gamma
        self.n_neighbors = params.n_neighbors
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.n_jobs = params.n_jobs
        self.alpha = params.alpha
        self.affinity_provider = params.affinity.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, kernel: Optional[String] = None, gamma: Optional[Float64] = None, n_neighbors: Optional[Int] = None, max_iter: Optional[Int] = None, tol: Optional[Float64] = None, n_jobs: Optional[Optional[Int]] = None, alpha: Optional[Float64] = None, affinity: Optional[Optional[Self.K]] = None) raises -> ref[self] Self:
        if kernel:
            self.kernel = kernel.value()
        if gamma:
            self.gamma = gamma.value()
        if n_neighbors:
            self.n_neighbors = n_neighbors.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if tol:
            self.tol = tol.value()
        if n_jobs:
            self.n_jobs = n_jobs.value()
        if alpha:
            self.alpha = alpha.value()
        if affinity:
            self.affinity_provider = affinity.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.kernel, gamma=self.gamma, n_neighbors=self.n_neighbors, max_iter=self.max_iter, tol=self.tol, n_jobs=self.n_jobs, alpha=self.alpha, affinity=self.affinity_provider.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _affinity(self, train: Self.Input, query: Self.Input) raises -> Self.Probabilities:
        var result: Self.Probabilities
        if self.kernel == "rbf":
            result = RBFAffinity[Self.dtype](self.gamma).affinity(train, query)
        elif self.kernel == "callable" and self.affinity_provider:
            result = self.affinity_provider.value().affinity(rebind[Self.K.Input](train), rebind[Self.K.Input](query))
        else:
            raise Error("ValueError: supported kernels are rbf and typed callable; knn is deferred")
        if result.ndim != 2 or result.shape[0] != train.shape[0] or result.shape[1] != query.shape[0]:
            raise Error("ValueError: affinity must have train rows and query columns")
        return result^

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        var targets = List[Optional[Self.L]]()
        for label in y:
            comptime if Self.L == Int:
                if rebind[Int](label) == -1:
                    targets.append(None)
                    continue
            targets.append(Optional(label.copy()))
        self.fit(X, targets)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: List[Optional[Self.L]]) raises -> ref[self] Self:
        self._fitted = False
        finite_features(X)
        validate_features(X, len(y))
        if not isfinite(self.gamma) or self.gamma < 0 or not isfinite(self.tol) or self.tol < 0 or self.max_iter < 1 or self.n_neighbors < 1:
            raise Error("ValueError: invalid graph parameters")
        comptime if Self.spreading:
            if not (self.alpha > 0 and self.alpha < 1):
                raise Error("ValueError: alpha must be in (0, 1)")
        self.classes_ = List[Self.L]()
        var targets = y.copy()
        for target in targets:
            if not target:
                continue
            var label = target.value().copy()
            comptime if Self.L == Int:
                if rebind[Int](label) == -1:
                    raise Error("ValueError: use None for an unlabeled tagged integer target")
            var found = False
            for existing in self.classes_:
                found = found or existing == label
            if not found:
                self.classes_.append(label.copy())
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                if self.classes_[j] < self.classes_[i]:
                    var temp = self.classes_[i].copy()
                    self.classes_[i] = self.classes_[j].copy()
                    self.classes_[j] = temp^
        var encoding = ClassEncoding(self.classes_)
        var codes = encoding.encode(targets)
        self.X_ = X.copy()
        self.n_features_in_ = X.shape[1]
        var n = len(y)
        var c = len(self.classes_)
        var graph = self._affinity(X, X)
        if self.kernel == "rbf":
            for i in range(n):
                graph.unsafe_set(i * Int(graph.strides[0]) + i * Int(graph.strides[1]), 1)
        comptime if Self.spreading:
            for i in range(n):
                graph.unsafe_set(i * Int(graph.strides[0]) + i * Int(graph.strides[1]), 0)
            var degrees = sum_axis2d(graph, 0)
            for i in range(n):
                var value = degrees.unsafe_get(i)
                degrees.unsafe_set(i, sqrt(value) if value != 0 else Float64(1))
            for i in range(n):
                for j in range(n):
                    var index = i * Int(graph.strides[0]) + j * Int(graph.strides[1])
                    graph.unsafe_set(index, graph.unsafe_get(index) / degrees.unsafe_get(i) / degrees.unsafe_get(j))
        else:
            graph = RowNormalizer().normalize(graph, ZeroMassPolicy.IEEE_DIVIDE)
        var initial = zeros[DType.float64](Shape(n, c))
        for i in range(n):
            if codes[i] != -1:
                initial.unsafe_set(i * c + codes[i], 1)
        var current = initial.copy()
        var previous = zeros[DType.float64](Shape(n, c))
        self.n_iter_ = 0
        self.converged_ = False
        for _ in range(self.max_iter):
            var change = Float64(0)
            for i in range(n * c):
                change += abs(current.unsafe_get(i) - previous.unsafe_get(i))
            if change < self.tol:
                self.converged_ = True
                break
            previous = current.copy()
            current = graph @ current
            comptime if Self.spreading:
                for i in range(n * c):
                    current.unsafe_set(i, self.alpha * current.unsafe_get(i) + (1 - self.alpha) * initial.unsafe_get(i))
            else:
                current = RowNormalizer().normalize(current, ZeroMassPolicy.PRESERVE)
                for i in range(n):
                    if codes[i] != -1:
                        for j in range(c):
                            current.unsafe_set(i * c + j, initial.unsafe_get(i * c + j))
            self.n_iter_ += 1
        if not self.converged_:
            print("ConvergenceWarning: max_iter=" + String(self.max_iter) + " was reached without convergence.")
        self.label_distributions_ = RowNormalizer().normalize(current, ZeroMassPolicy.PRESERVE)
        self.transduction_ = self._labels(self.label_distributions_)
        self._fitted = True
        return self

    def _labels(self, probabilities: Self.Probabilities) raises -> List[Self.L]:
        var result = List[Self.L]()
        for i in range(probabilities.shape[0]):
            var best = 0
            for j in range(1, len(self.classes_)):
                var candidate = probabilities.unsafe_get(i * len(self.classes_) + j)
                var current = probabilities.unsafe_get(i * len(self.classes_) + best)
                if (isnan(candidate) and not isnan(current)) or candidate > current:
                    best = j
            result.append(self.classes_[best].copy())
        return result^

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        if not self._fitted:
            raise Error("NotFittedError: fit the graph estimator first")
        finite_features(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: feature count differs from fit")
        var affinity = self._affinity(self.X_, X)
        var result = affinity.T() @ self.label_distributions_
        return RowNormalizer().normalize(result, ZeroMassPolicy.IEEE_DIVIDE)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self._labels(self.predict_proba(X))

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var predicted = self.predict(X)
        if len(y) != len(predicted):
            raise Error("ValueError: target length mismatch")
        if sample_weight:
            if len(sample_weight.value()) != len(y):
                raise Error("ValueError: sample_weight length mismatch")
        var correct = Float64(0)
        var total = Float64(0)
        for i in range(len(y)):
            var weight = sample_weight.value()[i] if sample_weight else Float64(1)
            if not isfinite(weight):
                raise Error("ValueError: sample_weight must contain only finite values")
            total += weight
            if y[i] == predicted[i]:
                correct += weight
        if total == 0:
            raise Error("ValueError: sample weights sum to zero")
        return correct / total

comptime LabelPropagation[dtype: DType = DType.float64, K: DenseAffinity & Copyable & Deinitable = RBFAffinity[dtype], L: Copyable & Comparable & Deinitable = Int] = GraphClassifier[False, dtype, K, L]
comptime LabelSpreading[dtype: DType = DType.float64, K: DenseAffinity & Copyable & Deinitable = RBFAffinity[dtype], L: Copyable & Comparable & Deinitable = Int] = GraphClassifier[True, dtype, K, L]
