# Adapted from scikit-learn; BSD-3-Clause, see algorithms/c_svm/SCIKIT_LEARN_LICENSE.
"""Dense kernel estimators using the family LIBSVM engine."""
import numojo as nm
from std.math import log
from moxie.svm._contracts import Dense, Indices, KernelModel, KernelSpec, KernelOptions, KernelProvider, TrainingSnapshot, validate_matrix, validate_vector
from moxie.svm._libsvm import LibSVM
from moxie.svm._classification import Gamma, Probability, class_scores, support_counts, linear_coefficients, public_support, checked_int32
from moxie.svm._linear import ClassWeight, _labels, _codes, _weights, _class_weights
from moxie.svm._base import ProbabilisticSVC, SupportVectorRegressor, OneClassSupportVector
from moxie._random import RandomState, entropy_state

@fieldwise_init
struct NuSVCParams[Label: Copyable & Comparable & Deinitable = Float64](Copyable, Movable):
    var nu: Float64
    var kernel: String
    var degree: Int
    var gamma: Gamma
    var coef0: Float64
    var tol: Float64
    var shrinking: Bool
    var cache_size: Float64
    var verbose: Bool
    var max_iter: Int
    var probability: Probability
    var class_weight: ClassWeight[Self.Label]
    var decision_function_shape: String
    var break_ties: Bool
    var random_state: Int


struct NuSVC[Label: Copyable & Comparable & Deinitable = Float64](ProbabilisticSVC):
    comptime Params = NuSVCParams[Self.Label]
    comptime Input = Dense
    comptime SampleWeight = Dense
    comptime Target = List[Self.Label]
    comptime Prediction = List[Self.Label]
    comptime Decision = Dense
    comptime Probabilities = Dense
    comptime SampleScores = Dense
    var nu: Float64
    var kernel: String
    var degree: Int
    var gamma: Gamma
    var coef0: Float64
    var tol: Float64
    var shrinking: Bool
    var cache_size: Float64
    var verbose: Bool
    var max_iter: Int
    var probability: Probability
    var class_weight: ClassWeight[Self.Label]
    var decision_function_shape: String
    var break_ties: Bool
    var random_state: Int
    var support_: nm.NDArray[DType.int32]
    var support_vectors_: Dense
    var n_support_: nm.NDArray[DType.int32]
    var dual_coef_: Dense
    var intercept_: Dense
    var n_features_in_: Int
    var n_iter_: nm.NDArray[DType.int32]
    var fit_status_: Int
    var shape_fit_: nm.Shape
    var _gamma: Float64
    var _kernel: KernelSpec
    var _model: Optional[KernelModel]
    var _snapshot: Optional[TrainingSnapshot]
    var _rng: RandomState
    var classes_: List[Self.Label]
    var class_weight_: Dense
    var probA_: Dense
    var probB_: Dense

    def __init__(out self, *, nu: Float64 = 0.5, kernel: String = "rbf", degree: Int = 3, gamma: Gamma = "scale", coef0: Float64 = 0.0, tol: Float64 = 1e-3, shrinking: Bool = True, cache_size: Float64 = 200.0, verbose: Bool = False, max_iter: Int = -1, probability: Probability = "deprecated", class_weight: ClassWeight[Self.Label] = "none", decision_function_shape: String = "ovr", break_ties: Bool = False, random_state: Int = -1) raises:
        self.nu = nu.copy()
        self.kernel = kernel.copy()
        self.degree = degree.copy()
        self.gamma = gamma.copy()
        self.coef0 = coef0.copy()
        self.tol = tol.copy()
        self.shrinking = shrinking.copy()
        self.cache_size = cache_size.copy()
        self.verbose = verbose.copy()
        self.max_iter = max_iter.copy()
        self.probability = probability.copy()
        self.class_weight = class_weight.copy()
        self.decision_function_shape = decision_function_shape.copy()
        self.break_ties = break_ties.copy()
        self.random_state = random_state.copy()
        self.support_ = nm.zeros[DType.int32](nm.Shape(0))
        self.support_vectors_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_support_ = nm.zeros[DType.int32](nm.Shape(0))
        self.dual_coef_ = nm.zeros[DType.float64](nm.Shape(0))
        self.intercept_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_features_in_ = 0
        self.n_iter_ = nm.zeros[DType.int32](nm.Shape(0))
        self.fit_status_ = 0
        self.shape_fit_ = nm.Shape(0, 0)
        self._gamma = 0
        self._kernel = KernelSpec("rbf", 0.0, 3, 0.0)
        self._model = None
        self._snapshot = None
        self._rng = entropy_state()
        self.classes_ = List[Self.Label]()
        self.class_weight_ = nm.zeros[DType.float64](nm.Shape(0))
        self.probA_ = nm.zeros[DType.float64](nm.Shape(0))
        self.probB_ = nm.zeros[DType.float64](nm.Shape(0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.nu.copy(), self.kernel.copy(), self.degree.copy(), self.gamma.copy(), self.coef0.copy(), self.tol.copy(), self.shrinking.copy(), self.cache_size.copy(), self.verbose.copy(), self.max_iter.copy(), self.probability.copy(), self.class_weight.copy(), self.decision_function_shape.copy(), self.break_ties.copy(), self.random_state.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.nu = params.nu.copy()
        self.kernel = params.kernel.copy()
        self.degree = params.degree.copy()
        self.gamma = params.gamma.copy()
        self.coef0 = params.coef0.copy()
        self.tol = params.tol.copy()
        self.shrinking = params.shrinking.copy()
        self.cache_size = params.cache_size.copy()
        self.verbose = params.verbose.copy()
        self.max_iter = params.max_iter.copy()
        self.probability = params.probability.copy()
        self.class_weight = params.class_weight.copy()
        self.decision_function_shape = params.decision_function_shape.copy()
        self.break_ties = params.break_ties.copy()
        self.random_state = params.random_state.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, nu: Optional[Float64] = None, kernel: Optional[String] = None, degree: Optional[Int] = None, gamma: Optional[Gamma] = None, coef0: Optional[Float64] = None, tol: Optional[Float64] = None, shrinking: Optional[Bool] = None, cache_size: Optional[Float64] = None, verbose: Optional[Bool] = None, max_iter: Optional[Int] = None, probability: Optional[Probability] = None, class_weight: Optional[ClassWeight[Self.Label]] = None, decision_function_shape: Optional[String] = None, break_ties: Optional[Bool] = None, random_state: Optional[Int] = None) raises -> ref[self] Self:
        if nu:
            self.nu = nu.value().copy()
        if kernel:
            self.kernel = kernel.value().copy()
        if degree:
            self.degree = degree.value().copy()
        if gamma:
            self.gamma = gamma.value().copy()
        if coef0:
            self.coef0 = coef0.value().copy()
        if tol:
            self.tol = tol.value().copy()
        if shrinking:
            self.shrinking = shrinking.value().copy()
        if cache_size:
            self.cache_size = cache_size.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        if probability:
            self.probability = probability.value().copy()
        if class_weight:
            self.class_weight = class_weight.value().copy()
        if decision_function_shape:
            self.decision_function_shape = decision_function_shape.value().copy()
        if break_ties:
            self.break_ties = break_ties.value().copy()
        if random_state:
            self.random_state = random_state.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(nu=self.nu.copy(), kernel=self.kernel.copy(), degree=self.degree.copy(), gamma=self.gamma.copy(), coef0=self.coef0.copy(), tol=self.tol.copy(), shrinking=self.shrinking.copy(), cache_size=self.cache_size.copy(), verbose=self.verbose.copy(), max_iter=self.max_iter.copy(), probability=self.probability.copy(), class_weight=self.class_weight.copy(), decision_function_shape=self.decision_function_shape.copy(), break_ties=self.break_ties.copy(), random_state=self.random_state.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._model)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, sample_weight: Optional[Dense]) raises -> ref[self] Self:
        var rng = self._rng.copy()
        if self.random_state < -1 or self.random_state > 4294967295:
            raise Error("ValueError: random_state must be None (-1) or uint32 seed")
        if self.random_state != -1:
            rng = RandomState(self.random_state)
        _ = self.fit_with_random_state(X, y, rng, sample_weight)
        if self.random_state == -1:
            self._rng = rng^
        return self

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        if self.kernel == "callable":
            raise Error("ValueError: callable kernel requires fit_with_kernel")
        var snapshot = TrainingSnapshot(X, self.kernel == "precomputed")
        var gamma = self.gamma.resolve(X, self.kernel)
        var spec = KernelSpec(self.kernel.copy(), gamma, self.degree, self.coef0)
        spec.validate()
        var weights = _weights(X.shape[0], sample_weight)
        if len(y) != X.shape[0]:
            raise Error("ValueError: inconsistent target length")
        if self.decision_function_shape != "ovr" and self.decision_function_shape != "ovo":
            raise Error("ValueError: decision_function_shape must be ovr or ovo")
        var classes = _labels(y)
        if len(classes) < 2:
            raise Error("ValueError: at least two classes required")
        var target = _codes(y, classes)
        var unweighted = nm.ones[DType.float64](nm.Shape(X.shape[0]))
        var costs = _class_weights(self.class_weight, classes, target, unweighted)
        var options = KernelOptions("nu_svc", spec.copy(), 0.0, self.nu, 0.0, self.tol, self.cache_size, self.shrinking, self.probability.enabled(), self.max_iter, rng.randint(2147483647))
        var engine = LibSVM()
        var model = engine.solve(X, target, weights, costs, options)
        self.support_ = public_support(model, weights)
        self.support_vectors_ = model.support_vectors.copy()
        self.n_support_ = support_counts(model)
        self.dual_coef_ = model.dual_coef.copy()
        self.intercept_ = model.rho * Float64(-1)
        if len(classes) == 2:
            self.dual_coef_ = self.dual_coef_ * Float64(-1)
            self.intercept_ = model.rho.copy()
        self.classes_ = classes^
        self.class_weight_ = costs^
        self.probA_ = model.prob_a.copy()
        self.probB_ = model.prob_b.copy()
        self.n_iter_ = checked_int32(model.iterations)
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = nm.Shape(X.shape[0], X.shape[1])
        self.fit_status_ = model.fit_status
        self._gamma = gamma
        self._kernel = spec^
        self._model = model^
        self._snapshot = snapshot^
        if self.fit_status_ != 0:
            print("ConvergenceWarning: LIBSVM reached max_iter")
        if self.verbose:
            print("[LibSVM] support vectors:", self.support_.shape[0])
        return self

    def _raw_decisions(self, X: Dense) raises -> Dense:
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        if self.kernel == "callable":
            raise Error("ValueError: callable inference requires a KernelProvider")
        self._snapshot.value().validate_query(X)
        var engine = LibSVM()
        return engine.decision_values(X, self._model.value(), self._kernel)

    def coef_(self) raises -> Dense:
        # sklearn's property checks the current kernel parameter before fitted state.
        if self.kernel != "linear":
            raise Error("AttributeError: coef_ is only available when using a linear kernel")
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        return linear_coefficients(self._model.value(), self._kernel, len(self.classes_))

    def decision_function(self, X: Dense) raises -> Dense:
        var result = self._raw_decisions(X)
        if len(self.classes_) == 2:
            return (result * Float64(-1)).reshape(nm.Shape(X.shape[0]))
        if self.decision_function_shape == "ovr":
            return class_scores(result, len(self.classes_), True)
        return result^

    def predict(self, X: Dense) raises -> Self.Prediction:
        if self.break_ties and self.decision_function_shape == "ovo":
            raise Error("ValueError: break_ties requires decision_function_shape=ovr")
        var raw = self._raw_decisions(X)
        var result = List[Self.Label]()
        var scores = class_scores(raw, len(self.classes_), self.break_ties and len(self.classes_) > 2)
        for i in range(X.shape[0]):
            var winner = 0
            for j in range(1, len(self.classes_)):
                if scores.unsafe_get(i * len(self.classes_) + j) > scores.unsafe_get(i * len(self.classes_) + winner):
                    winner = j
            result.append(self.classes_[winner].copy())
        return result^

    def predict_proba(self, X: Dense) raises -> Dense:
        if not self.probability.enabled():
            raise Error("AttributeError: predict_proba requires probability=True")
        var raw = self._raw_decisions(X)
        if self.probA_.size == 0:
            raise Error("NotFittedError: predict_proba is not available when fitted with probability=False")
        var engine = LibSVM()
        return engine.class_probabilities(raw, self._model.value())

    def predict_log_proba(self, X: Dense) raises -> Dense:
        var result = self.predict_proba(X)
        for i in range(result.size):
            result.unsafe_set(i, log(result.unsafe_get(i)))
        return result^

    def score(self, X: Dense, y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X)
        var weights = _weights(X.shape[0], sample_weight)
        if len(y) != X.shape[0]:
            raise Error("ValueError: inconsistent target length")
        var numerator = Float64(0)
        var denominator = Float64(0)
        for i in range(len(y)):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            denominator += w
            if prediction[i] == y[i]:
                numerator += w
        if denominator == 0:
            raise Error("ValueError: weights sum to zero")
        return numerator / denominator


    def score[Provider: KernelProvider](self, X: Dense, y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X, provider)
        var weights = _weights(X.shape[0], sample_weight)
        if len(y) != X.shape[0]:
            raise Error("ValueError: inconsistent target length")
        var numerator = Float64(0)
        var denominator = Float64(0)
        for i in range(len(y)):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            denominator += w
            if prediction[i] == y[i]:
                numerator += w
        if denominator == 0:
            raise Error("ValueError: weights sum to zero")
        return numerator / denominator


    def score(self, X: nm.NDArray[DType.float32], y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, sample_weight)

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        if self.kernel != "callable":
            raise Error("ValueError: fit_with_kernel requires kernel=callable")
        var snapshot = TrainingSnapshot(X)
        var gram = provider.evaluate(X, snapshot.x)
        validate_matrix(gram, X.shape[0])
        if gram.shape[0] != X.shape[0]:
            raise Error("ValueError: callable training kernel has wrong row count")
        var candidate = self.__sklearn_clone__()
        candidate._rng = self._rng.copy()
        candidate.kernel = "precomputed"
        _ = candidate.fit(gram, y, sample_weight)
        candidate.kernel = "callable"
        candidate._snapshot = snapshot^
        candidate.n_features_in_ = X.shape[1]
        candidate.shape_fit_ = nm.Shape(X.shape[0], X.shape[1])
        self = candidate^
        return self

    def _cross_kernel[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        if self.kernel != "callable":
            raise Error("ValueError: provider requires fitted callable kernel")
        self._snapshot.value().validate_query(X)
        var result = provider.evaluate(X, self._snapshot.value().x)
        validate_matrix(result, self._snapshot.value().n_train)
        if result.shape[0] != X.shape[0]:
            raise Error("ValueError: callable query kernel has wrong row count")
        return result^

    def _raw_with_kernel[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        var gram = self._cross_kernel(X, provider)
        var engine = LibSVM()
        return engine.decision_values(gram, self._model.value(), self._kernel)

    def decision_function[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        var result = self._raw_with_kernel(X, provider)
        if len(self.classes_) == 2:
            return (result * Float64(-1)).reshape(nm.Shape(X.shape[0]))
        if self.decision_function_shape == "ovr":
            return class_scores(result, len(self.classes_), True)
        return result^


    def predict[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Self.Prediction:
        if self.break_ties and self.decision_function_shape == "ovo":
            raise Error("ValueError: break_ties requires decision_function_shape=ovr")
        var raw = self._raw_with_kernel(X, provider)
        var result = List[Self.Label]()
        var scores = class_scores(raw, len(self.classes_), self.break_ties and len(self.classes_) > 2)
        for i in range(X.shape[0]):
            var winner = 0
            for j in range(1, len(self.classes_)):
                if scores.unsafe_get(i * len(self.classes_) + j) > scores.unsafe_get(i * len(self.classes_) + winner):
                    winner = j
            result.append(self.classes_[winner].copy())
        return result^


    def predict_proba[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        if not self.probability.enabled():
            raise Error("AttributeError: predict_proba requires probability=True")
        var raw = self._raw_with_kernel(X, provider)
        if self.probA_.size == 0:
            raise Error("NotFittedError: predict_proba is not available when fitted with probability=False")
        var engine = LibSVM()
        return engine.class_probabilities(raw, self._model.value())


    def predict_log_proba[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        var result = self.predict_proba(X, provider)
        for i in range(result.size):
            result.unsafe_set(i, log(result.unsafe_get(i)))
        return result^


    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y, sample_weight)
        return self

    def predict(self, X: nm.NDArray[DType.float32]) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64]())

    def decision_function(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.decision_function(X.astype[DType.float64]())

    def predict_proba(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.predict_proba(X.astype[DType.float64]())

    def predict_log_proba(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.predict_log_proba(X.astype[DType.float64]())

    def fit[dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: nm.NDArray[dtype], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        comptime assert Self.Label == Float64, "Numeric arrays require Float64 labels"
        var target = y.astype[DType.float64]()
        validate_vector(target, X.shape[0])
        var labels = List[Self.Label]()
        for i in range(target.shape[0]):
            labels.append(rebind[Self.Label](target.unsafe_get(i * Int(target.strides[0]))).copy())
        _ = self.fit(X, labels, sample_weight)
        return self

    def fit[dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: nm.NDArray[dtype], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y.astype[DType.float64](), sample_weight)
        return self

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit_with_kernel(X.astype[DType.float64](), y, provider, sample_weight)
        return self

    def predict[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64](), provider)

    def decision_function[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.decision_function(X.astype[DType.float64](), provider)

    def predict_proba[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.predict_proba(X.astype[DType.float64](), provider)

    def predict_log_proba[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.predict_log_proba(X.astype[DType.float64](), provider)

    def score[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, provider, sample_weight)


@fieldwise_init
struct NuSVRParams(Copyable, Movable):
    var nu: Float64
    var C: Float64
    var kernel: String
    var degree: Int
    var gamma: Gamma
    var coef0: Float64
    var tol: Float64
    var shrinking: Bool
    var cache_size: Float64
    var verbose: Bool
    var max_iter: Int


struct NuSVR(SupportVectorRegressor):
    comptime Params = NuSVRParams
    comptime Input = Dense
    comptime SampleWeight = Dense
    comptime Target = Dense
    comptime Prediction = Dense
    comptime Decision = Dense
    comptime Probabilities = Dense
    comptime SampleScores = Dense
    var nu: Float64
    var C: Float64
    var kernel: String
    var degree: Int
    var gamma: Gamma
    var coef0: Float64
    var tol: Float64
    var shrinking: Bool
    var cache_size: Float64
    var verbose: Bool
    var max_iter: Int
    var support_: nm.NDArray[DType.int32]
    var support_vectors_: Dense
    var n_support_: nm.NDArray[DType.int32]
    var dual_coef_: Dense
    var intercept_: Dense
    var n_features_in_: Int
    var n_iter_: Int
    var fit_status_: Int
    var shape_fit_: nm.Shape
    var _gamma: Float64
    var _kernel: KernelSpec
    var _model: Optional[KernelModel]
    var _snapshot: Optional[TrainingSnapshot]
    var _rng: RandomState

    def __init__(out self, *, nu: Float64 = 0.5, C: Float64 = 1.0, kernel: String = "rbf", degree: Int = 3, gamma: Gamma = "scale", coef0: Float64 = 0.0, tol: Float64 = 1e-3, shrinking: Bool = True, cache_size: Float64 = 200.0, verbose: Bool = False, max_iter: Int = -1) raises:
        self.nu = nu.copy()
        self.C = C.copy()
        self.kernel = kernel.copy()
        self.degree = degree.copy()
        self.gamma = gamma.copy()
        self.coef0 = coef0.copy()
        self.tol = tol.copy()
        self.shrinking = shrinking.copy()
        self.cache_size = cache_size.copy()
        self.verbose = verbose.copy()
        self.max_iter = max_iter.copy()
        self.support_ = nm.zeros[DType.int32](nm.Shape(0))
        self.support_vectors_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_support_ = nm.zeros[DType.int32](nm.Shape(0))
        self.dual_coef_ = nm.zeros[DType.float64](nm.Shape(0))
        self.intercept_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.fit_status_ = 0
        self.shape_fit_ = nm.Shape(0, 0)
        self._gamma = 0
        self._kernel = KernelSpec("rbf", 0.0, 3, 0.0)
        self._model = None
        self._snapshot = None
        self._rng = entropy_state()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.nu.copy(), self.C.copy(), self.kernel.copy(), self.degree.copy(), self.gamma.copy(), self.coef0.copy(), self.tol.copy(), self.shrinking.copy(), self.cache_size.copy(), self.verbose.copy(), self.max_iter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.nu = params.nu.copy()
        self.C = params.C.copy()
        self.kernel = params.kernel.copy()
        self.degree = params.degree.copy()
        self.gamma = params.gamma.copy()
        self.coef0 = params.coef0.copy()
        self.tol = params.tol.copy()
        self.shrinking = params.shrinking.copy()
        self.cache_size = params.cache_size.copy()
        self.verbose = params.verbose.copy()
        self.max_iter = params.max_iter.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, nu: Optional[Float64] = None, C: Optional[Float64] = None, kernel: Optional[String] = None, degree: Optional[Int] = None, gamma: Optional[Gamma] = None, coef0: Optional[Float64] = None, tol: Optional[Float64] = None, shrinking: Optional[Bool] = None, cache_size: Optional[Float64] = None, verbose: Optional[Bool] = None, max_iter: Optional[Int] = None) raises -> ref[self] Self:
        if nu:
            self.nu = nu.value().copy()
        if C:
            self.C = C.value().copy()
        if kernel:
            self.kernel = kernel.value().copy()
        if degree:
            self.degree = degree.value().copy()
        if gamma:
            self.gamma = gamma.value().copy()
        if coef0:
            self.coef0 = coef0.value().copy()
        if tol:
            self.tol = tol.value().copy()
        if shrinking:
            self.shrinking = shrinking.value().copy()
        if cache_size:
            self.cache_size = cache_size.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(nu=self.nu.copy(), C=self.C.copy(), kernel=self.kernel.copy(), degree=self.degree.copy(), gamma=self.gamma.copy(), coef0=self.coef0.copy(), tol=self.tol.copy(), shrinking=self.shrinking.copy(), cache_size=self.cache_size.copy(), verbose=self.verbose.copy(), max_iter=self.max_iter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._model)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, sample_weight: Optional[Dense]) raises -> ref[self] Self:
        var rng = self._rng.copy()
        _ = self.fit_with_random_state(X, y, rng, sample_weight)
        self._rng = rng^
        return self

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        if self.kernel == "callable":
            raise Error("ValueError: callable kernel requires fit_with_kernel")
        var snapshot = TrainingSnapshot(X, self.kernel == "precomputed")
        var gamma = self.gamma.resolve(X, self.kernel)
        var spec = KernelSpec(self.kernel.copy(), gamma, self.degree, self.coef0)
        spec.validate()
        var weights = _weights(X.shape[0], sample_weight)
        validate_vector(y, X.shape[0])
        var target = y.copy()
        var costs = nm.zeros[DType.float64](nm.Shape(0))
        var options = KernelOptions("nu_svr", spec.copy(), self.C, self.nu, 0.0, self.tol, self.cache_size, self.shrinking, False, self.max_iter, rng.randint(2147483647))
        var engine = LibSVM()
        var model = engine.solve(X, target, weights, costs, options)
        self.support_ = public_support(model, weights)
        self.support_vectors_ = model.support_vectors.copy()
        self.n_support_ = support_counts(model)
        self.dual_coef_ = model.dual_coef.copy()
        self.intercept_ = model.rho * Float64(-1)
        self.n_iter_ = Int(model.iterations.unsafe_get(0))
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = nm.Shape(X.shape[0], X.shape[1])
        self.fit_status_ = model.fit_status
        self._gamma = gamma
        self._kernel = spec^
        self._model = model^
        self._snapshot = snapshot^
        if self.fit_status_ != 0:
            print("ConvergenceWarning: LIBSVM reached max_iter")
        if self.verbose:
            print("[LibSVM] support vectors:", self.support_.shape[0])
        return self

    def _raw_decisions(self, X: Dense) raises -> Dense:
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        if self.kernel == "callable":
            raise Error("ValueError: callable inference requires a KernelProvider")
        self._snapshot.value().validate_query(X)
        var engine = LibSVM()
        return engine.decision_values(X, self._model.value(), self._kernel)

    def coef_(self) raises -> Dense:
        # sklearn's property checks the current kernel parameter before fitted state.
        if self.kernel != "linear":
            raise Error("AttributeError: coef_ is only available when using a linear kernel")
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        return linear_coefficients(self._model.value(), self._kernel, 0)

    def decision_function(self, X: Dense) raises -> Dense:
        var result = self._raw_decisions(X)
        return result.reshape(nm.Shape(X.shape[0]))

    def predict(self, X: Dense) raises -> Self.Prediction:
        return self.decision_function(X)

    def score(self, X: Dense, y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X)
        var weights = _weights(X.shape[0], sample_weight)
        validate_vector(y, X.shape[0])
        if X.shape[0] < 2:
            return Float64("nan")
        var total = Float64(0)
        var mean = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            total += w
            mean += w * y.unsafe_get(i * Int(y.strides[0]))
        if total == 0:
            raise Error("ValueError: weights sum to zero")
        mean /= total
        var residual = Float64(0)
        var variance = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            var actual = y.unsafe_get(i * Int(y.strides[0]))
            var error = actual - prediction.unsafe_get(i)
            residual += w * error * error
            error = actual - mean
            variance += w * error * error
        if residual == 0:
            return 1.0
        if variance == 0:
            return 0.0
        return 1.0 - residual / variance


    def score[Provider: KernelProvider](self, X: Dense, y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X, provider)
        var weights = _weights(X.shape[0], sample_weight)
        validate_vector(y, X.shape[0])
        if X.shape[0] < 2:
            return Float64("nan")
        var total = Float64(0)
        var mean = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            total += w
            mean += w * y.unsafe_get(i * Int(y.strides[0]))
        if total == 0:
            raise Error("ValueError: weights sum to zero")
        mean /= total
        var residual = Float64(0)
        var variance = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            var actual = y.unsafe_get(i * Int(y.strides[0]))
            var error = actual - prediction.unsafe_get(i)
            residual += w * error * error
            error = actual - mean
            variance += w * error * error
        if residual == 0:
            return 1.0
        if variance == 0:
            return 0.0
        return 1.0 - residual / variance


    def score(self, X: nm.NDArray[DType.float32], y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, sample_weight)

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        if self.kernel != "callable":
            raise Error("ValueError: fit_with_kernel requires kernel=callable")
        var snapshot = TrainingSnapshot(X)
        var gram = provider.evaluate(X, snapshot.x)
        validate_matrix(gram, X.shape[0])
        if gram.shape[0] != X.shape[0]:
            raise Error("ValueError: callable training kernel has wrong row count")
        var candidate = self.__sklearn_clone__()
        candidate._rng = self._rng.copy()
        candidate.kernel = "precomputed"
        _ = candidate.fit(gram, y, sample_weight)
        candidate.kernel = "callable"
        candidate._snapshot = snapshot^
        candidate.n_features_in_ = X.shape[1]
        candidate.shape_fit_ = nm.Shape(X.shape[0], X.shape[1])
        self = candidate^
        return self

    def _cross_kernel[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        if not self._model:
            raise Error("NotFittedError: estimator is not fitted")
        if self.kernel != "callable":
            raise Error("ValueError: provider requires fitted callable kernel")
        self._snapshot.value().validate_query(X)
        var result = provider.evaluate(X, self._snapshot.value().x)
        validate_matrix(result, self._snapshot.value().n_train)
        if result.shape[0] != X.shape[0]:
            raise Error("ValueError: callable query kernel has wrong row count")
        return result^

    def _raw_with_kernel[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        var gram = self._cross_kernel(X, provider)
        var engine = LibSVM()
        return engine.decision_values(gram, self._model.value(), self._kernel)

    def decision_function[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        var result = self._raw_with_kernel(X, provider)
        return result.reshape(nm.Shape(X.shape[0]))


    def predict[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Self.Prediction:
        return self.decision_function(X, provider)


    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y, sample_weight)
        return self

    def predict(self, X: nm.NDArray[DType.float32]) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64]())

    def decision_function(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.decision_function(X.astype[DType.float64]())

    def fit[dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: nm.NDArray[dtype], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y.astype[DType.float64](), sample_weight)
        return self

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit_with_kernel(X.astype[DType.float64](), y, provider, sample_weight)
        return self

    def predict[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64](), provider)

    def decision_function[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.decision_function(X.astype[DType.float64](), provider)

    def score[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], y: Self.Target, provider: Provider, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, provider, sample_weight)
