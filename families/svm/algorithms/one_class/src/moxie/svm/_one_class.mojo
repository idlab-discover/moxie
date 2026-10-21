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
struct OneClassSVMParams(Copyable, Movable):
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


struct OneClassSVM(OneClassSupportVector):
    comptime Params = OneClassSVMParams
    comptime Input = Dense
    comptime SampleWeight = Dense
    comptime Target = Dense
    comptime Prediction = Indices
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
    var offset_: Dense

    def __init__(out self, *, nu: Float64 = 0.5, kernel: String = "rbf", degree: Int = 3, gamma: Gamma = "scale", coef0: Float64 = 0.0, tol: Float64 = 1e-3, shrinking: Bool = True, cache_size: Float64 = 200.0, verbose: Bool = False, max_iter: Int = -1) raises:
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
        self.offset_ = nm.zeros[DType.float64](nm.Shape(0))

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.nu.copy(), self.kernel.copy(), self.degree.copy(), self.gamma.copy(), self.coef0.copy(), self.tol.copy(), self.shrinking.copy(), self.cache_size.copy(), self.verbose.copy(), self.max_iter.copy())

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
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, nu: Optional[Float64] = None, kernel: Optional[String] = None, degree: Optional[Int] = None, gamma: Optional[Gamma] = None, coef0: Optional[Float64] = None, tol: Optional[Float64] = None, shrinking: Optional[Bool] = None, cache_size: Optional[Float64] = None, verbose: Optional[Bool] = None, max_iter: Optional[Int] = None) raises -> ref[self] Self:
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
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(nu=self.nu.copy(), kernel=self.kernel.copy(), degree=self.degree.copy(), gamma=self.gamma.copy(), coef0=self.coef0.copy(), tol=self.tol.copy(), shrinking=self.shrinking.copy(), cache_size=self.cache_size.copy(), verbose=self.verbose.copy(), max_iter=self.max_iter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._model)

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, y: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, y: Optional[Dense], sample_weight: Optional[Dense]) raises -> ref[self] Self:
        var rng = self._rng.copy()
        _ = self.fit_with_random_state(X, rng, y, sample_weight)
        self._rng = rng^
        return self

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, mut rng: RandomState, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        if self.kernel == "callable":
            raise Error("ValueError: callable kernel requires fit_with_kernel")
        var snapshot = TrainingSnapshot(X, self.kernel == "precomputed")
        var gamma = self.gamma.resolve(X, self.kernel)
        var spec = KernelSpec(self.kernel.copy(), gamma, self.degree, self.coef0)
        spec.validate()
        var weights = _weights(X.shape[0], sample_weight)
        var target = nm.ones[DType.float64](nm.Shape(X.shape[0]))
        var costs = nm.zeros[DType.float64](nm.Shape(0))
        var options = KernelOptions("one_class", spec.copy(), 0.0, self.nu, 0.0, self.tol, self.cache_size, self.shrinking, False, self.max_iter, rng.randint(2147483647))
        var engine = LibSVM()
        var model = engine.solve(X, target, weights, costs, options)
        self.support_ = public_support(model, weights)
        self.support_vectors_ = model.support_vectors.copy()
        self.n_support_ = support_counts(model)
        self.dual_coef_ = model.dual_coef.copy()
        self.intercept_ = model.rho * Float64(-1)
        self.n_iter_ = Int(model.iterations.unsafe_get(0))
        self.offset_ = model.rho.copy()
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
        var scores = self.decision_function(X)
        var result = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for i in range(result.size):
            result.unsafe_set(i, Int64(1) if scores.unsafe_get(i) > 0 else Int64(-1))
        return result^

    def score_samples(self, X: Dense) raises -> Dense:
        return self.decision_function(X) + self.offset_.unsafe_get(0)

    def fit_predict(mut self, X: Dense, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> Indices:
        _ = self.fit(X, y, sample_weight)
        return self.predict(X)

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, provider: Provider, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
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
        var scores = self.decision_function(X, provider)
        var result = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for i in range(result.size):
            result.unsafe_set(i, Int64(1) if scores.unsafe_get(i) > 0 else Int64(-1))
        return result^


    def score_samples[Provider: KernelProvider](self, X: Dense, provider: Provider) raises -> Dense:
        return self.decision_function(X, provider) + self.offset_.unsafe_get(0)


    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y, sample_weight)
        return self

    def predict(self, X: nm.NDArray[DType.float32]) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64]())

    def decision_function(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.decision_function(X.astype[DType.float64]())

    def score_samples(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.score_samples(X.astype[DType.float64]())

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, *, sample_weight: Optional[Dense]) raises -> ref[self] Self:
        _ = self.fit(X, None, sample_weight)
        return self

    def fit_predict(mut self, X: nm.NDArray[DType.float32], y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> Indices:
        _ = self.fit(X, y, sample_weight)
        return self.predict(X)

    def fit_predict[Provider: KernelProvider](mut self, X: Dense, provider: Provider, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> Indices:
        _ = self.fit_with_kernel(X, provider, y, sample_weight)
        return self.predict(X, provider)

    def fit_with_kernel[Provider: KernelProvider, origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], provider: Provider, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit_with_kernel(X.astype[DType.float64](), provider, y, sample_weight)
        return self

    def predict[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64](), provider)

    def decision_function[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.decision_function(X.astype[DType.float64](), provider)

    def score_samples[Provider: KernelProvider](self, X: nm.NDArray[DType.float32], provider: Provider) raises -> Dense:
        return self.score_samples(X.astype[DType.float64](), provider)

    def fit_predict[Provider: KernelProvider](mut self, X: nm.NDArray[DType.float32], provider: Provider, y: Optional[Dense] = None, sample_weight: Optional[Dense] = None) raises -> Indices:
        _ = self.fit_with_kernel(X, provider, y, sample_weight)
        return self.predict(X, provider)
