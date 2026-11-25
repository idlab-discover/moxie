# Adapted from scikit-learn sklearn/linear_model/_logistic.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Dense binary and multinomial logistic regression."""

from std.math import exp, inf, isfinite, log
from numojo import NDArray, Shape, full, zeros
from moxie._random import check_random_state
from moxie._feature_metadata import InputFeatureCount
from moxie.base import NumericClassifierClasses, WeightedSupervisedEstimator
from moxie.linear_model._base import ProbabilisticLinearClassifier
from moxie.linear_model._contracts import validate_weights
from moxie.linear_model._linear_base import (
    _entry, _vector_entry, check_finite, check_predict_input, compact_matrix, compact_vector,
    linear_decision, public_prediction,
)
from moxie.linear_model._linear_loss import LogisticLoss, logistic_probability
from moxie.linear_model._logistic_solver import (
    LogisticSolution, logistic_lbfgs, logistic_newton, logistic_newton_cg, logistic_sag,
)
from moxie.linear_model._ridge_labels import (
    ClassWeight, LabelEncoding, _label, accuracy, fit_labels,
)
from moxie.svm._contracts import LinearModel, LinearOptions
from moxie.svm._liblinear import LibLinear
from moxie.utils.validation import check_is_fitted


def _check_numeric_labels[dtype: DType](y: NDArray[dtype]) raises:
    # The family label encoder and ClassWeight use signed Int keys.
    comptime if dtype.is_floating_point() or dtype == DType.uint64:
        if y.ndim != 1 and (y.ndim != 2 or y.shape[1] != 1):
            return
        for i in range(y.shape[0]):
            var value = y.unsafe_get(i * Int(y.strides[0]))
            comptime if dtype == DType.uint64:
                if value > Scalar[dtype](9223372036854775807):
                    raise Error("ValueError: LogisticRegression label exceeds the signed integer range")
            else:
                var numeric = Float64(value)
                if not isfinite(numeric):
                    raise Error("ValueError: LogisticRegression labels must be finite integers")
                if numeric < -9223372036854775808.0 or numeric >= 9223372036854775808.0:
                    raise Error("ValueError: LogisticRegression label exceeds the signed integer range")
                if Float64(Int(numeric)) != numeric:
                    raise Error("ValueError: LogisticRegression labels must be finite integers")


def _fit_weights[dtype: DType, label_dtype: DType](
    targets: List[Int], encoding: LabelEncoding[label_dtype],
    sample_weight: Optional[NDArray[dtype]], class_weight: Optional[ClassWeight],
    mut weights: NDArray[DType.float64],
) raises:
    var n = len(targets)
    if sample_weight:
        ref sw = sample_weight.value()
        validate_weights(sw, n, nonnegative=False)
        for i in range(n):
            weights.unsafe_set(i, Float64(sw.unsafe_get(i)))

    if class_weight:
        ref config = class_weight.value()
        var count = encoding.n_classes()
        var factors = List[Float64](length=count, fill=1)
        if config.balanced:
            var masses = List[Float64](length=count, fill=0)
            var total = Float64(0)
            for i in range(n):
                var weight = weights.unsafe_get(i)
                masses[targets[i]] += weight
                total += weight
            for k in range(count):
                if masses[k] == 0:
                    raise Error("ValueError: balanced class weights require nonzero mass for each class")
                factors[k] = total / (Float64(count) * masses[k])
        else:
            var matched = 0
            for k in range(count):
                var label = encoding.classes[k]
                if label in config.weights:
                    factors[k] = config.weights[label]
                    matched += 1
            if matched < count and matched != len(config.weights):
                raise Error("ValueError: class_weight contains an unknown label")
        for i in range(n):
            var factor = factors[targets[i]].cast[dtype]()
            var product = weights.unsafe_get(i).cast[dtype]() * factor
            weights.unsafe_set(i, Float64(product))

    for i in range(n):
        if not isfinite(weights.unsafe_get(i)):
            raise Error("ValueError: sample weights must be finite after class weighting")


@fieldwise_init
struct _FittedCoefficients[dtype: DType](Movable):
    var coef: NDArray[Self.dtype]
    var intercept: NDArray[Self.dtype]
    var iterations: Int


@fieldwise_init
struct LogisticRegressionParams(Copyable, Movable):
    var penalty: Optional[String]
    var C: Float64
    var l1_ratio: Optional[Float64]
    var dual: Bool
    var tol: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var class_weight: Optional[ClassWeight]
    var random_state: Optional[Int]
    var solver: String
    var max_iter: Int
    var verbose: Int
    var warm_start: Bool
    var n_jobs: Optional[Int]


struct LogisticRegression[dtype: DType = DType.float64, label_dtype: DType = DType.int64](
    Copyable, InputFeatureCount, NumericClassifierClasses,
    ProbabilisticLinearClassifier, WeightedSupervisedEstimator,
):
    """Dense logistic regression with sorted integer-valued numeric labels."""

    comptime Input = NDArray[Self.dtype]
    comptime ClassLabelDType = Self.label_dtype
    comptime Target = NDArray[Self.label_dtype]
    comptime Prediction = Self.Target
    comptime Decision = Self.Input
    comptime Probabilities = Self.Input
    comptime SampleWeight = Self.Input
    comptime Params = LogisticRegressionParams

    var penalty: Optional[String]
    var C: Float64
    var l1_ratio: Optional[Float64]
    var dual: Bool
    var tol: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var class_weight: Optional[ClassWeight]
    var random_state: Optional[Int]
    var solver: String
    var max_iter: Int
    var verbose: Int
    var warm_start: Bool
    var n_jobs: Optional[Int]
    var classes_: Self.Target
    var coef_: Self.Input
    var intercept_: Self.Input
    var n_iter_: NDArray[DType.int32]
    var n_features_in_: Int
    var _encoding: Optional[LabelEncoding[Self.label_dtype]]

    def __init__(
        out self, var penalty: Optional[String] = String("deprecated"), *,
        C: Float64 = 1.0, l1_ratio: Optional[Float64] = 0.0, dual: Bool = False,
        tol: Float64 = 1e-4, fit_intercept: Bool = True, intercept_scaling: Float64 = 1,
        var class_weight: Optional[ClassWeight] = None, random_state: Optional[Int] = None,
        var solver: String = "lbfgs", max_iter: Int = 100, verbose: Int = 0,
        warm_start: Bool = False, n_jobs: Optional[Int] = None,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime assert Self.label_dtype.is_integral() or Self.label_dtype.is_floating_point()
        self.penalty = penalty^
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight^
        self.random_state = random_state
        self.solver = solver^
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.classes_ = zeros[Self.label_dtype](Shape(0))
        self.coef_ = zeros[Self.dtype](Shape(0, 0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.n_iter_ = zeros[DType.int32](Shape(0))
        self.n_features_in_ = 0
        self._encoding = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.penalty.copy(), self.C, self.l1_ratio, self.dual, self.tol,
            self.fit_intercept, self.intercept_scaling, self.class_weight.copy(),
            self.random_state, self.solver.copy(), self.max_iter, self.verbose,
            self.warm_start, self.n_jobs,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.penalty = params.penalty.copy()
        self.C = params.C
        self.l1_ratio = params.l1_ratio
        self.dual = params.dual
        self.tol = params.tol
        self.fit_intercept = params.fit_intercept
        self.intercept_scaling = params.intercept_scaling
        self.class_weight = params.class_weight.copy()
        self.random_state = params.random_state
        self.solver = params.solver.copy()
        self.max_iter = params.max_iter
        self.verbose = params.verbose
        self.warm_start = params.warm_start
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, penalty: Optional[String] = None, C: Optional[Float64] = None,
        l1_ratio: Optional[Float64] = None, dual: Optional[Bool] = None,
        tol: Optional[Float64] = None, fit_intercept: Optional[Bool] = None,
        intercept_scaling: Optional[Float64] = None, class_weight: Optional[ClassWeight] = None,
        random_state: Optional[Int] = None, solver: Optional[String] = None,
        max_iter: Optional[Int] = None, verbose: Optional[Int] = None,
        warm_start: Optional[Bool] = None, n_jobs: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Update supplied keywords. Use the Params record to clear an optional field."""
        if penalty:
            self.penalty = penalty.copy()
        if C:
            ref value = C.value()
            self.C = value
        if l1_ratio:
            self.l1_ratio = l1_ratio
        if dual:
            ref value = dual.value()
            self.dual = value
        if tol:
            ref value = tol.value()
            self.tol = value
        if fit_intercept:
            ref value = fit_intercept.value()
            self.fit_intercept = value
        if intercept_scaling:
            ref value = intercept_scaling.value()
            self.intercept_scaling = value
        if class_weight:
            self.class_weight = class_weight.copy()
        if random_state:
            self.random_state = random_state
        if solver:
            ref value = solver.value()
            self.solver = value.copy()
        if max_iter:
            ref value = max_iter.value()
            self.max_iter = value
        if verbose:
            ref value = verbose.value()
            self.verbose = value
        if warm_start:
            ref value = warm_start.value()
            self.warm_start = value
        if n_jobs:
            self.n_jobs = n_jobs
        return self

    def __sklearn_clone__(self) raises -> Self:
        var model = Self()
        _ = model.set_params(self.get_params())
        return model^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._encoding)

    def get_classes(self) raises -> Self.Target:
        check_is_fitted(self)
        return self.classes_.copy()

    def get_n_features_in(self) raises -> Int:
        check_is_fitted(self)
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, Optional(compact_vector(sample_weight)))
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        *, ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        if X.ndim != 2:
            raise Error("ValueError: LogisticRegression requires a nonempty feature matrix")
        var weights = full[Self.dtype](Shape(X.shape[0]), sample_weight.cast[Self.dtype]())
        self._fit(X, y, Optional(weights^))
        return self

    def _regularization(self) raises -> Tuple[Float64, Float64]:
        if not (self.C > 0) or not (self.tol >= 0 and isfinite(self.tol)):
            raise Error("ValueError: LogisticRegression requires C > 0 and finite tol >= 0")
        if self.max_iter < 0 or self.verbose < 0:
            raise Error("ValueError: LogisticRegression requires max_iter and verbose >= 0")
        if not (self.intercept_scaling > 0 and isfinite(self.intercept_scaling)):
            raise Error("ValueError: intercept_scaling must be finite and positive")
        if self.random_state:
            ref seed = self.random_state.value()
            if seed < 0 or seed > 4294967295:
                raise Error("ValueError: random_state must be in [0, 2**32 - 1]")
        var liblinear = self.solver == "liblinear"
        var saga = self.solver == "saga"
        if not (
            liblinear or saga or self.solver == "sag" or self.solver == "lbfgs"
            or self.solver == "newton-cholesky" or self.solver == "newton-cg"
        ):
            raise Error(String("NotImplementedError: LogisticRegression solver ", self.solver))
        if self.dual and not liblinear:
            raise Error("ValueError: LogisticRegression requires dual=False for the selected solver")
        var ratio = self.l1_ratio.or_else(0)
        if not (ratio >= 0 and ratio <= 1):
            raise Error("ValueError: l1_ratio must lie in [0, 1]")
        var penalty = self.penalty.or_else(String("l2"))
        var C = self.C
        var unpenalized = not self.penalty
        if penalty == "deprecated":
            if not isfinite(C):
                penalty = "l2"
                unpenalized = True
            elif ratio == 0:
                penalty = "l2"
            elif ratio == 1:
                penalty = "l1"
            else:
                penalty = "elasticnet"
        if liblinear:
            if unpenalized or not isfinite(C) or (penalty != "l1" and penalty != "l2"):
                raise Error("ValueError: liblinear requires finite C and an L1 or L2 penalty")
            if self.dual and penalty == "l1":
                raise Error("ValueError: liblinear requires L2 regularization when dual=True")
        elif unpenalized:
            C = inf[DType.float64]()
        elif saga and penalty == "elasticnet":
            if not self.l1_ratio:
                raise Error("ValueError: elasticnet requires l1_ratio")
            return C, ratio
        elif saga and penalty == "l1":
            return C, 1.0
        elif penalty != "l2":
            raise Error("ValueError: LogisticRegression supports L2 or no penalty for the selected solver")
        return C, 1.0 if penalty == "l1" else 0.0

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        var C, ratio = self._regularization()
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: LogisticRegression requires a nonempty feature matrix")
        if (y.ndim != 1 and (y.ndim != 2 or y.shape[1] != 1)) or y.shape[0] != X.shape[0]:
            raise Error("ValueError: LogisticRegression requires one class label per sample")
        check_finite(X, "X", "LogisticRegression")
        _check_numeric_labels(y)
        var encoding = fit_labels(y, "LogisticRegression")
        var classes = encoding.n_classes()
        if classes < 2:
            raise Error("ValueError: LogisticRegression requires at least two classes")
        var liblinear = self.solver == "liblinear"
        if liblinear and classes != 2:
            raise Error("ValueError: liblinear supports binary classification only")
        var targets = List[Int](capacity=X.shape[0])
        for i in range(X.shape[0]):
            targets.append(encoding.index(_label(y, i, 0)))
        var weights = full[DType.float64](Shape(X.shape[0]), 1)
        _fit_weights(targets, encoding, sample_weight, self.class_weight, weights)
        var fitted: _FittedCoefficients[Self.dtype]
        if liblinear:
            fitted = self._fit_liblinear(X, targets, weights, C, ratio == 1)
        else:
            fitted = self._fit_native(X, targets^, weights^, classes, C, ratio)
        var iterations = full[DType.int32](Shape(1), Int32(fitted.iterations))
        var labels = encoding.classes_array()
        check_finite(fitted.coef, "coef_", "LogisticRegression")
        check_finite(fitted.intercept, "intercept_", "LogisticRegression")
        swap(self.coef_, fitted.coef)
        swap(self.intercept_, fitted.intercept)
        self.n_iter_ = iterations^
        self.classes_ = labels^
        self.n_features_in_ = X.shape[1]
        self._encoding = encoding^

    def _fit_liblinear(
        self, X: Self.Input, targets: List[Int], weights: NDArray[DType.float64],
        C: Float64, l1: Bool,
    ) raises -> _FittedCoefficients[Self.dtype]:
        var rows = X.shape[0]
        var columns = X.shape[1]
        for i in range(rows):
            for j in range(columns):
                if _entry(X, i, j) > Scalar[Self.dtype](1e30):
                    raise Error("ValueError: liblinear requires the maximum feature value to be <= 1e30")
        var labels = zeros[DType.float64](Shape(rows))
        for i in range(rows):
            labels.unsafe_set(i, Float64(targets[i]))
        var class_costs = full[DType.float64](Shape(2), 1)
        var solver_id = 6 if l1 else (7 if self.dual else 0)
        var seed: Int
        with check_random_state(self.random_state) as rng:
            seed = rng[].randint(2147483647)
        var options = LinearOptions(
            solver_id, C, 0.0, self.tol, self.fit_intercept, self.intercept_scaling, self.max_iter, seed,
        )
        var engine = LibLinear()
        var model: LinearModel
        comptime if Self.dtype == DType.float64:
            model = engine.solve(rebind[NDArray[DType.float64]](X), labels, weights, class_costs, options)
        else:
            # NuMojo astype traverses physical storage. Read the logical layout
            # so Float32 sliced and reversed views widen correctly.
            var widened = zeros[DType.float64](Shape(rows, columns))
            for i in range(rows):
                for j in range(columns):
                    widened.unsafe_set(i * columns + j, Float64(_entry(X, i, j)))
            model = engine.solve(widened, labels, weights, class_costs, options)
        return _FittedCoefficients(
            model.coef.astype[Self.dtype](), model.intercept.astype[Self.dtype](),
            Int(model.iterations.unsafe_get(0)),
        )

    def _fit_native(
        self, X: Self.Input, var targets: List[Int], var weights: NDArray[DType.float64],
        classes: Int, C: Float64, ratio: Float64,
    ) raises -> _FittedCoefficients[Self.dtype]:
        var features = X.shape[1]
        var outputs = 1 if classes == 2 else classes
        if features + Int(self.fit_intercept) > Int.MAX // 8 // outputs:
            raise Error("ValueError: LogisticRegression coefficient size overflows")
        var intercept_start = features * outputs
        var size = intercept_start + (outputs if self.fit_intercept else 0)
        var initial = zeros[DType.float64](Shape(size))
        if self.warm_start and self.__sklearn_is_fitted__():
            if self.coef_.ndim != 2 or self.coef_.shape[0] != outputs or self.coef_.shape[1] != features:
                raise Error("ValueError: warm-start coefficients do not match the new fit shape")
            for j in range(features):
                for k in range(outputs):
                    initial.unsafe_set(j * outputs + k, Float64(_entry(self.coef_, k, j)))
            if self.fit_intercept:
                if self.intercept_.ndim != 1 or self.intercept_.shape[0] != outputs:
                    raise Error("ValueError: warm-start intercepts do not match the new fit shape")
                for k in range(outputs):
                    initial.unsafe_set(intercept_start + k, Float64(_vector_entry(self.intercept_, k)))

        var newton = self.solver == "newton-cholesky"
        var saga = self.solver == "saga"
        var stochastic = saga or self.solver == "sag"
        var l2 = Float64(0)
        var solution: LogisticSolution
        if stochastic:
            var seed: UInt32
            with check_random_state(self.random_state) as rng:
                seed = UInt32(rng[].randint(1, 2147483647))
            solution = logistic_sag(
                compact_matrix(X), targets, weights, initial^, outputs, self.fit_intercept,
                (1 - ratio) / C, ratio / C, self.max_iter, self.tol, seed, saga,
            )
        else:
            var mass = Float64(0)
            for i in range(len(targets)):
                mass += weights.unsafe_get(i)
            if not isfinite(mass) or mass <= 0:
                raise Error("ValueError: logistic regression requires finite positive total weight")
            for i in range(len(targets)):
                weights.unsafe_set(i, weights.unsafe_get(i) / mass)
            l2 = (1 / mass) / C if C < 1 else (1 / C) / mass
            if not isfinite(l2):
                raise Error("ValueError: LogisticRegression L2 strength overflows")
            var objective = LogisticLoss(
                compact_matrix(X), targets^, weights^, outputs, l2, self.fit_intercept, newton and self.max_iter > 0,
            )
            if newton:
                solution = logistic_newton(
                    objective, initial^, outputs, features, self.fit_intercept, l2, self.max_iter, self.tol,
                )
            elif self.solver == "newton-cg":
                solution = logistic_newton_cg(objective, initial^, outputs, self.max_iter, self.tol)
            else:
                solution = logistic_lbfgs(objective, initial, self.max_iter, self.tol)

        ref fitted = solution.coefficients
        var coef = zeros[Self.dtype](Shape(outputs, features))
        var intercept = zeros[Self.dtype](Shape(outputs))
        for j in range(features):
            var mean = Float64(0)
            if newton and l2 == 0 and outputs > 1:
                for k in range(outputs):
                    mean += fitted.unsafe_get(j * outputs + k) / Float64(outputs)
            for k in range(outputs):
                var value = fitted.unsafe_get(j * outputs + k) - mean
                coef.unsafe_set(k * features + j, value.cast[Self.dtype]())
        if self.fit_intercept:
            var mean = Float64(0)
            if outputs > 1 and not stochastic:
                for k in range(outputs):
                    mean += fitted.unsafe_get(intercept_start + k) / Float64(outputs)
            for k in range(outputs):
                var value = fitted.unsafe_get(intercept_start + k) - mean
                intercept.unsafe_set(k, value.cast[Self.dtype]())

        return _FittedCoefficients(coef^, intercept^, solution.iterations)

    def _decision_matrix(self, X: Self.Input) raises -> Self.Decision:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "LogisticRegression")
        return linear_decision(X, self.coef_, self.intercept_)

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        var decision = self._decision_matrix(X)
        return public_prediction(decision, decision.shape[1] == 1)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var decision = self._decision_matrix(X)
        return self._encoding.value().labels_from_decision(decision)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var decision = self._decision_matrix(X)
        var rows = decision.shape[0]
        var columns = decision.shape[1]
        if columns == 1:
            var result = zeros[Self.dtype](Shape(rows, 2))
            for i in range(rows):
                var positive = logistic_probability(Float64(decision.unsafe_get(i))).cast[Self.dtype]()
                result.unsafe_set(i * 2, 1 - positive)
                result.unsafe_set(i * 2 + 1, positive)
            return result^

        for i in range(rows):
            var row = i * columns
            var peak = decision.unsafe_get(row)
            for k in range(1, columns):
                peak = max(peak, decision.unsafe_get(row + k))
            var total = Float64(0)
            for k in range(columns):
                var value = exp(Float64(decision.unsafe_get(row + k)) - Float64(peak)).cast[Self.dtype]()
                decision.unsafe_set(row + k, value)
                total += Float64(value)
            for k in range(columns):
                decision.unsafe_set(row + k, (Float64(decision.unsafe_get(row + k)) / total).cast[Self.dtype]())
        return decision^

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var probability = self.predict_proba(X)
        for i in range(probability.size):
            probability.unsafe_set(i, log(Float64(probability.unsafe_get(i))).cast[Self.dtype]())
        return probability^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var predicted = self.predict(X)
        _check_numeric_labels(y)
        return accuracy(y, predicted, sample_weight)
