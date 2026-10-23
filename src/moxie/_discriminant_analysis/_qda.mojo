# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""QuadraticDiscriminantAnalysis: per-class 'svd' and 'eigen' factorizations."""
from std.math import log, sqrt
from numojo import NDArray, Shape, zeros
from moxie.utils.validation import FittedState
from moxie._feature_metadata import InputFeatureCount
from moxie._discriminant_analysis_contracts import DiscriminantClassifier
from ._labels import ClassLabel, sorted_unique, encode_labels, class_counts
from ._work import (
    Work,
    get,
    put,
    matrix,
    vector,
    input_matrix,
    cast_array,
    widen,
    checked_svd,
    checked_eigh,
)
from ._covariance import (
    Shrinkage,
    CovarianceEstimator,
    NoCovarianceEstimator,
    estimate_covariance,
    column_means,
    centered,
    class_rows,
)
from ._lda import descending_order
from ._prediction import (
    row_argmax,
    log_softmax_rows,
    exp_array,
    select_labels,
    accuracy,
)

comptime ESTIMATOR = "QuadraticDiscriminantAnalysis"


@fieldwise_init
struct QuadraticDiscriminantAnalysisParams[C: CovarianceEstimator](Copyable, Movable):
    """Typed `get_params`/`set_params` record; constructor configuration only."""

    var solver: String
    var shrinkage: Shrinkage
    var priors: Optional[List[Float64]]
    var reg_param: Float64
    var store_covariance: Bool
    var tol: Float64
    var covariance_estimator: Optional[Self.C]


@fieldwise_init
struct _ClassFactor(Movable):
    """Per-class variances along principal axes, the axes, and optional covariance."""

    var scaling: Work
    var rotation: Work
    var covariance: Optional[Work]


def svd_factor(Xc: Work, reg_param: Float64, store_covariance: Bool) raises -> _ClassFactor:
    """QDA `_solve_svd`: scaling = (1-r) S^2/n + r, rotation = V."""
    var n = Xc.shape[0]
    var p = Xc.shape[1]
    var factor = checked_svd(Xc, ESTIMATOR + "(solver='svd') class SVD")
    var s = factor.singular_values()
    var vt = factor.vt()
    var k = s.shape[0]
    var scaling = vector(k)
    for i in range(k):
        var value = s.unsafe_get(i) * s.unsafe_get(i) / Float64(n)
        scaling.unsafe_set(i, (1.0 - reg_param) * value + reg_param)
    var rotation = matrix(p, k)
    for j in range(p):
        for i in range(k):
            put(rotation, j, i, get(vt, i, j))
    var covariance: Optional[Work] = None
    if store_covariance:
        # V diag(scaling) V^T
        var cov = matrix(p, p)
        for a in range(p):
            for b in range(p):
                var acc: Float64 = 0
                for i in range(k):
                    acc += get(rotation, a, i) * scaling.unsafe_get(i) * get(rotation, b, i)
                put(cov, a, b, acc)
        covariance = cov^
    return _ClassFactor(scaling^, rotation^, covariance^)


def eigen_factor[
    C: CovarianceEstimator
](X_class: Work, shrinkage: Shrinkage, estimator: Optional[C]) raises -> _ClassFactor:
    """QDA `_solve_eigen`: eigh of the (shrunk/estimated) class covariance, descending."""
    var p = X_class.shape[1]
    var cov = estimate_covariance(X_class, shrinkage, estimator)
    var eig = checked_eigh(cov, ESTIMATOR + "(solver='eigen') class covariance")
    var values = eig.eigenvalues()
    var vectors = eig.eigenvectors()
    var order = descending_order(values)
    var scaling = vector(p)
    var rotation = matrix(p, p)
    for i in range(p):
        scaling.unsafe_set(i, values.unsafe_get(order[i]))
        for j in range(p):
            put(rotation, j, i, get(vectors, j, order[i]))
    return _ClassFactor(scaling^, rotation^, cov^)


struct QuadraticDiscriminantAnalysis[
    dtype: DType = DType.float64,
    L: ClassLabel = Int,
    C: CovarianceEstimator = NoCovarianceEstimator,
](DiscriminantClassifier, InputFeatureCount):
    """Quadratic Discriminant Analysis (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis).

    Fits one Gaussian per class and classifies with Bayes' rule (quadratic boundary).

    Parameters:
        dtype: Float32 or Float64 input/output dtype (internal work is Float64).
        L: Class-label type (e.g. Int or String); `classes_` is sorted ascending.
        C: Native covariance estimator type used when `covariance_estimator` is set.

    Keyword-only constructor arguments keep sklearn's names and defaults:
    solver='svd', shrinkage=None, priors=None, reg_param=0.0,
    store_covariance=False, tol=1e-4, covariance_estimator=None.

    Learned attributes: covariance_ (Optional list), means_, priors_,
    rotations_ (list of (p, n_k)), scalings_ (list of (n_k,)), classes_, n_features_in_.
    """

    comptime Params = QuadraticDiscriminantAnalysisParams[Self.C]
    comptime Input = NDArray[Self.dtype]
    comptime Target = List[Self.L]
    comptime Prediction = List[Self.L]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Probabilities = NDArray[Self.dtype]
    comptime Decision = NDArray[Self.dtype]

    var solver: String
    var shrinkage: Shrinkage
    var priors: Optional[List[Float64]]
    var reg_param: Float64
    var store_covariance: Bool
    var tol: Float64
    var covariance_estimator: Optional[Self.C]

    var covariance_: Optional[List[NDArray[Self.dtype]]]
    var means_: NDArray[Self.dtype]
    var priors_: NDArray[Self.dtype]
    var rotations_: List[NDArray[Self.dtype]]
    var scalings_: List[NDArray[Self.dtype]]
    var classes_: List[Self.L]
    var n_features_in_: Int
    var _state: FittedState

    def __init__(
        out self,
        *,
        solver: String = "svd",
        shrinkage: Shrinkage = None,
        priors: Optional[List[Float64]] = None,
        reg_param: Float64 = 0.0,
        store_covariance: Bool = False,
        tol: Float64 = 1.0e-4,
        covariance_estimator: Optional[Self.C] = None,
    ) raises:
        """Store configuration only; validation happens in `fit` like sklearn."""
        self.solver = solver
        self.shrinkage = shrinkage.copy()
        self.priors = priors.copy()
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator.copy()
        self.covariance_ = None
        self.means_ = zeros[Self.dtype](Shape(0, 0))
        self.priors_ = zeros[Self.dtype](Shape(0))
        self.rotations_ = List[NDArray[Self.dtype]]()
        self.scalings_ = List[NDArray[Self.dtype]]()
        self.classes_ = List[Self.L]()
        self.n_features_in_ = 0
        self._state = FittedState()

    # ----- configuration -------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        """Return an independent copy of the constructor configuration."""
        return Self.Params(
            self.solver,
            self.shrinkage.copy(),
            self.priors.copy(),
            self.reg_param,
            self.store_covariance,
            self.tol,
            self.covariance_estimator.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        """Replace the whole configuration; learned state is left untouched."""
        self.solver = params.solver
        self.shrinkage = params.shrinkage.copy()
        self.priors = params.priors.copy()
        self.reg_param = params.reg_param
        self.store_covariance = params.store_covariance
        self.tol = params.tol
        self.covariance_estimator = params.covariance_estimator.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        solver: Optional[String] = None,
        shrinkage: Optional[Shrinkage] = None,
        priors: Optional[Optional[List[Float64]]] = None,
        reg_param: Optional[Float64] = None,
        store_covariance: Optional[Bool] = None,
        tol: Optional[Float64] = None,
        covariance_estimator: Optional[Optional[Self.C]] = None,
    ) raises -> ref[self] Self:
        """Keyword partial update; an outer None leaves that parameter unchanged."""
        if solver:
            self.solver = solver.value()
        if shrinkage:
            self.shrinkage = shrinkage.value().copy()
        if priors:
            self.priors = priors.value().copy()
        if reg_param:
            self.reg_param = reg_param.value()
        if store_covariance:
            self.store_covariance = store_covariance.value()
        if tol:
            self.tol = tol.value()
        if covariance_estimator:
            self.covariance_estimator = covariance_estimator.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        """Unfitted estimator with an independent copy of the configuration."""
        return Self(
            solver=self.solver,
            shrinkage=self.shrinkage.copy(),
            priors=self.priors.copy(),
            reg_param=self.reg_param,
            store_covariance=self.store_covariance,
            tol=self.tol,
            covariance_estimator=self.covariance_estimator.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` through the shared InputFeatureCount contract; raises before fit."""
        if not self._state.is_fitted():
            raise Error(
                "NotFittedError: This QuadraticDiscriminantAnalysis instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.n_features_in_

    def _validate_params(self) raises:
        if self.solver != "svd" and self.solver != "eigen":
            raise Error(
                "ValueError: The 'solver' parameter of QuadraticDiscriminantAnalysis must be a"
                " str among {'eigen', 'svd'}. Got '" + self.solver + "' instead."
            )
        self.shrinkage.validate(ESTIMATOR)
        if not (self.reg_param >= 0 and self.reg_param <= 1):
            raise Error(
                String("ValueError: The 'reg_param' parameter of QuadraticDiscriminantAnalysis")
                + String(" must be a float in the range [0, 1]. Got ", self.reg_param, " instead.")
            )
        if not (self.tol >= 0):
            raise Error(
                String("ValueError: The 'tol' parameter of QuadraticDiscriminantAnalysis must be")
                + String(" a float in the range [0, inf). Got ", self.tol, " instead.")
            )

    def _reset(mut self) raises:
        self.covariance_ = None
        self.means_ = zeros[Self.dtype](Shape(0, 0))
        self.priors_ = zeros[Self.dtype](Shape(0))
        self.rotations_ = List[NDArray[Self.dtype]]()
        self.scalings_ = List[NDArray[Self.dtype]]()
        self.classes_ = List[Self.L]()
        self.n_features_in_ = 0
        self._state.reset()

    # ----- fitting -------------------------------------------------------

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: NDArray[Self.dtype], ref[y_origin] y: List[Self.L]
    ) raises -> ref[self] Self:
        """Fit one Gaussian per class; replaces learned state (unfitted on failure)."""
        self._reset()
        self._validate_params()
        var work = input_matrix(X, ESTIMATOR)
        var n = work.shape[0]
        var p = work.shape[1]
        if len(y) != n:
            raise Error(
                String("ValueError: Found input variables with inconsistent numbers of samples: [")
                + String(n, ", ", len(y), "]")
            )
        var classes = sorted_unique(y)
        var codes = encode_labels(y, classes)
        var counts = class_counts(codes, len(classes))
        var k = len(classes)
        if k < 2:
            raise Error(
                String("ValueError: The number of classes has to be greater than one. Got ")
                + String(k, " class.")
            )
        var priors = vector(k)
        if self.priors:
            ref given = self.priors.value()
            if len(given) != k:
                raise Error(
                    String("ValueError: ", ESTIMATOR, " priors has ", len(given), " entries but y has ")
                    + String(k, " classes")
                )
            for c in range(k):
                priors.unsafe_set(c, given[c])
        else:
            for c in range(k):
                priors.unsafe_set(c, Float64(counts[c]) / Float64(n))
        if self.solver == "svd":
            if not self.shrinkage.is_none():
                raise Error("NotImplementedError: shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator:
                raise Error(
                    "ValueError: covariance_estimator is not supported with solver='svd'."
                    " Try solver='eigen' instead."
                )
        var means = matrix(k, p)
        var scalings = List[NDArray[Self.dtype]]()
        var rotations = List[NDArray[Self.dtype]]()
        var covariances = List[NDArray[Self.dtype]]()
        for c in range(k):
            if counts[c] == 1:
                raise Error(
                    String("ValueError: y has only 1 sample in class ", classes[c])
                    + ", covariance is ill defined."
                )
            var rows = class_rows(work, codes, c, counts[c])
            var mean = column_means(rows)
            for j in range(p):
                put(means, c, j, mean.unsafe_get(j))
            var factor: _ClassFactor
            if self.solver == "svd":
                factor = svd_factor(centered(rows, mean), self.reg_param, self.store_covariance)
            else:
                factor = eigen_factor(rows, self.shrinkage, self.covariance_estimator)
            var rank = 0
            for i in range(factor.scaling.shape[0]):
                if factor.scaling.unsafe_get(i) > self.tol:
                    rank += 1
            if rank < p:
                if self.solver == "svd" and counts[c] <= p:
                    raise Error(
                        String("LinAlgError: The covariance matrix of class ", classes[c])
                        + String(" is not full rank. When using `solver='svd'` the number of")
                        + String(" samples in each class should be more than the number of features,")
                        + String(" but class ", classes[c], " has ", counts[c], " samples and ")
                        + String(p, " features. Try using `solver='eigen'` and setting the")
                        + " parameter `shrinkage` for regularization."
                    )
                var param = "shrinkage" if self.solver == "eigen" else "reg_param"
                raise Error(
                    String("LinAlgError: The covariance matrix of class ", classes[c])
                    + String(" is not full rank. Increase the value of `", param)
                    + "` to reduce the collinearity."
                )
            scalings.append(cast_array[Self.dtype](factor.scaling))
            rotations.append(cast_array[Self.dtype](factor.rotation))
            if self.store_covariance:
                covariances.append(cast_array[Self.dtype](factor.covariance.value()))
        if self.store_covariance:
            self.covariance_ = covariances^
        self.means_ = cast_array[Self.dtype](means)
        self.priors_ = cast_array[Self.dtype](priors)
        self.scalings_ = scalings^
        self.rotations_ = rotations^
        self.classes_ = classes^
        self.n_features_in_ = p
        self._state.record_fit(n, p, n_targets=len(y))
        return self

    # ----- inference -----------------------------------------------------

    def _joint_log_likelihood(self, X: NDArray[Self.dtype]) raises -> Work:
        """Log posterior up to a constant (ESL eq. 4.12), shape (n, n_classes)."""
        if not self._state.is_fitted():
            raise Error(
                "NotFittedError: This QuadraticDiscriminantAnalysis instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        var work = input_matrix(X, ESTIMATOR)
        var n = work.shape[0]
        var p = work.shape[1]
        if p != self.n_features_in_:
            raise Error(
                String("ValueError: X has ", p, " features, but QuadraticDiscriminantAnalysis")
                + String(" is expecting ", self.n_features_in_, " features as input.")
            )
        var k = len(self.classes_)
        var scores = matrix(n, k)
        var means = widen(self.means_)
        for c in range(k):
            var rotation = widen(self.rotations_[c])
            var scaling = widen(self.scalings_[c])
            var m = scaling.shape[0]
            # Whitening matrix R * S^-1/2 and the log-determinant term sum(log S).
            var log_det: Float64 = 0
            for i in range(m):
                log_det += log(scaling.unsafe_get(i))
                var inv_root = 1.0 / sqrt(scaling.unsafe_get(i))
                for j in range(p):
                    rotation.unsafe_set(j * m + i, rotation.unsafe_get(j * m + i) * inv_root)
            var log_prior = log(Float64(self.priors_.unsafe_get(c)))
            var projected = List[Float64](length=m, fill=0.0)
            for s in range(n):
                for i in range(m):
                    projected[i] = 0.0
                for j in range(p):
                    var d = work.unsafe_get(s * p + j) - means.unsafe_get(c * p + j)
                    if d == 0:
                        continue
                    for i in range(m):
                        projected[i] += d * rotation.unsafe_get(j * m + i)
                var norm2: Float64 = 0
                for i in range(m):
                    norm2 += projected[i] * projected[i]
                scores.unsafe_set(s * k + c, -0.5 * (norm2 + log_det) + log_prior)
        return scores^

    def decision_function(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Log-likelihood ratio (n,) for two classes, else (n, n_classes) log posteriors."""
        var scores = self._joint_log_likelihood(X)
        if len(self.classes_) == 2:
            var flat = vector(scores.shape[0])
            for i in range(scores.shape[0]):
                flat.unsafe_set(i, get(scores, i, 1) - get(scores, i, 0))
            return cast_array[Self.dtype](flat)
        return cast_array[Self.dtype](scores)

    def predict(self, X: NDArray[Self.dtype]) raises -> List[Self.L]:
        """Class label with the largest posterior per sample."""
        return select_labels(self.classes_, row_argmax(self._joint_log_likelihood(X)))

    def predict_log_proba(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Normalized log posteriors (log-sum-exp), shape (n, n_classes)."""
        return cast_array[Self.dtype](log_softmax_rows(self._joint_log_likelihood(X)))

    def predict_proba(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Return exp(predict_log_proba(X)), shape (n, n_classes)."""
        return exp_array[Self.dtype](log_softmax_rows(self._joint_log_likelihood(X)))

    def score(
        self,
        X: NDArray[Self.dtype],
        y: List[Self.L],
        sample_weight: Optional[NDArray[Self.dtype]] = None,
    ) raises -> Float64:
        """Mean (optionally weighted) accuracy of `predict(X)` against y."""
        return accuracy(self.predict(X), y, sample_weight, ESTIMATOR)
