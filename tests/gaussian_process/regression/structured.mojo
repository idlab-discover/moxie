# Gallery fixture adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Structured GPR contracts and native values for the pinned sklearn oracle."""
from numojo import Shape, zeros
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises

from moxie.base import clone
from moxie.gaussian_process import (
    KernelContract, KernelEvaluation, StructuredGaussianProcessRegressor,
)
from moxie.gaussian_process.kernels import Hyperparameter
from moxie.gaussian_process._kernel_math import fexp, flog
from moxie.gaussian_process._plumbing import Dense64

# The pinned gallery's sequence kernel, with the derivative in log coordinates.
@fieldwise_init
struct SequenceKernel(KernelContract):
    comptime Input = List[String]
    comptime Params = Tuple[Float64, Bool, Int]

    var similarity: Float64
    var fixed: Bool
    var malformed: Int

    @staticmethod
    def empty_input() raises -> Self.Input:
        return []

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return (self.similarity, self.fixed, self.malformed)

    def set_params(mut self, params: Self.Params) raises:
        self.similarity, self.fixed, self.malformed = params

    def hyperparameters(self) raises -> List[Hyperparameter]:
        var bounds = self.bounds()
        if not self.fixed:
            bounds.unsafe_set(0, 0.1)
            bounds.unsafe_set(1, 0.9)
        return [Hyperparameter("baseline_similarity", 1, self.fixed, bounds)]

    def theta(self) raises -> Dense64:
        var theta = zeros[DType.float64](Shape(0 if self.fixed else 1))
        if not self.fixed:
            theta.unsafe_set(0, flog(self.similarity))
        return theta^

    def bounds(self) raises -> Dense64:
        var bounds = zeros[DType.float64](Shape(0 if self.fixed else 1, 2))
        if not self.fixed:
            bounds.unsafe_set(0, flog(0.1))
            bounds.unsafe_set(1, flog(0.9))
        return bounds^

    def set_theta(mut self, theta: Dense64) raises:
        if theta.shape != Shape(0 if self.fixed else 1):
            raise Error("invalid theta shape")
        if not self.fixed:
            self.similarity = fexp(theta.item(0))

    def clone_with_theta(self, theta: Dense64) raises -> Self:
        var kernel = self.copy()
        kernel.set_theta(theta)
        return kernel^

    def _pair(self, left: String, right: String) -> Tuple[Float64, Float64]:
        var matches = 0
        var pairs = 0
        for a in left.codepoints():
            for b in right.codepoints():
                pairs += 1
                matches += Int(a == b)
        var mismatch = self.similarity * Float64(pairs - matches)
        return (Float64(matches) + mismatch, mismatch)

    def evaluate(
        self, X: Self.Input, Y: Optional[Self.Input] = None, eval_gradient: Bool = False,
    ) raises -> KernelEvaluation:
        if Y and eval_gradient:
            raise Error("gradients require Y=None")
        var right = Y.value().copy() if Y else X.copy()
        var rows = len(X)
        var cols = len(right)
        if self.malformed == 1:
            return KernelEvaluation(covariance=zeros[DType.float64](Shape(rows)), gradient=None)
        var covariance = zeros[DType.float64](Shape(rows, cols))
        var gradient = zeros[DType.float64](Shape(rows, cols, Int(eval_gradient and not self.fixed)))
        for i in range(rows):
            for j in range(cols):
                var value, derivative = self._pair(X[i], right[j])
                covariance.unsafe_set(i * cols + j, value)
                if eval_gradient and not self.fixed:
                    gradient.unsafe_set(i * cols + j, derivative)
        if eval_gradient and self.malformed != 2:
            return KernelEvaluation(covariance=covariance^, gradient=gradient^)
        return KernelEvaluation(covariance=covariance^, gradient=None)

    def diag(self, X: Self.Input) raises -> Dense64:
        if self.malformed == 3:
            return zeros[DType.float64](Shape(len(X), 1))
        var diagonal = zeros[DType.float64](Shape(len(X)))
        for i in range(len(X)):
            diagonal.unsafe_set(i, self._pair(X[i], X[i])[0])
        return diagonal^

    def is_stationary(self) -> Bool:
        return False

    def requires_vector_input(self) -> Bool:
        return False


def vector(values: List[Float64]) raises -> Dense64:
    var result = zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def emit(case_name: String, key: String, values: Dense64) raises:
    var dims = List[String]()
    var data = List[String]()
    for axis in range(values.ndim):
        dims.append(String(values.shape[axis]))
    for i in range(values.size):
        data.append(String(values.unsafe_get(i)))
    print(case_name, key, ",".join(dims), " ".join(data), sep="|")


def parity_case(fixed: Bool) raises:
    comptime Model = StructuredGaussianProcessRegressor[SequenceKernel]
    var name = String("fixed") if fixed else String("optimized")
    var train: List[String] = ["AGCT", "AGC", "TAA", "AAA"]
    var all_samples: List[String] = ["AGCT", "AGC", "AACT", "TAA", "AAA", "GAACA"]
    var targets = vector([1.0, 1.0, 2.0, 3.0])
    var model = Model(SequenceKernel(0.5, fixed, 0), alpha=0.05, normalize_y=not fixed,
                      n_restarts_optimizer=2, random_state=31)
    emit(name, "gram", model.kernel.value().evaluate(all_samples).covariance)
    var prior = model.predict(all_samples, return_std=True)
    emit(name, "prior_mean", prior.mean)
    emit(name, "prior_std", prior.std.value())
    _ = model.fit(train, targets)
    emit(name, "theta", model.kernel_.theta())
    emit(name, "L", model.L_)
    emit(name, "alpha", model.alpha_)
    emit(name, "lml", vector([model.log_marginal_likelihood().value]))
    var posterior = model.predict(all_samples, return_std=True)
    emit(name, "mean", posterior.mean)
    emit(name, "std", posterior.std.value())
    emit(name, "cov", model.predict(all_samples, return_std=False, return_cov=True).covariance.value())
    emit(name, "score", vector([model.score(train, targets)]))
    var lml = model.log_marginal_likelihood(model.kernel_.theta(), eval_gradient=True)
    emit(name, "gradient", lml.gradient.value())
    assert_equal(model.sample_y(all_samples, 3).shape, Shape(len(all_samples), 3))

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.kernel.value().similarity, 0.5)
    var fitted_copy = model.copy()
    train[0] = "C"
    assert_equal(model.X_train_[0], "AGCT")
    assert_almost_equal(model.predict(all_samples).item(0), posterior.mean.item(0))
    _ = model.fit(train, targets)
    assert_equal(model.X_train_[0], "C")
    assert_equal(fitted_copy.X_train_[0], "AGCT")
    assert_almost_equal(fitted_copy.predict(all_samples).item(0), posterior.mean.item(0))
    with assert_raises(contains="at least one structured sample"):
        _ = model.fit(List[String](), targets)
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = model.fit(all_samples, targets)
    with assert_raises(contains="at least one structured sample"):
        _ = model.predict(List[String]())
    assert_equal(model.X_train_[0], "C")
    with assert_raises(contains="explicit non-vector kernel"):
        _ = model.set_params(kernel=Optional[SequenceKernel](None))
    assert_equal(model.kernel.value().similarity, 0.5)


def multi_target() raises:
    comptime Model = StructuredGaussianProcessRegressor[SequenceKernel, DType.float32]
    var X: List[String] = ["A", "C"]
    var y = zeros[DType.float32](Shape(2, 2))
    y.unsafe_set(0, 1.0)
    y.unsafe_set(3, 2.0)
    var model = Model(SequenceKernel(0.5, False, 0), optimizer=None, n_targets=2)
    _ = model.fit(X, y)
    assert_equal(model.predict(X).shape, Shape(2, 2))
    assert_equal(model.sample_y(X, 3).shape, Shape(2, 2, 3))
    assert_equal(model.kernel_.similarity, 0.5)


def invalid_kernel_outputs() raises:
    comptime Model = StructuredGaussianProcessRegressor[SequenceKernel]
    var X: List[String] = ["A", "AA"]
    var y = vector([1.0, 2.0])
    var bad_covariance = Model(SequenceKernel(0.5, True, 1))
    with assert_raises(contains="expected kernel covariance shape"):
        _ = bad_covariance.fit(X, y)
    var missing_gradient = Model(SequenceKernel(0.5, False, 2))
    with assert_raises(contains="did not return a requested gradient"):
        _ = missing_gradient.fit(X, y)
    var bad_diagonal = Model(SequenceKernel(0.5, True, 3))
    with assert_raises(contains="expected kernel diagonal shape"):
        _ = bad_diagonal.predict(X, return_std=True)


def main() raises:
    parity_case(True)
    parity_case(False)
    multi_target()
    invalid_kernel_outputs()
    print("PASS structured GPR contracts")
