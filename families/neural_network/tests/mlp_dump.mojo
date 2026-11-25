"""Public estimator fixtures for mlp_parity.py."""

from numojo import NDArray, Shape, zeros
from moxie.neural_network import MLPClassifier, MLPRegressor
from moxie.warnings import FitWarning


def emit[dtype: DType](values: NDArray[dtype]) raises:
    print(values.ndim)
    for axis in range(values.ndim):
        print(values.shape[axis])
    for i in range(values.size):
        print(Float64(values.unsafe_get(i)))


def emit_model[dtype: DType](
    counts: Tuple[Int, Int, Int, Int, Int], activation: String,
    loss: Float64, best_loss: Optional[Float64], weighted_score: Float64,
    var loss_curve: Optional[List[Float64]], coefs: List[NDArray[dtype]],
    intercepts: List[NDArray[dtype]], warnings: List[FitWarning],
) raises:
    print(counts[0], counts[1], counts[2], counts[3], counts[4], activation)
    print(loss, best_loss.or_else(0), weighted_score)
    var curve = (loss_curve^).or_else(List[Float64]())
    print(len(curve))
    for value in curve:
        print(value)
    for coef in coefs:
        emit(coef)
    for bias in intercepts:
        emit(bias)
    print(len(warnings))
    for warning in warnings:
        print(warning.category)
        # JSON-style escaping keeps multiline L-BFGS diagnostics on one line.
        print(warning.message.replace("\n", "\\n"))


def run_cases[dtype: DType]() raises:
    var X = zeros[dtype](Shape(8, 2))
    var weights = zeros[dtype](Shape(8))
    for row in range(8):
        X.unsafe_set(row * 2, Scalar[dtype](row - 3) / 4)
        X.unsafe_set(row * 2 + 1, Scalar[dtype](row * row % 7 - 3) / 4)
        weights.unsafe_set(row, Scalar[dtype](row % 3 + 1) / 2)

    comptime solvers = ("lbfgs", "sgd", "adam")
    comptime for i in range(3):
        comptime solver = solvers[i]
        comptime parameter_dtype = DType.float64 if solver == "lbfgs" else dtype
        for weighted_case in range(2):
            var weighted = weighted_case == 1
            for classes in range(1, 4):
                var y = zeros[DType.int64](Shape(8, 1))
                for row in range(8):
                    y.unsafe_set(row, Int64(3 * (row % classes) - 4))
                var clf = MLPClassifier[dtype](
                    [4, 3], "tanh", solver=solver, alpha=0.03,
                    batch_size=3, max_iter=6, random_state=42, tol=0.0,
                )
                if weighted:
                    _ = clf.fit(X, y, sample_weight=weights)
                else:
                    _ = clf.fit(X, y)
                print(clf.parameter_dtype() == DType.float64)
                emit_model(
                    (clf.n_iter_(), clf.t_(), clf.n_outputs_(), clf.n_layers_(),
                     clf.get_n_features_in()), clf.out_activation_(), clf.loss_(),
                    clf.best_loss_(), clf.score(X, y, Optional(weights.copy())),
                    clf.loss_curve_(), clf.coefs_[parameter_dtype](),
                    clf.intercepts_[parameter_dtype](), clf.fit_warnings,
                )
                emit(clf.classes_())
                emit(clf.predict(X))
                emit(clf.predict_proba(X))
                emit(clf.predict_log_proba(X))
                comptime if solver == "lbfgs" and dtype == DType.float32:
                    emit(clf.predict_proba[DType.float64](X))
                    emit(clf.predict_log_proba[DType.float64](X))

            for kind in range(3):
                var width = 2 if kind == 1 else 1
                var y = zeros[dtype](Shape(8, width) if width > 1 else Shape(8))
                for row in range(8):
                    var difference = X.unsafe_get(row * 2) - X.unsafe_get(row * 2 + 1)
                    for column in range(width):
                        y.unsafe_set(row * width + column, 1 + difference / Scalar[dtype](column + 1))
                var reg = MLPRegressor[dtype](
                    "poisson" if kind == 2 else "squared_error",
                    [4, 3], "tanh", solver=solver, alpha=0.03,
                    batch_size=3, max_iter=6, random_state=42, tol=0.0,
                )
                if weighted:
                    _ = reg.fit(X, y, sample_weight=weights)
                else:
                    _ = reg.fit(X, y)
                print(reg.parameter_dtype() == DType.float64)
                emit_model(
                    (reg.n_iter_(), reg.t_(), reg.n_outputs_(), reg.n_layers_(),
                     reg.get_n_features_in()), reg.out_activation_(), reg.loss_(),
                    reg.best_loss_(), reg.score(X, y, Optional(weights.copy())),
                    reg.loss_curve_(), reg.coefs_[parameter_dtype](),
                    reg.intercepts_[parameter_dtype](), reg.fit_warnings,
                )
                emit(reg.predict(X))
                comptime if solver == "lbfgs" and dtype == DType.float32:
                    emit(reg.predict[DType.float64](X))


def main() raises:
    run_cases[DType.float32]()
    run_cases[DType.float64]()
