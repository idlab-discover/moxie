"""Public fit/partial_fit sequences for mlp_incremental_parity.py."""

from numojo import NDArray, Shape, zeros
from moxie._random import RandomState
from moxie.neural_network import MLPClassifier, MLPRegressor
from mlp_dump import emit, emit_model


def data[dtype: DType](step: Int) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    var rows = 2 if step == 2 else 6
    var X = zeros[dtype](Shape(rows, 2))
    var weights = zeros[dtype](Shape(rows))
    for row in range(rows):
        X.unsafe_set(row * 2, Scalar[dtype](row + step - 3) / 4)
        X.unsafe_set(row * 2 + 1, Scalar[dtype]((row * row + step) % 7 - 3) / 4)
        weights.unsafe_set(row, Scalar[dtype](row % 3 + 1) / 2)
    return (X^, weights^)


def classifier[dtype: DType](solver: String, explicit: Bool, count: Int) raises:
    var rng = RandomState(42)
    var clf = MLPClassifier[dtype](
        [4, 3], "tanh", solver=solver, alpha=0.03,
        batch_size=3, max_iter=2, random_state=None if explicit else Optional(42),
        learning_rate="adaptive" if explicit else "invscaling", learning_rate_init=0.03,
        tol=10.0, n_iter_no_change=1,
    )
    var domain = zeros[DType.int64](Shape(count))
    for index in range(count):
        domain.unsafe_set(index, Int64(3 * (count - index - 1) - 4))
    for step in range(5):
        var batch = data[dtype](step)
        ref X = batch[0]
        ref weights = batch[1]
        var y = zeros[DType.int64](Shape(X.shape[0]))
        for row in range(y.size):
            # The first seeded batch omits the final declared class.
            var present = count - 1 if step == 0 and not explicit else count
            y.unsafe_set(row, Int64(3 * ((row + step) % present) - 4))
        if explicit:
            if step == 0:
                _ = clf.fit(X, y, rng, sample_weight=weights)
            else:
                _ = clf.partial_fit(X, y, rng, sample_weight=weights)
        elif step == 0:
            _ = clf.partial_fit(X, y, classes=domain.copy())
        else:
            _ = clf.partial_fit(X, y)

        print(clf.parameter_dtype() == DType.float64)
        emit_model(
            (clf.n_iter_(), clf.t_(), clf.n_outputs_(), clf.n_layers_(), clf.get_n_features_in()),
            clf.out_activation_(), clf.loss_(), clf.best_loss_(),
            clf.score(X, y, Optional(weights.copy())), clf.loss_curve_(), clf.coefs_(),
            clf.intercepts_(), clf.fit_warnings,
        )
        emit(clf.classes_())
        emit(clf.predict(X))
        emit(clf.predict_proba(X))
        emit(clf.predict_log_proba(X))
        if explicit:
            var probe = rng.copy()
            print(probe.next_uint32())


def regressor[dtype: DType](solver: String, explicit: Bool, poisson: Bool) raises:
    var rng = RandomState(42)
    var reg = MLPRegressor[dtype](
        "poisson" if poisson else "squared_error", [4, 3], "tanh",
        solver=solver, alpha=0.03, batch_size=3, max_iter=2,
        random_state=None if explicit else Optional(42),
        learning_rate="adaptive" if explicit else "invscaling", learning_rate_init=0.03,
        tol=10.0, n_iter_no_change=1,
    )
    for step in range(5):
        var batch = data[dtype](step)
        ref X = batch[0]
        ref weights = batch[1]
        var y = zeros[dtype](Shape(X.shape[0]) if poisson else Shape(X.shape[0], 2))
        for row in range(X.shape[0]):
            var diff = X.unsafe_get(row * 2) - X.unsafe_get(row * 2 + 1)
            if poisson:
                y.unsafe_set(row, 2 + diff)
            else:
                y.unsafe_set(row * 2, 2 + diff)
                y.unsafe_set(row * 2 + 1, 2 + diff / 2)
        if explicit:
            if step == 0:
                _ = reg.partial_fit(X, y, rng)
            else:
                _ = reg.partial_fit(X, y, rng, sample_weight=weights)
        elif step == 0:
            _ = reg.fit(X, y, sample_weight=weights)
        else:
            _ = reg.partial_fit(X, y, sample_weight=weights)

        print(reg.parameter_dtype() == DType.float64)
        emit_model(
            (reg.n_iter_(), reg.t_(), reg.n_outputs_(), reg.n_layers_(), reg.get_n_features_in()),
            reg.out_activation_(), reg.loss_(), reg.best_loss_(),
            reg.score(X, y, Optional(weights.copy())), reg.loss_curve_(), reg.coefs_(),
            reg.intercepts_(), reg.fit_warnings,
        )
        emit(reg.predict(X))
        if explicit:
            var probe = rng.copy()
            print(probe.next_uint32())


def cases[dtype: DType]() raises:
    for solver in ["sgd", "adam"]:
        for explicit in [False, True]:
            for count in [2, 3]:
                classifier[dtype](solver, explicit, count)
            for poisson in [False, True]:
                regressor[dtype](solver, explicit, poisson)


def main() raises:
    cases[DType.float32]()
    cases[DType.float64]()
