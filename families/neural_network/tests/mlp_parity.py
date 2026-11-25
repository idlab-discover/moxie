"""Compare dense public MLP estimators with the pinned scikit-learn classes."""

import argparse
from pathlib import Path
import runpy
import subprocess
import types
import warnings

import numpy as np


FAMILY = Path(__file__).resolve().parents[1]
ORACLE = FAMILY.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def load_reference():
    assert subprocess.check_output(
        ["git", "-C", ORACLE, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", ORACLE, "status", "--porcelain", "--", "sklearn/neural_network"],
        text=True,
    ).strip()
    module_dir = ORACLE / "sklearn/neural_network"
    path = module_dir / "_multilayer_perceptron.py"
    module = types.ModuleType("sklearn.neural_network._moxie_oracle")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    for filename, names in [
        ("_base.py", ("ACTIVATIONS", "DERIVATIVES", "LOSS_FUNCTIONS")),
        ("_stochastic_optimizers.py", ("AdamOptimizer", "SGDOptimizer")),
    ]:
        namespace = runpy.run_path(str(module_dir / filename))
        module.__dict__.update({name: namespace[name] for name in names})
    return module


def fixtures(module, warm_start, early_stopping):
    rows = np.arange(8)
    for dtype in (np.float32, np.float64):
        X = np.column_stack([(rows - 3) / 4, (rows**2 % 7 - 3) / 4]).astype(dtype)
        weights = ((rows % 3 + 1) / 2).astype(dtype)
        for solver in ("lbfgs", "sgd", "adam"):
            if early_stopping and solver == "lbfgs":
                continue
            params = dict(
                hidden_layer_sizes=(4, 3), activation="tanh", solver=solver,
                alpha=0.03, batch_size=3, max_iter=6, random_state=42, tol=0.0, warm_start=warm_start,
            )
            for weighted in (False, True):
                if early_stopping:
                    params.update(
                        early_stopping=True, validation_fraction=0.375, max_iter=12,
                        tol=0.01, n_iter_no_change=2,
                        learning_rate="adaptive" if weighted else "constant",
                    )
                for classes in (1, 2, 3):
                    y = 3 * (rows % classes) - 4
                    yield module.MLPClassifier(**params), X, y[:, None], weights, weighted, True
                for kind in range(3):
                    y = 1 + X[:, 0] - X[:, 1]
                    if kind == 1:
                        y = np.column_stack([y, 1 + (X[:, 0] - X[:, 1]) / 2])
                    loss = "poisson" if kind == 2 else "squared_error"
                    yield module.MLPRegressor(loss=loss, **params), X, y, weights, weighted, False


def fitted_snapshots(reference, *, warm_start=False, early_stopping=False):
    for model, X, y, weights, weighted, classifier in fixtures(reference, warm_start, early_stopping):
        for step in range(3 if warm_start else 1):
            if step == 1:
                model.set_params(alpha=0.06, learning_rate_init=0.02)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X, y, sample_weight=weights if weighted else None)
            yield model, X, y, weights, weighted, classifier, caught, None


def compare(actual, snapshots, *, expect_end=True):
    worst = {}
    previous_model = None
    for case, (model, X, y, weights, weighted, classifier, caught, rng_draw) in enumerate(snapshots):
        dtype = X.dtype
        tol = 2e-5 if dtype == np.float32 and model.solver == "lbfgs" else (
            3e-6 if dtype == np.float32 else 2e-11
        )
        # Roundoff in the first Float32 solve changes the starting point for
        # later line searches. Bound that drift separately from ordinary fits.
        warm_lbfgs32 = model is previous_model and model.solver == "lbfgs" and dtype == np.float32
        parameter_tol = 1e-3 if warm_lbfgs32 else tol
        prediction_tol = 5e-4 if warm_lbfgs32 else tol
        previous_model = model
        context = f"{case=} {dtype=} {model.solver=} {weighted=} {classifier=}"

        def close(observed, expected, *, atol=tol):
            np.testing.assert_allclose(observed, expected, rtol=tol, atol=atol, err_msg=context)
            error = np.max(np.abs(np.asarray(observed) - expected), initial=0)
            worst[dtype] = max(worst.get(dtype, 0), error)

        def stack(expected, *, exact=False, atol=tol):
            ndim = int(next(actual))
            shape = tuple(int(next(actual)) for _ in range(ndim))
            assert shape == expected.shape, context
            values = np.array([float(next(actual)) for _ in range(expected.size)]).reshape(shape)
            if exact:
                np.testing.assert_array_equal(values, expected, err_msg=context)
            else:
                close(values, expected, atol=atol)

        assert next(actual) == ("True" if model.coefs_[0].dtype == np.float64 else "False"), context
        metadata = (model.n_iter_, model.t_, model.n_outputs_, model.n_layers_, model.n_features_in_)
        assert next(actual).split() == [*map(str, metadata), model.out_activation_], context
        close(
            list(map(float, next(actual).split())),
            np.array([model.loss_, getattr(model, "best_loss_", None) or 0, model.score(X, y, sample_weight=weights)]),
        )
        curve = np.array(getattr(model, "loss_curve_", []))
        assert int(next(actual)) == len(curve), context
        close(np.array([float(next(actual)) for _ in curve]), curve)
        for coefficients in [*model.coefs_, *model.intercepts_]:
            stack(coefficients, atol=parameter_tol)
        # Dense rank-two column labels are normalized without a Python warning.
        fit_warnings = [item for item in caught if item.category.__name__ != "DataConversionWarning"]
        assert int(next(actual)) == len(fit_warnings), context
        for item in fit_warnings:
            assert next(actual) == item.category.__name__, context
            assert next(actual) == str(item.message).replace("\n", r"\n"), context
        if classifier:
            stack(model.classes_, exact=True)
        stack(model.predict(X), exact=classifier, atol=prediction_tol)
        if classifier:
            stack(model.predict_proba(X), atol=prediction_tol)
            stack(model.predict_log_proba(X), atol=prediction_tol)
        if model.solver == "lbfgs" and dtype == np.float32:
            if classifier:
                stack(model.predict_proba(X), atol=prediction_tol)
                stack(model.predict_log_proba(X), atol=prediction_tol)
            else:
                stack(model.predict(X), atol=prediction_tol)
        if model.early_stopping:
            length, best = next(actual).split()
            assert int(length) == len(model.validation_scores_), context
            close(float(best), model.best_validation_score_)
            close([float(next(actual)) for _ in range(int(length))], model.validation_scores_)
        if rng_draw is not None:
            assert int(next(actual)) == rng_draw, context
    if expect_end:
        assert next(actual, None) is None
    return case + 1, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=FAMILY / "builds/mlp_dump")
    args = parser.parse_args()
    actual = iter(subprocess.check_output([args.binary], text=True).splitlines())
    reference = load_reference()
    for warm_start in (False, True):
        count, worst = compare(
            actual, fitted_snapshots(reference, warm_start=warm_start), expect_end=False,
        )
        description = "warm-start sequence snapshots" if warm_start else "ordinary fits"
        print(f"PASS {count} public MLP {description} against pinned sklearn")
        print("Maximum absolute differences", {str(dtype): float(error) for dtype, error in worst.items()})

    split_oracle = runpy.run_path(str(
        FAMILY.parent / "model_selection/algorithms/splitters/tests/parity.py"
    ))
    namespace, _, _ = split_oracle["load_reference"]()
    reference.train_test_split = namespace["train_test_split"]
    count, worst = compare(actual, fitted_snapshots(reference, early_stopping=True))
    print(f"PASS {count} public MLP validation-stopping fits against pinned sklearn")
    print("Maximum absolute differences", {str(dtype): float(error) for dtype, error in worst.items()})


if __name__ == "__main__":
    main()
