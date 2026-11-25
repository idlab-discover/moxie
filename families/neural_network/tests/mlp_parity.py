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


def fixtures(module):
    rows = np.arange(8)
    for dtype in (np.float32, np.float64):
        X = np.column_stack([(rows - 3) / 4, (rows**2 % 7 - 3) / 4]).astype(dtype)
        weights = ((rows % 3 + 1) / 2).astype(dtype)
        for solver in ("lbfgs", "sgd", "adam"):
            params = dict(
                hidden_layer_sizes=(4, 3), activation="tanh", solver=solver,
                alpha=0.03, batch_size=3, max_iter=6, random_state=42, tol=0.0,
            )
            for weighted in (False, True):
                for classes in (1, 2, 3):
                    y = 3 * (rows % classes) - 4
                    yield module.MLPClassifier(**params), X, y[:, None], weights, weighted, True
                for kind in range(3):
                    y = 1 + X[:, 0] - X[:, 1]
                    if kind == 1:
                        y = np.column_stack([y, 1 + (X[:, 0] - X[:, 1]) / 2])
                    loss = "poisson" if kind == 2 else "squared_error"
                    yield module.MLPRegressor(loss=loss, **params), X, y, weights, weighted, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=FAMILY / "builds/mlp_dump")
    args = parser.parse_args()
    actual = iter(subprocess.check_output([args.binary], text=True).splitlines())
    reference = load_reference()
    worst = {}
    for case, (model, X, y, weights, weighted, classifier) in enumerate(fixtures(reference)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y, sample_weight=weights if weighted else None)
        dtype = X.dtype
        tol = 2e-5 if dtype == np.float32 and model.solver == "lbfgs" else (
            3e-6 if dtype == np.float32 else 2e-11
        )
        context = f"{case=} {dtype=} {model.solver=} {weighted=} {classifier=}"

        def close(observed, expected):
            np.testing.assert_allclose(observed, expected, rtol=tol, atol=tol, err_msg=context)
            error = np.max(np.abs(np.asarray(observed) - expected), initial=0)
            worst[dtype] = max(worst.get(dtype, 0), error)

        def stack(expected, *, exact=False):
            ndim = int(next(actual))
            shape = tuple(int(next(actual)) for _ in range(ndim))
            assert shape == expected.shape, context
            values = np.array([float(next(actual)) for _ in range(expected.size)]).reshape(shape)
            if exact:
                np.testing.assert_array_equal(values, expected, err_msg=context)
            else:
                close(values, expected)

        assert next(actual) == ("True" if model.coefs_[0].dtype == np.float64 else "False"), context
        metadata = (model.n_iter_, model.t_, model.n_outputs_, model.n_layers_, model.n_features_in_)
        assert next(actual).split() == [*map(str, metadata), model.out_activation_], context
        close(
            list(map(float, next(actual).split())),
            np.array([model.loss_, getattr(model, "best_loss_", 0), model.score(X, y, sample_weight=weights)]),
        )
        curve = np.array(getattr(model, "loss_curve_", []))
        assert int(next(actual)) == len(curve), context
        close(np.array([float(next(actual)) for _ in curve]), curve)
        for coefficients in [*model.coefs_, *model.intercepts_]:
            stack(coefficients)
        # Dense rank-two column labels are normalized without a Python warning.
        fit_warnings = [item for item in caught if item.category.__name__ != "DataConversionWarning"]
        assert int(next(actual)) == len(fit_warnings), context
        for item in fit_warnings:
            assert next(actual) == item.category.__name__, context
            assert next(actual) == str(item.message).replace("\n", r"\n"), context
        if classifier:
            stack(model.classes_, exact=True)
        stack(model.predict(X), exact=classifier)
        if classifier:
            stack(model.predict_proba(X))
            stack(model.predict_log_proba(X))
        if model.solver == "lbfgs" and dtype == np.float32:
            if classifier:
                stack(model.predict_proba(X))
                stack(model.predict_log_proba(X))
            else:
                stack(model.predict(X))
    assert next(actual, None) is None
    print(f"PASS {case + 1} public MLP fits against pinned sklearn")
    print("Maximum absolute differences", {str(dtype): float(error) for dtype, error in worst.items()})


if __name__ == "__main__":
    main()
