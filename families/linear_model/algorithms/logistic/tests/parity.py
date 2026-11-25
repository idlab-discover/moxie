"""Dense L-BFGS differential checks against the pinned sklearn estimator and loss."""
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from sklearn import __version__ as support_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _cd_fast
from sklearn.utils import _array_api, fixes

FAMILY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FAMILY / "tools"))
from parity_io import case_line, close, load_pinned, parse_results, pinned_code, verify_reference

LOSS = "sklearn/linear_model/_linear_loss.py"
LOGISTIC = "sklearn/linear_model/_logistic.py"
NEWTON = "sklearn/linear_model/_glm/_newton_solver.py"


def unsupported_oracle_branch(*args, **kwargs):
    raise AssertionError("the pinned oracle reached an import used only by LogisticRegressionCV")


def oracle():
    hashes = verify_reference([LOSS, LOGISTIC, NEWTON])
    print(f"scikit-learn source SHA256 {hashes}, support wheel {support_version}")
    # These symbols let the module import with the locked 1.9.1 support wheel.
    # None participates in the dense L-BFGS reference calculation.
    if not hasattr(_cd_fast, "enet_coordinate_descent_sparse"):
        _cd_fast.enet_coordinate_descent_sparse = _cd_fast.sparse_enet_coordinate_descent
    for module, name in [
        (_array_api, "_swapaxes"), (_array_api, "_unravel_index"), (fixes, "_is_gil_enabled")
    ]:
        if not hasattr(module, name):
            setattr(module, name, unsupported_oracle_branch)
    loss = load_pinned(LOSS, register=True)
    load_pinned(NEWTON, register=True)
    estimator = load_pinned(LOGISTIC)
    pinned_code(loss.LinearModelLoss.loss_gradient, LOSS)
    pinned_code(estimator._logistic_regression_path, LOGISTIC)
    pinned_code(estimator.LogisticRegression.fit, LOGISTIC)
    assert estimator.LinearModelLoss is loss.LinearModelLoss
    return estimator.LogisticRegression


def fixtures():
    rng = np.random.default_rng(842)
    for bits in (32, 64):
        dtype = np.float32 if bits == 32 else np.float64
        for classes in (2, 3, 5):
            for scenario in (
                "default", "no-intercept", "weighted", "balanced", "explicit",
                "unpenalized", "warm", "zero-iter", "one-iter", "wide", "signed",
            ):
                n, p = (14, 24) if scenario == "wide" else (87, 5)
                labels = np.array([-3, 8, 23, 157, 201])
                indices = rng.permutation(np.arange(n) % classes)
                y = labels[indices]
                if scenario == "default":
                    y = y[:, None]
                X = rng.normal(size=(n, p)).astype(dtype)
                X[np.arange(n), indices] += 0.8
                if scenario == "no-intercept":
                    X += 2
                Z = rng.normal(size=(13, p)).astype(dtype)
                arrays = {"X": X, "y": y, "Z": Z}
                if scenario in ("weighted", "balanced", "explicit", "signed"):
                    sw = rng.uniform(0.35, 2, size=n).astype(dtype)
                    sw[0] = -0.05 if scenario == "signed" else 0
                    arrays["sw"] = sw
                params = {
                    "C": 1.0 if scenario == "default" else 0.7,
                    "tol": 1e-4 if scenario == "default" else 1e-8,
                    "max_iter": {"zero-iter": 0, "one-iter": 1}.get(scenario, 100),
                    "intercept": int(scenario != "no-intercept"),
                    "weight": scenario if scenario in ("balanced", "explicit") else "none",
                    "unpenalized": int(scenario == "unpenalized"),
                }
                if scenario == "warm":
                    params["refit_C"] = 1.8
                yield f"{bits}-{classes}-{scenario}", bits, params, arrays

        params = {
            "C": 1.0, "tol": 1e-4, "max_iter": 100, "intercept": 1,
            "weight": "none", "unpenalized": 0, "invalid": 1,
        }
        X = np.eye(4, dtype=dtype)
        y = np.array([-3, 8, -3, 8])
        for scenario, options in {
            "one-class": {"y": np.full(4, -3)},
            "bad-weight-shape": {"sw": np.ones(3, dtype=dtype)},
            "multilabel": {"y": np.eye(4, dtype=np.int64)},
        }.items():
            arrays = {"X": X, "y": y, "Z": X} | options
            yield f"{bits}-invalid-{scenario}", bits, params, arrays


def reference(model_type, params, arrays):
    model = model_type(
        C=np.inf if params["unpenalized"] else params["C"],
        tol=params["tol"], max_iter=params["max_iter"], fit_intercept=bool(params["intercept"]),
        class_weight={"balanced": "balanced", "explicit": {-3: 2.0}}.get(params["weight"]),
        warm_start="refit_C" in params,
    )
    X, y, Z = (arrays[key] for key in ("X", "y", "Z"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(X, y, sample_weight=arrays.get("sw"))
        if "refit_C" in params:
            model.set_params(C=params["refit_C"]).fit(X, y)
    assert model.coef_.dtype == model.intercept_.dtype == X.dtype
    assert model.n_iter_.dtype == np.int32 and model.classes_.dtype == y.dtype
    return {
        "classes": model.classes_, "coef": model.coef_, "intercept": model.intercept_,
        "iterations": model.n_iter_, "decision": model.decision_function(Z),
        "predictions": model.predict(Z), "probability": model.predict_proba(Z),
        "log_probability": model.predict_log_proba(Z),
        "score": model.score(X, y, sample_weight=arrays.get("sw")), "features": model.n_features_in_,
    }


def main():
    model_type = oracle()
    cases = list(fixtures())
    output = FAMILY / "builds/logistic"
    fixture = output / "fixtures.txt"
    fixture.write_text("\n".join(case_line(*case) for case in cases) + "\n")
    completed = subprocess.run([output / "parity_driver", fixture], text=True, check=True, capture_output=True)
    (output / "native.txt").write_text(completed.stdout)
    observed = parse_results(completed.stdout)
    assert observed.keys() == {case[0] for case in cases}
    maxima = {}
    iteration_difference = 0
    invalid = 0
    for ident, bits, params, arrays in cases:
        if "invalid" in params:
            try:
                reference(model_type, params, arrays)
            except ValueError:
                pass
            else:
                raise AssertionError(f"pinned sklearn accepted {ident}")
            assert observed[ident]["error"].startswith("ValueError:"), observed[ident]
            invalid += 1
            continue

        expected = reference(model_type, params, arrays)
        assert "error" not in observed[ident], (ident, observed[ident])
        references = [("native-dtype", expected)]
        if bits == 32:
            widened = {key: value if key == "y" else value.astype(np.float64) for key, value in arrays.items()}
            references.append(("widened", reference(model_type, params, widened)))
        for kind, result in references:
            for key, wanted in result.items():
                actual = observed[ident][key]
                if key == "iterations" and params["max_iter"] > 1:
                    assert actual.shape == wanted.shape == (1,)
                    assert 0 < actual[0] <= params["max_iter"]
                    if kind == "native-dtype":
                        iteration_difference = max(iteration_difference, int(abs(actual[0] - wanted[0])))
                    continue
                if key in {"classes", "predictions", "features", "iterations"}:
                    atol = 0
                elif key == "score":
                    atol = 1e-7
                elif bits == 32 and kind == "native-dtype":
                    atol = 3e-3 if key in {"decision", "log_probability"} else 9e-4
                else:
                    atol = 5e-6 if key in {"decision", "log_probability"} else 2e-6
                error = close(actual, wanted, 0, atol, f"{ident}/{kind}/{key}")
                group = f"{bits}/{kind}/{key}"
                maxima[group] = max(maxima.get(group, 0), error)
    print(f"PASS: {len(cases) - invalid} pinned dense L-BFGS fits and {invalid} input rejections."
          f" Max unconstrained iteration-count difference {iteration_difference}.")
    for key, error in sorted(maxima.items()):
        if error > 0:
            print(f"  {key}: {error:.8g}")


if __name__ == "__main__":
    main()
