"""Check the native solve with sklearn's pinned _fit_lbfgs and locked SciPy."""
import argparse
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings

import numpy as np
import scipy
from packaging.version import parse

from objective_parity import ORACLE, ROOT, reference_functions


class ConvergenceWarning(UserWarning):
    pass


def load_reference():
    reference = reference_functions("_fit_lbfgs", "_loss_grad_lbfgs", "_unpack")
    optimize = ORACLE / "sklearn/utils/optimize.py"
    fixes = optimize.with_name("fixes.py")
    assert not subprocess.check_output(
        ["git", "-C", str(ORACLE), "status", "--porcelain", "--", str(optimize), str(fixes)],
        text=True,
    ).strip()
    for path, name in (
        (optimize, "_check_optimize_result"),
        (fixes, "_get_additional_lbfgs_options_dict"),
    ):
        tree = ast.parse(path.read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), reference)

    def minimize(*args, **kwargs):
        result = scipy.optimize.minimize(*args, **kwargs)
        reference["result"] = result
        return result

    reference.update(
        scipy=SimpleNamespace(optimize=SimpleNamespace(minimize=minimize)),
        warnings=warnings,
        ConvergenceWarning=ConvergenceWarning,
        sp_version=parse(scipy.__version__),
        parse_version=parse,
    )
    return reference


def expected_case(reference, kind, weighted, budget, dtype):
    units, hidden, output, loss = [
        ([2, 1], "identity", "identity", "squared_error"),
        ([2, 3, 1], "tanh", "logistic", "log_loss"),
        ([2, 3], "identity", "softmax", "log_loss"),
        ([2, 1], "identity", "exp", "poisson"),
        ([2, 2], "identity", "identity", "squared_error"),
    ][kind]
    width = units[-1]
    rows = np.arange(8)
    a, b = (rows - 3) / 4, (rows**2 % 7 - 3) / 4
    X = np.column_stack([a, b]).astype(dtype)
    y = 1 + (a - b)[:, None] / np.arange(1, width + 1)
    if kind == 1:
        y = (a + b > 0)[:, None]
    elif kind == 2:
        y = np.eye(width)[rows % width]
    elif kind == 3:
        y = (rows % 3)[:, None]
    y = y.astype(dtype)
    weights = (rows % 4 / 2).astype(dtype) if weighted else None

    model = SimpleNamespace(
        activation=hidden,
        out_activation_=output,
        loss=loss,
        n_layers_=len(units),
        _random_state=np.random.RandomState(42),
        verbose=False,
        max_iter=2 if budget == 1 else 100,
        max_fun=1 if budget == 2 else 15000,
        tol=100 if budget == 3 else 0.001,
        alpha=0.25,
    )
    for name in ("_forward_pass", "_compute_loss_grad", "_unpack", "_loss_grad_lbfgs", "_backprop"):
        setattr(model, name, reference[name].__get__(model))
    model.coefs_, model.intercepts_ = map(list, zip(*[
        reference["_init_coef"](model, n_in, n_out, dtype)
        for n_in, n_out in zip(units, units[1:])
    ]))
    count = len(units) - 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        reference["_fit_lbfgs"](
            model, X, y, weights, [X] + [None] * count,
            [None] * count, [None] * count, [None] * count, units,
        )
    warning = "\n".join(str(item.message) for item in caught)
    assert all(c.dtype == np.float64 for c in (*model.coefs_, *model.intercepts_))
    learned = reference["_pack"](model.coefs_, model.intercepts_)
    prediction = model._forward_pass([X] + [None] * count)[-1].ravel()
    return model.n_iter_, reference["result"].nfev, model.loss_, learned, prediction, warning


def main(binary):
    reference = load_reference()
    actual = subprocess.check_output([binary], text=True)
    cases = actual.split("CASE ")[1:]
    expected_cases = [
        (kind, weighted, budget, dtype)
        for kind in range(5)
        for weighted in (0, 1)
        for budget in range(1 if kind == 4 else 0, 4)
        for dtype in (np.float32, np.float64)
    ]
    assert len(cases) == len(expected_cases)
    for case, args in zip(cases, expected_cases):
        kind, weighted, budget, dtype = args
        data, warning = case.split("WARNING\n")
        title, numbers = data.split("\n", 1)
        assert title == f"{kind} {weighted} {budget} {dtype.__name__}"
        n_iter, nfev, loss, params, predictions, expected_warning = expected_case(reference, *args)
        values = np.fromstring(numbers, sep=" ")
        iterations, evaluations, actual_loss = values[:3]
        assert (iterations, evaluations) == (n_iter, nfev), (title, iterations, evaluations, n_iter, nfev)
        assert warning.removesuffix("\n") == expected_warning, (title, warning, expected_warning)
        parameter_tol = 1e-5 if dtype == np.float32 else 5e-12
        output_tol = 3e-6 if dtype == np.float32 else 2e-12
        assert values.size == 3 + params.size + predictions.size, title
        np.testing.assert_allclose(
            values[3:3 + params.size], params, rtol=parameter_tol, atol=parameter_tol, err_msg=title,
        )
        np.testing.assert_allclose(
            values[3 + params.size:], predictions, rtol=output_tol, atol=output_tol, err_msg=title,
        )
        np.testing.assert_allclose(actual_loss, loss, rtol=output_tol, atol=output_tol, err_msg=title)
    print(f"PASS {len(cases)} dense MLP L-BFGS-B fixtures against pinned sklearn and SciPy {scipy.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "builds/lbfgs_dump")
    main(parser.parse_args().binary)
