"""Compare native stochastic MLP training with pinned sklearn method bodies."""

import argparse
import ast
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import _indexing

from objective_parity import MODULE, ORACLE, ROOT, reference_functions

OPTIMIZERS = MODULE.with_name("_stochastic_optimizers.py")
INDEXING = ORACLE / "sklearn/utils/_indexing.py"
CHUNKING = ORACLE / "sklearn/utils/_chunking.py"


def load_functions(path, namespace, *names):
    tree = ast.parse(path.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    for function in functions:
        function.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


def load_reference():
    reference = reference_functions("_fit_stochastic", "_update_no_improvement_count")
    assert not subprocess.check_output(
        ["git", "-C", str(ORACLE), "status", "--porcelain", "--", OPTIMIZERS, INDEXING, CHUNKING],
        text=True,
    ).strip()
    indexing = vars(_indexing).copy()
    load_functions(INDEXING, indexing, "shuffle", "resample")
    load_functions(CHUNKING, reference, "gen_batches")
    reference.update(
        runpy.run_path(str(OPTIMIZERS)),
        shuffle=indexing["shuffle"],
        _safe_indexing=indexing["_safe_indexing"],
        ConvergenceWarning=ConvergenceWarning,
        warnings=warnings,
    )
    return reference


def expected_case(reference, kind, weighted, mode, dtype):
    width, hidden, output, loss = [
        (1, "relu", "identity", "squared_error"),
        (1, "tanh", "logistic", "log_loss"),
        (3, "logistic", "softmax", "log_loss"),
        (1, "identity", "exp", "poisson"),
        (2, "relu", "identity", "squared_error"),
    ][kind]
    units = (2, 9, width)
    size = 203 if mode == 6 else 8
    rows = np.arange(size)
    a, b = (rows % 8 - 3) / 4, (rows**2 % 7 - 3) / 4
    X = np.column_stack([a, b]).astype(dtype)
    y = 1 + (a - b)[:, None] / np.arange(1, width + 1)
    if kind == 1:
        y = (a + b > 0)[:, None]
    elif kind == 2:
        y = np.eye(width)[rows % width]
    elif kind == 3:
        y = (rows % 3)[:, None]
    y = y.astype(dtype)
    weights = ((rows % 4 + 1) / 2).astype(dtype) if weighted else None

    model = SimpleNamespace(
        activation=hidden,
        out_activation_=output,
        loss=loss,
        n_layers_=len(units),
        _random_state=np.random.RandomState(42),
        verbose=False,
        early_stopping=False,
        solver="sgd" if mode <= 3 else "adam",
        learning_rate_init=1e-5 if mode == 3 else 0.01,
        learning_rate={2: "invscaling", 3: "adaptive", 5: "adaptive"}.get(mode, "constant"),
        momentum=0 if mode == 0 else 0.65,
        nesterovs_momentum=mode != 1,
        power_t=0.35,
        beta_1=0.8,
        beta_2=0.96,
        epsilon=1e-5,
        batch_size={1: 20, 6: "auto"}.get(mode, 3),
        max_iter={3: 15, 4: 1, 5: 15, 6: 4}.get(mode, 6),
        shuffle=mode != 1,
        tol=10 if mode in (3, 5, 6) else 1e-4,
        n_iter_no_change={3: 2, 5: 2, 6: np.inf}.get(mode, 10),
        alpha=0.1,
        t_=0,
        loss_curve_=[],
        best_loss_=np.inf,
        _no_improvement_count=0,
    )
    for name in ("_forward_pass", "_compute_loss_grad", "_backprop", "_update_no_improvement_count"):
        setattr(model, name, reference[name].__get__(model))
    model.coefs_, model.intercepts_ = map(list, zip(*[
        reference["_init_coef"](model, n_in, n_out, dtype)
        for n_in, n_out in zip(units, units[1:])
    ]))
    count = len(units) - 1
    updates = 0
    for cycle in range({2: 2, 4: 3}.get(mode, 1)):
        if cycle == 1 and mode == 2:
            model.max_iter = 3
            updates = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference["_fit_stochastic"](
                model, X, y, weights, [X] + [None] * count,
                [None] * count, [None] * count, [None] * count, units, mode == 4,
            )
        messages = [(warning.category.__name__, str(warning.message)) for warning in caught]
        batch_warning = (
            "UserWarning", "Got `batch_size` less than 1 or larger than sample size. It is going to be clipped",
        )
        iteration_warning = (
            "ConvergenceWarning",
            f"Stochastic Optimizer: Maximum iterations ({model.max_iter}) reached and the optimization hasn't converged yet.",
        )
        assert all(message in (batch_warning, iteration_warning) for message in messages), messages
        batch_size = min(size, 200 if model.batch_size == "auto" else model.batch_size)
        updates += model.n_iter_ * ((size + batch_size - 1) // batch_size)
        if model.solver == "adam":
            assert model._optimizer.t == updates
        probe = np.random.RandomState()
        probe.set_state(model._random_state.get_state())
        assert all(c.dtype == dtype for c in (*model.coefs_, *model.intercepts_))
        metadata = [
            model.n_iter_, model.t_, updates, model._no_improvement_count,
            batch_warning in messages, iteration_warning in messages,
            probe.randint(2**32, dtype=np.uint32),
        ]
        values = np.concatenate((
            [model._optimizer.learning_rate, model.loss_, model.best_loss_],
            model.loss_curve_,
            reference["_pack"](model.coefs_, model.intercepts_),
            model._forward_pass([X] + [None] * count)[-1].ravel(),
        ))
        yield f"{kind} {weighted} {mode} {dtype.__name__} {cycle}", metadata, values


def main(binary):
    reference = load_reference()
    cases = subprocess.check_output([binary], text=True).split("CASE ")[1:]
    expected = [
        case
        for kind in range(5)
        for weighted in (0, 1)
        for mode in range(7)
        for dtype in (np.float32, np.float64)
        for case in expected_case(reference, kind, weighted, mode, dtype)
    ]
    assert len(cases) == len(expected)
    worst = {"float32": 0.0, "float64": 0.0}
    for actual, (title, metadata, values) in zip(cases, expected):
        label, numbers = actual.split("\n", 1)
        assert label == title
        result = np.fromstring(numbers, sep=" ")
        np.testing.assert_array_equal(result[:7], metadata, err_msg=title)
        np.testing.assert_allclose(result[7], values[0], rtol=1e-13, atol=1e-16, err_msg=title)
        dtype = title.split()[3]
        tolerance = 3e-6 if dtype == "float32" else 2e-12
        np.testing.assert_allclose(result[8:], values[1:], rtol=tolerance, atol=tolerance, err_msg=title)
        worst[dtype] = max(worst[dtype], float(np.max(np.abs(result[8:] - values[1:]))))
    print(f"PASS {len(cases)} dense MLP stochastic training snapshots against pinned sklearn")
    print("Maximum absolute loss, parameter or prediction error:", ", ".join(
        f"{name} {error:.3g}" for name, error in worst.items()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "builds/stochastic_dump")
    main(parser.parse_args().binary)
