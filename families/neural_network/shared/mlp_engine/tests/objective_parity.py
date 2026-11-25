"""Compare native dense batches with pinned sklearn forward and backprop."""
import ast
from itertools import chain
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT.parents[4] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
MODULE = ORACLE / "sklearn/neural_network/_multilayer_perceptron.py"
BASE = MODULE.with_name("_base.py")


def reference_functions(*additional_methods):
    assert subprocess.check_output(
        ["git", "-C", str(ORACLE), "rev-parse", "HEAD"], text=True
    ).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", str(ORACLE), "status", "--porcelain", "--", str(MODULE), str(BASE)],
        text=True,
    ).strip()
    tree = ast.parse(MODULE.read_text())
    estimator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseMultilayerPerceptron"
    )
    names = {"_init_coef", "_forward_pass", "_compute_loss_grad", "_backprop", *additional_methods}
    methods = [
        node for node in estimator.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    pack = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_pack"
    )
    namespace = runpy.run_path(str(BASE))
    # The dense branch of safe_sparse_dot is the NumPy matrix product.
    namespace.update(chain=chain, safe_sparse_dot=np.matmul)
    exec(compile(ast.Module(body=[*methods, pack], type_ignores=[]), str(MODULE), "exec"), namespace)
    return namespace


def expected_case(reference, hidden, kind, dtype):
    width, output, loss = [
        (1, "identity", "squared_error"),
        (2, "logistic", "log_loss"),
        (3, "softmax", "log_loss"),
        (2, "exp", "poisson"),
        (2, "identity", "squared_error"),
    ][kind]
    units = [2, 3, 2, width]
    X = np.fromfunction(lambda i, j: (2 * i - 3 * j - 2) / 4, (4, 2)).astype(dtype)
    y = np.fromfunction(lambda i, j: (i + j) / 4, (4, width)).astype(dtype)
    if kind == 1:
        y = np.fromfunction(lambda i, j: (i + j) % 2, (4, width)).astype(dtype)
    elif kind == 2:
        y = np.eye(width, dtype=dtype)[np.arange(4) % width]

    model = SimpleNamespace(
        activation=hidden, out_activation_=output, loss=loss,
        n_layers_=len(units), _random_state=np.random.RandomState(42),
    )
    for name in ("_forward_pass", "_compute_loss_grad"):
        setattr(model, name, reference[name].__get__(model))
    model.coefs_, model.intercepts_ = zip(*[
        reference["_init_coef"](model, n_in, n_out, dtype)
        for n_in, n_out in zip(units, units[1:])
    ])
    count = len(units) - 1
    yield model._forward_pass([X] + [None] * count)[-1].ravel()
    for model.alpha, weights in [(0.17, np.array([-1, 0, 2, 3], dtype=dtype)), (0.0, None)]:
        loss_value, coefs, biases = reference["_backprop"](
            model, X, y, weights, [X] + [None] * count,
            [None] * count, [None] * count, [None] * count,
        )
        yield [loss_value]
        yield reference["_pack"](coefs, biases)


def main():
    actual = np.fromstring(
        subprocess.check_output([str(ROOT / "builds/objective_dump")], text=True), sep="\n"
    )
    reference = reference_functions()
    offset = 0
    for hidden in ("identity", "logistic", "tanh", "relu"):
        for kind in range(5):
            for dtype in (np.float32, np.float64):
                tolerance = 3e-6 if dtype == np.float32 else 2e-12
                for block, expected in enumerate(expected_case(reference, hidden, kind, dtype)):
                    end = offset + len(expected)
                    np.testing.assert_allclose(
                        actual[offset:end], expected, rtol=tolerance, atol=tolerance,
                        err_msg=f"{hidden=}, {kind=}, {dtype=}, {block=}",
                    )
                    offset = end
    assert offset == len(actual)
    print("PASS 40 dense forward/gradient fixtures, weighted and unweighted, against pinned sklearn")


if __name__ == "__main__":
    main()
