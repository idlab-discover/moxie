"""Compare batch dictionary learning with the pinned function and its solvers."""

from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, assert_close, load_pinned, run_native

sys.path.insert(0, str(ROOT.parent / "linear_model/tools"))
from pinned_coordinate import load as load_coordinate


def main():
    reference = load_pinned("sklearn/decomposition/_dict_learning.py")
    reference.Lasso = load_coordinate(ROOT)[0].Lasso
    reference.LassoLars = load_pinned("sklearn/linear_model/_least_angle.py").LassoLars
    rng = np.random.RandomState(27)
    x = rng.normal(size=(9, 5))
    wide = rng.normal(size=(3, 5))
    code = rng.uniform(0, 0.3, size=(len(x), 3))
    dictionary = rng.uniform(0, 0.5, size=(3, x.shape[1]))
    cases = [
        ("default-rank", x, None, .35, 7, 0, False, 0., 1000),
        ("svd-initialization", x, 3, .35, 0, 0, False, 0., 1000),
        ("warm-initialization", x, 3, .35, 0, 2, False, 0., 1000),
        ("warm-truncation", x, 2, .35, 0, 2, False, 0., 1000),
        ("warm-padding", x, 5, .35, 2, 2, False, 0., 1000),
        ("zero-solver-steps", x, 3, .35, 2, 2, False, 0., 0),
        ("warm-sweep", x, 3, .35, 1, 2, False, 0., 1),
        ("ignored-lone-initializer", x, 3, .35, 1, 1, False, 0., 1000),
        ("positive-dictionary", x, 3, .35, 4, 2, True, 0., 1000),
        ("wide-overcomplete", wide, 6, .35, 4, 0, False, 0., 1000),
        ("seeded-resampling", x, 3, 100., 3, 2, False, 0., 1000),
        ("seeded-resampling-constant", np.zeros((4, 3)), 3, 100., 3, 0, False, 0., 1000),
        ("converged", x, 3, 100., 15, 2, False, 1e-8, 1000),
    ]
    count = 0
    for dtype in (np.float32, np.float64):
        for method in ("lars", "cd"):
            for name, matrix, k, alpha, iterations, init, positive_dict, tol, inner in cases:
                for positive_code in ((False, True) if method == "cd" else (False,)):
                    matrix = matrix.astype(dtype)
                    initial_code = code.astype(dtype)
                    initial_dict = dictionary.astype(dtype)
                    # sklearn pads with Float64 zeros, promoting the factors and
                    # leaving Float32 CD with incompatible buffers. Compare the
                    # same rounded inputs to a consistent Float64 reference.
                    oracle_dtype = (
                        np.float64 if name in ("warm-padding", "wide-overcomplete") else dtype
                    )
                    solver_steps = inner
                    if dtype == np.float32 and method == "cd" and oracle_dtype == np.float64:
                        # A bounded trajectory compares coefficients before
                        # precision differences accumulate across ill-conditioned sweeps.
                        solver_steps = 8
                    callbacks = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        expected_code, expected_dict, errors, n_iter = reference.dict_learning(
                            matrix.astype(oracle_dtype), k, alpha=alpha, max_iter=iterations, method=method,
                            code_init=initial_code.astype(oracle_dtype) if init else None,
                            dict_init=initial_dict.astype(oracle_dtype) if init == 2 else None,
                            positive_dict=positive_dict, positive_code=positive_code,
                            random_state=12, tol=tol, method_max_iter=solver_steps, return_n_iter=True,
                            callback=lambda state: callbacks.append((state["ii"], state["current_cost"])),
                        )
                    result = run_native(
                        ROOT / "builds/dictionary_driver", dtype, [matrix, initial_code, initial_dict],
                        k or 0, alpha, method, iterations, int(positive_dict), int(positive_code),
                        init, tol, solver_steps,
                    )
                    context = f"{dtype.__name__} {name} {method} {positive_code=}"
                    assert result["n_iter"].item() == n_iter, context
                    expected = {"code": expected_code, "dictionary": expected_dict,
                                "reconstruction": expected_code @ expected_dict}
                    assert np.array_equal(result.get("callback", []), [i for i, _ in callbacks]), context
                    if callbacks:
                        expected["callback_cost"] = [cost for _, cost in callbacks]
                    if n_iter:
                        expected["error"] = errors
                    tolerance = 4e-5 if dtype == np.float32 else 3e-7
                    for key, value in expected.items():
                        assert_close(result[key], value, tolerance, f"{context} {key}")
                    count += 1
    print(f"PASS: {count} batch dictionary learning cases against local sklearn {REVISION}")


if __name__ == "__main__":
    main()
