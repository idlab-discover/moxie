"""Compare the shared native vector kernel through pinned GPR and KernelRidge."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import sklearn.gaussian_process.kernels as support_gp

from parity import GPR, K, REFERENCE, ROOT, load_pinned

sys.path.insert(0, str(ROOT / "families/kernel_ridge/tests"))
from reference import KR, load_functions, ns


def reference(dtype):
    X = np.array([[0, 1], [2, 3], [1, 3], [4, 2]], dtype=dtype)
    y = np.array([1, -2, 3, 1], dtype=dtype)
    Z = np.array([[1, 2], [3, 4], [3, 0]], dtype=dtype)
    kernel = K.DotProduct(0.5, (0.1, 0.8)) + K.WhiteKernel(0.125, "fixed")
    # The native KernelContract computes covariance in Float64 for either dtype.
    gpr = GPR(kernel, alpha=0.5)
    _, prior_std = gpr.predict(Z.astype(np.float64), return_std=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gpr.fit(X.astype(np.float64), y)
    mean, std = gpr.predict(Z.astype(np.float64), return_std=True)
    _, covariance = gpr.predict(Z.astype(np.float64), return_cov=True)
    lml, gradient = gpr.log_marginal_likelihood(gpr.kernel_.theta, eval_gradient=True)
    with patch.object(support_gp, "Kernel", K.Kernel):
        ridge = KR(kernel=kernel, alpha=0.5).fit(X, y)
        ridge_predict = ridge.predict(Z)
    return dict(prior_std=prior_std, mean=mean, std=std, covariance=covariance,
                theta=gpr.kernel_.theta, L=gpr.L_, alpha=gpr.alpha_,
                lml=np.array([lml]), gradient=gradient, ridge_dual=ridge.dual_coef_,
                ridge_predict=ridge_predict)


def composition_reference():
    X = np.array([[0, 1], [2, 3], [1, 3], [4, 2]], dtype=np.float64)
    y = np.array([1, -2, 3, 1], dtype=np.float64)
    Z = np.array([[1, 2], [3, 4], [3, 0]], dtype=np.float64)
    leaf = K.DotProduct(0.5, (0.1, 0.8)) + K.WhiteKernel(0.125, "fixed")
    radial = K.RBF(1.2, (0.3, 2.0))
    kernel = (radial * leaf + 0.5) ** 2
    covariance, gradient = kernel(X, eval_gradient=True)
    theta = kernel.theta.copy()
    theta[1] = -1.2
    compound = K.CompoundKernel([leaf, radial])
    stacked, stacked_grad = compound(X, eval_gradient=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gpr = GPR(kernel, alpha=0.5).fit(X, y)
    gpr_mean, gpr_std = gpr.predict(Z, return_std=True)
    with patch.object(support_gp, "Kernel", K.Kernel):
        ridge_predict = KR(kernel=kernel, alpha=0.5).fit(X, y).predict(Z)

    gpc = load_pinned(
        "sklearn/gaussian_process/_gpc.py", "pinned_gp_gpc", "sklearn.gaussian_process",
        extra=[("from sklearn.gaussian_process.kernels import RBF, CompoundKernel, Kernel",
                "from pinned_gp_kernels import RBF, CompoundKernel, Kernel"),
               ("from sklearn.gaussian_process.kernels import ConstantKernel as C",
                "from pinned_gp_kernels import ConstantKernel as C")],
    ).GaussianProcessClassifier
    gpc_probability = gpc(kernel, optimizer=None).fit(X, [0, 1, 0, 1]).predict_proba(Z)
    return dict(covariance=covariance, gradient=gradient, cross=kernel(X, Z),
                diag=kernel.diag(Z), theta=kernel.theta, bounds=kernel.bounds,
                adjusted=kernel.clone_with_theta(theta)(X), stacked=stacked,
                stacked_grad=stacked_grad, stacked_diag=compound.diag(Z),
                gpr_mean=gpr_mean, gpr_std=gpr_std, gpr_theta=gpr.kernel_.theta,
                ridge_predict=ridge_predict, gpc_probability=gpc_probability)


def main():
    load_functions(REFERENCE / "sklearn/metrics/pairwise.py", ["_parallel_pairwise"], ns)
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
        ROOT / "builds/tests/gaussian_process/regression/vector_kernel"
    )
    lines = subprocess.check_output([binary], text=True).splitlines()
    assert lines.pop() == "PASS custom dense GP kernel contracts"
    actual = {}
    for line in lines:
        case, key, shape, values = line.split("|")
        actual[case, key] = np.fromstring(values, sep=" ").reshape(
            tuple(map(int, shape.split(",")))
        )
    expected = {(dtype, key): value for dtype in ("float32", "float64")
                for key, value in reference(dtype).items()}
    expected.update({("composition", key): value
                     for key, value in composition_reference().items()})
    assert actual.keys() == expected.keys()
    maximum = 0.0
    for key, value in expected.items():
        estimate = actual[key]
        assert estimate.dtype == value.dtype == np.float64
        assert estimate.shape == value.shape, (key, estimate.shape, value.shape)
        maximum = max(maximum, np.max(np.abs(estimate - value)))
        # Cholesky and solve order can differ in parallel libraries. The GP
        # fixture reaches the same bound in both optimizers.
        np.testing.assert_allclose(estimate, value, atol=2e-11, rtol=2e-11, err_msg=str(key))
    print(f"custom dense GP kernel parity: {len(expected)} checks passed, "
          f"max absolute error {maximum:.3e}")


if __name__ == "__main__":
    main()
