"""Compare native structured GP models to the pinned estimators and gallery kernel."""
import ast
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from parity import GPR, K, REFERENCE, load_pinned

ROOT = Path(__file__).resolve().parents[3]
GALLERY = "examples/gaussian_process/plot_gpr_on_structured_data.py"
assert not subprocess.check_output(
    ["git", "-C", REFERENCE, "status", "--porcelain", "--", GALLERY], text=True
).strip()
gallery = ast.parse((REFERENCE / GALLERY).read_text())
gallery_class = next(
    node for node in gallery.body
    if isinstance(node, ast.ClassDef) and node.name == "SequenceKernel"
)
namespace = {
    "np": np,
    "clone": GPR.fit.__globals__["clone"],
    **{name: getattr(K, name) for name in ("GenericKernelMixin", "Hyperparameter", "Kernel")},
}
exec(compile(ast.Module(body=[gallery_class], type_ignores=[]), GALLERY, "exec"), namespace)


class SequenceKernel(namespace["SequenceKernel"]):
    def __call__(self, X, Y=None, eval_gradient=False):
        result = super().__call__(X, Y, eval_gradient)
        if not eval_gradient:
            return result
        covariance, raw_gradient = result
        if self.hyperparameter_baseline_similarity.fixed:
            return covariance, raw_gradient[:, :, :0]
        # sklearn's optimizer expects the derivative with respect to log(theta).
        return covariance, raw_gradient * self.baseline_similarity


def reference(fixed):
    X = np.array(["AGCT", "AGC", "AACT", "TAA", "AAA", "GAACA"])
    train = X[[0, 1, 3, 4]]
    y = np.array([1.0, 1.0, 2.0, 3.0])
    kernel = SequenceKernel(0.5, "fixed" if fixed else (0.1, 0.9))
    model = GPR(kernel, alpha=0.05, normalize_y=not fixed,
                n_restarts_optimizer=2, random_state=31)
    prior_mean, prior_std = model.predict(X, return_std=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train, y)
    mean, std = model.predict(X, return_std=True)
    _, cov = model.predict(X, return_cov=True)
    _, gradient = model.log_marginal_likelihood(model.kernel_.theta, eval_gradient=True)
    return dict(gram=kernel(X), prior_mean=prior_mean, prior_std=prior_std,
                theta=model.kernel_.theta, L=model.L_, alpha=model.alpha_,
                lml=np.array([model.log_marginal_likelihood_value_]), mean=mean, std=std,
                cov=cov, score=np.array([model.score(train, y)]), gradient=gradient)


class IndexedSequenceKernel(SequenceKernel):
    """Decode numeric row IDs for sklearn's 2-D numeric OvO validation."""

    def __init__(self, sequences, baseline_similarity=0.5, baseline_similarity_bounds="fixed"):
        super().__init__(baseline_similarity, baseline_similarity_bounds)
        self.sequences = sequences

    def _decode(self, rows):
        return [self.sequences[int(row[0])] for row in rows]

    def __call__(self, X, Y=None, eval_gradient=False):
        return super().__call__(
            self._decode(X), None if Y is None else self._decode(Y), eval_gradient
        )

    def diag(self, X):
        return super().diag(self._decode(X))


def pinned_classifier():
    load_pinned("sklearn/multiclass.py", "pinned_gp_multiclass", "sklearn")
    return load_pinned(
        "sklearn/gaussian_process/_gpc.py", "pinned_gp_gpc", "sklearn.gaussian_process",
        extra=[
            ("from sklearn.gaussian_process.kernels import", "from pinned_gp_kernels import"),
            ("from sklearn.multiclass import", "from pinned_gp_multiclass import"),
        ],
    ).GaussianProcessClassifier


def classification_reference(name, gpc):
    binary = name in ("classification_gallery", "classification_optimized")
    fixed = name == "classification_gallery"
    strategy = "one_vs_one" if name == "classification_ovo" else "one_vs_rest"
    if binary:
        train = ["AGCT", "CGA", "TAAC", "TCG", "CTTT", "TGCT"]
        query = ["AAA", "ATAG", "CTC", "CT", "C"]
        y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    else:
        train = ["AAA", "CC", "GGG", "AAAT", "CCC", "GG"]
        query = ["AA", "CCC", "G", "AT", "TG"]
        y = np.array([9, -3, 4, 9, -3, 4], dtype=np.int64)
    kernel = SequenceKernel(0.5, "fixed" if fixed else (0.1, 0.9))
    if strategy == "one_vs_one":
        kernel = IndexedSequenceKernel(tuple(train + query), 0.5, (0.1, 0.9))
        train, query = np.split(np.arange(len(train) + len(query))[:, None], [len(train)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = gpc(
            kernel,
            optimizer="fmin_l_bfgs_b" if binary and not fixed else None,
            multi_class=strategy,
        ).fit(train, y)
    result = dict(
        classes=model.classes_,
        prediction=model.predict(query),
        score=np.array([model.score(train, y)]),
        lml=np.array([model.log_marginal_likelihood()]),
        theta=model.kernel_.theta,
    )
    if binary or strategy == "one_vs_rest":
        result["probability"] = model.predict_proba(query)
    if binary:
        result["mean"], result["variance"] = model.latent_mean_and_variance(query)
        _, result["gradient"] = model.log_marginal_likelihood(model.kernel_.theta, True)
    else:
        result["shared_lml"] = np.array([model.log_marginal_likelihood(np.array([-0.8]))])
        result["component_lml"] = np.array([
            model.log_marginal_likelihood(np.array([-0.8, -0.6, -0.4]))
        ])
    return result


def main():
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "builds/tests/gaussian_process/regression/structured"
    )
    lines = subprocess.check_output([binary], text=True).splitlines()
    assert lines.pop() == "PASS structured GP contracts"
    actual = {}
    for line in lines:
        case, key, shape, values = line.split("|")
        actual[case, key] = np.fromstring(values, sep=" ").reshape(
            tuple(map(int, shape.split(",")))
        )
    expected = {(case, key): value for case in ("fixed", "optimized")
                for key, value in reference(case == "fixed").items()}
    classifier = pinned_classifier()
    for name in ("classification_gallery", "classification_optimized",
                 "classification_ovr", "classification_ovo"):
        expected.update({(name, key): value
                         for key, value in classification_reference(name, classifier).items()})
    assert actual.keys() == expected.keys()
    maximum = 0.0
    for (case, key), value in expected.items():
        estimate = actual[case, key]
        assert estimate.shape == value.shape, (case, key, estimate.shape, value.shape)
        maximum = max(maximum, np.max(np.abs(estimate - value), initial=0.0))
        # Double-precision Cholesky and solve order can differ, as can scalar
        # probability approximation rounding. Both optimized fixtures reach
        # the same bound, so no optimizer resolution allowance is needed.
        if key in ("classes", "prediction"):
            np.testing.assert_array_equal(estimate, value, err_msg=f"{case} {key}")
        else:
            np.testing.assert_allclose(estimate, value, atol=2e-11, rtol=2e-11,
                                       err_msg=f"{case} {key}")
    print(f"structured GP parity: {len(expected)} checks passed, max absolute error {maximum:.3e}")


if __name__ == "__main__":
    main()
