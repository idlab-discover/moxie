"""Compare native GP kernel dispatch with the pinned sklearn KernelRidge."""
import builtins
import subprocess
from unittest.mock import patch

import sklearn.gaussian_process.kernels as support_gp

from reference import KR, REFERENCE, ROOT, compile, load_functions, np, ns, run


def encode(array):
    array = np.asarray(array)
    cols = array.shape[1] if array.ndim == 2 else 0
    values = ",".join(repr(float(value)) for value in array.flat)
    return f"{array.shape[0]}:{cols}:{values}"


def main():
    load_functions(
        REFERENCE / "sklearn/metrics/pairwise.py", ["_parallel_pairwise"], ns
    )
    relative = "sklearn/gaussian_process/kernels.py"
    assert not subprocess.check_output(
        ["git", "-C", str(REFERENCE), "status", "--porcelain", "--", relative],
        text=True,
    ).strip()
    source = REFERENCE / relative
    gp = {"__name__": "pinned_gp_kernels"}
    exec(builtins.compile(source.read_text(), str(source), "exec"), gp)
    periodic = gp["ExpSineSquared"]
    white = gp["WhiteKernel"]

    records = []
    # pairwise_kernels imports its GP base locally for dispatch.
    with patch.object(support_gp, "Kernel", gp["Kernel"]):
        for dtype in (np.float32, np.float64, np.int32, np.int64):
            X = np.array([[0, 1], [2, 3], [1, 3], [4, 2]], dtype=dtype)
            Z = np.array([[1, 2], [3, 4], [3, 0]], dtype=dtype)
            for name, kernel in (
                ("periodic", periodic(0.9, 4.5)),
                ("sum", periodic(0.9, 4.5) + white(5.0)),
                ("white", white(5.0)),
            ):
                for multioutput in (False, True):
                    y = np.array([1, -2, 3, 1], dtype=dtype)
                    if multioutput:
                        y = np.column_stack((y, 2 - y))
                    weights = [0.5, 1.0, 2.0, 1.5] if multioutput else None
                    model = KR(kernel=kernel, alpha=0.5).fit(X, y, sample_weight=weights)
                    prediction = model.predict(Z)
                    assert model.dual_coef_.dtype == prediction.dtype == np.float64
                    records.append("|".join((
                        np.dtype(dtype).name, name, encode(X), encode(y), encode(Z),
                        encode(model.dual_coef_), encode(prediction),
                    )))

    fixtures = ROOT / "builds" / "gp-kernel-fixtures.txt"
    fixtures.write_text("\n".join(records) + "\n")
    binary = ROOT / "builds" / "gp_kernel_parity_runner"
    compile(ROOT / "tests" / "gp_kernel_parity_runner.mojo", binary)
    run(binary, fixtures)


if __name__ == "__main__":
    main()
