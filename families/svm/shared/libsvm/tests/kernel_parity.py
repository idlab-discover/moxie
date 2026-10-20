"""Partial kernel evidence, not an optimizer acceptance gate.

Compiles the actual pinned local sklearn svm.cpp, with a scalar dot callback
standing in for the BLAS plumbing. No installed sklearn wheel is imported.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[4] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def run():
    revision = subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == REVISION, "Reference revision drift"
    paths = ["sklearn/svm/src/libsvm", "sklearn/svm/src/newrand"]
    assert not subprocess.check_output(
        ["git", "-C", str(REFERENCE), "status", "--porcelain", "--", *paths], text=True
    ), "Reference source is not clean"
    executable = ROOT / "builds/kernel_reference"
    command = ["c++", "-O2", "-ffp-contract=off", "-std=c++17",
               "-I", str(REFERENCE / paths[0]),
               str(ROOT / "tests/kernel_reference.cpp"), "-o", str(executable)]
    subprocess.run(command, check=True)
    cases = []
    for rows, cols in ((1, 1), (2, 3), (3, 2), (17, 7), (7, 17), (65, 33)):
        for kind_id, kind in enumerate(("linear", "poly", "rbf", "sigmoid")):
            raw = subprocess.check_output(
                [str(ROOT / "builds/evaluate_kernels"), str(rows), str(cols), kind], text=True
            )
            actual = [float(line.split()[1]) for line in raw.splitlines() if line.startswith("kernel ")]
            gamma = float(next(line.split()[1] for line in raw.splitlines() if line.startswith("gamma ")))
            expected = np.fromstring(subprocess.check_output(
                [str(executable), str(rows), str(cols), str(kind_id)], text=True
            ), sep="\n")
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            i, j = np.arange(rows)[:, None], np.arange(cols)[None, :]
            x = ((i * 17 + (2 * cols - 1 - 2 * j) * 3) % 31 - 15) / 97.0
            expected_gamma = 1 / (cols * x.var()) if x.var() else 1.0
            np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-12, atol=1e-12)
            cases.append([rows, cols, kind])
    files = [p for name in paths for p in (REFERENCE / name).rglob("*") if p.is_file()]
    report = {"scope": "training kernels and NumPy gamma composition only; no solver parity",
              "revision": revision, "compile_command": command, "cases": cases,
              "tolerance": "rtol=atol=1e-12 on bounded finite Float64 fixtures; allows reduction order differences",
              "reference_sha256": {str(p.relative_to(REFERENCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    destination = ROOT / "docs/private/libsvm/kernel-parity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(cases)} native/pinned-LIBSVM kernel arithmetic cases", flush=True)


if __name__ == "__main__":
    run()
