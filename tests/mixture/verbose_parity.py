"""Compare public mixture diagnostics byte for byte with pinned scikit-learn."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from compose_parity_io import load_pinned, verify_reference

RUNNER = ROOT / 'builds/tests/mixture/verbose_runner'
ELAPSED = re.compile(rb'(?<=time lapse )[0-9]+\.[0-9]{5}(?=s\t)')


def main():
    base = 'sklearn/mixture/_base.py'
    gaussian = 'sklearn/mixture/_gaussian_mixture.py'
    bayesian = 'sklearn/mixture/_bayesian_mixture.py'
    verify_reference([base, gaussian, bayesian])
    load_pinned(base, register=True)
    models = {
        'gaussian': load_pinned(gaussian, register=True).GaussianMixture,
        'bayesian': load_pinned(bayesian, register=True).BayesianGaussianMixture,
    }
    X = np.array([[-4.0], [-1.0], [0.0], [2.0], [6.0]])
    checks = 0
    for name, estimator in models.items():
        for verbose in (0, 1, 2, 3):
            for max_iter in (0, 1, 4):
                for interval in (1, 2):
                    model = estimator(
                        n_components=1, init_params='random', random_state=0, n_init=2,
                        verbose=verbose, max_iter=max_iter, verbose_interval=interval,
                    )
                    with redirect_stdout(StringIO()) as output, warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        model.fit(X)
                    expected = output.getvalue().encode()
                    native = subprocess.run(
                        [RUNNER, name, str(verbose), str(max_iter), str(interval)],
                        check=True, capture_output=True,
                    ).stdout
                    if verbose >= 2:
                        native = ELAPSED.sub(b'<elapsed>', native)
                        expected = ELAPSED.sub(b'<elapsed>', expected)
                    assert native == expected, (name, verbose, max_iter, interval, native, expected)
                    checks += 1
    print(f'PASS: {checks} mixture stdout comparisons against pinned scikit-learn')


if __name__ == '__main__':
    main()
