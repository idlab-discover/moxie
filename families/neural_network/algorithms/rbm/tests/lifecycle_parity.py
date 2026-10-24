"""Public BernoulliRBM lifecycle sequences against the pinned local RBM methods.

The sequences in lifecycle_dump.mojo are replayed with the original pinned
method bodies (see kernel_parity.py). Bernoulli samples (particles, gibbs) must
match exactly. Real arrays use the family tolerances: Float64 2e-12 absorbs
matmul/summation order over at most 16 terms and 4 epochs; Float32 3e-5 is
about 250 float32 ulps at unit scale for the same accumulations. Verbose lines
must match the reference text up to the elapsed time, which is wall-clock.
"""
from kernel_parity import RBM, ROOT, REVISION
import contextlib
import io
import re
import subprocess
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

output = subprocess.check_output([ROOT / 'builds/lifecycle_dump'], text=True).splitlines()
verbose = [line for line in output if line.startswith('[BernoulliRBM]')]
lines = iter(line for line in output if not line.startswith('[BernoulliRBM]'))
worst = {np.float64: 0.0, np.float32: 0.0}
reference_verbose = []


def compare(name, expected, dtype):
    fields = next(lines).split()
    assert fields[0] == name, (fields[0], name)
    actual = np.asarray(fields[1:], dtype=float)
    expected = np.asarray(expected).ravel(order='C')
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    if name in ('particles', 'gibbs'):
        assert_array_equal(actual, expected, err_msg=name)
    else:
        tolerance = 3e-5 if dtype == np.float32 else 2e-12
        assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, err_msg=name)
        worst[dtype] = max(worst[dtype], float(np.max(np.abs(actual - expected), initial=0)))


def state(model, X, dtype):
    for name, array in [('weights', model.components_), ('hidden_bias', model.intercept_hidden_),
                        ('visible_bias', model.intercept_visible_), ('particles', model.h_samples_),
                        ('transform', model.transform(X))]:
        compare(name, array, dtype)


def binary(n, d, dtype):
    i, j = np.indices((n, d))
    return ((i * 5 + j * 3 + i * j) % 7 < 3).astype(dtype)


for dtype in (np.float64, np.float32):
    X = binary(23, 12, dtype)
    model = RBM(n_components=16, learning_rate=0.05, batch_size=5, n_iter=4, random_state=3)
    model.fit(X)
    assert model.components_.dtype == dtype and model.transform(X).dtype == dtype
    state(model, X, dtype)
    compare('scores', model.score_samples(X), dtype)
    compare('gibbs', model.gibbs(X), dtype)
    if dtype == np.float64:
        model.partial_fit(X)
        state(model, X, dtype)
        compare('gibbs', model.gibbs(X), dtype)
    rng = np.random.RandomState(11)
    shared = RBM(n_components=6, batch_size=4, n_iter=3, random_state=rng)
    shared.fit(X)
    state(shared, X, dtype)
    compare('scores', shared.score_samples(X), dtype)
    compare('gibbs', shared.gibbs(X), dtype)
    compare('scores', shared.score_samples(X), dtype)
    if dtype == np.float64:
        shared.partial_fit(X)
        state(shared, X, dtype)
    noisy = RBM(n_components=4, batch_size=6, n_iter=2, verbose=1, random_state=np.random.RandomState(5))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        noisy.fit(X)
    reference_verbose.extend(printed.getvalue().splitlines())
    state(noisy, X, dtype)
    idle = RBM(n_components=3, n_iter=0, random_state=8)
    idle.fit(X)
    state(idle, X, dtype)
    model.set_params(n_components=4, learning_rate=0.2, batch_size=7)
    model.fit(X)
    state(model, X, dtype)
    view = X[:, ::-1]
    strided = RBM(n_components=5, batch_size=10, n_iter=2, random_state=1)
    strided.fit(view)
    state(strided, view, dtype)
    compare('scores', strided.score_samples(view), dtype)
assert next(lines, None) is None
assert len(verbose) == 4, verbose  # two epochs per dtype, one line each
pattern = re.compile(r'^(\[BernoulliRBM\] Iteration \d+, pseudo-likelihood = -?\d+\.\d\d), time = \d+\.\d\ds$')
for actual, expected in zip(verbose, reference_verbose, strict=True):
    a, e = pattern.match(actual), pattern.match(expected)
    assert a and e and a.group(1) == e.group(1), (actual, expected)
print('PASS RBM lifecycle parity (seeded and shared-context streams, partial_fit/Gibbs after fit, '
      'verbose, n_iter=0, set_params refit, strided input); worst |error| float64 %.3g float32 %.3g against %s'
      % (worst[np.float64], worst[np.float32], REVISION))
