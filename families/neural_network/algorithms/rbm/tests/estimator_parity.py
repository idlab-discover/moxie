"""Bounded public lifecycle comparisons against the local pinned RBM methods."""
from kernel_parity import RBM, ROOT, REVISION
import subprocess
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

lines = iter(subprocess.check_output([ROOT / 'builds/estimator_dump'], text=True).splitlines())

def compare(name, expected, dtype):
    fields = next(lines).split()
    assert fields[0] == name, fields
    actual = np.asarray(fields[1:], dtype=float)
    expected = expected.ravel(order='C')
    if name in ('particles', 'gibbs'):
        assert_array_equal(actual, expected)
    else:
        tolerance = 3e-5 if dtype == np.float32 else 2e-12
        assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, err_msg=name)

for dtype in (np.float64, np.float32):
    for n, d, h, batch in [(1, 1, 1, 1), (7, 3, 2, 4), (2, 10, 5, 8)]:
        X = (np.arange(n*d).reshape(n, d) % 7).astype(dtype) / dtype(6)
        model = RBM(n_components=h, batch_size=batch, n_iter=3, random_state=42)
        for repeat in range(2):
            transformed = model.fit(X).transform(X)
            for name, expected in [('weights', model.components_), ('hidden_bias', model.intercept_hidden_),
                                   ('visible_bias', model.intercept_visible_), ('particles', model.h_samples_),
                                   ('transform', transformed), ('scores', model.score_samples(X))]:
                compare(name, expected, dtype)
        compare('gibbs', model.gibbs(X), dtype)
        if dtype == np.float64:
            online = RBM(n_components=h, batch_size=batch, random_state=42)
            for step in range(2):
                online.partial_fit(X)
                compare('online_weights', online.components_, dtype)
                compare('online_transform', online.transform(X), dtype)
assert next(lines, None) is None
print('PASS six public RBM fit/refit/transform/score/Gibbs cases and three Float64 incremental cases against', REVISION)
