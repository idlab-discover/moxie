"""Exact selection and weighted relocation checks against unmodified NumPy/sklearn."""
import itertools
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cases():
    # Exhaustive duplicate/tie arrangements for the four-lane base network.
    for n in range(1, 5):
        for data in itertools.product([0., 1., 2.], repeat=n):
            yield np.asarray(data)
    rng = np.random.RandomState(917)
    for n in [5, 7, 8, 9, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129,
              255, 256, 257, 258, 271, 272, 300, 511, 512, 1023, 1024, 4097]:
        yield np.zeros(n)
        yield np.ones(n)
        yield np.arange(n, dtype=float) / 8
        yield np.arange(n, dtype=float)[::-1] / 8
        yield np.roll(np.arange(n, dtype=float), n // 2) / 8
        yield np.minimum(np.arange(n), np.arange(n)[::-1]).astype(float) / 8
        yield rng.randint(-8, 9, n).astype(float) / 4
        yield rng.randint(-64, 65, n).astype(float) / 8


def main(common):
    count = 0
    fallback_count = 0
    fallback = ROOT / 'builds/selection_fallback'
    subprocess.run(['c++', '-O2', '-ffp-contract=off', '-std=c++17',
                    str(ROOT / 'tests/selection_fallback.cpp'), '-o', str(fallback)], check=True)
    for name, dtype, tol in [('f32', np.float32, 3e-5), ('f64', np.float64, 1e-12)]:
        for raw in cases():
            x = raw.astype(dtype).reshape(-1, 1)
            n = len(x)
            # Exercise the emergency std::sort branch independently, even when
            # the median-of-four selection does not exhaust its depth budget.
            if n >= 16:
                args = [str(ROOT / 'builds/selection_driver'),
                        'sort32' if dtype == np.float32 else 'sort64', str(n), *map(str, raw)]
                output = subprocess.check_output(args, text=True)
                actual = [int(float(line.split()[1])) for line in output.splitlines() if line.startswith('index ')]
                expected = subprocess.check_output([str(fallback), name, *map(str, raw)], text=True)
                np.testing.assert_array_equal(actual, list(map(int, expected.split())),
                                              err_msg=str((name, n, 'std::sort fallback')))
                fallback_count += 1
            for empty in sorted({1, min(2, n), min(5, n), max(1, n // 2), n}):
                expected = np.argpartition((x * x).sum(axis=1), -empty)[:-empty-1:-1]
                args = [str(ROOT / 'builds/selection_driver'), name, str(empty),
                        *map(str, raw)]
                output = subprocess.check_output(args, text=True)
                fields = {'index': [], 'center': [], 'mass': []}
                for line in output.splitlines():
                    key, value = line.split()
                    fields[key].append(float(value))
                context = (name, n, empty, raw[:8].tolist())
                np.testing.assert_array_equal(fields['index'], expected, err_msg=str(context))
                if empty < n:
                    w = (np.arange(n) % 13 + 1).astype(dtype) / 4
                    old = np.zeros((empty + 1, 1), dtype=dtype)
                    centers = old.copy()
                    masses = np.zeros(empty + 1, dtype=dtype)
                    labels = np.zeros(n, dtype=np.int32)
                    # Same ordered accumulation as the Cython assignment step.
                    for i in range(n):
                        centers[0, 0] += x[i, 0] * w[i]
                        masses[0] += w[i]
                    common._relocate_empty_clusters_dense(x, w, old, centers, masses, labels)
                    # Reference normalization/fallback policy, including zero residuals.
                    largest = masses.argmax()
                    for i in range(empty + 1):
                        if masses[i] > 0:
                            centers[i] *= dtype(1) / masses[i]
                        else:
                            centers[i] = centers[largest]
                    np.testing.assert_array_equal(fields['mass'], masses, err_msg=str(context))
                    np.testing.assert_allclose(fields['center'], centers[:, 0], rtol=tol, atol=tol,
                                               err_msg=str(context))
                count += 1
    report = {'count': count, 'fallback_count': fallback_count, 'numpy': np.__version__, 'numpy_revision': np.version.git_revision,
              'selection': 'exact indices, unmodified NumPy argpartition',
              'relocation': 'exact masses; centers float32 3e-5 / float64 1e-12'}
    (ROOT / 'docs/private/centroid/selection-parity.json').write_text(json.dumps(report, indent=2))
    print(f'PASS {count} exact selection / weighted relocation cases; {fallback_count} std::sort fallbacks')


if __name__ == '__main__':
    from oracle import load
    main(load()[0])
