"""Compare precomputed/cross-gradient behavior, including reference errors."""
import subprocess
import numpy as np
from parity import K, ROOT, close


def catalog():
    pre = K.PairwiseKernel(metric='precomputed')
    fixed = K.PairwiseKernel(metric='precomputed', gamma_bounds='fixed')
    rbf = K.PairwiseKernel(metric='rbf', gamma=0.4)
    rbf_fixed = K.PairwiseKernel(metric='rbf', gamma_bounds='fixed')
    return [pre, fixed, pre + pre, pre * pre, pre ** 2,
            pre + fixed, pre * fixed, fixed ** 2,
            K.CompoundKernel([pre, pre]), K.CompoundKernel([fixed, fixed]),
            rbf, rbf_fixed, rbf + rbf, rbf * rbf, K.CompoundKernel([rbf, rbf]),
            pre + K.ConstantKernel(), pre + K.RBF()]


def datasets():
    for dtype in (np.float64, np.float32):
        for name, shape, yshape, has_y in [
            ('square', (3, 3), (3, 3), False),
            ('rectangular', (2, 3), (3, 5), True),
            ('features', (2, 3), (4, 3), True),
            ('nonsquare', (2, 3), (3, 3), False),
            ('singleton', (1, 1), (1, 4), True),
            ('column', (3, 1), (1, 1), True),
            ('row', (1, 3), (3, 3), True),
            ('empty_rows', (0, 3), (3, 3), True),
            ('empty_cols', (3, 0), (0, 3), True),
        ]:
            X = (np.arange(np.prod(shape)).reshape(shape) - 1).astype(dtype) / 8
            Y = np.ones(yshape, dtype=dtype)
            yield f'{name}_{dtype.__name__}', X, Y, has_y
    for name, array, idx, value in [('nan_X', 'X', 0, np.nan), ('inf_X', 'X', 1, np.inf),
                                   ('nan_Y', 'Y', 2, np.nan), ('inf_Y', 'Y', 0, -np.inf)]:
        X, Y = np.eye(3), np.eye(3)
        (X if array == 'X' else Y).flat[idx] = value
        yield name, X, Y, True


def main():
    cases = list(datasets())
    lines = []
    for name, X, Y, has_y in cases:
        tokens = [name, 'yes' if has_y else 'no', 'f32' if X.dtype == np.float32 else 'f64']
        for A in (X, Y):
            tokens.extend(map(str, A.shape))
            tokens.extend(map(str, A.ravel()))
        lines.append(' '.join(tokens))
    path = ROOT / 'builds/tests/gaussian_process/kernels/pairwise-cases.txt'
    path.write_text('\n'.join(lines))
    proc = subprocess.run([str(ROOT / 'builds/tests/gaussian_process/kernels/pairwise_driver'), str(path)],
                          text=True, capture_output=True, check=True)
    (ROOT / 'builds/tests/gaussian_process/kernels/pairwise-native.txt').write_text(proc.stdout)
    records = {}
    for line in proc.stdout.splitlines():
        parts = line.split('|')
        key = parts[0], int(parts[1]), parts[2]
        assert key not in records, key
        records[key] = None if len(parts) == 3 else np.fromstring(parts[4], sep=' ').reshape(
            tuple(map(int, parts[3].split(','))))
    passed = 0
    failures = []
    for name, X, Y, has_y in cases:
        for i, kernel in enumerate(catalog()):
            for operation in ('gradient', 'call', 'diag'):
                label = name, i, operation
                try:
                    with np.errstate(all='ignore'):
                        if operation == 'gradient':
                            # Native arithmetic widens Float32 before forward differences.
                            Xg, Yg = X.astype(float), Y.astype(float)
                            values = kernel(Xg, Yg if has_y else None, eval_gradient=True)
                            expected = dict(zip(('K', 'G'), values))
                        elif operation == 'call':
                            expected = {'call': kernel(X, Y if has_y else None)}
                        else:
                            expected = {'diag': kernel.diag(X)}
                except ValueError:
                    if (name, i, operation + '_error') in records:
                        passed += 1
                    else:
                        failures.append(f'{label}: expected ValueError')
                    continue
                for key, value in expected.items():
                    actual = records.get((name, i, key))
                    if actual is None:
                        failures.append(f'{label}: missing {key}')
                        continue
                    rtol, atol = (2e-5, 2e-5) if key == 'G' else (2e-6, 2e-6) if X.dtype == np.float32 else (1e-12, 1e-13)
                    ok, detail = close(actual, np.asarray(value), rtol, atol)
                    if ok:
                        passed += 1
                    else:
                        failures.append(f'{label}:{key}: {detail}')
    print(f'pairwise parity: {passed} checks passed, {len(failures)} failed, {len(cases)} cases')
    for failure in failures:
        print('FAIL', failure)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
