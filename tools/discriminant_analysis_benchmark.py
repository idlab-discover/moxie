"""Performance baseline: native fit+predict_proba vs the pinned sklearn oracle.

Run after `pixi run build`: `pixi run python tools/benchmark.py`. Results go to the
ignored benchmarks/ directory. Native timing is measured inside the process
(startup excluded); sklearn timing uses the same deterministic dataset.
Environment: MOXIE_BENCH_REPEATS (default 7).
"""
import json, os, platform, subprocess, sys, time, hashlib
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'  # single-threaded BLAS for the sklearn side, set before NumPy loads
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/discriminant_analysis'))
from oracle import load

REPEATS = int(os.environ.get('MOXIE_BENCH_REPEATS', '7'))
MODES = ['lda-svd', 'lda-lsqr', 'lda-eigen', 'qda-svd', 'qda-eigen']
SIZES = [(300, 5), (3000, 20), (20000, 50)]


def dataset(n, p):
    """Same LCG stream as tests/benchmark.mojo (seed 42)."""
    mask = (1 << 64) - 1
    state = 42
    X = np.zeros((n, p))
    y = np.arange(n) % 3
    for i in range(n):
        for j in range(p):
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            X[i, j] = (state >> 11) / 9007199254740992.0 + y[i] * (0.5 if j % 2 == 0 else 0.1)
    return X, y


def main():
    da = load()['da']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    records = []
    for n, p in SIZES:
        X, y = dataset(n, p)
        for mode in MODES:
            loops = max(1, int(3000 * 20 / (n * p)))
            native = []
            for _ in range(REPEATS):
                out = subprocess.check_output([str(ROOT / 'builds/tests/discriminant_analysis/benchmark'), mode, str(n), str(p), str(loops)],
                                              text=True, env=env)
                native.append(float(out.splitlines()[0]))
            cls = da.LinearDiscriminantAnalysis if mode.startswith('lda') else da.QuadraticDiscriminantAnalysis
            reference = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                for _ in range(loops):
                    cls(solver=mode[4:]).fit(X, y).predict_proba(X)
                reference.append((time.perf_counter() - start) / loops)
            records.append(dict(mode=mode, n_samples=n, n_features=p, loops=loops,
                                native_median_s=float(np.median(native)), sklearn_median_s=float(np.median(reference)),
                                native_raw=native, sklearn_raw=reference))
            print(f'{mode:10s} n={n:6d} p={p:3d} native={np.median(native) * 1e3:9.3f} ms '
                  f'sklearn={np.median(reference) * 1e3:9.3f} ms')
    out = ROOT / 'benchmarks/discriminant_analysis'
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        'platform': platform.platform(), 'cpu': subprocess.check_output(['lscpu'], text=True),
        'compiler': subprocess.check_output(['mojo', '--version'], text=True).strip(),
        'flags': os.environ.get('MOXIE_MOJO_FLAGS', '-O3 --fp-mode=contract=off'),
        'threads': 'native serial; sklearn BLAS single-threaded (OMP/OPENBLAS/MKL_NUM_THREADS=1)',
        'numpy': np.__version__, 'seed': 42, 'repeats': REPEATS,
        'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted((ROOT / 'src').rglob('*.mojo'))},
    }
    (out / 'baseline.json').write_text(json.dumps(dict(metadata=metadata, records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
