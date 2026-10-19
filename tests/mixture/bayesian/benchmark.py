"""Small performance baseline: native BayesianGaussianMixture.fit vs sklearn (1 thread).

Same data generator in both (moxie RandomState == NumPy legacy RandomState).
Writes benchmarks/baseline.json with hardware, flags, versions and raw times.
Environment: MOXIE_BENCH_REPEATS (default 5).
"""
from pathlib import Path
import json, os, platform, subprocess, time
import numpy as np
import sklearn
from sklearn.mixture import BayesianGaussianMixture

ROOT = Path(__file__).resolve().parents[3]
REPEATS = int(os.environ.get('MOXIE_BENCH_REPEATS', '5'))
CASES = [(1000, 2, 3), (10000, 5, 4), (20000, 10, 5)]


def data(n, d, k):
    rng = np.random.RandomState(0)
    X = np.empty((n, d))
    for i in range(n):
        for f in range(d):
            X[i, f] = (i % k) * 5 + rng.standard_normal()
    return X


results = []
for n, d, k in CASES:
    X = data(n, d, k)
    for kind in ['full', 'tied', 'diag', 'spherical']:
        raw = subprocess.check_output([str(ROOT / 'builds/tests/mixture/bayesian/benchmark'), str(n), str(d), str(k), kind, str(REPEATS)], text=True)
        native = [float(line.split()[0]) for line in raw.splitlines()]
        native_iters = int(raw.splitlines()[0].split()[1])
        reference = []
        for _ in range(REPEATS):
            model = BayesianGaussianMixture(n_components=k, covariance_type=kind, random_state=0)
            start = time.perf_counter()
            model.fit(X)
            reference.append(time.perf_counter() - start)
        results.append(dict(n=n, d=d, k=k, covariance_type=kind, native_s=native, sklearn_s=reference,
                            native_median=float(np.median(native)), sklearn_median=float(np.median(reference)),
                            native_n_iter=native_iters, sklearn_n_iter=model.n_iter_))
        print(f'{n:6d} {d:3d} {k:2d} {kind:9s} native {np.median(native):.4f}s sklearn {np.median(reference):.4f}s')
report = dict(hardware=platform.processor() or platform.machine(), cpu=subprocess.check_output(['sh', '-c', "grep -m1 'model name' /proc/cpuinfo"], text=True).strip(),
              flags=['-O3', '--fp-mode=contract=off'], mojo=subprocess.check_output(['mojo', '--version'], text=True).strip(),
              sklearn=sklearn.__version__, numpy=np.__version__, threads=dict(OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'), OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS')),
              seeds={'data': 0, 'random_state': 0}, repeats=REPEATS, results=results)
(ROOT / 'benchmarks/mixture/bayesian').mkdir(parents=True, exist_ok=True)
(ROOT / 'benchmarks/mixture/bayesian/baseline.json').write_text(json.dumps(report, indent=2))
