"""GPR timing baseline (native vs installed sklearn); writes benchmarks/baseline.json.

Env: GP_BENCH_N (default 150), GP_BENCH_D (3), GP_BENCH_REPEATS (3). Best-of-repeats,
single thread. Inputs have the same shape/model but different random values, and
optimizer iteration counts can differ, so fit timings are indicative only.
"""
from pathlib import Path
import json, os, platform, subprocess, time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
n = int(os.environ.get('GP_BENCH_N', 150)); d = int(os.environ.get('GP_BENCH_D', 3)); repeats = int(os.environ.get('GP_BENCH_REPEATS', 3))
out = subprocess.run([str(ROOT / 'builds/tests/gaussian_process/regression/benchmark'), str(n), str(d), str(repeats)], check=True, text=True, capture_output=True).stdout
native = {l.split('|')[1]: float(l.split('|')[2]) for l in out.splitlines() if l.startswith('T|')}
X = np.random.RandomState(0).uniform(-3, 3, (n, d)); y = np.sin(X[:, 0])
ref = {'fit': float('inf'), 'predict_std': float('inf')}
for _ in range(repeats):
    g = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2))
    t = time.perf_counter(); g.fit(X, y); ref['fit'] = min(ref['fit'], time.perf_counter() - t)
    t = time.perf_counter(); g.predict(X, return_std=True); ref['predict_std'] = min(ref['predict_std'], time.perf_counter() - t)
result = dict(n=n, d=d, repeats=repeats, native_seconds=native, sklearn_seconds=ref, host=platform.machine(),
              flags='-O3 --fp-mode=contract=off', threads='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
(ROOT / 'benchmarks/gaussian_process/regression').mkdir(parents=True, exist_ok=True)
(ROOT / 'benchmarks/gaussian_process/regression/baseline.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
