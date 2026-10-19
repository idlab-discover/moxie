"""Small timing baseline: native kernels (with gradient) vs installed sklearn.

Writes benchmarks/baseline.json (ignored). Environment: GP_BENCH_N, GP_BENCH_D,
GP_BENCH_REPEATS. Best-of-repeats wall time, single thread, same inputs shape
(values differ: native LCG vs NumPy uniform; timings do not depend on values).
"""
from pathlib import Path
import json, os, platform, subprocess, time
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
n = int(os.environ.get('GP_BENCH_N', 300)); d = int(os.environ.get('GP_BENCH_D', 5))
repeats = int(os.environ.get('GP_BENCH_REPEATS', 5))
out = subprocess.run([str(ROOT / 'builds/tests/gaussian_process/kernels/benchmark'), str(n), str(d), str(repeats)], check=True, text=True, capture_output=True).stdout
native = {line.split('|')[1]: float(line.split('|')[2]) for line in out.splitlines() if line.startswith('T|')}
X = np.random.RandomState(0).uniform(size=(n, d))
reference = {}
for name, k in [('C*RBF+White', ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)),
                ('Matern2.5', Matern(1.0, nu=2.5)), ('Matern0.75', Matern(1.0, nu=0.75))]:
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter(); k(X, eval_gradient=True); best = min(best, time.perf_counter() - start)
    reference[name] = best
result = dict(n=n, d=d, repeats=repeats, native_seconds=native, sklearn_seconds=reference,
              host=platform.processor() or platform.machine(), flags='-O3 --fp-mode=contract=off',
              threads='OMP_NUM_THREADS=1', note='best-of-repeats wall time; native includes Float64 input copy')
(ROOT / 'benchmarks/gaussian_process/kernels').mkdir(parents=True, exist_ok=True)
(ROOT / 'benchmarks/gaussian_process/kernels/baseline.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
