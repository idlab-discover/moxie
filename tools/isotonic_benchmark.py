"""Run after parity. pyperf records native regions; GNU time records process RSS."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pyperf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'MODULAR_CACHE_DIR': '.pixi/modular-cache',
                    'PIXI_CACHE_DIR': '.pixi/pixi-cache', 'TMPDIR': 'builds/tmp'}.items():
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'
binary = ROOT / 'builds/isotonic/benchmark'
output = ROOT / 'benchmarks/isotonic'
output.mkdir(parents=True, exist_ok=True)
if '--worker' not in sys.argv:
    subprocess.run([sys.executable, 'tools/isotonic_checks.py', 'build'], check=True)
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', 'builds/deps',
                    'tests/isotonic/benchmark.mojo', '-o', str(binary)], check=True)
    metadata = {'command': sys.argv, 'compiler': subprocess.check_output(['mojo', '--version'], text=True),
                'flags': '-O3 --fp-mode=contract=off', 'hardware': platform.platform(),
                'cpu': Path('/proc/cpuinfo').read_text(), 'seed': 'deterministic arithmetic inputs',
                'threads': 1, 'pyperf': pyperf.__version__,
                'source_sha256': hashlib.sha256((ROOT/'src/moxie/isotonic.mojo').read_bytes()).hexdigest(),
                'memory_note': 'GNU time whole-process maximum RSS, including input allocation'}
    (output/'environment.json').write_text(json.dumps(metadata, indent=2))
    for n in (16, 1024, 65536):
        with (output/f'rss-{n}.txt').open('w') as log:
            subprocess.run(['time', '-v', str(binary), 'float64', str(n), '1', 'fit'],
                           stdout=log, stderr=log, check=True)

def elapsed(loops, dtype, n, mode):
    result = subprocess.check_output([str(binary), dtype, str(n), str(loops), mode], text=True)
    return float(result.splitlines()[0])

runner = pyperf.Runner()
runner.metadata['timing_region'] = 'native fit+predict or weighted PAVA; excludes startup and input creation'
for dtype in ('float32', 'float64'):
    for n in (16, 1024, 65536):
        for mode in ('kernel', 'fit'):
            runner.bench_time_func(f'{dtype}-{mode}-{n}', elapsed, dtype, n, mode)
