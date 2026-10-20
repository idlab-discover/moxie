"""Small reproducible timing baselines for children (writes ignored benchmarks/).

`record(root, name, rows)` stores rows with host/toolchain metadata. Native
timings come from each child's compiled benchmark driver (median of repeats,
fit only, data generation excluded); sklearn timings use the installed support
wheel with `timeit` (best of repeats) on same-shaped data, single-threaded BLAS
(MOXIE_THREADS, default 1). These are baselines, not tuned speed claims.
"""
from pathlib import Path
import json
import platform
import subprocess
import timeit

from childcheck import FLAGS, environment


def native(root, *args):
    out = subprocess.run([str(root / 'builds/benchmark'), *map(str, args)], text=True, capture_output=True,
                         check=True, env=environment(root)).stdout
    rows = []
    for line in out.splitlines():
        dtype, n, p, seconds = line.split()[:4]
        rows.append(dict(dtype=dtype, n=int(n), p=int(p), seconds=float(seconds)))
    return rows


def best(function, repeats=5):
    return min(timeit.repeat(function, number=1, repeat=repeats))


def record(root, name, rows):
    cpu = next((l.split(':', 1)[1].strip() for l in subprocess.check_output(['lscpu'], text=True).splitlines()
                if l.startswith('Model name')), platform.processor())
    import numpy, scipy, sklearn
    meta = dict(benchmark=name, cpu=cpu, platform=platform.platform(), mojo_flags=FLAGS,
                mojo=subprocess.check_output(['mojo', '--version'], text=True).strip(),
                numpy=numpy.__version__, scipy=scipy.__version__, sklearn_support=sklearn.__version__,
                threads=environment(root)['OMP_NUM_THREADS'], rows=rows)
    folder = root / 'benchmarks'
    folder.mkdir(exist_ok=True)
    (folder / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    for row in rows:
        print(row)
