"""Compare native adapter and backend costs with pyperf; no timing substitute."""
from pathlib import Path
import hashlib
import platform
import subprocess

import pyperf

ROOT = Path(__file__).resolve().parents[1]


def timed(loops, bits, mode, n, layout):
    out = subprocess.check_output([ROOT/'builds/benchmark',str(bits),mode,str(n),str(loops),layout],text=True)
    return float(out.splitlines()[0])


def main():
    runner = pyperf.Runner()
    runner.metadata.update(dict(compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),
        flags='-O3', platform=platform.platform(), thread_policy='serial Cholesky and solves; pyperf CPU affinity',
        workload='deterministic SPD n+1 diagonal, 0.25 offdiagonal, four RHS, alternating first diagonal',
        source_sha256=hashlib.sha256((ROOT/'src/moxie/_linalg.mojo').read_bytes()).hexdigest(),
        dependency_sha256=hashlib.sha256((ROOT/'DEPENDENCIES.lock.json').read_bytes()).hexdigest(),
        environment_sha256=hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()))
    for bits in (32,64):
        for n in (8,64,128):
            for layout in ('contiguous','strided'):
                for mode in ('backend','adapter'):
                    runner.bench_time_func(f'{mode}_{bits}_{n}_{layout}',timed,bits,mode,n,layout)


if __name__ == '__main__':
    main()
