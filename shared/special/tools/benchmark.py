"""Reproducible native scalar latency using pyperf, with startup excluded."""
from pathlib import Path
import hashlib
import platform
import subprocess
import pyperf

ROOT=Path(__file__).resolve().parents[1]


def timed(loops,a,scale):
    out=subprocess.check_output([ROOT/'builds/benchmark',str(a),str(scale),str(loops)],text=True)
    return float(out.splitlines()[0])


def main():
    runner=pyperf.Runner()
    runner.metadata.update(dict(compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),
        flags='-O3',platform=platform.platform(),thread_policy='serial scalar; pyperf affinity',
        workload='x=a*scale+(i%2)*0.01, sum results; initialization and process startup excluded',
        source_sha256=hashlib.sha256((ROOT/'src/moxie/_special.mojo').read_bytes()).hexdigest(),
        dependency_sha256=hashlib.sha256((ROOT/'DEPENDENCIES.lock.json').read_bytes()).hexdigest(),
        environment_sha256=hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()))
    for a,scale in ((.5,.1),(.5,1),(.5,10),(5,1),(500,1),(50000,1),(1e8,1)):
        runner.bench_time_func(f'gammainc_a{a}_scale{scale}',timed,a,scale)


if __name__ == '__main__':
    main()
