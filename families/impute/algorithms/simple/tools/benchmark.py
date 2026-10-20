"""pyperf drives native timed fit/transform regions, excluding process startup."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def timed(loops, rows):
    result=subprocess.check_output([ROOT/'builds/benchmark',str(rows),str(loops)],text=True)
    return float(result.splitlines()[0])
def main():
    runner=pyperf.Runner()
    runner.metadata.update(dict(compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),flags='-O3 --fp-mode=contract=off',platform=platform.platform(),seed='deterministic i%101, missing i%7==0',threads='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',source_sha256=hashlib.sha256((ROOT/'src/moxie/impute/_simple.mojo').read_bytes()).hexdigest(),lock_sha256=hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()))
    for rows in (16,1024): runner.bench_time_func(f'mean_fit_transform_{rows}x16',timed,rows)
if __name__=='__main__': main()
