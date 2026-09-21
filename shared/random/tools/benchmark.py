"""pyperf calibration around native elapsed regions, with startup excluded."""
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys
import pyperf
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'builds/benchmark'
def timed(loops,mode,n):
    output=subprocess.check_output([BINARY,mode,str(n),str(loops)],text=True)
    return float(output.splitlines()[0])
def main():
    runner=pyperf.Runner()
    runner.metadata.update({'compiler':subprocess.check_output(['mojo','--version'],text=True).strip(),
        'flags':'-O3 --fp-mode=contract=off','seed':42,'rng_state_bytes':624*4+8+8+1,
        'platform':platform.platform(),'thread_policy':'serial RNG; one CPU affinity',
        'numpy':__import__('numpy').__version__,
        'source_sha256':hashlib.sha256((ROOT/'src/moxie/_random.mojo').read_bytes()).hexdigest(),
        'lock_sha256':hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()})
    for mode in ['uniform','normal','native','permutation','log_intrinsic','log_stdlib']:
        for n in [16,10000]: runner.bench_time_func(f'{mode}_{n}',timed,mode,n)
if __name__=='__main__': main()
