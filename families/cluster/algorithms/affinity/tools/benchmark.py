"""pyperf baseline for complete dense fitting; see docs/api.md."""
from pathlib import Path
import hashlib, json, os, platform, subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]

def timed(loops,dtype,n):
    out=subprocess.check_output([ROOT/'builds/benchmark',dtype,str(n),str(loops)],text=True)
    return float(out.splitlines()[0])

def main():
    runner=pyperf.Runner()
    runner.metadata.update({'compiler':subprocess.check_output(['mojo','--version'],text=True).strip(),
        'flags':'-O3 --fp-mode=contract=off','seed':42,'max_iter':40,
        'platform':platform.platform(),'thread_policy':'one CPU affinity; MAX inner matmul',
        'numpy':__import__('numpy').__version__,'pyperf':pyperf.__version__,
        'source_sha256':hashlib.sha256((ROOT/'src/moxie/cluster/_affinity_propagation.mojo').read_bytes()).hexdigest(),
        'dependency_manifest_sha256':hashlib.sha256((ROOT/'DEPENDENCIES.json').read_bytes()).hexdigest(),
        'lock_sha256':hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()})
    for dtype in ['32','64']:
        for n in [8,64,256]:runner.bench_time_func(f'fit_{dtype}_{n}',timed,dtype,n)
if __name__=='__main__': main()
