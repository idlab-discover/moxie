"""pyperf driver for native timed regions; includes runtime certificates."""
from pathlib import Path
import hashlib, os, platform, subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def timed(loops,dtype,m,n,mode):
    result=subprocess.check_output([ROOT/'builds/benchmark',dtype,str(m),str(n),str(loops),mode],text=True)
    return float(result.split()[0])
runner=pyperf.Runner()
runner.metadata.update({'compiler':subprocess.check_output(['mojo','--version'],text=True).strip(),'flags':'-O3 --fp-mode=contract=off','seed':42,'platform':platform.platform(),'thread_policy':'single CPU affinity; MAX inner workers bounded by affinity','source_sha256':hashlib.sha256((ROOT/'src/moxie/decomposition/_svd_engine.mojo').read_bytes()).hexdigest(),'lock_sha256':hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest()})
for dtype in ['f32','f64']:
    for m,n in [(4,3),(64,16),(128,32)]:
        for mode in ['full','gesvd','random']:
            runner.bench_time_func(f'{dtype}_{m}x{n}_{mode}',timed,dtype,m,n,mode)
