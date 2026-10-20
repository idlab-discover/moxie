"""pyperf measures native elapsed solve regions, including optimizer allocations."""
from pathlib import Path
import hashlib
import math
import subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]

def measure(loops,n,p,solver):
    output=subprocess.check_output([str(ROOT/'builds/benchmark'),str(n),str(p),str(solver),str(loops)],text=True)
    elapsed,checksum=map(float,output.split())
    if not math.isfinite(checksum): raise ValueError('nonfinite checksum')
    return elapsed

if __name__=='__main__':
    runner=pyperf.Runner()
    runner.metadata.update({'compiler':'Mojo 1.0.0 -O3 --fp-mode=contract=off','storage':'NuMojo 0.10.0 Float64','MAX':'26.5.0','seed':42,'timing':'native perf_counter_ns; output/workspace allocation included; input creation/startup excluded','threads':'one CPU via pyperf affinity; optimizer sequential','source_sha256':hashlib.sha256((ROOT/'src/moxie/svm/_liblinear.mojo').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'DEPENDENCIES.json').read_bytes()).hexdigest()})
    for n,p in ((8,3),(128,16)):
        for solver in (0,1,2,3,4,5,6,7,11,12,13):
            runner.bench_time_func(f'solver-{solver}/{n}x{p}',measure,n,p,solver)
