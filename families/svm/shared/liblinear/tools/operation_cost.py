"""pyperf comparison for the bounded private TRON reduction."""
from pathlib import Path
import math
import subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def measure(loops,p,mode):
    raw=subprocess.check_output([str(ROOT/'builds/operation_cost'),str(p),str(loops),mode],text=True)
    elapsed,checksum=map(float,raw.split());assert math.isfinite(checksum)
    return elapsed
if __name__=='__main__':
    runner=pyperf.Runner()
    runner.metadata.update({'compiler':'Mojo 1.0.0 -O3 --fp-mode=contract=off','scope':'native timed region; contiguous Float64 private TRON contraction','threads':'one CPU via affinity'})
    for p in (4,64,1024):
        for mode in ('native','numojo'):
            runner.bench_time_func(f'{p}/{mode}',measure,p,mode)
