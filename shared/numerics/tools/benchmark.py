"""pyperf consumes native elapsed seconds, excluding subprocess startup."""
from pathlib import Path
import math
import os
import subprocess
import sys
import pyperf

ROOT=Path(__file__).resolve().parents[1]

def measure(loops,dtype,mode,rows,cols,layout):
    raw=subprocess.check_output([str(ROOT/'builds/benchmark'),dtype,mode,str(rows),str(cols),layout,str(loops)],text=True)
    elapsed,checksum=map(float,raw.split())
    if not math.isfinite(checksum): raise RuntimeError('nonfinite checksum')
    return elapsed

if __name__=='__main__':
    runner=pyperf.Runner()
    runner.metadata.update({'timing':'native perf_counter_ns region; allocation of output included; construction excluded','compiler':'Mojo 1.0.0 -O3','seed':'deterministic modular integers / 8','threads':'CPU affinity; see pyperf metadata','storage':'NuMojo live source; see assembly receipt'})
    for dtype in ('f32','f64'):
        for rows,cols in ((2,4),(64,32),(256,256)):
            for layout in ('C','S'):
                baseline=None
                for mode in ('list','fused','expression'):
                    if '--worker' not in sys.argv:
                        raw=subprocess.check_output([str(ROOT/'builds/benchmark'),dtype,mode,str(rows),str(cols),layout,'3'],text=True)
                        checksum=float(raw.split()[1])
                        if baseline is None: baseline=checksum
                        assert math.isclose(checksum,baseline,rel_tol=3e-5,abs_tol=3e-5)
                    runner.bench_time_func(f'{dtype}/{rows}x{cols}/{layout}/{mode}',measure,dtype,mode,rows,cols,layout)
