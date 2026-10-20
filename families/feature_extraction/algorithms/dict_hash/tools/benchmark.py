"""Short pyperf baseline: native elapsed region, output allocation included."""
import math,subprocess
from pathlib import Path
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def measure(loops,dtype,rows,columns):
    raw=subprocess.check_output([str(ROOT/'builds/benchmark'),dtype,str(rows),str(columns),str(loops)],text=True)
    elapsed,checksum=map(float,raw.split());assert math.isfinite(checksum)
    return elapsed
if __name__=='__main__':
    runner=pyperf.Runner()
    runner.metadata.update(compiler='Mojo 1.0.0 -O3 --fp-mode=contract=off',runtime='MAX 26.5.0; NuMojo 515fb2856f0ecf3d2740a34d958fe168183e1129',seed='deterministic (row+column)%7',threads='OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1',timing='native transform including output allocation; input construction and startup excluded')
    for dtype in ['float32','float64']:
        for rows,columns in [(2,4),(64,32),(256,256)]:
            runner.bench_time_func(f'{dtype}/{rows}x{columns}',measure,dtype,rows,columns)
