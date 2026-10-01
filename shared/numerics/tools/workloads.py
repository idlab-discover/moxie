"""Centering, shared axis sum and complete Gram expression; use --affinity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark import measure
import pyperf
if __name__=='__main__':
 runner=pyperf.Runner()
 runner.metadata.update({'timing':'native steady-state region','compiler':'Mojo 1.0.0 -O3','threads':'affinity sweep; no unverified MAX thread environment variable','seed':'deterministic integers'})
 for dtype in ('f32','f64'):
  for rows,cols in ((2,4),(64,32),(256,256)):
   for layout in ('C','S'):
    for mode in ('sum','center','matmul'):
     runner.bench_time_func(f'{dtype}/{rows}x{cols}/{layout}/{mode}',measure,dtype,mode,rows,cols,layout)
