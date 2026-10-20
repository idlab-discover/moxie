"""pyperf times native regions; startup and source compilation excluded."""
from pathlib import Path
import sys,subprocess,pyperf
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check import ROOT,env,FLAGS
r=pyperf.Runner()
r.metadata.update(compiler='Mojo 1.0.0',max_version='26.5.0',flags=' '.join(FLAGS),threads=1,seed='none; exhaustive')
def bench(loops,n,kind):
    s=subprocess.check_output([str(ROOT/'builds/test_benchmark'),str(loops),str(n),kind],env=env(),text=True)
    return float(s.splitlines()[0])
for kind in ['patch','graph']:
    for n in [4,16,32]:r.bench_time_func(f'{kind}-{n}x{n}',bench,n,kind)
