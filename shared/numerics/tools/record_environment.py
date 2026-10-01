"""Capture reproducibility metadata without changing system settings."""
from pathlib import Path
import hashlib,json,os,platform,subprocess
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'pixi.lock',ROOT/'DEPENDENCIES.lock.json',*sorted((ROOT/'src').rglob('*.mojo')),*sorted((ROOT/'tests').rglob('*.mojo')),ROOT/'builds/benchmark']
data={'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'flags':'-O3','seed':'deterministic modular integer formula in tests/benchmark.mojo','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MODULAR_NUM_THREADS','MOJO_NUM_THREADS')}}
for cmd in (['mojo','--version'],['python','-m','pyperf','--version'],['valgrind','--version'],['perf','--version'],['lscpu'],['ldd',str(ROOT/'builds/benchmark')]):
 p=subprocess.run(cmd,capture_output=True,text=True); data[' '.join(cmd)]=p.stdout+p.stderr
(ROOT/'benchmarks/environment.json').write_text(json.dumps(data,indent=2)+'\n')
