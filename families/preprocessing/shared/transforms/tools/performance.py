"""Short reproducible baseline, no throughput claims; run after parity."""
from pathlib import Path
import subprocess,sys,json,platform,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'benchmarks/performance';out.mkdir(parents=True,exist_ok=True)
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
commands=[]
for dtype in ('f32','f64'):
 for n,p in ((2,4),(256,256)):
  for mode in ('copy','baseline','reference'):
   path=out/f'{dtype}-{n}-{p}-{mode}.json';path.unlink(missing_ok=True)
   cmd=[sys.executable,str(ROOT/'tools/benchmark.py'),'--dtype',dtype,'--mode',mode,'--rows',str(n),'--cols',str(p),'--processes','2','--values','3','--warmups','1','--min-time','0.01','-o',str(path)]
   result=subprocess.run(cmd,text=True,capture_output=True,check=True)
   path.with_suffix('.log').write_text(result.stdout+result.stderr);commands.append(cmd)
metadata=dict(commands=commands,platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text(),toolchain=subprocess.check_output(['mojo','--version'],text=True),flags='-O3 --fp-mode=contract=off',seed='deterministic modular pattern',threads=1,environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},contention='Concurrent runner peers possible; short baseline only',source_sha256=hashlib.sha256((ROOT/'src/moxie/preprocessing/_dense_transforms.mojo').read_bytes()).hexdigest())
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved 12 short pyperf baselines')
