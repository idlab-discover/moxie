"""Native transform versus simple loop; native timed region excludes startup."""
from pathlib import Path
import os, subprocess, re
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def add_cmd(cmd,args):
 for k in ('dtype','mode','rows','cols'):cmd.extend(['--'+k,str(getattr(args,k))])
runner=pyperf.Runner(add_cmdline_args=add_cmd)
runner.argparser.add_argument('--dtype',choices=['f32','f64'],default='f64')
runner.argparser.add_argument('--mode',choices=['copy','baseline','reference'],default='copy')
runner.argparser.add_argument('--rows',type=int,default=256)
runner.argparser.add_argument('--cols',type=int,default=256)
args=runner.parse_args()
runner.metadata['compiler']='Mojo 1.0.0 -O3 --fp-mode=contract=off'
runner.metadata['threads']=1
runner.metadata['seed']='deterministic modular pattern'
runner.metadata['contention']='Runner peers may be active; no speedup claim'
if args.mode=='reference':
 import numpy as np, time
 x=(np.arange(args.rows*args.cols)%31-15).astype(np.float32 if args.dtype=='f32' else np.float64).reshape(args.rows,args.cols)/8
 def timed(loops):
  start=time.perf_counter()
  for _ in range(loops):
   y=x.copy();y-=.25;y/=2
   float(y.sum(dtype=np.float64))
  return time.perf_counter()-start
else:
 def timed(loops):
  output=subprocess.check_output([str(ROOT/'builds/workload'),args.dtype,args.mode,str(args.rows),str(args.cols),'0','65536',str(loops)],text=True)
  return float(re.search(r'seconds (\S+)',output)[1])
runner.bench_time_func(f'{args.dtype}-{args.rows}x{args.cols}-{args.mode}',timed)
