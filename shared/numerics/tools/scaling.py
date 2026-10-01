"""Repeat Gram workload with CPU affinity and concurrent callers; native times."""
from pathlib import Path
import concurrent.futures,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
records=[]

def once(affinity):
 command=['taskset','-c',affinity,str(ROOT/'builds/benchmark'),'f64','matmul','256','256','C','30']
 p=subprocess.run(command,capture_output=True,text=True,check=True)
 seconds,checksum=map(float,p.stdout.split())
 return {'command':command,'native_seconds':seconds,'checksum':checksum}
for affinity,callers in (('0',1),('0-3',1),('0-3',4)):
 for repeat in range(3):
  with concurrent.futures.ThreadPoolExecutor(max_workers=callers) as pool:
   runs=list(pool.map(once,[affinity]*callers))
  assert all(r['checksum']==runs[0]['checksum'] for r in runs)
  records.append({'affinity':affinity,'callers':callers,'repeat':repeat,'runs':runs})
(ROOT/'benchmarks/scaling.json').write_text(json.dumps(records,indent=2)+'\n')
for affinity in ('0','0-3'):
 command=['taskset','-c',affinity,'perf','stat','-e','task-clock,context-switches,cpu-migrations','-o',str(ROOT/f'profiling/parallel-{affinity}.perf'),str(ROOT/'builds/benchmark'),'f64','matmul','256','256','C','30']
 subprocess.run(command,check=True,capture_output=True,text=True)
