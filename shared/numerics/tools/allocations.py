"""DHAT allocation differences between zero and ten workload repetitions."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'profiling';out.mkdir(exist_ok=True)
results=[]
for dtype in ('f32','f64'):
 for layout in ('C','S'):
  for mode in ('list','fused','expression'):
   measurements=[]
   for reps in (0,10):
    stem=f'{dtype}-{layout}-{mode}-{reps}'
    command=['taskset','-c','1','valgrind','--tool=dhat',f'--dhat-out-file={out/stem}.dhat',str(ROOT/'builds/benchmark'),dtype,mode,'64','32',layout,str(reps)]
    p=subprocess.run(command,capture_output=True,text=True)
    (out/f'{stem}.log').write_text(p.stdout+p.stderr)
    if p.returncode: raise RuntimeError(p.stderr)
    j=json.loads((out/f'{stem}.dhat').read_text())
    measurements.append({'reps':reps,'bytes':sum(x['tb'] for x in j['pps']),'blocks':sum(x['tbk'] for x in j['pps']),'command':command})
   results.append({'dtype':dtype,'layout':layout,'mode':mode,'bytes_per_call':(measurements[1]['bytes']-measurements[0]['bytes'])/10,'blocks_per_call':(measurements[1]['blocks']-measurements[0]['blocks'])/10,'raw':measurements})
report={'status':'INCONCLUSIVE_FOR_CUSTOM_ALLOCATOR','reason':'DHAT may not intercept Modular TCMalloc; compare Callgrind tc_new counts before interpreting deltas.','observations':results}
(out/'allocations.json').write_text(json.dumps(report,indent=2)+'\n')
print([(x['dtype'],x['layout'],x['mode'],x['bytes_per_call'],x['blocks_per_call']) for x in results])
