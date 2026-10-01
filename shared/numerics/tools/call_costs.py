"""Count custom allocator calls using Callgrind, which sees TCMalloc directly."""
from pathlib import Path
import json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'profiling'

def calls(path):
 lines=path.read_text().splitlines();names={}
 for line in lines:
  m=re.match(r'(?:c?fn)=\((\d+)\) (.+)',line)
  if m:names[m[1]]=m[2]
 current='';counts={'allocate':0,'delete':0,'copy':0}
 for line in lines:
  m=re.match(r'cfn=\((\d+)\)',line)
  if m:current=names.get(m[1],'')
  if line.startswith('calls='):
   n=int(line.split()[0][6:])
   if 'TCMallocGlobals::tc_new(' in current:counts['allocate']+=n
   if 'TCMallocGlobals::tc_delete(' in current:counts['delete']+=n
   if 'memcpy' in current:counts['copy']+=n
 return counts
results=[]
for dtype in ('f32','f64'):
 for layout in ('C','S'):
  for mode in ('list','fused','expression'):
   totals=[]
   for reps in (0,10):
    stem=f'calls-{dtype}-{layout}-{mode}-{reps}'
    path=out/f'{stem}.callgrind'
    command=['taskset','-c','1','valgrind','--tool=callgrind',f'--callgrind-out-file={path}',str(ROOT/'builds/benchmark'),dtype,mode,'64','32',layout,str(reps)]
    p=subprocess.run(command,capture_output=True,text=True)
    (out/f'{stem}.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError(p.stderr)
    totals.append(calls(path))
   results.append({'dtype':dtype,'layout':layout,'mode':mode,'per_call':{k:(totals[1][k]-totals[0][k])/10 for k in totals[0]},'raw_counts':totals})
(out/'call-costs.json').write_text(json.dumps(results,indent=2)+'\n')
print(results)
