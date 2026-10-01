"""Whole-process peak RSS, not a substitute for allocation counts."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'profiling';records=[]
for dtype in ('f32','f64'):
 for rows,cols in ((2,4),(64,32),(2048,512)):
  for mode in ('list','fused','expression'):
   log=out/f'rss-{dtype}-{rows}-{cols}-{mode}.txt'
   command=['taskset','-c','1',str(ROOT/'.pixi/envs/default/bin/time'),'-v','-o',str(log),str(ROOT/'builds/benchmark'),dtype,mode,str(rows),str(cols),'S','3']
   p=subprocess.run(command,capture_output=True,text=True,check=True)
   records.append({'command':command,'stdout':p.stdout,'time_output':log.read_text()})
(out/'rss.json').write_text(json.dumps(records,indent=2)+'\n')
