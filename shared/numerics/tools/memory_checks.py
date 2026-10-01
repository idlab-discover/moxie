"""Run Memcheck with any caller-configured debuginfod service and local cache."""
from pathlib import Path
import json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'profiling';out.mkdir(exist_ok=True)
env=dict(os.environ)
env['DEBUGINFOD_CACHE_PATH']=str(ROOT/'.pixi/debuginfod')
cases=[['contracts'],['edges'],['exchange']]
for dtype in ('f32','f64'):
 for layout in ('C','S','T','O','R'):
  cases.append(['evaluate',dtype,'17','7',layout])
 for mode in ('list','fused','expression','matmul','center'):
  cases.append(['benchmark',dtype,mode,'64','32','S','3'])
records=[]
for i,case in enumerate(cases):
 cmd=['valgrind','--tool=memcheck','--leak-check=full','--error-exitcode=99',f'--log-file={out}/memory-check-{i}.log',str(ROOT/'builds'/case[0]),*case[1:]]
 p=subprocess.run(cmd,capture_output=True,text=True,env=env)
 records.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
(out/'memory-checks.json').write_text(json.dumps({'debuginfod_urls':env.get('DEBUGINFOD_URLS'),'runs':records},indent=2)+'\n')
assert all(r['returncode']==0 for r in records), 'See profiling/memory-checks.json'
print(f'PASS: {len(cases)} Memcheck cases')
