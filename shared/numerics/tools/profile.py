"""Keep existing-profiler output and hardware/tool metadata locally."""
from pathlib import Path
import hashlib,json,os,platform,subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'profiling'; out.mkdir(exist_ok=True)
metadata={'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'binary_sha256':hashlib.sha256((ROOT/'builds/benchmark').read_bytes()).hexdigest(),'flags':'-O3','seed':'deterministic modular integers / 8','runs':[]}
for cmd in (['mojo','--version'],['valgrind','--version'],['perf','--version'],['lscpu'],['ldd',str(ROOT/'builds/benchmark')]):
    p=subprocess.run(cmd,capture_output=True,text=True)
    metadata[' '.join(cmd)]=p.stdout+p.stderr
for mode in ('list','fused','expression'):
    cmd=['taskset','-c','1','valgrind','--tool=memcheck','--leak-check=full','--error-exitcode=99',f'--log-file={out/mode}.memcheck',str(ROOT/'builds/benchmark'),'f64',mode,'64','32','S','10']
    p=subprocess.run(cmd,capture_output=True,text=True)
    metadata['runs'].append({'command':cmd,'status':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    cmd=['taskset','-c','1','valgrind','--tool=massif',f'--massif-out-file={out/mode}.massif',str(ROOT/'builds/benchmark'),'f64',mode,'256','256','S','10']
    p=subprocess.run(cmd,capture_output=True,text=True)
    metadata['runs'].append({'command':cmd,'status':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print([(r['command'][4],r['status']) for r in metadata['runs']])
