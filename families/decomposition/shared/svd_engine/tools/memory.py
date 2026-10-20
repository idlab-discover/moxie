"""Record GNU time whole-process RSS; no custom profiler."""
from pathlib import Path
import json, os, platform, subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks';OUT.mkdir(exist_ok=True)
for dtype in ('f32','f64'):
    for m,n in [(4,3),(64,16),(128,32),(2048,256)]:
        cmd=['time','-v',str(ROOT/'builds/benchmark'),dtype,str(m),str(n),'1','random']
        p=subprocess.run(cmd,capture_output=True,text=True,check=True)
        (OUT/f'memory-{dtype}-{m}x{n}.txt').write_text(' '.join(cmd)+'\n'+p.stdout+p.stderr)
(OUT/'environment.json').write_text(json.dumps({'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'compiler':subprocess.check_output(['mojo','--version'],text=True).strip(),'flags':'-O3 --fp-mode=contract=off','time':subprocess.check_output(['time','--version'],text=True).splitlines()[0]},indent=2)+'\n')
