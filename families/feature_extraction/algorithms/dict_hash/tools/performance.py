"""Compile and capture repeatable short timing/memory baselines after parity."""
from pathlib import Path
import hashlib,json,os,platform,sys
from datetime import datetime, timezone
from check import ROOT,compile,run
out=ROOT/'benchmarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
compile(ROOT/'tools/benchmark.mojo',ROOT/'builds/benchmark')
command=[sys.executable,ROOT/'tools/benchmark.py','--processes','1','--values','2','--warmups','1','--min-time','0.01','-o',out/'transform.json']
print(run(*command).stdout)
records=[]
for rows,cols in [(64,32),(256,256),(1024,512)]:
    result=run('time','-v',ROOT/'builds/benchmark','float64',str(rows),str(cols),'1')
    (out/f'memory-{rows}x{cols}.txt').write_text(result.stdout+result.stderr)
    records.append(dict(rows=rows,columns=cols,dense_output_bytes=rows*cols*8))
metadata=dict(source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'tools/benchmark.mojo',ROOT/'src/moxie/feature_extraction/_dict_vectorizer.mojo',ROOT/'pixi.lock']},load_average=os.getloadavg(),command=list(map(str,command)),platform=platform.platform(),cpu=run('lscpu').stdout,compiler=run('mojo','--version').stdout,flags=['-O3','--fp-mode=contract=off'],memory=records,memory_scope='GNU time whole-process max RSS includes metadata input; not an isolated array allocation count',warning='short smoke baseline; no performance advantage claim')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
