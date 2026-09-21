"""GNU time memory-scale baseline; raw reports remain local."""
from pathlib import Path
import hashlib, json, os, platform, subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks'; OUT.mkdir(exist_ok=True)
metadata={'platform':platform.platform(),'cpu':subprocess.check_output(['lscpu'],text=True),
 'compiler':subprocess.check_output(['mojo','--version'],text=True),'flags':'-O3 --fp-mode=contract=off',
 'source_sha256':hashlib.sha256((ROOT/'src/moxie/_random.mojo').read_bytes()).hexdigest(),'lock_sha256':hashlib.sha256((ROOT/'pixi.lock').read_bytes()).hexdigest(),'time_version':subprocess.check_output(['time','--version'],text=True),'seed':42,'threads':'serial','affinity':sorted(os.sched_getaffinity(0))}
for n in [16,10000,1000000]:
 args=['time','-v','-o',str(OUT/f'memory-{n}.txt'),str(ROOT/'builds/benchmark'),'permutation',str(n),'1']
 subprocess.run(args,check=True,stdout=subprocess.DEVNULL)
 metadata[f'command_{n}']=args
(OUT/'environment.json').write_text(json.dumps(metadata,indent=2))
