"""Whole-process RSS baseline via GNU time; includes runtime overhead."""
from pathlib import Path
import json, platform, subprocess, hashlib
from check import ROOT, build, env
build()
out=ROOT/'benchmarks';out.mkdir(exist_ok=True)
metadata={'platform':platform.platform(),'processor':platform.processor(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('\n\n')[0],
          'compiler':subprocess.check_output(['mojo','--version'],env=env(),text=True).strip(),
          'flags':'-O3 --fp-mode=contract=off','threads':1,'seed':'none',
          'source_sha256':hashlib.sha256((ROOT/'src/moxie/feature_extraction/image.mojo').read_bytes()).hexdigest(),
          'dependency_manifest_sha256':hashlib.sha256((ROOT/'DEPENDENCIES.json').read_bytes()).hexdigest(),
          'dependency_assembly':json.loads((ROOT/'builds/deps/ASSEMBLY.json').read_text()),
          'binary_sha256':hashlib.sha256((ROOT/'builds/test_benchmark').read_bytes()).hexdigest()}
(out/'environment.json').write_text(json.dumps(metadata,indent=2))
for kind,sizes in [('patch',[4,128,1024]),('graph',[4,16,64])]:
    for n in sizes:
        with (out/f'memory-{kind}-{n}.txt').open('w') as log:
            subprocess.run(['time','-f','peak_rss_kib=%M elapsed=%e',str(ROOT/'builds/test_benchmark'),'1',str(n),kind],env=env(),stdout=log,stderr=log,check=True)
