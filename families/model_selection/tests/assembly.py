"""Exercise this family's manifests against the shared assembler."""
from pathlib import Path
import copy, hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY=ROOT.parents[1]/'shared/assembly'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
consumers=[ROOT/'tests/consumers'/n for n in ('a','b')]
for module in ('model_selection/_evaluation','model_selection/_indices','_dense','_random'):
    a,b=[c/f'builds/deps/moxie/{module}.mojo' for c in consumers]
    assert sha(a)==sha(b)
    assert a.stat().st_ino!=b.stat().st_ino
base=json.loads((ROOT/'DEPENDENCIES.json').read_text())
for p in base['projects']: p['path']=str((ROOT/p['path']).resolve())
ASSEMBLY=Path(next(p for p in base['projects'] if p['id']=='assembly')['path'])
report=ROOT/'builds/deps/ASSEMBLY.json'; before=sha(report)
def rejected(lock, diagnostic):
    path=ROOT/'builds/rejected-lock.json';path.write_text(json.dumps(lock))
    proc=subprocess.run([sys.executable,str(ASSEMBLY/'tools/consume_family.py'),'--consumer',str(ROOT),'--lock',str(path),'--tool',str(ASSEMBLY/'tools/assemble_family.py')],capture_output=True,text=True)
    assert proc.returncode and diagnostic.lower() in proc.stderr.lower(),proc.stderr
    assert sha(report)==before
# Live inputs: a declared source file that does not exist must fail and keep output.
bad=copy.deepcopy(base)
next(project for project in bad['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
rejected(bad,'missing file')
bad=copy.deepcopy(base)
other=copy.deepcopy(next(p for p in bad['projects'] if p['id']=='model_selection'))
other['id']='conflicting-owner';bad['projects'].append(other)
rejected(bad,'module collision')
bad=copy.deepcopy(base)
p=next(p for p in bad['projects'] if p['id']=='model_selection')
p['exports']['packages']['moxie']['clone']='moxie.model_selection._evaluation'
rejected(bad,'export')
print('PASS: isolated copies, missing live sources, conflicting ownership and exports rejected')
