"""Explicit local source pin update; never part of build or test."""
from pathlib import Path
import hashlib,json,subprocess
r=Path(__file__).resolve().parents[1]
p=r/'DEPENDENCIES.lock.json'; d=json.loads(p.read_text())
files=['src/moxie/decomposition/_svd_engine.mojo','EXPORTS.json','tools/check.py','docs/api.md','docs/scope.md','LICENSES/scikit-learn.txt']
e={'id':'svd_engine','path':'.','revision':subprocess.check_output(['git','-C',str(r.parent.parent),'rev-parse','HEAD'],text=True).strip(),'files':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files},'modules':{'moxie.decomposition._svd_engine':{'file':files[0]}},'assets':{},'exports':json.loads((r/'EXPORTS.json').read_text())}
d['projects']=[x for x in d['projects'] if x['id']!='svd_engine']+[e]
p.write_text(json.dumps(d,indent=2)+'\n')
# Flatten root and both consumer contributions, retaining upstream evidence pins.
import os
for name in ('a','b'):
    c=r/'tests/consumers'/name
    child=json.loads(json.dumps(d))
    for entry in child['projects']:
        owner=(r/entry['path']).resolve()
        entry['path']=os.path.relpath(owner,c)
        if entry['id']=='svd_engine':
            for f in ['DEPENDENCIES.lock.json','AGENTS.md','pixi.lock']:
                entry['files'][f]=hashlib.sha256((r/f).read_bytes()).hexdigest()
    for other in ('a','b'):
        owner=r/'tests/consumers'/other
        f='src/moxie/svd_consumer_'+other+'.mojo'
        child['projects'].append(dict(id='consumer_'+other,path=os.path.relpath(owner,c),revision=e['revision'],files={f:hashlib.sha256((owner/f).read_bytes()).hexdigest()},modules={'moxie.svd_consumer_'+other:{'file':f}},assets={},exports={'schema':1,'packages':{'moxie':{}}}))
    (c/'DEPENDENCIES.lock.json').write_text(json.dumps(child,indent=2)+'\n')
