"""Explicit initial/update pin operation; never run by checks."""
from pathlib import Path
import hashlib, json, os, subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def head(p): return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
def pin():
    original=ROOT.parent/'random/tests/consumers/a/DEPENDENCIES.lock.json'
    base=[p for p in json.loads(original.read_text())['projects'] if p['id']!='consumer_a']
    for consumer in [ROOT]+[ROOT/'tests/consumers'/n for n in ['a','b']]:
        projects=json.loads(json.dumps(base))
        for p in projects:
            source=(original.parent/p['path']).resolve()
            p['path']=os.path.relpath(source,consumer)
            # Preserve hashes AND recorded revisions from the completed owner.
        files=['src/moxie/_readiness.mojo','src/moxie/__init__.mojo','EXPORTS.json']
        if consumer!=ROOT: files+=['DEPENDENCIES.lock.json','pixi.lock','docs/readiness.md','docs/scope.md']
        projects.append(dict(id='readiness',path=os.path.relpath(ROOT,consumer),revision=head(ROOT),
            files={f:sha(ROOT/f) for f in files},modules={'moxie._readiness':{'file':files[0]}},
            assets={},exports=json.loads((ROOT/'EXPORTS.json').read_text())))
        if consumer!=ROOT:
            f=f'src/moxie/consumer_{consumer.name}.mojo'
            projects.append(dict(id='consumer',path='.',revision=head(consumer),files={f:sha(consumer/f)},
                modules={f'moxie.consumer_{consumer.name}':{'file':f}},assets={},
                exports={'schema':1,'packages':{'moxie':{}}}))
        (consumer/'DEPENDENCIES.lock.json').write_text(json.dumps(dict(schema=1,projects=projects,overrides=[]),indent=2,sort_keys=True)+'\n')
if __name__=='__main__': pin()
