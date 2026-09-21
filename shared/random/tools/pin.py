"""Explicit maintainer pinning. Never called by normal checks."""
from pathlib import Path
import hashlib, json, os, subprocess
ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parent / 'assembly'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def head(p): return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'], text=True).strip()
def write(p, data): p.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
def owner(root, consumer, ident, modules, files):
    return dict(id=ident,path=os.path.relpath(root,consumer),revision=head(root),
                files={p:sha(root/p) for p in files},modules=modules,assets={},
                exports={'schema':1,'packages':{'moxie':{}}})
def main():
    original=ASSEMBLY/'tests/fixtures/consumer_a/DEPENDENCIES.lock.json'
    base=json.loads(original.read_text())['projects']
    base=[p for p in base if p['id'] != 'consumer']
    for consumer in [ROOT]+[ROOT/'tests/consumers'/x for x in ['a','b']]:
        projects=json.loads(json.dumps(base))
        for p in projects:
            source=(original.parent/p['path']).resolve()
            p['path']=os.path.relpath(source,consumer)
            # Preserve completed pins. Current repository HEAD is the source revision.
            p['revision']=head(source)
        modules={'moxie._random':{'file':'src/moxie/_random.mojo'}}
        files=['src/moxie/_random.mojo','src/moxie/__init__.mojo','EXPORTS.json']
        if consumer!=ROOT: files+=['DEPENDENCIES.lock.json','pixi.lock','docs/random.md','docs/scope.md']
        notices=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'LICENSES').iterdir() if p.is_file())
        contribution=owner(ROOT,consumer,'random',modules,files+notices)
        contribution['assets']={f'licenses/random-{Path(p).name}':p for p in notices}
        projects.append(contribution)
        if consumer!=ROOT:
            name='consumer_'+consumer.name
            path=f'src/moxie/{name}.mojo'
            projects.append(owner(consumer,consumer,name,{f'moxie.{name}':{'file':path}},[path]))
        write(consumer/'DEPENDENCIES.lock.json',dict(schema=1,projects=projects,overrides=[]))
if __name__=='__main__': main()
