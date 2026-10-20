"""Explicit reviewed lock update; never invoked by build/test."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'moxie/shared/readiness/DEPENDENCIES.lock.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def head(p):
    return subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()

def entry(owner, root, files, modules, exports, consumer):
    return dict(id=owner, path=os.path.relpath(root, consumer), revision=head(root),
                files={f: sha(root/f) for f in files}, modules=modules, assets={}, exports=exports)

def main():
    exports = {'schema': 1, 'packages': {'moxie': {}, 'moxie.decomposition': {
        name: 'moxie.decomposition._base' for name in
        ['DecompositionTransformer', 'InvertibleDecomposition', 'IncrementalDecomposition']}}}
    (ROOT/'EXPORTS.json').write_text(json.dumps(exports, indent=2)+'\n')
    base = json.loads(SOURCE.read_text())['projects']
    for consumer in [ROOT] + [ROOT/'tests/consumers'/n for n in ('a', 'b')]:
        projects = json.loads(json.dumps([p for p in base if p['id'] != 'readiness']))
        for p in projects:
            p['path'] = os.path.relpath((SOURCE.parent/p['path']).resolve(), consumer)
        evidence = ['DEPENDENCIES.lock.json', 'docs/readiness.md', 'docs/scope.md', 'pixi.lock']
        projects.append(entry('readiness', SOURCE.parent, evidence, {}, None, consumer))
        files = ['src/moxie/decomposition/_base.mojo', 'src/moxie/decomposition/_contracts.mojo',
                 'src/moxie/decomposition/__init__.mojo', 'EXPORTS.json', 'SCAFFOLD_BASE']
        if consumer != ROOT:
            files += ['DEPENDENCIES.lock.json', 'docs/plumbing.md', 'docs/scope.md', 'AGENTS.md', 'pixi.lock']
        projects.append(entry('decomposition', ROOT, files,
            {'moxie.decomposition.'+n: {'file': 'src/moxie/decomposition/'+n+'.mojo'}
             for n in ('_base', '_contracts')}, exports, consumer))
        if consumer != ROOT:
            for name in ('a', 'b'):
                source = ROOT/'tests/consumers'/name
                f = 'src/moxie/probe_'+name+'.mojo'
                projects.append(entry('probe_'+name, source, [f],
                    {'moxie.probe_'+name: {'file': f}},
                    {'schema': 1, 'packages': {'moxie': {}}}, consumer))
        (consumer/'DEPENDENCIES.lock.json').write_text(json.dumps(
            dict(schema=1, projects=projects, overrides=[]), indent=2, sort_keys=True)+'\n')
if __name__ == '__main__':
    main()
