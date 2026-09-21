"""Explicit maintainer re-pin command; never invoked by build/test/parity."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parents[1]
NUMERICS = ROOT.parent / 'numerics'
DUMMY = PARENT.parent / 'moxie-dummy'
BASELINE = 'ec75e4cf4a4b1d244fffa6f8b3b55016635e5914'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def project(owner, root, consumer, files, modules=None, packages=None, assets=None):
    return dict(id=owner, path=os.path.relpath(root, consumer), revision=head(root),
                files={name: sha(root / name) for name in files}, modules=modules or {},
                assets=assets or {}, exports={'schema': 1, 'packages': packages or {}})


def dependencies(consumer):
    baseline = {'moxie.base': {'file': 'src/moxie/base.mojo', 'baseline': BASELINE},
                'moxie.utils.validation': {'file': 'src/moxie/utils/validation.mojo', 'baseline': BASELINE}}
    packages = {'moxie': {symbol: 'moxie.base' for symbol in [
        'BaseEstimator', 'SupervisedEstimator', 'UnsupervisedEstimator', 'Predictor',
        'ClassifierMixin', 'RegressorMixin', 'TransformerMixin', 'clone']},
        'moxie.utils': {symbol: 'moxie.utils.validation' for symbol in ['FittedState', 'check_is_fitted']}}
    parent = project('moxie', PARENT, consumer,
                     ['src/moxie/base.mojo', 'src/moxie/utils/validation.mojo',
                      'src/moxie/__init__.mojo', 'src/moxie/utils/__init__.mojo', 'docs/interfaces.md', 'docs/families.md'], baseline, packages)
    numeric = project('numerics', NUMERICS, consumer,
                      ['src/moxie/_dense.mojo', 'DEPENDENCIES.lock.json', 'pixi.lock', 'docs/numerics.md'],
                      {'moxie._dense': {'file': 'src/moxie/_dense.mojo'}})
    original = json.loads((NUMERICS / 'DEPENDENCIES.lock.json').read_text())['numojo']
    source = (NUMERICS / original['path']).resolve()
    # Bootstrap immutable hashes, never silently re-pin the completed evaluation.
    numojo = dict(id='numojo', path=os.path.relpath(source, consumer), revision=original['revision'],
                  files=original['files'], modules={}, exports=None,
                  assets={name if name != 'LICENSE' else 'licenses/NuMojo-LICENSE': name for name in original['files'] if name.startswith('numojo/') or name == 'LICENSE'})
    tool = project('assembly', ROOT, consumer, ['tools/assemble.py', 'tools/consume.py', 'schemas/assembly-v1.schema.json',
                   'schemas/exports-v1.schema.json', 'EXPORTS.json'])
    return [parent, numeric, numojo, tool]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def main():
    for letter in ('a', 'b'):
        consumer = ROOT / 'tests/fixtures' / f'consumer_{letter}'
        projects = dependencies(consumer)
        name = f'moxie.probe_{letter}'
        modules = {name: {'file': f'src/moxie/probe_{letter}.mojo'}}
        exports = {f'evaluate_{letter}': name}
        overrides = []
        if letter == 'b':
            family = project('dummy-family', DUMMY, consumer,
                ['src/moxie/base.mojo', 'src/moxie/utils/validation.mojo', 'src/moxie/dummy.mojo', 'src/moxie/__init__.mojo'],
                {**projects[0]['modules'], 'moxie.dummy': {'file': 'src/moxie/dummy.mojo'}},
                {'moxie': {'DummyClassifier': 'moxie.dummy', 'DummyRegressor': 'moxie.dummy'}})
            projects.append(family)
            modules['moxie.dummy'] = {'file': 'src/moxie/dummy.mojo'}
            exports['replacement_marker'] = 'moxie.dummy'
            overrides = [dict(module='moxie.dummy', **{'from': ['dummy-family']},
                              to='consumer', reason='Test explicit inherited dummy relocation with a marker; no estimator claim.')]
        projects.append(project('consumer', consumer, consumer, [v['file'] for v in modules.values()], modules, {'moxie': exports}))
        write(consumer / 'DEPENDENCIES.lock.json', dict(schema=1, projects=projects, overrides=overrides))
    projects = dependencies(ROOT)
    # This owner's self source changes during development; downstream locks pin the tool.
    projects = [p for p in projects if p['id'] != 'assembly']
    write(ROOT / 'DEPENDENCIES.lock.json', dict(schema=1, projects=projects, overrides=[]))


if __name__ == '__main__':
    main()
