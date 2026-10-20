"""Validate neighbors ownership and an isolated consumer of Moxie's namespace."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'families/neighbors'
DEPS = ROOT / 'builds/deps'

def run(*args):
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT)

manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
for project in manifest['projects']:
    assert 'moxie-neighbors' not in project['path'], project
    for module, record in project.get('modules', {}).items():
        if module.startswith('moxie.neighbors.'):
            source = (ROOT / project['path'] / record['file']).resolve()
            assert source.is_relative_to(FAMILY), source
run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
    '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
for project in manifest['projects']:
    for module, record in project.get('modules', {}).items():
        if module.startswith('moxie.neighbors.'):
            assert receipt['modules'][module] == [project['id']]
            source = ROOT / project['path'] / record['file']
            assert (DEPS / (module.replace('.', '/') + '.mojo')).read_bytes() == source.read_bytes()
for name in ['sklearn', 'scipy']:
    assert (DEPS / f'licenses/neighbors-{name}.txt').read_bytes() == (FAMILY / f'LICENSES/{name}.txt').read_bytes()
run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/neighbors/core.mojoc')
with tempfile.TemporaryDirectory(prefix='moxie-neighbors-consumer-') as temp:
    consumer = Path(temp)
    shutil.copytree(DEPS, consumer / 'deps')
    # These public tests cover the estimator traits plus centroid, NCA and trees.
    for source in [FAMILY / 'tests/test_integration.mojo',
                   *[FAMILY / f'tests/consumers/{name}.mojo' for name in ['centroid', 'nca', 'kdtree', 'balltree']]]:
        shutil.copy2(source, consumer / 'main.mojo')
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', consumer / 'deps',
            consumer / 'main.mojo', '-o', consumer / 'probe')
        run(consumer / 'probe')
print('PASS: Moxie namespace, family ownership, license assets and isolated public consumers')
