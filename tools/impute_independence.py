"""Verify impute ownership and public consumers of the complete Moxie assembly."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
from compiler_limits import compiler_command

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'families/impute'
DEPS = ROOT / 'builds/deps'


def run(*args):
    subprocess.run(compiler_command(args), check=True, cwd=ROOT)


manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
for path in [ROOT / 'DEPENDENCIES.json', *FAMILY.rglob('DEPENDENCIES.json')]:
    if 'builds' in path.parts:
        continue
    for project in json.loads(path.read_text())['projects']:
        assert 'moxie-impute' not in project['path'], project
        assert 'moxie-neighbors' not in project['path'], project
        assert (path.parent / project['path']).resolve().exists(), project
run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
    '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
project = next(p for p in manifest['projects'] if p['id'] == 'impute')
for module, record in project['modules'].items():
    source = (ROOT / project['path'] / record['file']).resolve()
    assert source.is_relative_to(FAMILY), source
    assert receipt['modules'][module] == ['impute']
    assert (DEPS / (module.replace('.', '/') + '.mojo')).read_bytes() == source.read_bytes()
assert (DEPS / 'licenses/impute-sklearn.txt').read_bytes() == (FAMILY / 'LICENSES/sklearn.txt').read_bytes()
(FAMILY / 'builds').mkdir(parents=True, exist_ok=True)
run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', FAMILY / 'builds/core.mojoc')
with tempfile.TemporaryDirectory(prefix='moxie-impute-consumer-') as temp:
    consumer = Path(temp)
    shutil.copytree(DEPS, consumer / 'deps')
    for source in [FAMILY / 'tests/test_public_api.mojo',
                   FAMILY / 'algorithms/simple/tests/test_integer.mojo',
                   FAMILY / 'algorithms/simple/tests/test_string.mojo',
                   FAMILY / 'algorithms/simple/tests/test_string_indicator.mojo',
                   FAMILY / 'algorithms/knn/tests/test_knn.mojo']:
        shutil.copy2(source, consumer / 'main.mojo')
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', consumer / 'deps',
            consumer / 'main.mojo', '-o', consumer / 'probe')
        run(consumer / 'probe')
print('PASS: complete Moxie namespace, impute ownership, license assets and isolated public consumers')
