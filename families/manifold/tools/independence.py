"""Verify ownership and run public consumers with only copied assembly inputs."""
import json
from pathlib import Path
import shutil
import tempfile
from check import ROOT, CORE, FLAGS, assemble, run

assemble(ROOT)
manifest = json.loads((CORE / 'DEPENDENCIES.json').read_text())
for project in manifest['projects']:
    path = (CORE / project['path']).resolve()
    assert 'moxie-manifold' not in path.parts
    assert 'retired-moxie-manifold' not in path.parts
receipt = json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())
for source in (ROOT / 'src/moxie/manifold').glob('*.mojo'):
    if source.stem == '__init__':
        continue
    assert receipt['modules']['moxie.manifold.' + source.stem] == ['manifold']
    assert source.read_bytes() == (ROOT / 'builds/deps/moxie/manifold' / source.name).read_bytes()
assert (ROOT / 'LICENSES/scikit-learn.txt').read_bytes() == (ROOT / 'builds/deps/licenses/manifold-scikit-learn.txt').read_bytes()
with tempfile.TemporaryDirectory(prefix='moxie-manifold-independent-') as folder:
    isolated = Path(folder)
    shutil.copytree(ROOT / 'builds/deps', isolated / 'deps')
    assert not any(p.is_symlink() for p in (isolated / 'deps').rglob('*'))
    for name in ('classical_mds', 'trustworthiness', 'deferred'):
        source = isolated / (name + '.mojo')
        shutil.copy2(ROOT / 'tests' / ('test_' + name + '.mojo'), source)
        binary = isolated / name
        run('mojo', 'build', *FLAGS, '-I', isolated / 'deps', source, '-o', binary, consumer=isolated)
        run(binary, consumer=isolated)
print('PASS: manifold ownership, license assets, and isolated public consumers without standalone source imports')
