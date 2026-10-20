"""Focused checks against the assembled public Moxie namespace."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command

def run(*args, cwd=CORE):
    subprocess.run(compiler_command(args), cwd=cwd, check=True)

def independence():
    manifest = json.loads((CORE / 'DEPENDENCIES.json').read_text())
    manifests = [CORE / 'DEPENDENCIES.json', *(
        path for path in ROOT.rglob('DEPENDENCIES.json')
        if not {'builds', '.pixi'}.intersection(path.relative_to(ROOT).parts))]
    for path in manifests:
        for project in json.loads(path.read_text())['projects']:
            target = (path.parent / project['path']).resolve()
            assert target.is_dir(), target
            assert 'moxie-feature_extraction' not in target.parts
            assert 'retired-moxie-feature_extraction' not in target.parts
    run(sys.executable, CORE / 'shared/assembly/tools/assemble_family.py', '--consumer', ROOT,
        '--lock', CORE / 'DEPENDENCIES.json', '--output', ROOT / 'builds/deps')
    receipt = json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())
    for project in manifest['projects']:
        if project['id'] not in ('feature_extraction', 'feature-extraction-dict_hash', 'feature-extraction-image'):
            continue
        owner = CORE / project['path']
        for module, spec in project['modules'].items():
            assert receipt['modules'][module] == [project['id']]
            assembled = ROOT / 'builds/deps' / Path(*module.split('.')).with_suffix('.mojo')
            assert assembled.read_bytes() == (owner / spec['file']).read_bytes()
        for dest, source in project['assets'].items():
            assert (ROOT / 'builds/deps' / dest).read_bytes() == (owner / source).read_bytes()
    with tempfile.TemporaryDirectory(prefix='feature-extraction-independent-') as folder:
        isolated = Path(folder)
        shutil.copytree(ROOT / 'builds/deps', isolated / 'deps')
        assert not any(p.is_symlink() for p in (isolated / 'deps').rglob('*'))
        for name, source in [('dictionary', ROOT / 'algorithms/dict_hash/examples/cities.mojo'),
                             ('image', ROOT / 'algorithms/image/examples/patches.mojo'),
                             ('deferred', ROOT / 'tests/test_deferred.mojo')]:
            shutil.copy2(source, isolated / 'main.mojo')
            run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', isolated / 'deps',
                isolated / 'main.mojo', '-o', isolated / name, cwd=isolated)
            run(isolated / name, cwd=isolated)
    print('PASS: ownership, license assets, standalone-free paths and isolated public consumers')

def main():
    action = sys.argv[1]
    if action == 'independence':
        independence()
    elif action in ('build', 'test'):
        for project in [ROOT, ROOT / 'algorithms/dict_hash', ROOT / 'algorithms/image']:
            run(sys.executable, project / 'tools/check.py', action)
    elif action == 'parity':
        for child in ('dict_hash', 'image'):
            project = ROOT / 'algorithms' / child
            run(sys.executable, CORE / 'shared/assembly/tools/assemble_family.py',
                '--consumer', project, '--lock', CORE / 'DEPENDENCIES.json',
                '--output', project / 'builds/deps')
            run(sys.executable, project / 'tests/parity.py')
    elif action == 'start':
        for project, name in [(ROOT, 'plumbing'), (ROOT / 'algorithms/dict_hash', 'cities'),
                              (ROOT / 'algorithms/image', 'patches')]:
            run(project / 'builds' / name)
    else:
        raise ValueError(action)
if __name__ == '__main__':
    main()
