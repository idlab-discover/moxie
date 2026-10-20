"""Verify core ownership and run a public consumer in a detached namespace."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
from check import ROOT, assemble, run


def main():
    assemble(ROOT)
    core = ROOT.parents[1]
    manifest = json.loads((core / 'DEPENDENCIES.json').read_text())
    receipt = json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())
    assert any(p['id'] == 'semi-supervised' for p in manifest['projects'])
    for project in manifest['projects']:
        owner = (core / project['path']).resolve()
        assert not {'moxie-semi_supervised', 'retired-moxie-semi_supervised'} & set(owner.parts)
        if project['id'] != 'semi-supervised':
            continue
        assert owner == ROOT
        for module, spec in project['modules'].items():
            assert receipt['modules'][module] == ['semi-supervised']
            target = ROOT / 'builds/deps' / Path(*module.split('.')).with_suffix('.mojo')
            assert target.read_bytes() == (ROOT / spec['file']).read_bytes()
        for target, source in project['assets'].items():
            assert (ROOT / 'builds/deps' / target).read_bytes() == (ROOT / source).read_bytes()
    with tempfile.TemporaryDirectory(prefix='semi-supervised-consumer-') as directory:
        detached = Path(directory)
        shutil.copytree(ROOT / 'builds/deps', detached / 'deps')
        shutil.copy2(ROOT / 'examples/semi_supervised.mojo', detached / 'main.mojo')
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', detached / 'deps',
            detached / 'main.mojo', '-o', detached / 'probe', root=detached)
        actual = run(detached / 'probe', root=detached).stdout
        expected = run(ROOT / 'builds/semi_supervised_example').stdout
        assert actual == expected
    print('PASS core ownership, source/license bytes, detached public consumer')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stdout, error.stderr)
        raise
