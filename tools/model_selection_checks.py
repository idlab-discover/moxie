"""Validate model selection through Moxie's assembled public namespace."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
from compiler_limits import compiler_command

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'families/model_selection'
DEPS = ROOT / 'builds/deps'


def run(*args, cwd=ROOT):
    subprocess.run(compiler_command(args), cwd=cwd, check=True)


def main():
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity', 'all', 'independence'):
        raise ValueError(action)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
        '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    for project in manifest['projects']:
        assert 'moxie-model_selection' not in project['path'], project
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for project in manifest['projects']:
        if project['id'] in ('model_selection', 'splitters', 'search', 'validation'):
            owner = (ROOT / project['path']).resolve()
            assert owner.is_relative_to(FAMILY)
            for module, spec in project['modules'].items():
                assert receipt['modules'][module] == [project['id']]
                assembled = DEPS / (module.replace('.', '/') + '.mojo')
                assert assembled.read_bytes() == (owner / spec['file']).read_bytes()
            for target, source in project['assets'].items():
                assert (DEPS / target).read_bytes() == (owner / source).read_bytes()
    if action == 'independence':
        with tempfile.TemporaryDirectory(prefix='moxie-model-selection-') as temporary:
            isolated = Path(temporary)
            shutil.copytree(DEPS, isolated / 'deps')
            (isolated / 'builds').mkdir()
            for source in (FAMILY / 'algorithms/validation/tests/displays.mojo',
                           FAMILY / 'examples/splitters.mojo'):
                shutil.copy2(source, isolated / 'main.mojo')
                run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', isolated / 'deps',
                    isolated / 'main.mojo', '-o', isolated / 'consumer')
                run(isolated / 'consumer', cwd=isolated)
            search = FAMILY / 'algorithms/search/tests'
            shutil.copy2(search / 'search_support.mojo', isolated / 'search_support.mojo')
            shutil.copy2(search / 'compile_fail/probability.mojo', isolated / 'invalid.mojo')
            result = subprocess.run(compiler_command(['mojo', 'build', '-I', isolated / 'deps',
                '-I', isolated, isolated / 'invalid.mojo', '-o', isolated / 'invalid']),
                text=True, capture_output=True)
            assert result.returncode and 'violated constraint' in result.stderr, result.stderr
        print('PASS isolated public consumer and source ownership')
        return
    run('mojo', 'precompile', '-I', DEPS, DEPS / 'moxie', '-o', ROOT / 'builds/model-selection.mojoc')
    for child in (sys.argv[2:] or ('splitters', 'search', 'validation')):
        project = FAMILY / 'algorithms' / child
        build = project / 'builds'
        build.mkdir(exist_ok=True)
        shutil.copytree(DEPS, build / 'deps', dirs_exist_ok=True)
        for source in sorted((project / 'tests').glob('*.mojo')) + sorted((project / 'examples').glob('*.mojo')):
            binary = build / source.stem
            run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                '-I', project / 'tests', source, '-o', binary)
            if action in ('test', 'all'):
                run(binary, cwd=project)
        if child == 'search' and action in ('test', 'all'):
            probe = project / 'tests/compile_fail/probability.mojo'
            result = subprocess.run(compiler_command(['mojo', 'build', '-I', DEPS,
                '-I', project / 'tests', probe, '-o', build / 'invalid_probability']),
                text=True, capture_output=True)
            assert result.returncode and 'violated constraint' in result.stderr, result.stderr
            print('PASS search probability capability rejected at compile time')
        if action in ('parity', 'all'):
            run(sys.executable, project / 'tests/parity.py', cwd=project)
            if child == 'search':
                run(sys.executable, project / 'tests/search_parity.py', cwd=project)
    print('PASS model_selection', action, 'using Moxie-owned sources')


if __name__ == '__main__':
    main()
