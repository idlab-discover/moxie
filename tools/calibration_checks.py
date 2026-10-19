"""Calibration gates using Moxie's assembled namespace and pinned oracle wheels."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/calibration'
DEPS = ROOT / 'builds/deps'
SUPPORT = ROOT / 'examples/calibration_support'


def sources():
    return sorted(TESTS.glob('*.mojo'))


def ownership():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie').glob('*calibration*.mojo'):
        assert receipt['modules']['moxie.' + source.stem] == ['moxie']
        assert (DEPS / 'moxie' / source.name).read_bytes() == source.read_bytes()
    assert (DEPS / 'licenses/calibration-scikit-learn.txt').read_bytes() == (ROOT / 'LICENSES/calibration-scikit-learn.txt').read_bytes()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-calibration' not in p['path'] for p in manifest['projects'])


def isolated(run):
    import shutil
    with tempfile.TemporaryDirectory(prefix='moxie-calibration-consumer-') as folder:
        consumer = Path(folder)
        # Copy only assembled dependencies and the public consumer. No sibling
        # or repository source include path is available to this compiler.
        shutil.copytree(DEPS, consumer / 'deps')
        shutil.copy2(TESTS / 'test_public.mojo', consumer / 'main.mojo')
        binary = consumer / 'consumer'
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', consumer / 'deps',
            consumer / 'main.mojo', '-o', binary)
        run(binary, consumer / 'reliability.svg')
        import xml.etree.ElementTree as ET
        svg = ET.parse(consumer / 'reliability.svg').getroot()
        assert svg.findall('.//{http://www.w3.org/2000/svg}polyline')


def test(run, executable):
    ownership()
    for source in sources():
        if source.stem != 'parity_runner':
            run(executable(source))
    run(executable(ROOT / 'examples/calibration.mojo'))
    run(sys.executable, TESTS / 'check_svg.py')
    result = subprocess.run(['mojo', 'build', '-I', str(DEPS),
        str(TESTS / 'compile_fail/missing_predict.mojo'), '-o', str(ROOT / 'builds/invalid-calibration')],
        capture_output=True, text=True)
    (ROOT / 'builds/calibration-missing-predict.log').write_text(result.stderr)
    assert result.returncode and "required function 'predict' is not implemented" in result.stderr, result.stderr
    isolated(run)
    print('PASS: calibration contracts, ownership, isolated consumer and negative compilation')


def parity(run):
    ownership()
    environment = ROOT / 'builds/calibration-oracle'
    python = environment / 'bin/python'
    if not python.exists():
        run('uv', 'venv', '--python', sys.executable, environment)
    run('uv', 'pip', 'sync', '--python', python, TESTS / 'requirements-parity.txt')
    run(python, TESTS / 'check_parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + [ROOT / 'examples/calibration.mojo']:
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                      '-I', SUPPORT, '-I', ROOT / 'examples', source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
