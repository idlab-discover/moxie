"""Mixture contracts, public consumers, source ownership and sklearn comparisons."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/mixture'
DEPS = ROOT / 'builds/deps'


def sources():
    return sorted(TESTS.glob('*.mojo')) + sorted(TESTS.glob('gaussian/*.mojo')) + sorted(TESTS.glob('bayesian/*.mojo'))


def examples():
    return [ROOT / 'examples' / name for name in
            ('gaussian_mixture.mojo', 'bayesian_mixture.mojo', 'mixture_plumbing.mojo')]


def ownership(output=DEPS):
    receipt = json.loads((output / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie/mixture').glob('*.mojo'):
        if source.stem == '__init__':
            continue
        assert receipt['modules']['moxie.mixture.' + source.stem] == ['moxie']
        assert (output / 'moxie/mixture' / source.name).read_bytes() == source.read_bytes()
    for name in ('_api', '_api_types'):
        assert not (output / f'moxie/mixture/{name}.mojo').exists()
    exports = (output / 'moxie/mixture/__init__.mojo').read_text()
    for name, module in [('GaussianMixture', '_gaussian_mixture'), ('BayesianGaussianMixture', '_bayesian_mixture')]:
        assert f'from moxie.mixture.{module} import {name}' in exports
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-mixture' not in project['path'] for project in manifest['projects'])
    linalg = next(p for p in manifest['projects'] if p['id'] == 'mixture-linalg')
    for name, spec in linalg['modules'].items():
        assert receipt['modules'][name] == ['mixture-linalg']
        assembled = output / (name.replace('.', '/') + '.mojo')
        assert assembled.read_bytes() == (ROOT / linalg['path'] / spec['file']).read_bytes()


def isolated(run):
    with tempfile.TemporaryDirectory(prefix='moxie-mixture-') as folder:
        base = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
        for name in ('a', 'b'):
            manifest['projects'].append(dict(
                id='mixture-probe-' + name, path=str(TESTS / 'consumers' / name),
                modules={'moxie.probe_' + name: {'file': 'probe_' + name + '.mojo'}},
                assets={}, exports={'schema': 1, 'packages': {'moxie': {'Probe' + name.upper(): 'moxie.probe_' + name}}}))
        outputs = []
        for name in ('a', 'b'):
            consumer = base / name
            consumer.mkdir()
            run('git', 'init', '-q', consumer)
            (consumer / '.gitignore').write_text('builds/\n')
            lock = consumer / 'DEPENDENCIES.json'
            lock.write_text(json.dumps(manifest))
            output = consumer / 'builds/deps'
            command = [sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                       '--consumer', consumer, '--lock', lock, '--output', output]
            run(*command)
            ownership(output)
            outputs.append(output)
            for fixture in (TESTS / f'consumers/{name}/main.mojo', TESTS / 'test_public.mojo'):
                source = consumer / 'consumer.mojo'
                source.write_bytes(fixture.read_bytes())
                binary = consumer / 'consumer'
                run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', output, source, '-o', binary)
                run(binary)
            before = (output / 'ASSEMBLY.json').read_bytes()
            for kind in ('missing', 'module', 'export'):
                bad = json.loads(json.dumps(manifest))
                if kind == 'missing':
                    bad['projects'][0]['modules']['moxie.mixture._gaussian_mixture']['file'] = 'missing.mojo'
                elif kind == 'module':
                    bad['projects'].append(dict(bad['projects'][-1], id='duplicate'))
                else:
                    bad['projects'][-1]['exports']['packages']['moxie']['ProbeA'] = 'moxie.probe_b'
                lock.write_text(json.dumps(bad))
                result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
                assert result.returncode and any(marker in result.stderr.lower() for marker in
                    ('missing file', 'conflict', 'duplicate', 'collision')), result.stderr
                assert (output / 'ASSEMBLY.json').read_bytes() == before
        assert (outputs[0] / 'moxie/mixture/_contracts.mojo').stat().st_ino != (outputs[1] / 'moxie/mixture/_contracts.mojo').stat().st_ino


def test(run, executable):
    ownership()
    for source in sources():
        if source.stem.startswith('test_'):
            run(executable(source))
    for source in examples():
        run(executable(source))
    for name, marker in [('missing_predict', "required function 'predict'"),
                         ('bayesian_information_criteria', 'InformationCriteria'),
                         ('private_base', 'BaseMixture')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS),
            str(TESTS / 'compile_fail' / f'{name}.mojo'), '-o', str(ROOT / 'builds/invalid-mixture')],
            capture_output=True, text=True)
        (ROOT / f'builds/mixture-{name}.log').write_text(result.stderr)
        assert result.returncode and marker in result.stderr, result.stderr
    isolated(run)
    print('PASS: mixture contracts, public estimators, ownership, isolated consumers and rejection checks')


def parity(run):
    ownership()
    known = json.loads((TESTS / 'known_float32_exceptions.json').read_text())
    for name in ('gaussian', 'bayesian'):
        run(sys.executable, TESTS / name / 'parity.py')
        report_path = Path(os.environ.get('MOXIE_PARITY_REPORT', ROOT / f'builds/mixture/{name}/parity/report.json'))
        report = json.loads(report_path.read_text())
        expected = known[name]
        counts = report['counts']
        assert counts['ok'] + counts[expected['exception_counter']] == expected['numerical_cases']
        assert counts['error'] == expected['error_cases']
        actual = {case['case'] for case in report['diverged']}
        assert actual <= set(expected['cases']), f'New {name} numerical exceptions: {actual - set(expected["cases"])}'
        print(f'{name}: {counts["ok"]} strict numerical passes; {counts["error"]} error-path passes; {len(actual)} known Float32 exceptions')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
            '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + examples():
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS, source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
