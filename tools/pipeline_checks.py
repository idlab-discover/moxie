"""Pipeline contracts and isolated consumers using Moxie's live assembly workflow."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/pipeline'
DEPS = ROOT / 'builds/deps'
MODULES = ('pipeline', '_pipeline', '_pipeline_api', '_pipeline_api_types',
           '_pipeline_contracts', '_pipeline_sequential', '_pipeline_union', '_feature_metadata')


def sources():
    return sorted(TESTS.glob('test_*.mojo')) + [TESTS / 'union_parity_driver.mojo']


def isolated(run):
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    for project in manifest['projects']:
        project['path'] = str((ROOT / project['path']).resolve())
    for name in ('a', 'b'):
        manifest['projects'].append(dict(
            id=f'consumer-{name}', path=str(TESTS / 'consumers' / name),
            modules={f'moxie.consumer_{name}': {'file': f'src/moxie/consumer_{name}.mojo'}},
            assets={}, exports={'schema': 1, 'packages': {
                'moxie': {f'Probe{name.upper()}': f'moxie.consumer_{name}'}}}))
    for name in ('a', 'b'):
        folder = ROOT / 'builds/tests/pipeline/consumers' / name
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / 'DEPENDENCIES.json'
        lock.write_text(json.dumps(manifest, indent=2) + '\n')
        deps = folder / 'builds/deps'
        command = [sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                   '--consumer', folder, '--lock', lock, '--output', deps]
        run(*command)
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', deps,
            '-I', TESTS, TESTS / 'consumers' / name / 'main.mojo', '-o', folder / 'probe')
        run(folder / 'probe')
        receipt = json.loads((deps / 'ASSEMBLY.json').read_text())
        assert receipt['modules']['moxie._pipeline'] == ['moxie']
        assert receipt['modules']['moxie._dense'] == ['numerics']
        assert (deps / 'moxie/_dense.mojo').read_bytes() == (DEPS / 'moxie/_dense.mojo').read_bytes()
        before = hashlib.sha256((deps / 'ASSEMBLY.json').read_bytes()).hexdigest()
        for kind in ('module', 'export'):
            bad = copy.deepcopy(manifest)
            extra = copy.deepcopy(bad['projects'][-1])
            extra['id'] = 'conflicting-owner'
            if kind == 'export':
                extra['modules'] = {'moxie.other_probe': {'file': 'src/moxie/consumer_b.mojo'}}
                extra['exports']['packages']['moxie']['ProbeB'] = 'moxie.other_probe'
            bad['projects'].append(extra)
            lock.write_text(json.dumps(bad))
            result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True, text=True)
            (folder / f'{kind}-conflict.log').write_text(result.stderr)
            assert result.returncode and any(word in result.stderr.lower()
                for word in ('conflict', 'duplicate', 'collision')), result.stderr
            assert hashlib.sha256((deps / 'ASSEMBLY.json').read_bytes()).hexdigest() == before
        lock.write_text(json.dumps(manifest, indent=2) + '\n')
    a, b = [ROOT / 'builds/tests/pipeline/consumers' / n / 'builds/deps/moxie/_dense.mojo' for n in ('a', 'b')]
    assert a.stat().st_ino != b.stat().st_ino


def test(run, executable):
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for module in MODULES:
        assert receipt['modules'][f'moxie.{module}'] == ['moxie']
    for source in sources():
        if source.stem != 'union_parity_driver':
            run(executable(source))
    run(executable(ROOT / 'examples/feature_union.mojo'))
    for name, diagnostic in [
        ('step_predict', "'predict'"), ('pipeline_samples', "'score_samples'"),
        ('pipeline_decision', "'decision_function'"), ('pipeline_score', "'score'"),
        ('pipeline_log_probability', "'predict_log_proba'"),
        ('pipeline_probability', "'predict_proba'"),
        ('pipeline_type_mismatch', 'pipeline intermediate types must match'),
        ('union_predict', "'predict'"), ('union_dtype', 'union input must be NDArray[dtype]'),
        ('union_missing_transform', 'TransformerMixin'),
    ]:
        result = subprocess.run(
            ['mojo', 'build', '-I', str(DEPS), '-I', str(TESTS),
             str(TESTS / 'negative' / f'{name}.mojo'),
             '-o', str(ROOT / 'builds/invalid-pipeline')],
            cwd=ROOT, capture_output=True, text=True)
        (ROOT / f'builds/pipeline-{name}.log').write_text(result.stderr)
        if result.returncode == 0 or diagnostic not in result.stderr:
            raise RuntimeError(f'Unexpected negative compilation result for {name}: {result.stderr}')
    isolated(run)
    print('PASS: Pipeline contracts, capability rejection, isolated consumers and ownership conflicts')
