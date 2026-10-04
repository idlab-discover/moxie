"""FrozenEstimator contracts against the assembled Moxie namespace."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/frozen'
EXAMPLES = ROOT / 'examples/frozen'
DEPS = ROOT / 'builds/deps'


def sources():
    return sorted(TESTS.glob('*.mojo')) + [EXAMPLES / 'frozen.mojo']


def test(run, executable):
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for name in ('_base', '_frozen'):
        assert receipt['modules'][f'moxie.frozen.{name}'] == ['moxie']
    for source in sorted(TESTS.glob('test_*.mojo')):
        run(executable(source))
    run(executable(EXAMPLES / 'frozen.mojo'))
    for name, diagnostic in [
        ('nested_params', "no matching method in call to 'set_params'"),
        ('inner_params', "no matching method in call to 'set_params'"),
        ('mutation', "operand mutability 'False' doesn't match expected mutability 'True'"),
        ('param_mutation', "operand mutability 'False' doesn't match expected mutability 'True'"),
        ('fit_predict', "has no attribute 'fit_predict'"),
        ('fit_transform', "has no attribute 'fit_transform'"),
        ('proba', "invalid call to 'predict_proba': violated constraint"),
        ('transform', "invalid call to 'transform': violated constraint"),
        ('partial_fit', "no matching method in call to 'partial_fit'"),
    ]:
        result = subprocess.run(
            ['mojo', 'build', '-I', str(DEPS), '-I', str(TESTS), '-I', str(EXAMPLES),
             str(TESTS / 'negative' / f'frozen_{name}.mojo'),
             '-o', str(ROOT / 'builds/invalid-frozen')],
            cwd=ROOT, capture_output=True, text=True)
        (ROOT / f'builds/frozen-{name}.log').write_text(result.stderr)
        if result.returncode == 0 or diagnostic not in result.stderr:
            raise RuntimeError(f'Unexpected negative compilation result for {name}: {result.stderr}')
    print('PASS: Frozen contracts, shared ownership and nine capability restrictions')


def parity(run):
    run(sys.executable, TESTS / 'compare_reference.py')
    run(sys.executable, TESTS / 'compare_frozen_updates.py',
        '--sklearn-root', ROOT.parent / 'scikit-learn',
        '--native', ROOT / 'builds/tests/frozen/test_frozen_updates')
