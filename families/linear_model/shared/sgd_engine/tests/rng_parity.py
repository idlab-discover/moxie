"""Compile pinned sklearn our_rand_r and demonstrate that MT19937 is not the SGD shuffle stream.

The Cython wrapper calls the unmodified local inline primitive. This is a
blocker reproducer, not a replacement reference implementation or SGD parity.
"""
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT.parents[4] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/utils/_random.pxd', 'sklearn/utils/_typedefs.pxd',
         'sklearn/utils/_seq_dataset.pyx.tp',
         'sklearn/linear_model/_sgd_fast.pyx.tp']


def main():
    assert subprocess.check_output(['git', '-C', str(ORACLE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(ORACLE), 'status', '--porcelain', '--', *FILES], text=True)
    target = ROOT / 'builds/oracle'
    package = target / 'sklearn/utils'
    package.mkdir(parents=True, exist_ok=True)
    for name in FILES[:2]:
        shutil.copyfile(ORACLE / name, target / name)
    (target / 'sklearn/__init__.pxd').write_text('')
    (package / '__init__.pxd').write_text('')
    (target / 'sklearn/__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    # Compile the unedited pxd declarations to provide DEFAULT_SEED's C API.
    (package / '_random.pyx').write_text('# Declaration-only host for the pinned pxd.\n')
    (target / 'rng_oracle.pyx').write_text('''from sklearn.utils._random cimport our_rand_r
from sklearn.utils._typedefs cimport uint32_t

def words(uint32_t seed, int n):
    result = []
    for i in range(n):
        result.append(int(our_rand_r(&seed)))
    return result

def order(uint32_t seed, int n):
    result = list(range(n))
    for i in range(n - 1):
        j = i + our_rand_r(&seed) % (n - i)
        result[i], result[j] = result[j], result[i]
    return result
''')
    (target / 'setup.py').write_text("from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize([Extension('sklearn.utils._random',['sklearn/utils/_random.pyx']), Extension('rng_oracle',['rng_oracle.pyx'])], compiler_directives={'language_level':3}))\n")
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=target, check=True)
    sys.path.insert(0, str(target))
    oracle = importlib.import_module('rng_oracle')
    assert Path(oracle.__file__).is_relative_to(target)
    raw = subprocess.check_output([ROOT / 'builds/rng_probe'], text=True)
    actual = {}
    for line in raw.splitlines():
        kind, seed, *values = line.split()
        actual[kind, int(seed)] = [int(v) for v in values]
    comparisons = []
    for seed in (0, 1, 42):
        expected = oracle.words(seed, 1)
        order = oracle.order(seed, 3)
        comparisons.append(dict(seed=seed, expected_our_rand_r=expected,
                                actual_mt19937=actual['word', seed],
                                expected_dataset_order=order,
                                actual_mt19937_permutation=actual['order', seed]))
    report = {'revision': REVISION, 'oracle_binary': oracle.__file__,
              'hashes': {f: hashlib.sha256((ORACLE / f).read_bytes()).hexdigest() for f in FILES},
              'comparisons': comparisons}
    destination = ROOT / 'docs/private/sgd_engine/rng-report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)
    raise SystemExit('UNIMPLEMENTED: SGD must integrate shared our_rand_r. This historical probe only demonstrates that MT19937 substitution fails.')


if __name__ == '__main__':
    main()
