"""Compile the exact pinned sklearn RNG primitive, never a wheel substitute."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[4]/'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == REV
files = ['sklearn/utils/_random.pxd', 'sklearn/utils/_typedefs.pxd', 'sklearn/linear_model/_cd_fast.pyx']
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*files],text=True)
target = ROOT/'builds/rng_oracle'
package = target/'sklearn/utils'
package.mkdir(parents=True, exist_ok=True)
for f in files[:2]:
    shutil.copyfile(SOURCE/f, target/f)
(target/'sklearn/__init__.py').write_text('')
(package/'__init__.py').write_text('')
(target/'probe.pyx').write_text('''include "sklearn/utils/_random.pxd"

def draws(uint32_t state, int size):
    result = []
    for i in range(size):
        value = our_rand_r(&state)
        result.append((int(value), int(state)))
    return result
''')
(target/'setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize('probe.pyx', compiler_directives={'language_level':3}))\n")
subprocess.run([sys.executable,'setup.py','build_ext','--inplace'], cwd=target, check=True, stdout=subprocess.DEVNULL)
sys.path.insert(0,str(target))
import probe
assert str(target) in probe.__file__
output = subprocess.check_output([ROOT/'builds/rng_probe'], text=True)
lines = output.splitlines()
seed = int(lines[0].split()[1])
assert seed == int(np.random.RandomState(0).randint(0, 2147483647))
expected = [v % 3 for v, state in probe.draws(seed, 8)]
mt = [int(line.split()[1]) for line in lines if line.startswith('mt ')]
reseeded = [int(line.split()[2]) for line in lines if line.startswith('mt ')]
assert mt != expected and reseeded != expected
assert probe.draws(0,1) == probe.draws(1,1) == [(270369,270369)]
result = {'revision': REV, 'source_hashes': {f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files}, 'seed': seed, 'expected_coordinate_indices': expected, 'shared_mt_continued':mt, 'shared_mt_reseeded':reseeded, 'zero_seed':probe.draws(0,1), 'native_output':output, 'interpretation':'MT19937 differs from our_rand_r; shared our_rand_r now exists but random coordinate selection remains unimplemented.'}
p = ROOT/'docs/private/coordinate_engine/rng-evidence.json'
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
