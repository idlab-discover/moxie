"""Execute the pinned local sklearn _init_coef/_pack bodies, not a wheel."""
import ast
from itertools import chain
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT.parents[4]/'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SOURCE = ORACLE/'sklearn/neural_network/_multilayer_perceptron.py'
assert subprocess.check_output(['git', '-C', str(ORACLE), 'rev-parse', 'HEAD'],text=True).strip() == REVISION
assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',str(SOURCE)],text=True).strip()
tree = ast.parse(SOURCE.read_text())
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseMultilayerPerceptron')
init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_init_coef')
pack = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pack')
namespace = {'np':np, 'chain':chain}
exec(compile(ast.Module(body=[init,pack],type_ignores=[]),str(SOURCE),'exec'),namespace)
assert namespace['_init_coef'].__code__.co_filename == str(SOURCE)
assert namespace['_pack'].__code__.co_filename == str(SOURCE)
actual = np.fromstring(subprocess.check_output([str(ROOT/'builds/parameter_dump')],text=True),sep='\n')
k = 0
for activation in ['identity','logistic','tanh','relu']:
    for seed in [0,42]:
        for widths in [[2,3,1],[1,1]]:
            for dtype in [np.float32,np.float64]:
                model = SimpleNamespace(activation=activation,_random_state=np.random.RandomState(seed))
                tensors = [namespace['_init_coef'](model,a,b,dtype) for a,b in zip(widths,widths[1:])]
                expected = namespace['_pack']([x[0] for x in tensors],[x[1] for x in tensors]).astype(np.float64)
                expected = np.append(expected,model._random_state.random_sample())
                # Native Float64 printing preserves enough decimal digits for this tolerance.
                np.testing.assert_allclose(actual[k:k+len(expected)],expected,rtol=1e-14,atol=1e-15)
                k += len(expected)
assert k == len(actual)
print('PASS 32 initialization/packing cases and subsequent RNG draws against pinned local sklearn')
