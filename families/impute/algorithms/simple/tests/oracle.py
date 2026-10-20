"""Load pinned estimator bodies; a wheel supplies only supporting infrastructure."""
import sys
sys.dont_write_bytecode = True
import inspect
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sklearn
ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'../../../../../scikit-learn').resolve()
REVISION='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/impute/_base.py','sklearn/impute/__init__.py','sklearn/impute/tests/test_impute.py','doc/modules/impute.rst','sklearn/base.py','sklearn/utils/validation.py']
def load():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True)
    name='sklearn.impute._simple_pinned_oracle'
    path=SOURCE/'sklearn/impute/_base.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SimpleImputer._dense_fit.__code__.co_filename==str(path)
    assert inspect.unwrap(module.MissingIndicator.transform).__code__.co_filename==str(path)
    evidence = ROOT/'docs/private/impute.simple'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/'oracle.json').write_text(json.dumps(dict(revision=REVISION,support_sklearn=sklearn.__version__,files={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}),indent=2)+'\n')
    return module
if __name__=='__main__': print(load().SimpleImputer)
