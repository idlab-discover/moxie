"""Run original pinned CV bodies with sklearn 1.9.1 validation infrastructure."""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[4]
ORACLE=WORKSPACE/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
if '--oracle' not in sys.argv:
    # Moxie's reference environment (sklearn 1.9.1) supplies Python-only infrastructure.
    python=Path(os.environ.get('MOXIE_REFERENCE_PYTHON',str(WORKSPACE/'moxie/.pixi/envs/default/bin/python')))
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run([str(python),str(Path(__file__).resolve()),'--oracle'],env=env,check=True)
    raise SystemExit()
sys.dont_write_bytecode=True
import numpy as np
import sklearn
import sklearn.model_selection._validation as infrastructure
import sklearn.utils._metadata_requests as metadata
assert sklearn.__version__=='1.9.1',sklearn.__version__
assert subprocess.check_output(['git','-C',str(ORACLE),'rev-parse','HEAD'],text=True).strip()==REV
files=['sklearn/model_selection/_validation.py','sklearn/utils/_metadata_requests.py','sklearn/dummy.py','sklearn/naive_bayes.py','sklearn/model_selection/_plot.py']
assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',*files],text=True)
ns=vars(infrastructure).copy()
ns['array_device']=lambda x: None
ns['move_to']=lambda x,**kwargs: np.asarray(x)
body=next(n for n in ast.parse((ORACLE/files[1]).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_manual_routing')
meta=vars(metadata).copy();exec(compile(ast.Module(body=[body],type_ignores=[]),str(ORACLE/files[1]),'exec'),meta);ns['_manual_routing']=meta['_manual_routing']
functions=[n for n in ast.parse((ORACLE/files[0]).read_text()).body if isinstance(n,ast.FunctionDef)]
for function in functions:function.decorator_list=[]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(ORACLE/files[0]),'exec'),ns)
spec=importlib.util.spec_from_file_location('sklearn._moxie_dummy_reference',ORACLE/files[2]);dummy=importlib.util.module_from_spec(spec);spec.loader.exec_module(dummy)
X=np.zeros((8,2));y=np.arange(8,dtype=float)
from sklearn.model_selection import KFold
cv=list(KFold(4).split(X))
model=dummy.DummyRegressor()
expected_scores=ns['cross_val_score'](model,X,y,cv=cv)
expected_prediction=ns['cross_val_predict'](model,X,y,cv=cv)
result=ns['cross_validate'](model,X,y,cv=cv,return_train_score=True,return_estimator=True,return_indices=True)
assert len(result['estimator'])==4
actual_scores=[];actual_prediction=[]
for line in subprocess.check_output([str(ROOT/'builds/evaluation')],text=True).splitlines():
    fields=line.split()
    if fields[0]=='score':actual_scores.append(float(fields[2]))
    if fields[0]=='prediction':actual_prediction.append(float(fields[2]))
np.testing.assert_allclose(actual_scores,expected_scores,rtol=1e-13,atol=1e-13)
np.testing.assert_allclose(actual_prediction,expected_prediction,rtol=1e-13,atol=1e-13)
from curve_parity import check_curves
curve_report=check_curves(ns,dummy,ROOT)
from learning_parity import check_learning
learning_report=check_learning(ns,dummy,ROOT)
from permutation_parity import check_permutations
permutation_report=check_permutations(ns,dummy,ROOT)
from prediction_parity import check_predictions
prediction_report=check_predictions(ns,dummy,ROOT)
from incremental_display_parity import check_incremental_display
extra = check_incremental_display(ns, ROOT, ORACLE)
report=dict(incremental_display=extra,class_predictions=prediction_report,permutations=permutation_report,learning=learning_report,curves=curve_report,reference=REV,support_sklearn=sklearn.__version__,files={f:hashlib.sha256((ORACLE/f).read_bytes()).hexdigest() for f in files},scores=len(actual_scores),predictions=len(actual_prediction),rtol=1e-13,atol=1e-13)
dest=ROOT/'docs/private/reference/parity.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
print('PASS original CV bodies and real DummyRegressor: 4 scores, 8 predictions; tolerance 1e-13')
