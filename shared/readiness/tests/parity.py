"""Differential integration only: execute pinned original dense function bodies."""
from pathlib import Path
import ast, hashlib, json, numbers, subprocess, sys, types
import numpy as np
from scipy.sparse import issparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check import CONSUMERS, run
SKLEARN=ROOT.parents[2]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/dummy.py','sklearn/utils/multiclass.py','sklearn/utils/validation.py',
       'sklearn/tests/test_dummy.py','sklearn/utils/random.py','sklearn/base.py',
       'doc/modules/model_evaluation.rst','COPYING']
def load():
    assert subprocess.check_output(['git','-C',str(SKLEARN),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SKLEARN),'status','--porcelain','--',*FILES],text=True)
    ns=dict(np=np,numbers=numbers,issparse=issparse,check_is_fitted=lambda _:None,_num_samples=len)
    wanted={'sklearn/utils/validation.py':['check_random_state'],
            'sklearn/utils/multiclass.py':['class_distribution'],
            'sklearn/dummy.py':['predict','predict_proba']}
    for file,names in wanted.items():
        path=SKLEARN/file; tree=ast.parse(path.read_text())
        nodes=tree.body
        if file.endswith('dummy.py'):
            nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name=='DummyClassifier').body
        selected=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(selected)==len(names)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
        for name in names: assert ns[name].__code__.co_filename==str(path)
    Dummy=type('ReferenceMethods',(),{n:ns[n] for n in ['predict','predict_proba']})
    return ns['class_distribution'],Dummy

def fixture(weighted,singleton):
    vocab=[np.array([-7,2,11]),np.array(['amsterdam','paris','東京'])]
    ids=np.array([2,0,1,2,0,1])
    y=[vocab[k][ids%(1 if singleton else 3-k)][:,None] for k in range(2)]
    weights=np.arange(6)%4/2 if weighted else np.ones(6)
    return y,weights

def parse(output):
    blocks=[]; current={}; words=[]
    for line in output.splitlines():
        key,*values=line.split()
        if key=='WORD':
            words.append(int(values[0]))
            if len(words)==624:
                current['words']=words; blocks.append(current); current={}; words=[]
        elif key=='POSITION': current['position']=int(values[0])
        else: current.setdefault((key,int(values[0])),[]).append(values[1])
    assert not current and not words
    return blocks

def check_case(consumer,distribution,Dummy,seed,n,strategy,weighted,singleton,repeats):
    actual=parse(run(consumer/'builds/probe',seed,n,strategy,int(weighted),int(singleton),repeats,
                     consumer=consumer,capture=True).stdout)
    assert len(actual)==repeats
    y,weights=fixture(weighted,singleton)
    distributions=[distribution(column,weights) for column in y]
    classes=[d[0][0] for d in distributions]; priors=[d[2][0] for d in distributions]
    dummy=Dummy(); dummy.classes_=classes; dummy.n_classes_=[len(c) for c in classes]
    dummy.class_prior_=priors; dummy.n_outputs_=2; dummy.constant=None
    dummy._strategy=strategy; dummy.sparse_output_=False; dummy.random_state=np.random.RandomState(seed)
    X=np.zeros((n,1))
    for block in actual:
        prior_state=dummy.random_state.get_state()
        prediction=dummy.predict(X)
        assert prediction.shape==(n,2)
        after=dummy.random_state.get_state()
        dummy.random_state.set_state(prior_state)
        probabilities=dummy.predict_proba(X)
        dummy.random_state.set_state(after)
        for k in range(2):
            np.testing.assert_array_equal(block['CLASS',k],classes[k].astype(str))
            _, inverse=np.unique(y[k],return_inverse=True)
            counts=np.bincount(inverse.ravel(),weights=weights)
            np.testing.assert_array_equal(np.array(block['COUNT',k],float),counts)
            np.testing.assert_allclose(np.array(block['PRIOR',k],float),priors[k],rtol=2e-15,atol=0)
            np.testing.assert_array_equal(block.get(('LABEL',k),[]),prediction[:,k].astype(str))
            indexes=np.array(block.get(('INDEX',k),[]),int)
            np.testing.assert_array_equal(classes[k][indexes].astype(str),prediction[:,k].astype(str))
            if strategy=='stratified':
                draws=np.array(block.get(('DRAW',k),[]),int).reshape(n,len(classes[k]))
                np.testing.assert_array_equal(draws,probabilities[k])
        np.testing.assert_array_equal(block['words'],after[1]); assert block['position']==after[2]
    return len(actual)

def main():
    distribution,Dummy=load()
    consumers=CONSUMERS if len(sys.argv)==1 else [Path(sys.argv[2]).resolve()]
    count=0
    for consumer in consumers:
        for seed in [0,42,4294967295]:
            for n in [0,1,17,257]:
                for strategy in ['uniform','stratified']:
                    for weighted,singleton in [(False,False),(True,False),(True,True)]:
                        count+=check_case(consumer,distribution,Dummy,seed,n,strategy,weighted,singleton,2)
    evidence={'revision':REV,'numpy':np.__version__,'batches':count,
        'files':{f:hashlib.sha256((SKLEARN/f).read_bytes()).hexdigest() for f in FILES},
        'reference':'original AST function bodies; validation plumbing only; no wheel target',
        'tolerance':'exact counts, draws, labels and MT state; priors rtol=2e-15, atol=0'}
    (ROOT/'docs/private/readiness/parity.json').write_text(json.dumps(evidence,indent=2))
    print(f'PASS: {count} dense consumer batches against pinned original sklearn methods')
if __name__=='__main__': main()
