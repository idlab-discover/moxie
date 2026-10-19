"""Structural end-to-end parity against a wheel built from the local pinned source."""
from pathlib import Path
import subprocess,json
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
ROOT=Path(__file__).resolve().parents[2]
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__,sklearn.__file__
    source=ROOT.parent/'scikit-learn'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==REV
    for name in ['_classes.py','_criterion.pyx','_splitter.pyx','_partitioner.pyx','_tree.pyx']:
        data=subprocess.check_output(['git','-C',str(source),'show',f'{REV}:sklearn/tree/{name}'])
        assert (ROOT/'builds/tree/reference/sklearn/tree'/name).read_bytes()==data
    output=subprocess.check_output([ROOT/'builds/tree/mae_driver'],text=True)
    dest=ROOT/'docs/private/tree/absolute_error';dest.mkdir(parents=True,exist_ok=True)
    (dest/'native.txt').write_text(output)
    actual={}
    dimensions={'CASE':2,'NODE':3,'VALUE':4,'PRED':3,'LEAF':3,'IMPORTANCE':3,'SCORE':2,'PATH':3}
    for line in output.splitlines():
        tag,*fields=line.split();count=dimensions[tag]
        actual[tag,*map(int,fields[:count])]=list(map(float,fields[count:]))
    failures=[];checks=0
    def check(key,want,exact=False):
        nonlocal checks
        checks+=1
        try:np.testing.assert_allclose(actual[key],want,rtol=0 if exact else 2e-12,atol=0 if exact else 2e-12,equal_nan=True)
        except (AssertionError,KeyError) as e:failures.append({'key':key,'got':actual.get(key),'want':np.asarray(want).tolist()})
    for tag in (2,6):
      kind=tag%4
      for case in range(26):
        n=8+case;p=1+case%4;outputs=1 if case>=18 else 1+case%3
        X=np.array([[((i*(j+3)+j*5+case)%23)/4 for j in range(p)] for i in range(n)],dtype=np.float32)
        if False:X[::4,:]=np.nan
        if case%7==0:X[:,0]=1
        y=np.array([[10+5*((i+i//3+k)%(2+k%2)) if kind<2 else ((i*7+k*3+case)%17)/4 for k in range(outputs)] for i in range(n)],dtype=np.float64)
        weights=np.array([0 if i%7==0 else 1+i%3 for i in range(n)],dtype=float)
        if case>=24:weights[::7]=-.125
        kw=dict(criterion='absolute_error',random_state=case,max_depth=4,min_samples_leaf=1+case%2,max_features=1+case%p,splitter='random' if case%3==1 else 'best',ccp_alpha=.02 if case%4==2 else 0)
        if tag>=4:
            kw['splitter']='random'
            if case%3==0:kw.pop('max_features')
        if case%6==2:kw['max_leaf_nodes']=4
        if kind<2 and case%4==3:kw['class_weight']='balanced'
        if case>=18 and case<24:
            kw['monotonic_cst']=[1 if j%2==0 else -1 for j in range(p)]
            if case==18:kw['max_features']='sqrt'
            elif case==19:kw['max_features']='log2'
            elif case==21:kw.update(min_samples_leaf=.15,min_samples_split=.3)
            elif case==22:kw['max_features']=.6
            elif case==23:kw['min_impurity_decrease']=.015
        if case==20:kw['monotonic_cst']=None
        cls=(ExtraTreeClassifier if kind<2 else ExtraTreeRegressor) if tag>=4 else (DecisionTreeClassifier if kind<2 else DecisionTreeRegressor)
        estimator=cls(**kw).fit(X,y,sample_weight=weights)
        t=estimator.tree_
        check(('CASE',tag,case),[t.node_count,estimator.get_depth(),estimator.get_n_leaves()],True)
        for i in range(t.node_count):
            check(('NODE',tag,case,i),[t.children_left[i],t.children_right[i],t.feature[i],t.threshold[i],t.impurity[i],t.n_node_samples[i],t.weighted_n_node_samples[i],t.missing_go_to_left[i]])
            for j,v in enumerate(t.value[i].flat):check(('VALUE',tag,case,i,j),[v])
        for i,v in enumerate(estimator.predict(X).flat):check(('PRED',tag,case,i),[v],kind<2)
        for i,v in enumerate(estimator.apply(X)):check(('LEAF',tag,case,i),[v],True)
        for i,v in enumerate(estimator.feature_importances_):check(('IMPORTANCE',tag,case,i),[v])
        if kind>=2 or outputs==1:check(('SCORE',tag,case),[estimator.score(X,y,sample_weight=weights)])
        path=estimator.cost_complexity_pruning_path(X,y,sample_weight=weights)
        native_length=sum(key[:3]==('PATH',tag,case) for key in actual)
        if native_length!=len(path.ccp_alphas):failures.append({'key':('PATH_LENGTH',tag,case),'got':native_length,'want':len(path.ccp_alphas)})
        for i,(a,r) in enumerate(zip(path.ccp_alphas,path.impurities)):check(('PATH',tag,case,i),[a,r])
    report={'reference_revision':REV,'checks':checks,'failures':failures,'tolerance':'Float64 rtol=atol=2e-12; bounded rational inputs; tree IDs exact'}
    (dest/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':checks,'failures':len(failures),'first':failures[:8]},indent=2))
    assert not failures
if __name__=='__main__':main()
