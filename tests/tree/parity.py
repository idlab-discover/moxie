"""Structural end-to-end parity against a wheel built from the local pinned source."""
from pathlib import Path
import subprocess,json,hashlib,sys
from types import SimpleNamespace
import numpy as np
import sklearn
import treelite,treelite.gtil,treelite.sklearn
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor
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
    assert treelite.__version__=='4.6.1',treelite.__version__
    exports=ROOT/'builds/tree/treelite/cases';exports.mkdir(parents=True,exist_ok=True)
    for old in exports.glob('*.tl'):old.unlink()
    output=subprocess.check_output([ROOT/'builds/tree/parity_driver',exports],text=True)
    dest=ROOT/'docs/private/tree/dense-trees';dest.mkdir(parents=True,exist_ok=True)
    (dest/'native.txt').write_text(output)
    actual={}
    dimensions={'CASE':2,'NODE':3,'VALUE':4,'PRED':3,'LEAF':3,'IMPORTANCE':3,'SCORE':2,'PATH':3}
    for line in output.splitlines():
        tag,*fields=line.split();count=dimensions[tag]
        actual[tag,*map(int,fields[:count])]=list(map(float,fields[count:]))
    failures=[];checks=0;treelite_cases=0;nan_divergent=0
    def check(key,want,exact=False):
        nonlocal checks
        checks+=1
        try:np.testing.assert_allclose(actual[key],want,rtol=0 if exact else 2e-12,atol=0 if exact else 2e-12,equal_nan=True)
        except (AssertionError,KeyError) as e:failures.append({'key':key,'got':actual.get(key),'want':np.asarray(want).tolist()})
    for tag in range(8):
      kind=tag%4
      for case in range(26):
        n=8+case;p=1+case%4;outputs=1 if case>=18 else 1+case%3
        X=np.array([[((i*(j+3)+j*5+case)%23)/4 for j in range(p)] for i in range(n)],dtype=np.float32)
        if case%5==0:X[::4,:]=np.nan
        if case%7==0:X[:,0]=1
        y=np.array([[10+5*((i+i//3+k)%(2+k%2)) if kind<2 else ((i*7+k*3+case)%17)/4 for k in range(outputs)] for i in range(n)],dtype=np.float64)
        weights=np.array([0 if i%7==0 else 1+i%3 for i in range(n)],dtype=float)
        if case>=24:weights[::7]=-.125
        kw=dict(criterion=['gini','entropy','squared_error','poisson'][kind],random_state=case,max_depth=4,min_samples_leaf=1+case%2,max_features=1+case%p,splitter='random' if case%3==1 else 'best',ccp_alpha=.02 if case%4==2 else 0)
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
        # Treelite: exact bytes of the 4.6.1 sklearn loader fed moxie's own tree, then finite-row predictions.
        native=SimpleNamespace(node_count=t.node_count,
            children_left=np.array([actual['NODE',tag,case,i][0] for i in range(t.node_count)],dtype=np.int64),
            children_right=np.array([actual['NODE',tag,case,i][1] for i in range(t.node_count)],dtype=np.int64),
            feature=np.array([actual['NODE',tag,case,i][2] for i in range(t.node_count)],dtype=np.int64),
            threshold=np.array([actual['NODE',tag,case,i][3] for i in range(t.node_count)],dtype=np.float64),
            impurity=np.array([actual['NODE',tag,case,i][4] for i in range(t.node_count)],dtype=np.float64),
            n_node_samples=np.array([actual['NODE',tag,case,i][5] for i in range(t.node_count)],dtype=np.int64),
            weighted_n_node_samples=np.array([actual['NODE',tag,case,i][6] for i in range(t.node_count)],dtype=np.float64),
            value=np.array([actual['VALUE',tag,case,i,j][0] for i in range(t.node_count) for j in range(t.value[0].size)],dtype=np.float64).reshape(t.value.shape))
        wrapper=(ExtraTreesClassifier if kind<2 else ExtraTreesRegressor)(n_estimators=1) if tag>=4 else (RandomForestClassifier if kind<2 else RandomForestRegressor)(n_estimators=1)
        wrapper.estimators_=[SimpleNamespace(tree_=native)];wrapper.n_features_in_=p;wrapper.n_outputs_=estimator.n_outputs_
        if kind<2:wrapper.n_classes_=estimator.n_classes_
        exported=(exports/f'{tag}_{case}.tl').read_bytes()
        checks+=1;treelite_cases+=1
        reference=bytes(treelite.sklearn.import_model(wrapper).serialize_bytes())
        if reference!=exported:
            failures.append({'key':('TREELITE_BYTES',tag,case)})
            (exports/f'{tag}_{case}.reference.tl').write_bytes(reference)
        else:
            got=treelite.gtil.predict(treelite.Model.deserialize_bytes(exported),X.astype(np.float64))
            if kind<2:
                # Treelite pads every output to max_n_classes with zero probabilities.
                proba=estimator.predict_proba(X) if outputs>1 else [estimator.predict_proba(X)]
                want=np.zeros((n,outputs,t.value.shape[2]))
                for k,column in enumerate(proba):want[:,k,:column.shape[1]]=column
            else:want=estimator.predict(X).reshape(n,outputs,1)
            finite=~np.isnan(X).any(axis=1)
            checks+=1
            try:np.testing.assert_allclose(got.reshape(want.shape)[finite],want[finite],rtol=2e-12,atol=2e-12)
            except AssertionError:failures.append({'key':('TREELITE_PREDICT',tag,case)})
            nan_divergent+=int((~np.isclose(got.reshape(want.shape)[~finite],want[~finite],rtol=2e-12,atol=2e-12)).any(axis=(1,2)).sum())
        path=estimator.cost_complexity_pruning_path(X,y,sample_weight=weights)
        native_length=sum(key[:3]==('PATH',tag,case) for key in actual)
        if native_length!=len(path.ccp_alphas):failures.append({'key':('PATH_LENGTH',tag,case),'got':native_length,'want':len(path.ccp_alphas)})
        for i,(a,r) in enumerate(zip(path.ccp_alphas,path.impurities)):check(('PATH',tag,case,i),[a,r])
    report={'reference_revision':REV,'sklearn':sklearn.__version__,'module':sklearn.__file__,'checks':checks,'failures':failures,'tolerance':'2e-12 Float64: small rational inputs, permits accumulation/log roundoff; IDs and predictions for classification exact','treelite':{'version':treelite.__version__,'cases':treelite_cases,'bytes':'exact vs treelite.sklearn.import_model of the native tree','finite_predictions':'gtil vs sklearn, 2e-12','nan_rows_routed_differently':nan_divergent}}
    (dest/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':checks,'failures':len(failures),'treelite_cases':treelite_cases,'nan_rows_routed_differently':nan_divergent,'first':failures[:8]},indent=2))
    assert not failures
if __name__=='__main__':main()
