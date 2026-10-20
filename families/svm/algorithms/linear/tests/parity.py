"""Execute pinned sklearn wrapper source with its pinned C++ LIBLINEAR engine.

The installed sklearn supplies validation/inheritance infrastructure only.
AST extraction retains the local wrapper, dispatch and fit function bodies.
C++ is a validation subprocess, never linked to the Mojo library.
"""
from pathlib import Path
import ast, importlib.util, json, subprocess, types, warnings
import numpy as np
import sklearn.svm._base as support_base
import sklearn.svm._classes as support_classes
import sklearn.svm._bounds as support_bounds
ROOT=Path(__file__).resolve().parents[1]
FAMILY=ROOT.parents[1]
REF=FAMILY.parents[2]/'scikit-learn'
ENGINE=FAMILY/'shared/liblinear'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'

def load(path,names,namespace):
    tree=ast.parse(path.read_text(),filename=str(path))
    tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert len(tree.body)==len(names)
    exec(compile(tree,str(path),'exec'),namespace)
    for name in names:
        obj=namespace[name]
        if isinstance(obj,type):obj=obj.fit.__wrapped__ if hasattr(obj.fit,'__wrapped__') else obj.fit
        while hasattr(obj,'__wrapped__'):obj=obj.__wrapped__
        assert obj.__code__.co_filename==str(path)

spec=importlib.util.spec_from_file_location('engine_parity',ENGINE/'tests/parity.py')
ep=importlib.util.module_from_spec(spec);spec.loader.exec_module(ep)

def train_wrap(X,y,sparse,solver,tol,bias,C,cw,limit,seed,epsilon,sw):
    assert not sparse
    case=ROOT/'builds/reference-case.txt'
    ep.write_case(case,X,y,sw,cw,solver,C,epsilon,tol,bias>0,bias if bias>0 else 1.,limit,seed)
    w,b,iterations,status=ep.invoke(ENGINE/'builds/oracle',case)
    raw=np.column_stack([w,b/bias]) if bias>0 else w
    return raw,iterations

def reference():
    assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(REF),'status','--porcelain','--','sklearn/svm','sklearn/utils/class_weight.py'],text=True)
    ep.reference()
    base=dict(vars(support_base))
    base['liblinear']=types.SimpleNamespace(train_wrap=train_wrap,set_verbosity_wrap=lambda value:None)
    load(REF/'sklearn/svm/_base.py',{'_fit_liblinear','_get_liblinear_solver_type'},base)
    classes=dict(vars(support_classes));classes.update({k:base[k] for k in ('_fit_liblinear','_get_liblinear_solver_type')})
    load(REF/'sklearn/svm/_classes.py',{'LinearSVC','LinearSVR','_validate_dual_parameter'},classes)
    return classes

def native(x,y,sw,config,regression):
    c=config.copy();cw=c.get('class_weight');mode='balanced' if cw=='balanced' else 'explicit' if isinstance(cw,dict) else 'none'
    k=len(cw) if isinstance(cw,dict) else 0
    header=[*x.shape,int(regression),c['loss'],c.get('penalty','l2'),str(c['dual']).lower(),c.get('multi_class','ovr'),mode,int(c['fit_intercept']),c['intercept_scaling'],c['C'],c.get('epsilon',0),c['tol'],c['max_iter'],c['random_state'],int(x.dtype==np.float32),k]
    values=[*header,*x.ravel(),*y,*sw]
    if k:
        for label,weight in cw.items():values.extend([label,weight])
    case=ROOT/'builds/wrapper-case.txt';case.write_text(' '.join(map(str,values))+'\n')
    proc=subprocess.run([str(ROOT/'builds/driver'),str(case)],text=True,capture_output=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    lines=proc.stdout.splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('RESULT '));_,r,p,it,status=lines[start].split();r,p,it,status=map(int,(r,p,it,status))
    a=np.array([float(s) for s in lines[start+1:]]);i=r*p
    coef=a[:i].reshape(r,p);intercept=a[i:i+r];i+=r
    decision=a[i:i+len(y)*r].reshape(len(y),r);i+=len(y)*r
    predicted=None if regression else a[i:i+len(y)]
    return coef,intercept,decision,predicted,a[-1],it,status

def main():
    refs=reference();rng=np.random.RandomState(45);results=[]
    modes=[('squared_hinge','l2','auto','ovr'),('squared_hinge','l2',False,'ovr'),('squared_hinge','l2',True,'ovr'),('hinge','l2','auto','ovr'),('hinge','l2',True,'ovr'),('squared_hinge','l1','auto','ovr'),('squared_hinge','l1',False,'ovr'),('hinge','l1',False,'crammer_singer'),('epsilon_insensitive','l2','auto','ovr'),('epsilon_insensitive','l2',True,'ovr'),('squared_epsilon_insensitive','l2','auto','ovr'),('squared_epsilon_insensitive','l2',True,'ovr'),('squared_epsilon_insensitive','l2',False,'ovr')]
    for mode in modes:
      loss,penalty,dual,multi=mode;regression='insensitive' in loss
      for n,p,k in [(12,3,2),(6,11,2),(18,4,3)]:
       for fit in (False,True):
        for weighting in ('none','balanced','explicit'):
         if regression and weighting!='none':continue
         x=rng.randn(n,p);y=rng.randn(n) if regression else np.array([-7,4,19])[np.arange(n)%k]
         sw=rng.uniform(.4,2,n);sw[-1]=0
         config=dict(loss=loss,dual=dual,C=.7,tol=1e-8,fit_intercept=fit,intercept_scaling=2.3,max_iter=10000,random_state=42)
         if regression:config['epsilon']=.12
         else:config.update(penalty=penalty,multi_class=multi,class_weight=None if weighting=='none' else 'balanced' if weighting=='balanced' else {-7:.4,4:1.7,19:2.2})
         if len(results)%2:x=x.astype(np.float32)
         expected_model=refs['LinearSVR' if regression else 'LinearSVC'](**config).fit(x,y,sample_weight=sw)
         got=native(x,y,sw,config,regression)
         w=np.atleast_2d(expected_model.coef_);b=np.broadcast_to(expected_model.intercept_,got[1].shape)
         np.testing.assert_allclose(got[0],w,rtol=5e-5,atol=3e-6,err_msg=str(config))
         np.testing.assert_allclose(got[1],b,rtol=5e-5,atol=3e-6)
         expected_decision=(expected_model.predict(x) if regression else expected_model.decision_function(x)).reshape(len(y),-1)
         np.testing.assert_allclose(got[2],expected_decision,rtol=7e-5,atol=2e-5)
         ambiguous=0
         if not regression:
          predicted=expected_model.predict(x); mismatches=got[3]!=predicted
          if expected_decision.shape[1]==1:
           gaps=np.abs(expected_decision[:,0])
          else:
           ordered=np.sort(expected_decision,axis=1);gaps=ordered[:,-1]-ordered[:,-2]
          assert np.all(gaps[mismatches]<2e-5),(config,gaps[mismatches])
          ambiguous=int(mismatches.sum())
          # Exact argmax away from numerically tied decision boundaries.
          np.testing.assert_array_equal(got[3][gaps>=2e-5],predicted[gaps>=2e-5])
          np.testing.assert_allclose(got[4],np.average(got[3]==y,weights=sw),atol=1e-14)
         if not ambiguous:
          np.testing.assert_allclose(got[4],expected_model.score(x,y,sample_weight=sw),rtol=1e-6,atol=1e-6)
         solver=refs['_get_liblinear_solver_type'](multi,penalty,loss,refs['_validate_dual_parameter'](dual,loss,penalty,multi,x))
         codes=np.unique(y,return_inverse=True)[1].astype(float) if not regression else y
         cw=np.empty(0) if regression else support_base.compute_class_weight(config['class_weight'],classes=np.unique(y),y=y,sample_weight=sw)
         aw,ab=got[:2];ew=w;eb=b
         if solver==4 and k==2:
          aw=np.concatenate([-aw/2,aw/2]);ab=np.concatenate([-ab/2,ab/2])
          ew=np.concatenate([-ew/2,ew/2]);eb=np.concatenate([-eb/2,eb/2])
         ao=ep.objective(aw,ab,x,codes,sw,cw,solver,config['C'],config.get('epsilon',0),config['intercept_scaling'],fit)
         eo=ep.objective(ew,eb,x,codes,sw,cw,solver,config['C'],config.get('epsilon',0),config['intercept_scaling'],fit)
         np.testing.assert_allclose(ao,eo,rtol=1e-7,atol=1e-8)
         results.append(dict(mode=mode,shape=[n,p],classes=k,intercept=fit,weighting=weighting,iterations=got[5],ambiguous_predictions=ambiguous,max_coef_error=float(np.max(abs(got[0]-w)))))
    bounds=dict(vars(support_bounds))
    load(REF/'sklearn/svm/_bounds.py',{'l1_min_c'},bounds)
    bound_count=0
    for k in (1,2,4):
     for loss in ('squared_hinge','log'):
      for fit in (False,True):
       for shape in ((9,3),(5,1),(4,7)):
        x=rng.randn(*shape);y=np.arange(shape[0])%k
        expected=bounds['l1_min_c'](x,y,loss=loss,fit_intercept=fit,intercept_scaling=3.4)
        case=ROOT/'builds/bounds-case.txt'
        case.write_text(' '.join(map(str,[*shape,loss,int(fit),3.4,*x.ravel(),*y]))+'\n')
        actual=float(subprocess.check_output([str(ROOT/'builds/bounds_driver'),str(case)],text=True))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
        bound_count+=1
    # Zero/collinear designs and early stopping: prediction consistency and
    # solver output are checked even when a capped fit has not converged.
    edges=0
    for regression in (False,True):
     for limit in (0,1,3,10000):
      for kind in ('zero','collinear','signed','small_c'):
       x=np.zeros((6,2)) if kind=='zero' else np.arange(12,dtype=float).reshape(6,2)/7-1
       if kind=='collinear':x[:,1]=x[:,0]
       y=np.linspace(-1,1,6) if regression else np.array([-7,-7,-7,4,4,4])
       sw=np.ones(6)
       if kind=='signed':sw[1]=0;sw[4]=-1
       config=dict(loss='squared_epsilon_insensitive' if regression else 'squared_hinge',dual=True,C=1e-6 if kind=='small_c' else 1.,tol=1e-9,fit_intercept=True,intercept_scaling=1.,max_iter=limit,random_state=17)
       expected_model=refs['LinearSVR' if regression else 'LinearSVC'](**config).fit(x,y,sample_weight=sw)
       got=native(x,y,sw,config,regression)
       np.testing.assert_allclose(got[0],np.atleast_2d(expected_model.coef_),rtol=5e-5,atol=3e-6)
       np.testing.assert_allclose(got[1],expected_model.intercept_,rtol=5e-5,atol=3e-6)
       assert got[6]==int(got[5]>=limit)
       edges+=1
    (ROOT/'builds/parity-results.json').write_text(json.dumps(dict(revision=REV,cases=results,bounds_cases=bound_count,edge_cases=edges),indent=2)+'\n')
    print(f'PASS {len(results)} public linear-wrapper comparisons, {edges} edge cases, {bound_count} l1_min_c cases against local sklearn source and C++ engine')

if __name__=='__main__':main()
