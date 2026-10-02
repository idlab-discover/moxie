"""Execute exact pinned dense function bodies; validation plumbing is tooling only."""
from pathlib import Path
import ast, json, subprocess, sys, types, warnings
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip()==REV
paths=['sklearn/preprocessing/_data.py','sklearn/utils/extmath.py']
assert not subprocess.check_output(['git','-C',str(REFERENCE),'diff','HEAD','--',*paths],text=True)
namespace={'get_namespace_and_device':lambda *a:(np,True,None), '_find_matching_floating_dtype':lambda x,*a,**k:x.dtype, 'np':np,'sparse':sparse,'get_namespace':lambda *a:(np,True),'_is_numpy_namespace':lambda xp:True,'array_device':lambda x:None,
 '_array_api':types.SimpleNamespace(supported_float_dtypes=lambda *a,**k:(np.float32,np.float64)),
 'check_array':lambda x,**k:np.array(x,copy=True) if k.get('copy') else x}
for file,names in [(paths[1],['row_norms']),(paths[0],['_handle_zeros_in_scale','normalize','binarize'])]:
    path=REFERENCE/file;tree=ast.parse(path.read_text());body=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node.decorator_list=[];body.append(node)
    assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    for name in names:assert namespace[name].__code__.co_filename==str(path)

# Extract the actual scaler methods too, without constructing public estimators.
namespace.update(check_is_fitted=lambda model:None, FLOAT_DTYPES=(np.float32,np.float64),
 validate_data=lambda model,x,**k:namespace['check_array'](x,**k),
 _modify_in_place_if_numpy=lambda xp,func,*args,**kwargs:func(*args,**kwargs))
path=REFERENCE/paths[0]
for node in ast.parse(path.read_text()).body:
 if isinstance(node,ast.ClassDef) and node.name in ('RobustScaler','MinMaxScaler'):
  methods=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('transform','inverse_transform')]
  for m in methods:
   m.decorator_list=[]
   original=m.name;m.name=node.name+'_'+original
   exec(compile(ast.Module(body=[m],type_ignores=[]),str(path),'exec'),namespace)

def data(dtype,n,p,scenario):
    x=(((np.arange(n*p)*37)%101-50)/7).astype(dtype).reshape(n,p);x[0]=0
    if scenario==1:x[:]=1e20 if dtype==np.float32 else 1e200
    elif scenario==2:x[:]=1e-30
    elif scenario==3:x[:]=1e-8
    elif scenario==4:x[:,0]=np.nan
    return x

def reference(x,mode,axis,norm,mixed=False,scenario=0):
    lower,upper=(np.nan,.75) if scenario==5 else ((.75,-.5) if scenario==6 else (-.5,.75))
    if mode>=6:return namespace['normalize'](x,norm=['','l1','l2','max'][norm],axis=axis,return_norm=True)
    if mode==5:return namespace['binarize'](x,threshold=lower),None
    count=x.shape[1-axis];a=((np.arange(count)%5-2)/3).astype(np.float64 if mixed else x.dtype);b=((np.arange(count)%7+1)/7).astype(np.float64 if mixed else x.dtype)
    model=types.SimpleNamespace(copy=True,with_centering=True,with_scaling=True,center_=a,scale_=b,min_=a,clip=True,feature_range=(lower,upper))
    z=x.T if axis==1 else x
    if mode==0:y=namespace['RobustScaler_transform'](model,z)
    elif mode==1:y=namespace['RobustScaler_inverse_transform'](model,z)
    elif mode==2:y=namespace['MinMaxScaler_transform'](model,z)
    elif mode==3:y=namespace['MinMaxScaler_inverse_transform'](model,z)
    else:y=np.clip(z,lower,upper)
    return (y.T if axis==1 else y),None

count=0
with warnings.catch_warnings(),np.errstate(all='ignore'):
 warnings.simplefilter('ignore')
 for dtype,label in [(np.float32,'f32'),(np.float64,'f64'),(np.float32,'mixed')]:
  for n,p in [(1,1),(3,5),(65,7),(3,257)]:
   for layout in (0,1):
    for axis in (0,1):
     for mode,norm in [(m,2) for m in range(6)]+[(6,k) for k in (1,2,3)]+([(7,k) for k in (1,2,3)] if axis==0 else []):
      for tile in (8,65536):
       scenarios=range(7) if (n,p)==(3,5) else (0,)
       for scenario in scenarios:
        args=[label,n,p,layout,mode,axis,norm,tile,scenario]
        output=subprocess.check_output([str(ROOT/'builds/differential'),*map(str,args)],text=True)
        values=[float(l.split()[1]) for l in output.splitlines() if l.startswith('V ')]
        norms=[float(l.split()[1]) for l in output.splitlines() if l.startswith('N ')]
        expected,en=reference(data(dtype,n,p,scenario),mode,axis,norm,label=='mixed',scenario)
        tol=4e-6 if dtype==np.float32 else 2e-13
        np.testing.assert_allclose(np.array(values).reshape(n,p),expected,rtol=tol,atol=tol,equal_nan=True,err_msg=str(args))
        if mode<6:np.testing.assert_array_equal(np.array(values,dtype=dtype).reshape(n,p),expected,err_msg=str(args))
        if en is not None:np.testing.assert_allclose(norms,en,rtol=tol,atol=tol,equal_nan=True,err_msg=str(args))
        count+=1
out=ROOT/'docs/private/transforms';out.mkdir(parents=True,exist_ok=True)
(out/'parity.json').write_text(json.dumps({'reference':REV,'cases':count,'float32_tolerance':4e-6,'float64_tolerance':2e-13,'oracle':'AST-extracted normalize/binarize/row_norms; RobustScaler/MinMaxScaler transform/inverse bodies'},indent=2)+'\n')
print(f'PASS parity: {count} dtype/layout/axis/operation/tile/edge cases')
