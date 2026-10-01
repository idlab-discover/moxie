"""Dense candidate differential checks; pinned sklearn row_norms source executes.

The isolated AST function uses NumPy namespace plumbing because importing the
whole unbuilt sklearn checkout requires compiled extensions. No wheel replaces
the function under test. This does not test sparse or array-API dispatch.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT/'DEPENDENCIES.lock.json').read_text())
sk = (ROOT/lock['sklearn']['path']).resolve()
assert subprocess.check_output(['git','-C',str(sk),'rev-parse','HEAD'], text=True).strip() == lock['sklearn']['revision']
for name,digest in lock['sklearn']['files'].items():
    assert hashlib.sha256((sk/name).read_bytes()).hexdigest()==digest, f'Reference drift: {name}'
source = sk/'sklearn/utils/extmath.py'
assert not subprocess.check_output(['git','-C',str(sk),'diff','HEAD','--',str(source)],text=True)
module = ast.parse(source.read_text(), filename=str(source))
function = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name == 'row_norms')
namespace = {'np':np,'sparse':sparse,'get_namespace':lambda x:(np,False),'_is_numpy_namespace':lambda xp:xp is np}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'), namespace)
row_norms = namespace['row_norms']
assert row_norms.__code__.co_filename == str(source)
records=[]
for dtype in ('f32','f64'):
    typ = np.float32 if dtype == 'f32' else np.float64
    tolerance = 3e-5 if dtype == 'f32' else 1e-12
    for rows,cols in ((1,1),(2,3),(3,2),(17,7),(7,17),(65,33)):
        for layout in ('C','S','T','O','R'):
            raw=subprocess.check_output([str(ROOT/'builds/evaluate'),dtype,str(rows),str(cols),layout],text=True)
            actual={}
            shapes={}
            for line in raw.splitlines():
                parts=line.split()
                if parts[0]=='shape':
                    shapes[parts[1]]=tuple(map(int,parts[2:]))
                else:
                    name,index,value=parts
                    assert int(index)==len(actual.setdefault(name,[]))
                    actual[name].append(float(value))
            owner=((np.arange(rows)[:,None]*17+np.arange(cols*2)[None,:]*3)%31-15).astype(typ)/97
            x=owner[:,::2]
            if layout=='O': x=owner[:,1::2]
            if layout=='R': x=x[:,::-1]
            if layout=='C': x=np.ascontiguousarray(x)
            if layout=='T': x=x.T
            weights=((np.arange(x.shape[1])[:,None]*7+np.arange(5)[None,:]*11)%23-11).astype(typ)/53
            expected={'linear':x@weights,'input':x,'total':np.sum(x),'sum0':x.sum(axis=0),'mean0':x.mean(axis=0),'center':x-x.mean(axis=0),'gram':x@x.T,'norm':row_norms(x,squared=True)}
            for name,value in expected.items():
                np.testing.assert_allclose(actual[name],np.ravel(value),rtol=tolerance,atol=tolerance,err_msg=f'{dtype}/{rows}/{cols}/{layout}/{name}')
                if name!='total': assert shapes[name]==(value.ndim,value.size)
            records.append([dtype,rows,cols,layout])
report={'cases':records,'reference':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'numpy':np.__version__,'tolerances':{'float32':'rtol=atol=3e-5','float64':'rtol=atol=1e-12'},'scope':'dense finite inputs; local sklearn row_norms AST with NumPy namespace'}
(ROOT/'docs/private/numojo/parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {len(records)} native/NumPy/pinned-sklearn dense cases')
