"""Execute pinned local sklearn image bodies; NumPy/SciPy supply infrastructure."""
from pathlib import Path
import ast, hashlib, itertools, json, numbers, re, subprocess, sys
from collections import defaultdict
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check import run,FLAGS
REF=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip()==REV
files=['sklearn/feature_extraction/image.py','sklearn/feature_extraction/tests/test_image.py','sklearn/base.py']
assert not subprocess.check_output(['git','-C',str(REF),'status','--porcelain','--',*files],text=True).strip()
source=REF/files[0]
class BaseEstimator:
    def get_params(self, deep=True):
        return vars(self).copy()
base_source = REF/'sklearn/base.py'
base_tree = ast.parse(base_source.read_text())
base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseEstimator')
set_params = next(n for n in base_class.body if isinstance(n, ast.FunctionDef) and n.name == 'set_params')
base_ns = dict(defaultdict=defaultdict)
exec(compile(ast.Module(body=[set_params], type_ignores=[]), str(base_source), 'exec'), base_ns)
BaseEstimator.set_params = base_ns['set_params']
class TransformerMixin: pass
def check_array(x,**kw):
    a=np.asarray(x)
    if not np.isfinite(a).all(): raise ValueError('nonfinite')
    return a
def check_random_state(seed):
    return seed if isinstance(seed,np.random.RandomState) else np.random.RandomState(seed)
ns=dict(Interval=lambda *a,**k:None, RealNotInt=numbers.Real, np=np,sparse=sparse,product=itertools.product,Integral=numbers.Integral,Real=numbers.Real,Number=numbers.Number,
        as_strided=np.lib.stride_tricks.as_strided,BaseEstimator=BaseEstimator,TransformerMixin=TransformerMixin,
        check_array=check_array,check_random_state=check_random_state,validate_data=lambda self,X,**kw:check_array(X))
tree=ast.parse(source.read_text(),filename=str(source))
body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
for node in body:
    for n in ast.walk(node):
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)): n.decorator_list=[]
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),ns)
assert ns['extract_patches_2d'].__code__.co_filename==str(source)
lines=['from std.testing import assert_equal, assert_true','from numojo import zeros, Shape',
       'from moxie.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, PatchExtractor, PatchLimit, grid_to_graph, img_to_graph',
       'from moxie._random import RandomState','']
case=0
DT={str(np.dtype(x)):f'DType.{x}' for x in ['float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool']}
def emit(s):lines.append('    '+s)
def arr(name,a):
    emit(f'var {name} = zeros[{DT[str(a.dtype)]}](Shape({", ".join(map(str,a.shape))}))')
    for i,v in enumerate(a.ravel()):
        if a.dtype==bool: value='True' if v else 'False'
        elif a.dtype.kind in 'iu': value=f'SIMD[{DT[str(a.dtype)]},1]({int(v)})'
        else: value=f'Float64({float(v)!r}).cast[{DT[str(a.dtype)]}]()'
        emit(f'{name}.unsafe_set({i}, {value})')
def expect(a):
    emit(f'assert_equal(got.ndim,{a.ndim})')
    for i,n in enumerate(a.shape):emit(f'assert_equal(got.shape[{i}],{n})')
    for i,v in enumerate(a.ravel()):
        emit(f'assert_true(abs(Float64(got.unsafe_get({i})) - Float64({float(v)!r})) <= 1e-12)')
def start():
    global case
    case+=1;lines.append(f'def case_{case}() raises:')
def end():lines.append('')
for dt in [np.float32,np.float64,np.uint8,np.int16,np.int64,np.bool_]:
    for shape in [(3,4),(3,4,1),(3,4,2)]:
        x=(np.arange(np.prod(shape))%7).reshape(shape).astype(dt)
        for limit in [None,2,99,0.01,0.5]:
            start();arr('x',x);emit('var rng = RandomState(42)')
            lim='PatchLimit()' if limit is None else f'PatchLimit(1,{limit})' if isinstance(limit,int) else f'PatchLimit(2,0,{limit})'
            emit(f'var got = extract_patches_2d(x,(2,2),max_patches={lim},random_state=rng)')
            rng=np.random.RandomState(42)
            expected=ns['extract_patches_2d'](x,(2,2),max_patches=limit,random_state=rng)
            expect(expected)
            emit(f'assert_equal(rng.next_uint32(), UInt32({rng.randint(0,2**32,dtype=np.uint32)}))');end()
        for count in [0,1,6,8]:
            start();patches=ns['extract_patches_2d'](x,(2,2))
            patches=np.concatenate([patches,patches[:2]])[:count]
            arr('p',patches)
            # Reconstruct into collapsed singleton-channel shape, matching raw output.
            outshape=shape[:2] if shape[-1:]==(1,) else shape
            emit(f'var got = reconstruct_from_patches_2d(p,Shape({", ".join(map(str,outshape))}))')
            expect(ns['reconstruct_from_patches_2d'](patches,outshape));end()
for dt in [np.float32,np.float64,np.int8,np.uint8,np.int16,np.int32,np.int64,np.uint16,np.uint32,np.uint64]:
    for shape in [(2,3),(2,2,2)]:
        x=(np.arange(np.prod(shape))*17).reshape(shape).astype(dt)
        for masked in [False,True]:
            start();arr('x',x)
            mask=(np.arange(x.size)%3!=1).reshape(shape) if masked else None
            args=''
            if masked:arr('mask',mask);args='mask=mask,'
            emit(f'var got = img_to_graph(x,{args}return_as="dense")')
            expect(ns['img_to_graph'](x,mask=mask,return_as=np.ndarray));end()
for dtype in [np.int64,np.bool_,np.float32,np.float64]:
    for shape in [(1,1,1),(2,3,1),(2,2,2)]:
        for mode in ['all','none','isolated']:
            start();mask=np.ones(shape,dtype=bool)
            if mode=='none':mask[:]=False
            if mode=='isolated':mask.ravel()[1:-1]=False
            arr('mask',mask)
            emit(f'var got = grid_to_graph[{DT[str(np.dtype(dtype))]}]({",".join(map(str,shape))},mask=mask,return_as="dense")')
            expect(ns['grid_to_graph'](*shape,mask=mask,return_as=np.ndarray,dtype=dtype));end()
for shape in [(2,3,4),(2,3,4,2),(1,10,10)]:
    for limit in [None,2,0.5]:
        start();x=np.arange(np.prod(shape),dtype=np.float64).reshape(shape);arr('x',x)
        lim='PatchLimit()' if limit is None else f'PatchLimit(1,{limit})' if isinstance(limit,int) else f'PatchLimit(2,0,{limit})'
        default=shape==(1,10,10)
        emit(f'var model = PatchExtractor(patch_size={"(-1,-1)" if default else "(2,2)"},max_patches={lim},random_state=7)')
        emit('var got = model.transform(x)')
        model=ns['PatchExtractor'](patch_size=None if default else (2,2),max_patches=limit,random_state=7)
        expect(model.transform(x));end()
for shape,ps in [((2,2),(2,2)),((1,4),(1,4)),((2,4),(2,2))]:
    start();x=np.arange(np.prod(shape),dtype=np.float64).reshape(shape);arr('x',x)
    emit('var rng = RandomState(1)')
    emit(f'var got = extract_patches_2d(x,{ps},max_patches=PatchLimit(1,99),random_state=rng)')
    rng=np.random.RandomState(1)
    expect(ns['extract_patches_2d'](x,ps,max_patches=99,random_state=rng))
    emit(f'assert_equal(rng.next_uint32(),UInt32({rng.randint(0,2**32,dtype=np.uint32)}))');end()
for dt in [np.int8,np.int16,np.int32,np.uint8,np.uint16,np.uint32]:
    start();info=np.iinfo(dt);x=np.array([[info.min,info.max],[0,1]],dtype=dt);arr('x',x)
    emit('var got = img_to_graph(x,return_as="dense")')
    expect(ns['img_to_graph'](x,return_as=np.ndarray));end()
# Named partial updates and repeated borrowed streams use the reference estimator.
for limit, native_limit in [(None, 'PatchLimit()'), (2, 'PatchLimit(1,2)'), (0.5, 'PatchLimit(2,0,0.5)')]:
    for call in range(2):
        start()
        x = np.arange(48, dtype=np.float64).reshape(2,3,4,2)
        arr('x', x)
        emit('var model = PatchExtractor(patch_size=(1,1),random_state=7)')
        emit(f'model.set_params(patch_size=(2,2),max_patches={native_limit})')
        emit('var rng = RandomState(42)')
        if call:
            emit('_ = model.fit_transform(x,random_state=rng)')
        emit('var got = model.fit_transform(x,random_state=rng)')
        rng = np.random.RandomState(42)
        model = ns['PatchExtractor'](patch_size=(1,1),random_state=rng)
        assert model.set_params(patch_size=(2,2),max_patches=limit) is model
        if call:
            model.fit(x).transform(x)
        expect(model.fit(x).transform(x))
        emit(f'assert_equal(rng.next_uint32(),UInt32({rng.randint(0,2**32,dtype=np.uint32)}))')
        end()
start()
x = np.arange(100, dtype=np.float64).reshape(1,10,10)
arr('x',x)
emit('var model = PatchExtractor(patch_size=(2,2),max_patches=PatchLimit(1,2),random_state=42)')
emit('var got = model.set_params(patch_size=(-1,-1),max_patches=PatchLimit(),random_state=0).fit(x).transform(x)')
model = ns['PatchExtractor'](patch_size=(2,2),max_patches=2,random_state=42)
expect(model.set_params(patch_size=None,max_patches=None,random_state=0).fit(x).transform(x))
end()
lines.append('def main() raises:')
for i in range(1,case+1):emit(f'case_{i}()')
emit(f'print("PASS {case} local sklearn differential cases")')
p=ROOT/'builds/parity.mojo';p.write_text('\n'.join(lines)+'\n')
# Keep the generated differential cases within the shared compiler memory limit.
text = p.read_text()
functions = list(re.finditer(r"(?m)^def (\w+)\(", text))
assert functions[-1].group(1) == 'main'
entries = [f'case_{i}' for i in range(1, case+1)]
blocks = {match.group(1): text[match.start():functions[i+1].start()]
          for i, match in enumerate(functions[:-1])}
assert len(blocks) == len(entries) and set(blocks) == set(entries)
preamble = text[:functions[0].start()]
for start_index in range(0, len(entries), 28):
    names = entries[start_index:start_index+28]
    batch = ROOT/'builds'/f'parity_batch_{start_index//28:02d}.mojo'
    batch.write_text(preamble + ''.join(blocks[name] for name in names)
                     + 'def main() raises:\n'
                     + ''.join(f'    {name}()\n' for name in names))
    run('mojo','build',*FLAGS,'-I',ROOT/'builds/deps',batch,'-o',batch.with_suffix(''))
    run(batch.with_suffix(''))
    print(f'PASS: image parity cases {start_index+1}-{start_index+len(names)}', flush=True)
print(f'PASS {case} local sklearn differential cases')
(ROOT/'builds/parity.json').write_text(json.dumps(dict(cases=case,reference=REV,compiler_batch_size=28,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tolerance=1e-12),indent=2))
