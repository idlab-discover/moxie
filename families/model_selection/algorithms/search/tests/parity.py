"""Pinned ParameterGrid/ParameterSampler bodies and unedited Cython RNG oracle."""
import ast
from collections.abc import Mapping, Iterable, Sequence
from functools import partial, reduce
from itertools import product
import hashlib, json, numbers, operator, os, shutil, subprocess, sys, warnings
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ORACLE=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'

def load():
    files=['sklearn/model_selection/_search.py','sklearn/utils/_random.pyx','sklearn/utils/_random.pxd','sklearn/utils/_typedefs.pxd','sklearn/utils/validation.py']
    assert subprocess.check_output(['git','-C',str(ORACLE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',*files],text=True)
    target=ROOT/'builds/oracle'; package=target/'sklearn/utils';package.mkdir(parents=True,exist_ok=True)
    for f in files[1:4]:shutil.copyfile(ORACLE/f,target/f)
    body=next(n for n in ast.parse((ORACLE/files[4]).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_random_state')
    (target/'sklearn/__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    (package/'validation.py').write_text('import numpy as np\nimport numbers\n'+ast.unparse(body)+'\n')
    (target/'setup.py').write_text("from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize([Extension('sklearn.utils._random',['sklearn/utils/_random.pyx'])], compiler_directives={'language_level':3}))\n")
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True,stdout=subprocess.DEVNULL)
    sys.path.insert(0,str(target))
    from sklearn.utils._random import sample_without_replacement
    from sklearn.utils.validation import check_random_state
    source=ORACLE/files[0]
    classes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in ('ParameterGrid','ParameterSampler')]
    ns=dict(np=np,Mapping=Mapping,Iterable=Iterable,Sequence=Sequence,partial=partial,reduce=reduce,product=product,operator=operator,numbers=numbers,warnings=warnings,sample_without_replacement=sample_without_replacement,check_random_state=check_random_state)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),ns)
    return ns,{f:hashlib.sha256((ORACLE/f).read_bytes()).hexdigest() for f in files}

class Discrete:
    def __init__(self,values):self.values=values
    def rvs(self,random_state):return self.values[random_state.randint(len(self.values))]

def main():
    ns,hashes=load(); cases=[]
    for grids in ([{}],[],[{'z':[4,5],'a':[1,2,3]}],[{'a':[9]},{'b':[2,3],'a':[1,4]}],[{'a':list(range(200))}]):
        cases.append(('grid',grids,0,0))
        for seed in (0,42,113):
            for n in (1,3,200):cases.append(('sample',grids,n,seed))
    for spaces in ([{'z':[4,5],'a':Discrete([1,2,3])}],[{'a':Discrete([9,3])},{'b':[2,3],'a':[1,4]}],[{}, {'z':Discrete([4,7]),'a':[2]}]):
        for seed in (0,42,113):cases.append(('sample',spaces,12,seed))
    code='''from moxie.model_selection import ParameterGrid, ParameterSampler, ParameterDomain, DiscreteDistribution
from moxie._random import RandomState

def emit[P: moxie.model_selection.ParameterCandidates & Deinitable](mut cursor: P, case_id: Int) raises:
    var i = 0
    while True:
        var candidate = cursor.next()
        if not candidate:
            break
        for key in candidate.value().keys():
            print(case_id, i, key, candidate.value()[key])
        print(case_id, i, "END", 0)
        i += 1

import moxie.model_selection

def main() raises:
'''
    body=code[code.index('def emit['):code.index('import moxie.model_selection')]
    grid_emit=body.replace('def emit[P: moxie.model_selection.ParameterCandidates & Deinitable](mut cursor: P,', 'def emit(mut cursor: ParameterGrid[Int],')
    sampler_emit=body.replace('def emit[P: moxie.model_selection.ParameterCandidates & Deinitable](mut cursor: P,', 'def emit(mut cursor: ParameterSampler[DiscreteDistribution[Int]],')
    code=code.replace(body,grid_emit+sampler_emit)
    expected={};states={}
    for c,(kind,spaces,n,seed) in enumerate(cases):
        rng=np.random.RandomState(seed)
        values=list(ns['ParameterGrid'](spaces)) if kind=='grid' else list(ns['ParameterSampler'](spaces,n,random_state=rng))
        for i,v in enumerate(values):expected[c,i]=v
        if kind!='grid':states[c]=int(rng.randint(0,2**32,dtype=np.uint32))
        mixed=any(isinstance(v,Discrete) for d in spaces for v in d.values())
        if mixed:
            code+=f'    var spaces{c} = List[List[ParameterDomain[DiscreteDistribution[Int]]]]()\n'
            for g,space in enumerate(spaces):
                code+=f'    var domains{c}_{g} = List[ParameterDomain[DiscreteDistribution[Int]]]()\n'
                for key,value in space.items():
                    val='DiscreteDistribution[Int]('+repr(value.values)+')' if isinstance(value,Discrete) else repr(value)
                    code+=f'    domains{c}_{g}.append(ParameterDomain[DiscreteDistribution[Int]]("{key}", {val}))\n'
                code+=f'    spaces{c}.append(domains{c}_{g}^)\n'
            argument=f'spaces{c}'
        else:
            code+=f'    var spaces{c} = List[Dict[String, List[Int]]]()\n'
            for g,space in enumerate(spaces):
                code+=f'    var mapping{c}_{g} = Dict[String, List[Int]]()\n'
                for key,value in space.items():code+=f'    mapping{c}_{g}["{key}"] = {value}\n'
                code+=f'    spaces{c}.append(mapping{c}_{g}^)\n'
            code+=f'    var grid{c} = ParameterGrid[Int](spaces{c})\n'
            argument=f'grid{c}'
        if kind=='grid':code+=f'    emit(grid{c}, {c})\n'
        else:
            code+=f'    var rng{c} = RandomState({seed})\n    var sampler{c} = ParameterSampler[DiscreteDistribution[Int]]({argument}, {n}, rng{c})\n    emit(sampler{c}, {c})\n    print({c}, 0, "STATE", Int(rng{c}.next_uint32()))\n'
    source=ROOT/'builds/parameter_parity.mojo';source.write_text(code);binary=ROOT/'builds/parameter_parity'
    subprocess.run(['mojo','build','-O3','--fp-mode=contract=off','-I',str(ROOT/'builds/deps'),str(source),'-o',str(binary)],check=True)
    actual={};got_states={}
    for line in subprocess.check_output([str(binary)],text=True).splitlines():
        c,i,key,val=line.split();c=int(c);i=int(i)
        if key=='STATE':got_states[c]=int(val)
        else:
            row=actual.setdefault((c,i),{})
            if key!='END':row[key]=int(val)
    assert actual==expected,next(((k,actual.get(k),v) for k,v in expected.items() if actual.get(k)!=v),'extra candidates')
    assert got_states==states,(got_states,states)
    report=dict(revision=REV,files=hashes,cases=len(cases),candidates=len(expected),rng_states=len(states),tolerance=0)
    dest=ROOT/'docs/private/reference/parity.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS {len(cases)} cases, {len(expected)} exact candidates, {len(states)} exact subsequent RNG words')
if __name__=='__main__':main()
