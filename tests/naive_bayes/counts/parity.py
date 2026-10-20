"""Differential fixtures execute the pinned local Naive Bayes source."""
from pathlib import Path
import argparse
import importlib.util
import inspect
import os
import sys
import subprocess
import warnings
import numpy as np
import sklearn
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--deps', type=Path, default=ROOT/'builds/deps')
args = parser.parse_args()
DEPS = args.deps.resolve()
REFERENCE=ROOT.parent/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip()==REV
assert not subprocess.check_output(['git','-C',str(REFERENCE),'status','--porcelain','--','sklearn/naive_bayes.py','sklearn/tests/test_naive_bayes.py','doc/modules/naive_bayes.rst'])
source=REFERENCE/'sklearn/naive_bayes.py'
spec=importlib.util.spec_from_file_location('sklearn.naive_bayes',source)
nb=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=nb
spec.loader.exec_module(nb)
assert Path(inspect.getfile(nb.MultinomialNB))==source

def literal(x):
    if isinstance(x,(bool,np.bool_)): return str(bool(x))
    if isinstance(x,(int,np.integer)): return str(int(x))
    if isinstance(x,(float,np.floating)):
        if np.isnan(x): return 'Float64("nan")'
        if np.isinf(x): return ('-' if x<0 else '')+'Float64("inf")'
        return repr(float(x))
    return '['+', '.join(map(literal,x))+']'

lines=['''import numojo as nm
from std.math import isnan, isfinite
from std.testing import assert_true, assert_equal
from moxie._nb_counts import MultinomialNB, ComplementNB

@no_inline
def close(a: Float64, b: Float64) raises:
    if isnan(b):
        assert_true(isnan(a))
    elif not isfinite(b):
        assert_equal(a, b)
    else:
        assert_true(isfinite(a))
        if abs(a-b) > 1e-7 + 1e-7 * abs(b):
            print("mismatch", a, b)
        assert_true(abs(a-b) <= 1e-7 + 1e-7 * abs(b))
''']
rng=np.random.default_rng(912)
cases=[]
for model in ('MultinomialNB','ComplementNB'):
 for n,f,c in [(1,1,1),(3,2,2),(7,5,3)]:
  for alpha in [0.,1.,np.arange(f,dtype=float)*.3, np.full(f,1e-12)]:
   for norm in ([False,True] if model=='ComplementNB' else [False]):
    for force in [False,True]:
     cases.append((model,n,f,c,alpha,norm,force))
# Preserve the original cases, then exercise layouts independently of dtype and
# lifecycle. Padding differs from logical alpha so flat reads cannot pass.
cases = [case + ('contiguous', i % 2 == 1, 'float32' if i % 2 else 'float64')
         for i, case in enumerate(cases)]
for model in ('MultinomialNB', 'ComplementNB'):
 for norm in ([False, True] if model == 'ComplementNB' else [False]):
  for layout in ('strided', 'reversed'):
   for force in (False, True):
    for incremental in (False, True):
     for dtype in ('float32', 'float64'):
      cases.append((model, 7, 5, 3, np.array([0., .2, 1e-12, 2., .7]),
                    norm, force, layout, incremental, dtype))
with warnings.catch_warnings(),np.errstate(all='ignore'):
 warnings.simplefilter('ignore')
 for index,(model,n,f,c,alpha,norm,force,layout,incremental,dtype) in enumerate(cases):
    X=rng.integers(0,5,(n,f)).astype(float)
    if n==1: X[:]=0
    y=(np.arange(n)%c)*7-3
    weights=np.arange(n,dtype=float)+.5
    if index%4==0: X=X/2
    if layout != 'contiguous':
        stride = 2 if layout == 'strided' else -2
        offset = 1 if stride > 0 else 2 * f - 1
        backing = np.full(2 * f + 2, 9.)
        backing[offset + np.arange(f) * stride] = alpha
        alpha = backing[1:2*f:2] if stride > 0 else backing[2*f-1:0:-2]
        assert alpha.size == f
    kwargs={'alpha':alpha,'force_alpha':force,'fit_prior':index%3!=0}
    if model=='ComplementNB': kwargs['norm']=norm
    if index%5==0: kwargs['class_prior']=np.arange(1,c+1,dtype=float)/(c*(c+1)/2)
    oracle=getattr(nb,model)(**kwargs)
    if incremental:
        oracle.partial_fit(X,y,classes=np.arange(c)*7-3,sample_weight=weights)
        oracle.partial_fit(X,y,sample_weight=weights)
    else: oracle.fit(X,y,sample_weight=weights)
    body=[f'@no_inline\ndef case_{index}() raises:',f'    var X = nm.zeros[DType.{dtype}](nm.Shape({n}, {f}))',f'    var y: List[Int] = {literal(y)}',f'    var weights: List[Float64] = {literal(weights)}']
    for i in range(n):
     for j in range(f): body.append(f'    X[nm.Item({i}, {j})] = {literal(X[i,j])}')
    args=[f'force_alpha={force}',f'fit_prior={kwargs["fit_prior"]}']
    if np.ndim(alpha):
        if layout == 'contiguous':
            body.append(f'    var alpha = nm.zeros[DType.float64](nm.Shape({f}))')
        else:
            stride = 2 if layout == 'strided' else -2
            offset = 1 if stride > 0 else 2 * f - 1
            body.append(f'    var backing = nm.full[DType.float64](nm.Shape({2*f+2}), 9.)')
            body.append(f'    var alpha = backing.view_with_layout(nm.Shape({f}), nm.NDArrayStrides({stride}), {offset})')
        for j in range(f): body.append(f'    alpha[nm.Item({j})] = {literal(alpha[j])}')
        args.append('alpha_values=alpha^')
    else: args.append('alpha='+literal(alpha))
    if model=='ComplementNB': args.append(f'norm={norm}')
    if 'class_prior' in kwargs:
        body.append('    var prior: List[Float64] = '+literal(kwargs['class_prior']))
        args.append('class_prior=prior^')
    body.append(f'    var model = {model}('+', '.join(args)+')')
    if incremental:
        body.append(f'    var classes: List[Int] = {literal(np.arange(c)*7-3)}')
        body.append('    model.partial_fit(X, y, classes.copy(), sample_weight=weights.copy())')
        body.append('    model.partial_fit(X, y, sample_weight=weights.copy())')
    else: body.append('    model.fit(X, y, sample_weight=weights.copy())')
    for k,v in enumerate(oracle.class_count_): body.append(f'    close(model.class_count_.item({k}), {literal(v)})')
    for k,v in enumerate(oracle.class_log_prior_): body.append(f'    close(model.class_log_prior_[{k}], {literal(v)})')
    for field in ['feature_count_','feature_log_prob_']:
        for k,row in enumerate(getattr(oracle,field)):
            for j,v in enumerate(row): body.append(f'    close(model.{field}.item({k}, {j}), {literal(v)})')
    if model=='ComplementNB':
        for j,v in enumerate(oracle.feature_all_): body.append(f'    close(model.feature_all_.item({j}), {literal(v)})')
    for method in ['predict_joint_log_proba','predict_proba','predict_log_proba']:
        body.append(f'    var {method} = model.{method}(X)')
        for i,row in enumerate(getattr(oracle,method)(X)):
            for k,v in enumerate(row): body.append(f'    close({method}[{i}][{k}], {literal(v)})')
    body.append('    var predicted = model.predict(X)')
    for i,v in enumerate(oracle.predict(X)): body.append(f'    assert_equal(predicted[{i}], {literal(v)})')
    body.append(f'    close(model.score(X, y), {literal(oracle.score(X,y))})')
    lines.append('\n'.join(body))
# Keep generated sources away from stale standalone moxie.mojoc packages.
# Bounded batches also avoid one large optimized compilation for every fixture.
output_dir = ROOT/'builds/naive-bayes-counts-parity'
output_dir.mkdir(parents=True, exist_ok=True)
for start in range(0, len(cases), 12):
    stop = min(start + 12, len(cases))
    main = 'def main() raises:\n' + ''.join(
        f'    case_{i}()\n' for i in range(start, stop))
    main += f'    print("PASS parity cases {start}..{stop-1}")\n'
    fixture = output_dir/f'cases_{start}.mojo'
    binary = output_dir/f'cases_{start}'
    fixture.write_text('\n\n'.join([lines[0], *lines[start+1:stop+1], main]))
    subprocess.run(['mojo','build','-O3','--fp-mode=contract=off','-I',
                    str(DEPS),str(fixture),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
print(f'PASS {len(cases)} estimator parity cases')
print('Oracle:',source,REV,'support:',sklearn.__version__)
