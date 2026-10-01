"""Scalar SciPy parity and gamma substitution into the pinned sklearn sample body."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import scipy
import mpmath as mp
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
SKLEARN = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SEED = 723619


def native(a, x, bits=64):
    dtype = np.float64 if bits == 64 else np.float32
    a, x = np.broadcast_arrays(np.asarray(a,dtype=dtype), np.asarray(x,dtype=dtype))
    path = ROOT/'builds/fixtures.txt'
    path.write_text(''.join(f'{bits} {float(ai):.17g} {float(xi):.17g}\n'
                            for ai,xi in zip(a.flat,x.flat)))
    lines = subprocess.check_output([ROOT/'builds/driver',path],text=True).splitlines()
    assert len(lines) == a.size
    failures = [(float(ai),float(xi),s) for ai,xi,s in zip(a.flat,x.flat,lines)
                if 'Error' in s]
    assert not failures, failures[:10]
    return np.array(lines,dtype=dtype).reshape(a.shape)


def sklearn_sample():
    path = SKLEARN/'sklearn/neighbors/_kde.py'
    assert subprocess.check_output(['git','-C',SKLEARN,'rev-parse','HEAD'],text=True).strip() == REVISION
    assert not subprocess.check_output(['git','-C',SKLEARN,'diff','HEAD','--',path],text=True)
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n,ast.ClassDef) and n.name == 'KernelDensity')
    node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'sample')
    namespace = dict(np=np, gammainc=gammainc,
                     check_is_fitted=lambda obj: None,
                     check_random_state=np.random.RandomState,
                     row_norms=lambda x,squared: np.einsum('ij,ij->i',x,x))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    assert namespace['sample'].__code__.co_filename == str(path)
    return namespace,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng=np.random.RandomState(SEED)
    report={'seed':SEED,'numpy':np.__version__,'scipy':scipy.__version__,'cases':{}}
    for bits in (32,64):
        dtype=np.float64 if bits == 64 else np.float32
        shapes=np.concatenate(([np.nextafter(dtype(0),dtype(1)),1e-30,1e-10,0.1,0.5,1,1.5,2,15.999,16,16.001],
                               np.geomspace(1e-5,1e8,80)))
        pairs=[(a,x) for a in shapes for x in np.r_[0,np.geomspace(1e-35,1e10,80),np.inf]]
        # Branch boundaries, central region and large deviations.
        for a in shapes:
            pairs.extend((a,x) for x in np.r_[a, np.nextafter(a,a+1),a+1,
                                              np.nextafter(a+1,a),np.nextafter(a+1,np.inf),
                                              a*np.array([.01,.5,.749999,.75,.750001,.99999,1.00001,1.249999,1.25,1.250001,2,100])])
            pairs.extend((a,max(0,a+z*np.sqrt(a))) for z in (-40,-15,-8,-2,2,8,15,40))
        aa=10**rng.uniform(-12,8,5000)
        xx=10**rng.uniform(-35,10,5000)
        pairs.extend(zip(aa,xx))
        if bits == 64:
            pairs.extend((a,x) for a in (1e-300,1e-100,0.5,1,2,100,1e8)
                         for x in (5e-324,1e-310,1e-300,1e-100,1e100,1e300,1.7976931348623157e308))
        a,x=np.asarray(pairs,dtype=dtype).T
        actual=native(a,x,bits)
        expected=gammainc(a,x)
        # SciPy's 2000-term series truncates the large-a lower tail outside
        # its asymptotic region. Independently integrate the defining integral
        # at 60 digits after t=x*exp(-u/sqrt(a)); never tune to our kernel.
        reference_overrides=[]
        for i in np.flatnonzero((a > 1e5) & (x < a-4.5*np.sqrt(a)) &
                                (x >= a-40*np.sqrt(a))):
            with mp.workdps(60):
                aa=mp.mpf(float(a[i])); xx=mp.mpf(float(x[i])); root=mp.sqrt(aa)
                integrand=lambda u: mp.exp(-root*u-xx*mp.expm1(-u/root))
                integral=mp.quad(integrand,[0,1,4,16,64])
                value=mp.exp(aa*mp.log(xx)-xx-mp.loggamma(aa)-mp.log(root))*integral
                reference_overrides.append(dict(a=float(aa),x=float(xx),scipy=float(expected[i]),
                                                quadrature=float(value)))
                expected[i]=float(value)
        # Relative tolerance for normal probabilities; absolute tolerance only
        # for subnormals, where rounding to the nearest representable value wins.
        rtol=1e-11 if bits == 64 else 3e-7
        atol=16*float(np.nextafter(dtype(0),dtype(1)))
        # The SciPy prefactor can underflow for subnormal shapes. mpmath
        # directly evaluates the regularized function without that underflow.
        for i in np.flatnonzero((a < 1e-280) | ((expected == 0) & (actual > 0))):
            with mp.workdps(60):
                value=float(mp.gammainc(mp.mpf(float(a[i])),0,
                                       mp.mpf(float(x[i])),regularized=True))
                reference_overrides.append(dict(a=float(a[i]),x=float(x[i]),
                                                scipy=float(expected[i]),mpmath=value))
                expected[i]=value
        relative=np.abs(actual.astype(float)-expected)/np.maximum(expected,np.finfo(dtype).tiny)
        bad=np.abs(actual.astype(float)-expected) > atol+rtol*np.abs(expected)
        assert not np.any(bad), list(zip(a[bad],x[bad],actual[bad],expected[bad],relative[bad]))[:15]
        assert np.all((actual>=0)&(actual<=1))
        report['cases'][str(bits)]={'count':len(a),'max_relative_normal':float(relative.max()),'rtol':rtol,'atol':atol,'independent_tail_references':reference_overrides}
        # Monotonic CDF, including both evaluation branches.
        xs=np.geomspace(1e-8,1e9,500)
        for shape in (0.5,1.5,16,100,10000,1e8):
            assert np.all(np.diff(native(shape,xs,bits).astype(float)) >= 0)
    # Independent checks of cancellation-sensitive, non-subnormal deep tails.
    aa=np.array([.5,1.5,16,100,1689.8497868124552,3605.471154250499,10000.,16.])
    xx=np.r_[aa[:-1]/2,9e-20]
    with mp.workdps(60):
        reference=np.array([float(mp.gammainc(mp.mpf(float(a)),0,mp.mpf(float(x)),regularized=True))
                            for a,x in zip(aa,xx)])
    np.testing.assert_allclose(native(aa,xx),reference,rtol=3e-12,atol=16*np.nextafter(0.,1.))
    namespace,source_hash=sklearn_sample()
    for dim in (1,2,3,10,100,1000):
        data=rng.normal(size=(5,dim))
        for weights in (None,np.array([1,2,0,3,4.])):
            model=SimpleNamespace(kernel='tophat',bandwidth_=0.37,
                                  tree_=SimpleNamespace(data=data,sample_weight=weights))
            namespace['gammainc']=gammainc
            expected=namespace['sample'](model,64,random_state=42)
            namespace['gammainc']=native
            actual=namespace['sample'](model,64,random_state=42)
            np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-14)
    report['sklearn']={'revision':REVISION,'source_sha256':source_hash,'sample_cases':12,
                       'scope':'actual sample method, replacing only the gamma call; no native estimator claim'}
    (ROOT/'builds/parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print("PASS", {bits: data["count"] for bits,data in report["cases"].items()},
          "scalar cases; 12 sklearn sample cases; details: builds/parity.json")


if __name__ == '__main__':
    main()
