"""Execute the pinned local DictVectorizer class, with tooling-only plumbing.

The class and its methods are unchanged AST nodes. Decorator/base/validation
plumbing below replaces unavailable compiled sklearn imports, not the encoder.
Finite nonempty rank-2 inverse cases exercise actual inverse_transform body;
its external check_array error semantics are independently native-tested.
"""
import ast
from array import array
from collections.abc import Iterable, Mapping
from numbers import Number
from operator import itemgetter
from pathlib import Path
import hashlib,json,subprocess,types
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
SK=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def load():
    assert subprocess.check_output(['git','-C',str(SK),'rev-parse','HEAD'],text=True).strip()==REV
    source=SK/'sklearn/feature_extraction/_dict_vectorizer.py'
    assert not subprocess.check_output(['git','-C',str(SK),'diff','HEAD','--',str(source)],text=True)
    pins=json.loads((ROOT/'tests/ORACLE.json').read_text())
    for path,digest in pins.items():
        assert hashlib.sha256((SK/path).read_bytes()).hexdigest()==digest,path
    class BaseEstimator: pass
    class TransformerMixin: pass
    def check_is_fitted(obj,names):
        for name in ([names] if isinstance(names,str) else names):
            if not hasattr(obj,name):raise ValueError('NotFittedError')
    def check_array(x,**kwargs):
        x=np.asarray(x)
        if x.ndim!=2 or not min(x.shape) or not np.isfinite(x).all():raise ValueError('invalid inverse array')
        return x
    ns=dict(globals(),BaseEstimator=BaseEstimator,TransformerMixin=TransformerMixin,
            _fit_context=lambda **kw:lambda f:f,check_is_fitted=check_is_fitted,
            check_array=check_array,metadata_routing=types.SimpleNamespace(UNUSED=None),
            _align_api_if_sparse=lambda x:x)
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DictVectorizer')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    result=ns['DictVectorizer']
    assert result._transform.__code__.co_filename==str(source)
    return result
