"""Reproduce required-but-unimplemented heterogeneous categorical capabilities."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from oracle import load
oracle=load()
X=np.array([['a'],[1],[None]],dtype=object)
expected=oracle.SimpleImputer(strategy='most_frequent',missing_values=None).fit_transform(X)
mask=oracle.MissingIndicator(missing_values=None,sparse=False).fit_transform(X)
(ROOT/'docs/private/impute.simple/gaps.json').write_text(json.dumps(dict(categorical=dict(input=[['a'],[1],[None]],expected=expected.tolist(),indicator_expected=mask.tolist(),actual='heterogeneous categories are outside StringMatrix')),indent=2)+'\n')
print('Confirmed unfinished heterogeneous categorical estimator APIs')
