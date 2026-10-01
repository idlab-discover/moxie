"""Compare native named inputs with pinned sklearn's named-column semantics.

Pandas is solely an oracle input adapter. No native runtime dependency or Python
binding is introduced. Exact equality is required: these cases only select and
concatenate numeric columns without arithmetic.
"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/columns"
sys.path.insert(0, str(ROOT / 'tools'))
import compose_parity_io as io
from compose_parity_io import environment

sources = io.verify_reference(['sklearn/compose/_column_transformer.py'])
module = io.load_pinned('sklearn/compose/_column_transformer.py')
io.pinned_code(module.ColumnTransformer.transform, 'sklearn/compose/_column_transformer.py')
native = subprocess.check_output([str(BUILD / 'named_parity_driver')], text=True, env=environment(ROOT))
results = io.parse_results(native)
names = {line.split('|')[1]: line.split('|')[2].split(',') for line in native.splitlines() if line.startswith('names|')}
report = []
for dtype in (np.float32, np.float64):
    X = pd.DataFrame(np.arange(3, dtype=dtype)[None, :] + 10 * np.arange(3, dtype=dtype)[:, None], columns=['age', 'income', 'unused'])
    Z = pd.DataFrame([[102, 100, 101, 999], [112, 110, 111, 999]], columns=['unused', 'age', 'income', 'extra'], dtype=dtype)
    keys = [['income', 'age'], slice('age', 'income'), module.make_column_selector(dtype_include=['floating']),
            module.make_column_selector(dtype_include=['float32']), slice(2, None, -1), [1, 0], [False, True, False]]
    for case, key in enumerate(keys):
        ident = f'{case}_{np.dtype(dtype).itemsize * 8}'
        ct = module.ColumnTransformer([('selected', 'passthrough', key)], remainder='passthrough')
        out = ct.fit_transform(X)
        tz = ct.transform(Z)
        np.testing.assert_array_equal(results[ident]['out'], out)
        np.testing.assert_array_equal(results[ident]['tz'], tz)
        assert names[ident] == list(ct.get_feature_names_out()), ident
        report.append(ident)
folder = BUILD / 'parity'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'named-report.json').write_text(json.dumps(dict(revision=io.REVISION, sources=sources, cases=report, pandas=pd.__version__), indent=2) + '\n')
print(f'Named-column parity passed: {len(report)} cases, exact outputs and feature names')
