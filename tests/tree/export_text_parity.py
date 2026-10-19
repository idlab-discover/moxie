"""Exact text and binary64 decimal formatting checks against the pinned oracle."""
from pathlib import Path
import json
import struct
import subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    revision = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
    expected = subprocess.check_output(['git', '-C', str(ROOT.parent / 'scikit-learn'), 'show', f'{revision}:sklearn/tree/_export.py'])
    assert (ROOT / 'builds/tree/reference/sklearn/tree/_export.py').read_bytes() == expected
    output = subprocess.check_output([ROOT / 'builds/tree/export_text_driver'], text=True)
    reports, formats = output.split('FORMATS\n')
    actual = {}
    for block in reports.split('REPORT ')[1:]:
        key, report = block.split('\n', 1)
        actual[tuple(map(int, key.split()))] = report
    failures = []
    checks = 0
    for tag, cls in enumerate((DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier)):
        classifier = tag % 2 == 1
        for case in range(20):
            n = 16 + case
            outputs = 2 if case % 3 == 0 else 1
            X = np.array([[((i * 3 + case) % 13) / 4, (i % 7) / 4] for i in range(n)], dtype=np.float32)
            y = np.array([[(i + i // 3 + k) % (1 if case == 1 else 3) if classifier else ((i * 7 + case + k) % 13) / 4 for k in range(outputs)] for i in range(n)], dtype=float)
            w = np.array([(1 + i % 3) / 4 for i in range(n)])
            model = cls(random_state=case, max_depth=4).fit(X, y, sample_weight=w)
            for mode in range(4):
                kw = dict(feature_names=['height', 'colour intensity'] if mode == 1 else None, class_names=['first', 'second', 'third'] if classifier and outputs == 1 and case != 1 and mode == 1 else None, max_depth=0 if mode == 2 else 10, spacing=1 if mode == 3 else 3, decimals=mode, show_weights=mode % 2 == 1)
                want = export_text(model, **kw)
                got = actual[tag, case, mode]
                checks += 1
                if got != want:
                    failures.append(dict(key=[tag, case, mode], got=got, want=want))
    for line in formats.splitlines():
        bits, decimals, got = line.split()
        value = struct.unpack('d', struct.pack('Q', int(bits)))[0]
        want = f'{value:.{decimals}f}'
        checks += 1
        if got != want:
            failures.append(dict(bits=bits, decimals=decimals, got=got, want=want))
    dest = ROOT / 'docs/private/tree/export_text'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'native.txt').write_text(output)
    (dest / 'results.json').write_text(json.dumps(dict(checks=checks, failures=failures), indent=2))
    print(f'{checks} exact export-text/format comparisons; {len(failures)} mismatches')
    assert not failures, failures[:2]

if __name__ == '__main__':
    main()
