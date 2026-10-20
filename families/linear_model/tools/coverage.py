"""Account for reference public names against actual assembled exports.

This reports export availability, not numerical or behavioral parity. Declaration
stubs in the historical _api module never count as implemented exports.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn/sklearn/linear_model/__init__.py'


def report():
    tree = ast.parse(REFERENCE.read_text())
    names = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets))
    package = ROOT / 'builds/deps/moxie/linear_model/__init__.mojo'
    exports = {}
    for line in package.read_text().splitlines():
        if line.startswith('from '):
            _, module, _, symbol = line.split()
            exports[symbol] = module
    rows = []
    for name in sorted(names):
        module = exports.get(name)
        if module and module.endswith('._api'):
            raise ValueError(f'{name}: declaration stub exported as an implementation')
        rows.append({'name': name, 'module': module,
                     'status': 'exported; behavior requires parity evidence' if module else 'unimplemented'})
    return {'reference': str(REFERENCE), 'public_names': len(rows),
            'exported_names': sum(r['module'] is not None for r in rows), 'apis': rows}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
