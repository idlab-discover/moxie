"""Integrated family modules must be owned by Moxie and match live sources."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MOXIE = ROOT.parents[1]
receipt = json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())
assert receipt['modules']['moxie.kernel_ridge'] == ['moxie']
assert receipt['modules']['moxie.kernel_ridge_serialization'] == ['moxie']
for module, owners in receipt['modules'].items():
    if module.startswith(('moxie.kernel_ridge', 'moxie._kernel_ridge.')):
        assert owners == ['moxie'], (module, owners)
        relative = Path(*module.split('.')).with_suffix('.mojo')
        assert (ROOT / 'builds/deps' / relative).read_bytes() == (MOXIE / 'src' / relative).read_bytes()
assert all('moxie-kernel_ridge' not in project['path'] for project in json.loads((MOXIE / 'DEPENDENCIES.json').read_text())['projects'])
print('KernelRidge live source ownership checks passed')
