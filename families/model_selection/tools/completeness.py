"""Fail visibly while queued public APIs remain declaration-only."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
exports=json.loads((root/'EXPORTS.json').read_text())['packages']['moxie.model_selection']
missing=[name for name, module in exports.items() if module.endswith('._api')]
if missing:
    raise SystemExit('INCOMPLETE public APIs: '+', '.join(missing))
print('PASS implemented export ownership (behavioral parity is a separate gate)')
