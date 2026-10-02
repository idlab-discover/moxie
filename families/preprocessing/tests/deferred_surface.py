"""Report public declarations that still raise the API-scaffold NotImplementedError.

With --require-complete (used once the whole surface is implemented) any
remaining deferred entry point is a failure.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
remaining = []
for path in sorted((ROOT.parents[1] / 'src/moxie/preprocessing').glob('_*.mojo')):
    for match in re.finditer(r'NotImplementedError: (\w+(?:\.\w+)?) is an API scaffold', path.read_text()):
        remaining.append(f'{path.name}:{match.group(1)}')
if remaining and '--require-complete' in sys.argv:
    raise SystemExit('deferred public API remains:\n' + '\n'.join(remaining))
print(f'PASS deferred-surface report: {len(remaining)} scaffold entry points remain')
