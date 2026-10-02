"""Regenerate isolated-consumer manifests from the root DEPENDENCIES.json.

The root manifest is the single source of truth; consumers receive every project
with paths rebased to their own directory. Run after editing the root manifest.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = [ROOT / 'tests/consumers' / name for name in ('a', 'b')]


def rebased(consumer):
    root = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    for project in root['projects']:
        target = (ROOT / project['path']).resolve()
        project['path'] = os.path.relpath(target, consumer) if target != consumer else '.'
    return root


def main(check=False):
    stale = []
    for consumer in CONSUMERS:
        text = json.dumps(rebased(consumer), indent=2) + '\n'
        path = consumer / 'DEPENDENCIES.json'
        if path.read_text() != text:
            stale.append(path)
            if not check:
                path.write_text(text)
    return stale


if __name__ == '__main__':
    import sys
    stale = main(check='--check' in sys.argv)
    if stale and '--check' in sys.argv:
        raise SystemExit(f'stale consumer manifests: {stale}; run python tools/sync_manifests.py')
