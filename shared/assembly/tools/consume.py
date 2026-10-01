"""Small launcher: execute the shared tool at its declared local path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def consume(consumer, lock, tool):
    consumer, lock, tool = Path(consumer).resolve(), Path(lock).resolve(), Path(tool).resolve()
    data = json.loads(lock.read_text())
    pins = [p for p in data['projects'] if p['id'] == 'assembly']
    if len(pins) != 1:
        raise ValueError('exactly one assembly project required')
    pin = pins[0]
    expected = (lock.parent / pin['path'] / 'tools/assemble.py').resolve()
    if tool != expected:
        raise ValueError('wrong assembly tool path')
    subprocess.run([sys.executable, str(tool), '--consumer', str(consumer),
                    '--lock', str(lock), '--output', str(consumer / 'builds/deps')], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--consumer', required=True, type=Path)
    p.add_argument('--lock', required=True, type=Path)
    p.add_argument('--tool', required=True, type=Path)
    args = p.parse_args()
    consume(args.consumer, args.lock, args.tool)
