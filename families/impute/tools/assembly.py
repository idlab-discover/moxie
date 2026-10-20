"""Launch the live shared assembler from the local Moxie checkout.

Every build reads the current files at each declared project path, including
uncommitted edits in Moxie. There are no source revision or hash gates.
"""
import json
from pathlib import Path
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]
ASSEMBLY = (FAMILY.parents[1] / 'shared/assembly').resolve()


def assemble(consumer, lock, env, check=True, capture=False):
    consumer, lock = Path(consumer).resolve(), Path(lock).resolve()
    data = json.loads(lock.read_text())
    owner = (lock.parent / next(p for p in data['projects'] if p['id'] == 'assembly')['path']).resolve()
    if owner != ASSEMBLY:
        raise ValueError(f'{lock}: assembly project must point at {ASSEMBLY}, not {owner}')
    return subprocess.run([sys.executable, str(ASSEMBLY / 'tools/consume_family.py'),
                           '--consumer', str(consumer), '--lock', str(lock),
                           '--tool', str(ASSEMBLY / 'tools/assemble_family.py')],
                          cwd=consumer, env=env, check=check,
                          capture_output=capture, text=True)
