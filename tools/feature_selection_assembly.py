"""Launch the live shared assembler from the local Moxie checkout.

Every build reads the current files at each declared project path, including
uncommitted edits in ../moxie. There are no source revision or hash gates.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = (ROOT / 'shared/assembly').resolve()


def assemble(consumer, run, lock_path=None, check=True, capture=False):
    consumer = Path(consumer).resolve()
    lock_path = Path(lock_path or consumer / 'DEPENDENCIES.json').resolve()
    lock = json.loads(lock_path.read_text())
    owner = (lock_path.parent / next(p for p in lock['projects'] if p['id'] == 'assembly')['path']).resolve()
    if owner != ASSEMBLY:
        raise ValueError(f'{lock_path}: assembly project must point at {ASSEMBLY}, not {owner}')
    return run(sys.executable, ASSEMBLY / 'tools/consume_family.py',
               '--consumer', consumer, '--lock', lock_path,
               '--tool', ASSEMBLY / 'tools/assemble_family.py', check=check, capture=capture)
