"""Run the current shared assembler against declared live source paths."""
import json
from pathlib import Path
import sys


def assembly_command(consumer, lock=None):
    consumer = Path(consumer).resolve()
    manifest = Path(lock).resolve() if lock else consumer / 'DEPENDENCIES.json'
    projects = [p for p in json.loads(manifest.read_text())['projects'] if p['id'] == 'assembly']
    if len(projects) != 1:
        raise ValueError('exactly one assembly project required')
    source = (manifest.parent / projects[0]['path']).resolve()
    tool = source / 'tools/assemble_family.py'
    if not tool.is_file():
        raise FileNotFoundError(f'missing current shared assembler: {tool}')
    return [sys.executable, str(tool), '--consumer', str(consumer),
            '--lock', str(manifest), '--output', str(consumer / 'builds/deps')]
