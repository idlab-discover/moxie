"""Unsupported sparse-returning API must remain unavailable."""
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'builds/tree/storage/no_decision_path.mojo'
source.write_text('from moxie.tree._tree import Tree\ndef main() raises:\n    var engine = Tree()\n    engine.decision_path()\n')
result=subprocess.run(['mojo','build','-I',str(ROOT/'builds/deps'),str(source),'-o',str(ROOT/'builds/tree/storage/no_decision_path')],text=True,capture_output=True)
assert result.returncode != 0 and 'decision_path' in result.stderr and 'attribute' in result.stderr, result.stderr
print('PASS decision_path unavailable')
