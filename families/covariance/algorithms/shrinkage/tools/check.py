"""Shrinkage child checks through the family's shared child driver."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from child_check import main

SOURCES = ['tests/contracts.mojo', 'tests/parity_driver.mojo', 'examples/shrinkage.mojo']
if __name__ == '__main__':
    main(ROOT, SOURCES, tests=['contracts', 'shrinkage'], parity='tests/parity.py')
