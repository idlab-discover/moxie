"""build/test/parity for LinearRegression through the family's verified assembly."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from childcheck import main

NATIVE = ['tests/contracts.mojo', 'examples/linear_regression.mojo', 'tests/parity_driver.mojo', 'tests/benchmark.mojo']
if __name__ == '__main__':
    main(ROOT, NATIVE, tests=['contracts', 'linear_regression'], parity='tests/parity.py')
