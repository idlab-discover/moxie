from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from childcheck import main
if __name__ == '__main__':
    main(ROOT, ['tests/contracts.mojo', 'tests/parity_driver.mojo', 'examples/elastic_net.mojo'], ['contracts', 'elastic_net'], 'tests/parity.py')
