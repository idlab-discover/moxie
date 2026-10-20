"""Build/validate dense IncrementalPCA in the isolated Numax environment."""
import sys
import os
from check_numax_pca import build as build_pca
from family import ROOT, run


def build():
    build_pca(pca_tests=False)
    includes = ['-I', ROOT / 'builds/numax-deps', '-I', ROOT.parents[2] / 'numax',
                '-I', ROOT / 'backends/numax',
                '-I', ROOT / 'tests/support']
    for source in sorted((ROOT / 'tests/numax_incremental_pca').glob('*.mojo')) + [ROOT / 'examples/numax_incremental_pca.mojo']:
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', *includes, source,
            '-o', ROOT / 'builds' / ('ipca_' + source.stem))


if __name__ == '__main__':
    action = sys.argv[1]
    build()
    if action == 'test':
        run(ROOT / 'builds/ipca_contracts')
        run(sys.executable, ROOT / 'tools/check_incremental_consumers.py')
    elif action == 'parity':
        os.environ['MOXIE_IPCA_DRIVER'] = str(ROOT / 'builds/ipca_driver')
        run(sys.executable, ROOT / 'tests/incremental_pca/parity.py')
    elif action != 'build': raise ValueError(action)
