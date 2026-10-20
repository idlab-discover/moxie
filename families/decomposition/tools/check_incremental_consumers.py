"""Verify IncrementalPCA from two separately assembled native namespaces."""
import hashlib
import sys
from family import ROOT, ASSEMBLY, run, working_lock


def check():
    lock = working_lock(ROOT / "DEPENDENCIES.json")
    copies = []
    for name in ('a', 'b'):
        consumer = ROOT / 'builds' / ('ipca-consumer-' + name)
        consumer.mkdir(parents=True, exist_ok=True)
        run(sys.executable, ASSEMBLY / 'tools/consume_family.py', '--consumer', consumer,
            '--lock', lock, '--tool', ASSEMBLY / 'tools/assemble_family.py')
        deps = consumer / 'builds/deps'
        run('patch', '--batch', '--fuzz=0', '-p1', '-d', deps, '-i', ROOT / 'tools/numojo_mojo11.patch')
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', deps,
            '-I', ROOT.parents[2] / 'numax',
            '-I', ROOT / 'backends/numax', ROOT / 'examples/numax_incremental_pca.mojo',
            '-o', consumer / 'probe')
        run(consumer / 'probe')
        copies.append(deps / 'moxie/decomposition/_incremental_pca.mojo')
    assert copies[0].stat().st_ino != copies[1].stat().st_ino
    assert hashlib.sha256(copies[0].read_bytes()).digest() == hashlib.sha256(copies[1].read_bytes()).digest()
    print('PASS: IncrementalPCA in two isolated assembled consumers')


if __name__ == '__main__': check()
