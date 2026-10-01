"""Build current shared sources; Python is validation tooling only."""
from pathlib import Path
import subprocess, sys, json
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY=ROOT.parent/'assembly'
def run(*args): subprocess.run([str(a) for a in args],cwd=ROOT,check=True)
def assemble(consumer):
    run(sys.executable, ASSEMBLY/'tools/consume.py','--consumer',consumer,
        '--lock',consumer/'DEPENDENCIES.lock.json','--tool',ASSEMBLY/'tools/assemble.py')
def build():
    assert subprocess.check_output(['mojo','--version'],text=True).startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble(ROOT)
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/assembled.mojoc')
    run('mojo','precompile',ROOT/'src/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/random.mojoc')
    for source,output in [('tests/contracts.mojo','contracts'),('tests/normalization.mojo','normalization'),('tests/normalization_draws.mojo','normalization_draws'),('tests/draws.mojo','draws'),('tests/benchmark.mojo','benchmark'),('examples/seeded.mojo','example')]:
        run('mojo','build','-O3','--fp-mode=contract=off','-I',ROOT/'builds/deps',ROOT/source,'-o',ROOT/'builds'/output)
    for name in ['a','b']:
        consumer=ROOT/'tests/consumers'/name
        assemble(consumer)
        run('mojo','build','-O3','--fp-mode=contract=off','-I',consumer/'builds/deps',consumer/'main.mojo','-o',consumer/'builds/probe')
def main():
    build()
    if sys.argv[1]=='test':
        run(ROOT/'builds/contracts'); run(ROOT/'builds/normalization'); run(ROOT/'builds/example')
        outputs=[subprocess.check_output([ROOT/'tests/consumers'/n/'builds/probe']) for n in ['a','b']]
        assert outputs[0]==outputs[1]
        rejected=subprocess.run(['mojo','build','-I',str(ROOT/'builds/deps'),str(ROOT/'tests/negative/sparse.mojo'),'-o',str(ROOT/'builds/unsupported')],capture_output=True,text=True)
        assert rejected.returncode != 0 and '_random_choice_csc' in rejected.stderr
        print('PASS: sparse capability is unavailable')
        for name, diagnostic in [('copy', 'copy'), ('escape', 'origin'),
                                 ('alias', 'aliasing'), ('pointer_escape', 'origin'),
                                 ('moved_source', 'uninitialized')]:
            source = ROOT/'tests/negative'/f'normalization_{name}.mojo'
            rejected = subprocess.run(['mojo','build','-I',str(ROOT/'builds/deps'),
                                       str(source),'-o',str(ROOT/'builds/unsupported')],
                                      capture_output=True,text=True)
            (ROOT/'builds'/f'normalization-{name}.log').write_text(rejected.stderr)
            assert rejected.returncode != 0 and diagnostic in rejected.stderr, rejected.stderr
        print('PASS: normalization rejects copying, aliasing and escaped references')
    elif sys.argv[1]=='parity': run(sys.executable,ROOT/'tests/parity.py')
if __name__=='__main__': main()
