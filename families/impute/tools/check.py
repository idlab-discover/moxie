"""Family validation runner against live local Moxie HEAD via the shared assembler."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = [ROOT/'tests/consumers'/name for name in ('a', 'b')]
FLAGS = ['-O3', '--fp-mode=contract=off']
SIMPLE = ROOT/'algorithms/simple'
KNN = ROOT/'algorithms/knn'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def environment(consumer):
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME':'.pixi/cache', 'PIXI_CACHE_DIR':'.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR':'.pixi/modular-cache', 'TMPDIR':'builds/tmp'}.items():
        path = consumer/folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    return env

def run(*args, consumer=ROOT, check=True, capture=False):
    return subprocess.run([str(a) for a in args], cwd=consumer, env=environment(consumer),
                          check=check, capture_output=capture, text=True)

def assemble(consumer, lock=None, check=True):
    lock = lock or consumer/'DEPENDENCIES.json'
    from assembly import assemble as assemble_live
    return assemble_live(consumer, lock, environment(consumer), check=check, capture=not check)

def compile_source(source, output, consumer=ROOT):
    run('mojo', 'build', *FLAGS, '-I', consumer/'builds/deps', '-I', ROOT/'examples',
        source, '-o', output, consumer=consumer)

def compile_consumer(consumer):
    assemble(consumer)
    run('mojo', 'precompile', consumer/'builds/deps/moxie', '-I', consumer/'builds/deps',
        '-o', consumer/'builds/assembled.mojoc', consumer=consumer)
    compile_source(consumer/'main.mojo', consumer/'builds/probe', consumer)

def build():
    if not run('mojo', '--version', capture=True).stdout.startswith('Mojo 1.0.0 '):
        raise RuntimeError('compiler mismatch')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(ROOT)
    run('mojo', 'precompile', ROOT/'builds/deps/moxie', '-I', ROOT/'builds/deps',
        '-o', ROOT/'builds/moxie.mojoc')
    for source in sorted((ROOT/'tests').glob('*.mojo')):
        compile_source(source, ROOT/'builds'/source.stem)
    for source in sorted((ROOT/'examples').glob('*.mojo')):
        compile_source(source, ROOT/'builds'/source.stem)
    for algorithm in (SIMPLE, KNN):
        output = ROOT/'builds'/algorithm.name
        output.mkdir(exist_ok=True)
        for folder in ('tests', 'examples'):
            for source in sorted((algorithm/folder).glob('*.mojo')):
                compile_source(source, output/source.stem)
    for consumer in CONSUMERS:
        compile_consumer(consumer)

def negative_compilation():
    for name, expected in [('missing_predict', "required function 'predict' is not implemented"),
                           ('unsupported_matrix', 'missing_mask')]:
        proc = run('mojo', 'build', '-I', ROOT/'builds/deps',
                   ROOT/'tests/compile_fail'/f'{name}.mojo', '-o', ROOT/'builds/invalid',
                   check=False, capture=True)
        (ROOT/'builds'/f'{name}.log').write_text(proc.stdout+proc.stderr)
        assert proc.returncode != 0 and expected in proc.stderr, proc.stderr

def assembly_contracts():
    reports = []
    for consumer in CONSUMERS:
        run(consumer/'builds/probe', consumer=consumer)
        report = json.loads((consumer/'builds/deps/ASSEMBLY.json').read_text())
        reports.append(report)
        assert report['modules']['moxie.impute._plumbing'] == ['impute']
        assert report['modules']['moxie.impute._simple'] == ['impute']
        assert report['modules']['moxie.impute._string'] == ['impute']
        assert report['modules']['moxie.impute._indicator'] == ['impute']
        assert report['modules']['moxie.impute._knn'] == ['impute']
        assert report['modules']['moxie.neighbors._query_engine'] == ['query_engine']
        assert report['modules']['moxie.neighbors._contracts'] == ['neighbors']
        assert report['modules']['moxie._dense'] == ['numerics']
        assert report['modules']['moxie.probe_a'] == ['probe_a']
        assert report['modules']['moxie.probe_b'] == ['probe_b']
        assert sha(consumer/'builds/deps/moxie/impute/_plumbing.mojo') == sha(ROOT/'src/moxie/impute/_plumbing.mojo')
        assert not (consumer/'builds/deps/moxie/_sparse.mojo').exists()
    shared = 'builds/deps/moxie/impute/_plumbing.mojo'
    assert (CONSUMERS[0]/shared).stat().st_ino != (CONSUMERS[1]/shared).stat().st_ino
    consumer = CONSUMERS[0]
    original = json.loads((consumer/'DEPENDENCIES.json').read_text())
    before = sha(consumer/'builds/deps/ASSEMBLY.json')
    for mode in ('missing', 'owner', 'export'):
        lock = json.loads(json.dumps(original))
        family = next(p for p in lock['projects'] if p['id'] == 'impute')
        probe = next(p for p in lock['projects'] if p['id'] == 'probe_b')
        if mode == 'missing':
            next(project for project in lock['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file'] = 'src/moxie/absent.mojo'
        elif mode == 'owner':
            duplicate = json.loads(json.dumps(family))
            duplicate['id'] = 'conflicting-family'
            lock['projects'].append(duplicate)
        else:
            probe['exports']['packages']['moxie']['ProbeA'] = 'moxie.probe_b'
        # Preserve relative source resolution: the alternate lock is next to the real one.
        path = consumer/'invalid.lock.json'
        try:
            path.write_text(json.dumps(lock))
            proc = assemble(consumer, path, check=False)
            assert proc.returncode != 0, mode
            assert any(term in proc.stderr.lower() for term in
                       ('missing file', 'conflict', 'duplicate', 'ownership', 'collision')), proc.stderr
            assert sha(consumer/'builds/deps/ASSEMBLY.json') == before
        finally:
            path.unlink(missing_ok=True)

def test():
    for source in sorted((ROOT/'tests').glob('*.mojo')):
        run(ROOT/'builds'/source.stem)
    for algorithm in (SIMPLE, KNN):
        for source in sorted((algorithm/'tests').glob('test_*.mojo')):
            run(ROOT/'builds'/algorithm.name/source.stem)
    negative_compilation()
    assembly_contracts()
    print('PASS: inherited contracts, dense plumbing, isolated consumers and assembly failures')

def main():
    action = sys.argv[1]
    if action not in ('test-only', 'parity-only'):
        build()
    if action in ('test', 'test-only'):
        test()
    elif action == 'start':
        run(ROOT/'builds/simple/simple')
    elif action in ('parity', 'parity-only'):
        run('pixi', 'run', '--manifest-path', ROOT.parents[1]/'pixi.toml', 'python', SIMPLE/'tests/parity.py', '--build-dir', ROOT/'builds')
        run('pixi', 'run', '--manifest-path', ROOT.parents[1]/'pixi.toml', 'python', SIMPLE/'tests/integer_parity.py', '--build-dir', ROOT/'builds')
        run('pixi', 'run', '--manifest-path', ROOT.parents[1]/'pixi.toml', 'python', SIMPLE/'tests/string_parity.py', '--build-dir', ROOT/'builds')
        run('pixi', 'run', '--manifest-path', ROOT.parents[1]/'pixi.toml', 'python', KNN/'tests/parity.py')
    elif action != 'build':
        raise ValueError(action)
    print(f'PASS: {action}')

if __name__ == '__main__':
    main()
