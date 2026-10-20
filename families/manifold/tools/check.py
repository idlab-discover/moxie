"""Local checks invoking the current Moxie assembler and live source dependencies."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
ASSEMBLY = CORE / 'shared/assembly'
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command
CONSUMERS = [ROOT / 'tests/consumers' / name for name in ('a', 'b')]
FLAGS = ['-O3', '--fp-mode=contract=off']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def environment(consumer):
    env = os.environ.copy()
    for name, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                         'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = consumer / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env

def run(*args, consumer=ROOT, check=True):
    return subprocess.run(compiler_command(args), cwd=consumer, env=environment(consumer),
                          check=check, text=True, capture_output=not check)

def assemble(consumer, lock=None, check=True):
    if consumer == ROOT and lock is None:
        return run(sys.executable, ASSEMBLY / 'tools/assemble_family.py', '--consumer', ROOT,
                   '--lock', CORE / 'DEPENDENCIES.json', '--output', ROOT / 'builds/deps', check=check)
    lock = lock or consumer / 'DEPENDENCIES.lock.json'
    data = json.loads(lock.read_text())
    for project in data['projects']:
        if project['id'] == 'numojo':
            project['path'] = str(CORE.parent / 'NuMojo')
        else:
            project['path'] = str((lock.parent / project['path']).resolve())
        if project['id'] == 'manifold':
            project['modules'] = {
                'moxie.manifold.' + path.stem: {'file': str(path.relative_to(ROOT))}
                for path in (ROOT / 'src/moxie/manifold').glob('*.mojo')
                if path.stem != '__init__'}
            project['exports'] = json.loads((ROOT / 'EXPORTS.json').read_text())
    # Follow the RNG owner's current transitive dependency declarations.
    random_manifest = CORE / 'shared/random/DEPENDENCIES.lock.json'
    known = {project['id'] for project in data['projects']}
    for project in json.loads(random_manifest.read_text())['projects']:
        if project['id'] not in known:
            project['path'] = str((random_manifest.parent / project['path']).resolve())
            data['projects'].append(project)
            known.add(project['id'])
    for project in json.loads((ROOT / 'NUMERICAL-DEPENDENCIES.json').read_text())['projects']:
        project['path'] = str((ROOT / project['path']).resolve())
        data['projects'].append(project)
    generated = consumer / 'builds/dependencies.json'
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(data))
    return run(sys.executable, ASSEMBLY / 'tools/assemble_family.py', '--consumer', consumer,
               '--lock', generated, '--output', consumer / 'builds/deps', consumer=consumer, check=check)

def compile_file(source, output, consumer=ROOT):
    run('mojo', 'build', *FLAGS, '-I', consumer / 'builds/deps', '-I', ROOT / 'examples', '-I', ROOT / 'tests',
        source, '-o', output, consumer=consumer)

def build_consumer(consumer):
    assemble(consumer)
    run('mojo', 'precompile', consumer / 'builds/deps/moxie', '-I', consumer / 'builds/deps',
        '-o', consumer / 'builds/package.mojoc', consumer=consumer)
    compile_file(consumer / 'main.mojo', consumer / 'builds/probe', consumer)

def build():
    assemble(ROOT)
    run('mojo', 'precompile', ROOT / 'builds/deps/moxie', '-I', ROOT / 'builds/deps',
        '-o', ROOT / 'builds/moxie.mojoc')
    compile_file(ROOT / 'examples/estimator.mojo', ROOT / 'builds/example')
    compile_file(ROOT / 'examples/dense_embedding_contract.mojo', ROOT / 'builds/dense_example')
    for name in ('contracts', 'capabilities', 'family', 'plumbing', 'trustworthiness', 'classical_mds', 'deferred'):
        compile_file(ROOT / f'tests/test_{name}.mojo', ROOT / f'builds/test_{name}')
    compile_file(ROOT / 'examples/classical_mds.mojo', ROOT / 'builds/classical_example')
    for consumer in CONSUMERS:
        build_consumer(consumer)

def test():
    for name in ('contracts', 'capabilities', 'family', 'plumbing', 'trustworthiness', 'classical_mds', 'deferred'):
        run(ROOT / f'builds/test_{name}')
    for name, message in [('missing_predict', "required function 'predict' is not implemented")]:
        result = run('mojo', 'build', '-I', ROOT / 'builds/deps',
                     ROOT / f'tests/compile_fail/{name}.mojo', '-o', ROOT / 'builds/invalid', check=False)
        assert result.returncode and message in result.stderr, result.stderr
    for consumer in CONSUMERS:
        run(consumer / 'builds/probe', consumer=consumer)
        report = json.loads((consumer / 'builds/deps/ASSEMBLY.json').read_text())
        for module, owner in [('moxie._dense', 'numerics'), ('moxie._random', 'random'),
                              ('moxie.manifold._plumbing', 'manifold'),
                              ('moxie.probe_a', 'probe_a'), ('moxie.probe_b', 'probe_b')]:
            assert report['modules'][module] == [owner]
        assert not (consumer / 'builds/deps/moxie/_sparse.mojo').exists()
    a, b = CONSUMERS
    path = 'builds/deps/moxie/manifold/_plumbing.mojo'
    assert sha(a / path) == sha(b / path) == sha(ROOT / 'src/moxie/manifold/_plumbing.mojo')
    assert (a / path).stat().st_ino != (b / path).stat().st_ino
    # A stale v1 pin must not restore historical Moxie bytes.
    data = json.loads((a / 'DEPENDENCIES.lock.json').read_text())
    foundation = next(project for project in data['projects'] if project['id'] == 'moxie')
    foundation['revision'] = '0' * 40
    foundation['files'] = {'src/moxie/base.mojo': '0' * 64}
    live = a / 'live-lock.json'
    try:
        live.write_text(json.dumps(data))
        assemble(a, live)
        assert (a / 'builds/deps/moxie/base.mojo').read_bytes() == (
            CORE / 'src/moxie/base.mojo').read_bytes()
        # Modify a disposable dependency to prove consecutive builds reread it.
        probe = next(project for project in data['projects'] if project['id'] == 'probe_b')
        source = a / 'builds/live-probe'
        source.mkdir(exist_ok=True)
        probe['path'] = str(source)
        probe['modules']['moxie.probe_b']['file'] = 'probe.mojo'
        live.write_text(json.dumps(data))
        original = (b / 'src/moxie/probe_b.mojo').read_bytes()
        for suffix in (b'\n# First live build\n', b'\n# Second live build\n'):
            (source / 'probe.mojo').write_bytes(original + suffix)
            assemble(a, live)
            assert (a / 'builds/deps/moxie/probe_b.mojo').read_bytes() == original + suffix
    finally:
        live.unlink(missing_ok=True)
        assemble(a)
    # Missing inputs and conflicting ownership/exports must fail atomically.
    for mode in ('missing', 'module', 'export'):
        data = json.loads((a / 'DEPENDENCIES.lock.json').read_text())
        probe = next(p for p in data['projects'] if p['id'] == 'probe_b')
        if mode == 'missing':
            probe['modules']['moxie.probe_b']['file'] = 'src/moxie/missing.mojo'
        elif mode == 'module':
            probe['modules']['moxie.probe_a'] = {'file': 'src/moxie/probe_b.mojo'}
        else:
            probe['exports']['packages']['moxie']['BaseEstimator'] = 'moxie.probe_b'
        # Relative paths require the temporary lock to remain beside the original.
        bad = a / 'bad-lock.json'
        before = sha(a / 'builds/deps/ASSEMBLY.json')
        try:
            bad.write_text(json.dumps(data))
            result = assemble(a, bad, check=False)
            assert result.returncode, mode
            expected = {'missing': 'missing file', 'module': 'module collision', 'export': 'export collision'}[mode]
            assert expected in result.stderr.lower(), result.stderr
            assert sha(a / 'builds/deps/ASSEMBLY.json') == before
        finally:
            bad.unlink(missing_ok=True)
    print('PASS: live assembly, missing inputs, ownership/export conflicts, negative capabilities')

if __name__ == '__main__':
    action = sys.argv[1]
    if Path.cwd().resolve() in CONSUMERS:
        consumer = Path.cwd().resolve()
        build_consumer(consumer)
        if action == 'test':
            run(consumer / 'builds/probe', consumer=consumer)
    else:
        build()
        if action == 'test':
            test()
        elif action == 'start':
            run(ROOT / 'builds/classical_example')
        elif action != 'build':
            raise ValueError(action)
