"""Assemble live Moxie HEAD sources with the current shared assembler, then check."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[1]/'shared/assembly'
CONSUMERS = [ROOT/'tests/consumers'/name for name in ('a', 'b')]
CORE = ROOT.parents[1]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command

FLAGS = ['-O3', '--fp-mode=contract=off']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def environment(consumer):
    env = os.environ.copy()
    for key, folder in {'PIXI_CACHE_DIR':'.pixi/cache', 'XDG_CACHE_HOME':'.pixi/xdg',
                        'MODULAR_CACHE_DIR':'.pixi/modular', 'TMPDIR':'builds/tmp'}.items():
        path = consumer/folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    return env

def run(*args, consumer=ROOT, check=True, capture=False):
    return subprocess.run(compiler_command(args), cwd=consumer, env=environment(consumer),
                          check=check, capture_output=capture, text=True)

def assembly(consumer, manifest=None, check=True):
    # Current shared assembler; reads live declared sources on every build.
    return run(sys.executable, ASSEMBLY/'tools/assemble_family.py', '--consumer', consumer,
               '--lock', manifest or consumer/'DEPENDENCIES.json',
               '--output', consumer/'builds/deps', consumer=consumer, check=check,
               capture=not check)

def compile_file(consumer, source, target):
    run('mojo', 'build', *FLAGS, '-I', consumer/'builds/deps', '-I', ROOT/'examples',
        source, '-o', consumer/'builds'/target, consumer=consumer)

def build(consumer):
    assembly(consumer)
    run('mojo', 'precompile', '-I', consumer/'builds/deps', consumer/'builds/deps/moxie',
        '-o', consumer/'builds/moxie.mojoc', consumer=consumer)
    if consumer != ROOT:
        compile_file(consumer, consumer/'main.mojo', 'probe')
        return
    compile_file(ROOT, ROOT/'examples/estimator.mojo', 'example')
    compile_file(ROOT, ROOT/'examples/layers.mojo', 'layers')
    for name in ('contracts', 'capabilities', 'family', 'plumbing'):
        compile_file(ROOT, ROOT/f'tests/test_{name}.mojo', 'test_'+name)
    for child in CONSUMERS:
        build(child)

def negative_compile(source, message):
    result = run('mojo','build',*FLAGS,'-I',ROOT/'builds/deps',source,
                 '-o',ROOT/'builds/invalid',check=False,capture=True)
    assert result.returncode and message in result.stderr, result.stderr

def reject_manifest(consumer, lock, expected):
    # Relative dependency paths are relative to the manifest's directory.
    bad = consumer/'rejected-manifest.json'
    previous = sha(consumer/'builds/deps/ASSEMBLY.json')
    try:
        bad.write_text(json.dumps(lock))
        result = assembly(consumer, bad, check=False)
        assert result.returncode and expected in result.stderr.lower(), result.stderr
        assert sha(consumer/'builds/deps/ASSEMBLY.json') == previous
    finally:
        bad.unlink(missing_ok=True)

def test(consumer):
    if consumer != ROOT:
        run(consumer/'builds/probe', consumer=consumer)
        return
    run(ROOT/'builds/layers')
    for name in ('contracts','capabilities','family','plumbing'):
        run(ROOT/'builds'/('test_'+name))
    negative_compile(ROOT/'tests/compile_fail/missing_predict.mojo', "required function 'predict' is not implemented")
    negative_compile(ROOT/'tests/compile_fail/integer_matrix.mojo', 'comptime assert')
    # MLP declarations remain explicitly unimplemented; tested by the public probe.
    results = [run(c/'builds/probe', consumer=c, capture=True).stdout for c in CONSUMERS]
    assert results[0] == results[1]
    for relative in ('moxie/_dense.mojo','moxie/_random.mojo','moxie/neural_network/_plumbing.mojo'):
        paths = [c/'builds/deps'/relative for c in CONSUMERS]
        assert sha(paths[0]) == sha(paths[1])
        assert paths[0].stat().st_ino != paths[1].stat().st_ino
    for c in CONSUMERS:
        report = json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie._dense'] == ['numerics']
        assert report['modules']['moxie._random'] == ['random']
        assert report['modules']['moxie.neural_network._plumbing'] == ['neural_network.scaffold']
        assert report['modules']['moxie.probe_a'] == ['probe_a']
        assert report['modules']['moxie.probe_b'] == ['probe_b']
    c = CONSUMERS[0]
    original = json.loads((c/'DEPENDENCIES.json').read_text())
    lock = json.loads(json.dumps(original))
    # Live inputs: a declared source that does not exist must fail.
    next(project for project in lock['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file'] = 'src/moxie/missing.mojo'
    reject_manifest(c, lock, 'missing file')
    lock = json.loads(json.dumps(original))
    duplicate = json.loads(json.dumps(next(p for p in lock['projects'] if p['id']=='probe_a')))
    duplicate['id'] = 'collision'
    lock['projects'].append(duplicate)
    reject_manifest(c, lock, 'collision')
    lock = json.loads(json.dumps(original))
    for name in ('a','b'):
        entry = next(p for p in lock['projects'] if p['id']=='probe_'+name)
        entry['exports']['packages']['moxie']['Collision'] = 'moxie.probe_'+name
    reject_manifest(c, lock, 'collision')
    print(results[0].strip())
    print('PASS missing-source, module and export conflict rejection')

def main():
    action = sys.argv[1]
    consumer = Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT
    assert consumer in [ROOT, *CONSUMERS]
    assert run('mojo','--version',consumer=consumer,capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    build(consumer)
    if action == 'test':
        test(consumer)
    elif action == 'example':
        run(ROOT/'builds/example')
    elif action != 'build':
        raise ValueError(action)
    print('PASS', action, consumer.name)

if __name__ == '__main__':
    main()
