"""Family gates assembling live Moxie HEAD with the current shared assembler."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from dependencies import verify_external_dependencies

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
ASSEMBLY = CORE / 'shared/assembly'
CONSUMERS = [ROOT/'tests/consumers/a', ROOT/'tests/consumers/b']
sys.path.insert(0, str(ROOT.parents[1]/'tools'))
from compiler_limits import compiler_command
FLAGS = ['-O3', '--fp-mode=contract=off']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(consumer):
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = consumer/folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def run(*args, consumer=ROOT, check=True):
    result = subprocess.run(compiler_command(args), cwd=consumer,
        env=environment(consumer), capture_output=True, text=True)
    if check and result.returncode:
        print(result.stdout, result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, args)
    return result


def assemble(consumer, lock=None, check=True):
    if consumer == ROOT and lock is None:
        return run(sys.executable, ASSEMBLY/'tools/assemble_family.py', '--consumer', ROOT, '--lock', CORE/'DEPENDENCIES.json', '--output', ROOT/'builds/deps', check=check)
    lock = lock or consumer/'DEPENDENCIES.json'
    verify_external_dependencies(lock)
    return run(sys.executable, ASSEMBLY/'tools/consume_family.py', '--consumer', consumer,
        '--lock', lock, '--tool', ASSEMBLY/'tools/assemble_family.py', consumer=consumer, check=check)


def compile_main(consumer, source, output):
    run('mojo', 'build', *FLAGS, '-I', consumer/'builds/deps', '-I', ROOT/'examples',
        source, '-o', output, consumer=consumer)


def build():
    assert run('mojo', '--version').stdout.startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(ROOT)
    run('mojo', 'precompile', ROOT/'builds/deps/moxie', '-I', ROOT/'builds/deps',
        '-o', ROOT/'builds/moxie.mojoc')
    for source in sorted((ROOT/'tests').glob('test_*.mojo')):
        compile_main(ROOT, source, ROOT/'builds'/source.stem)
    for source in sorted((ROOT/'examples').glob('*.mojo')):
        compile_main(ROOT, source, ROOT/'builds'/source.stem)
    for consumer in CONSUMERS:
        assemble(consumer)
        run('mojo', 'precompile', consumer/'builds/deps/moxie', '-I', consumer/'builds/deps',
            '-o', consumer/'builds/moxie.mojoc', consumer=consumer)
        compile_main(consumer, consumer/'main.mojo', consumer/'builds/probe')
    print('PASS: build')


def assembly_failures():
    consumer = CONSUMERS[0]
    original = json.loads((consumer/'DEPENDENCIES.json').read_text())
    before = sha(consumer/'builds/deps/ASSEMBLY.json')
    for failure in ['missing', 'module', 'export']:
        data = json.loads(json.dumps(original))
        family = next(p for p in data['projects'] if p['id'] == 'feature_extraction')
        if failure == 'missing':
            next(project for project in data['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file'] = 'src/moxie/missing.mojo'
        elif failure == 'module':
            duplicate = json.loads(json.dumps(family))
            duplicate['id'] = 'conflicting-owner'
            data['projects'].append(duplicate)
        else:
            a = next(p for p in data['projects'] if p['id'] == 'probe_a')
            b = next(p for p in data['projects'] if p['id'] == 'probe_b')
            for p, module in [(a, 'a'), (b, 'b')]:
                p['exports']['packages']['moxie.feature_extraction']['Collision'] = 'moxie.feature_extraction._probe_'+module
        # Put the temporary manifest beside the normal one so relative paths retain meaning.
        badlock = consumer/'negative-manifest.json'
        try:
            badlock.write_text(json.dumps(data))
            result = assemble(consumer, badlock, check=False)
            assert result.returncode, failure
            assert any(word in result.stderr.lower() for word in ['missing', 'conflict', 'duplicate', 'collision']), result.stderr
            assert sha(consumer/'builds/deps/ASSEMBLY.json') == before
        finally:
            badlock.unlink(missing_ok=True)


def negative_imports():
    imports = [('moxie._sparse', 'CSR')]
    for module, symbol in imports:
        path = ROOT/'builds'/f'negative_{symbol}.mojo'
        path.write_text(f'from {module} import {symbol}\n\ndef main():\n    pass\n')
        result = run('mojo', 'build', '-I', ROOT/'builds/deps', path,
                     '-o', ROOT/'builds/negative', check=False)
        assert result.returncode and (symbol in result.stderr or module.split('.')[-1] in result.stderr), result.stderr
    result = run('mojo', 'build', '-I', ROOT/'builds/deps',
        ROOT/'tests/compile_fail/missing_predict.mojo', '-o', ROOT/'builds/negative', check=False)
    assert result.returncode and "required function 'predict' is not implemented" in result.stderr, result.stderr


def test():
    run(sys.executable, ROOT/'tools/test_dependencies.py')
    for source in sorted((ROOT/'tests').glob('test_*.mojo')):
        print(run(ROOT/'builds'/source.stem).stdout, end='')
    for consumer in CONSUMERS:
        print(run(consumer/'builds/probe', consumer=consumer).stdout, end='')
        report = json.loads((consumer/'builds/deps/ASSEMBLY.json').read_text())
        for module, owner, source in [
            ('moxie._dense', 'numerics', CORE/'shared/numerics/src/moxie/_dense.mojo'),
            ('moxie.feature_extraction._mapping', 'feature_extraction', ROOT/'src/moxie/feature_extraction/_mapping.mojo'),
            ('moxie.feature_extraction._validation', 'feature_extraction', ROOT/'src/moxie/feature_extraction/_validation.mojo')]:
            assert report['modules'][module] == [owner]
            assembled = consumer/'builds/deps'/Path(*module.split('.')).with_suffix('.mojo')
            assert sha(assembled) == sha(source)
        assert report['modules']['moxie.feature_extraction._probe_a'] == ['probe_a']
        assert report['modules']['moxie.feature_extraction._probe_b'] == ['probe_b']
        assert not (consumer/'builds/deps/moxie/_sparse.mojo').exists()
    a, b = [c/'builds/deps/moxie/feature_extraction/_mapping.mojo' for c in CONSUMERS]
    assert a.stat().st_ino != b.stat().st_ino
    assembly_failures()
    negative_imports()
    print('PASS: test; isolated consumers, ownership/export conflicts, missing sources and unavailable capabilities')


def main():
    action = sys.argv[1]
    if len(sys.argv) == 4 and sys.argv[2] == '--consumer':
        consumer = ROOT/'tests/consumers'/sys.argv[3]
        if consumer not in CONSUMERS or action not in ('build', 'test'):
            raise ValueError(sys.argv[1:])
        assemble(consumer)
        run('mojo', 'precompile', consumer/'builds/deps/moxie', '-I', consumer/'builds/deps',
            '-o', consumer/'builds/moxie.mojoc', consumer=consumer)
        compile_main(consumer, consumer/'main.mojo', consumer/'builds/probe')
        if action == 'test':
            print(run(consumer/'builds/probe', consumer=consumer).stdout, end='')
        print('PASS:', action, 'isolated consumer', consumer.name)
        return
    build()
    if action == 'test':
        test()
    elif action == 'start':
        print(run(ROOT/'builds/plumbing').stdout, end='')
    elif action != 'build':
        raise ValueError(action)


if __name__ == '__main__':
    main()
