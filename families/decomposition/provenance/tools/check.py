"""Project-local checks using the pinned foundation assembler read-only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parent / 'moxie/shared/assembly'
CONSUMERS = [ROOT / 'tests/consumers' / name for name in ('a', 'b')]
FLAGS = ['-O3', '--fp-mode=contract=off']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def environment(root):
    env = os.environ.copy()
    for key, name in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                      'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env

def run(*args, root=ROOT, check=True):
    return subprocess.run([str(a) for a in args], cwd=root, env=environment(root),
                          check=check, text=True, capture_output=True)

def assemble(root, lock=None, check=True):
    lock = lock or root / 'DEPENDENCIES.lock.json'
    data = json.loads(lock.read_text())
    owner = next(p for p in data['projects'] if p['id'] == 'assembly')
    assert sha(ASSEMBLY / 'tools/consume_family.py') == owner['files']['tools/consume_family.py']
    return run(sys.executable, ASSEMBLY / 'tools/consume_family.py', '--consumer', root,
               '--lock', lock, '--tool', ASSEMBLY / 'tools/assemble_family.py', root=root, check=check)

def compile_file(source, output, root=ROOT, check=True):
    return run('mojo', 'build', *FLAGS, '-I', root / 'builds/deps', '-I', ROOT / 'examples',
               source, '-o', output, root=root, check=check)

def build(root=ROOT):
    assert run('mojo', '--version').stdout.startswith('Mojo 1.0.0 ')
    metadata = list((Path(sys.prefix) / 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(root)
    run('mojo', 'precompile', root / 'builds/deps/moxie', '-I', root / 'builds/deps',
        '-o', root / 'builds/moxie.mojoc', root=root)
    if root != ROOT:
        compile_file(root / 'main.mojo', root / 'builds/probe', root)
        return
    for source in sorted((ROOT / 'tests').glob('test_*.mojo')):
        compile_file(source, ROOT / 'builds' / source.stem)
    for source in sorted((ROOT / 'examples').glob('*.mojo')):
        compile_file(source, ROOT / 'builds' / source.stem)
    for consumer in CONSUMERS:
        build(consumer)

def test():
    for source in sorted((ROOT / 'tests').glob('test_*.mojo')):
        print(run(ROOT / 'builds' / source.stem).stdout, end='')
    for consumer in CONSUMERS:
        print(run(consumer / 'builds/probe', root=consumer).stdout, end='')
        report = json.loads((consumer / 'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie.decomposition._contracts'] == ['decomposition']
        assert report['modules']['moxie._dense'] == ['numerics']
        assert report['modules']['moxie._random'] == ['random']
        assert 'moxie.probe_a' in report['modules'] and 'moxie.probe_b' in report['modules']
    for name in ('moxie/_dense.mojo', 'moxie/decomposition/_contracts.mojo'):
        a, b = [c / 'builds/deps' / name for c in CONSUMERS]
        assert sha(a) == sha(b) and a.stat().st_ino != b.stat().st_ino
    for file, expected in [('missing_predict.mojo', "required function 'predict'"),
                            ('unsupported_sparse.mojo', '_sparse'), ('unsupported_pca.mojo', 'PCA')]:
        result = compile_file(ROOT / 'tests/compile_fail' / file, ROOT / 'builds/invalid', check=False)
        assert result.returncode and expected in result.stderr, result.stderr
    # Test the actual family configuration's drift and conflicting module/export failures.
    consumer = CONSUMERS[0]
    original = json.loads((consumer / 'DEPENDENCIES.lock.json').read_text())
    before = sha(consumer / 'builds/deps/ASSEMBLY.json')
    for mode in ('drift', 'module', 'export'):
        data = json.loads(json.dumps(original))
        family = next(p for p in data['projects'] if p['id'] == 'decomposition')
        if mode == 'drift':
            next(project for project in data['projects'] if project['id'] == 'moxie')['files']['src/moxie/base.mojo'] = '0' * 64
        else:
            other = next(p for p in data['projects'] if p['id'] == 'probe_a')
            if mode == 'module':
                other['modules']['moxie.decomposition._contracts'] = {'file': 'src/moxie/probe_a.mojo'}
            else:
                other['exports']['packages']['moxie.decomposition'] = {'DecompositionTransformer': 'moxie.probe_a'}
        # Keep relative project paths anchored at the original consumer.
        bad = consumer / 'bad-lock.json'
        try:
            bad.write_text(json.dumps(data))
            result = assemble(consumer, bad, check=False)
            assert result.returncode, mode
            expected = {'drift': 'drift', 'module': 'module collision', 'export': 'export collision'}[mode]
            assert expected in result.stderr.lower(), result.stderr
            assert sha(consumer / 'builds/deps/ASSEMBLY.json') == before
        finally:
            bad.unlink(missing_ok=True)
    print('PASS: two isolated composed consumers, negative capabilities, drift and conflicts')

if __name__ == '__main__':
    try:
        action = sys.argv[1]
        selected = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT
        build(selected)
        if action == 'test':
            if selected == ROOT:
                test()
            else:
                print(run(selected / 'builds/probe', root=selected).stdout, end='')
        elif action == 'start':
            print(run(ROOT / 'builds/plumbing').stdout, end='')
        elif action != 'build':
            raise ValueError(action)
        print('PASS:', action)
    except subprocess.CalledProcessError as error:
        print(error.stdout, error.stderr, file=sys.stderr)
        raise
