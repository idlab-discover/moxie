"""Resolve Moxie HEAD assembly and compile/run contained family consumers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from dependencies import moxie_head, prepare
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from compiler_limits import compiler_command

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = [ROOT / 'tests/consumers' / n for n in ('a', 'b')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def environment(root):
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env

def run(*args, root=ROOT, check=True):
    return subprocess.run(compiler_command(args), cwd=root, env=environment(root), check=check,
                          capture_output=True, text=True)

def assemble(root, lock=None, check=True, revision=None):
    if root == ROOT and lock is None:
        core = ROOT.parents[1]
        return run(sys.executable, core / 'shared/assembly/tools/assemble_family.py',
                   '--consumer', ROOT, '--lock', core / 'DEPENDENCIES.json',
                   '--output', ROOT / 'builds/deps', root=root, check=check)
    lock = lock or root / 'DEPENDENCIES.lock.json'
    try:
        effective, assembly = prepare(root, lock, revision)
        return run(sys.executable, assembly / 'tools/consume_family.py', '--consumer', root,
                   '--lock', effective, '--tool', assembly / 'tools/assemble_family.py', root=root, check=check)
    except ValueError as error:
        if check:
            raise
        return subprocess.CompletedProcess([], 1, '', str(error))

def compile_file(root, source, name):
    run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', root/'builds/deps',
        '-I', ROOT/'examples', source, '-o', root/'builds'/name, root=root)

def build():
    print(run(sys.executable, ROOT/'tests/test_dependencies.py').stderr, end='')
    assert run('mojo', '--version').stdout.startswith('Mojo 1.0.0 ')
    revision = moxie_head()
    print(f'Moxie HEAD: {revision}', flush=True)
    for root in [ROOT, *CONSUMERS]:
        assemble(root, revision=revision)
        run('mojo', 'precompile', root/'builds/deps/moxie', '-I', root/'builds/deps',
            '-o', root/'builds/moxie.mojoc', root=root)
        if root == ROOT:
            for test in ('test_contracts', 'test_capabilities', 'test_family', 'test_plumbing', 'test_estimators', 'test_fluent'):
                compile_file(root, ROOT/'tests'/f'{test}.mojo', test)
            compile_file(root, ROOT/'examples/estimator.mojo', 'example')
            compile_file(root, ROOT/'examples/semi_supervised.mojo', 'semi_supervised_example')
        else:
            compile_file(root, root/'main.mojo', 'probe')
    print('PASS build: family, inherited contracts, two isolated namespaces')

def test():
    for name in ('test_contracts', 'test_capabilities', 'test_family', 'test_plumbing', 'test_estimators', 'test_fluent', 'example', 'semi_supervised_example'):
        print(run(ROOT/'builds'/name).stdout, end='')
    outputs = [run(c/'builds/probe', root=c).stdout for c in CONSUMERS]
    assert outputs[0] == outputs[1]
    for c in CONSUMERS:
        report = json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie._dense'] == ['numerics']
        assert report['modules']['moxie.semi_supervised._contracts'] == ['semi-supervised']
        assert sha(c/'builds/deps/moxie/semi_supervised/_contracts.mojo') == sha(ROOT/'src/moxie/semi_supervised/_contracts.mojo')
        assert not (c/'builds/deps/moxie/_sparse.mojo').exists()
        assert 'moxie.semi_supervised._probe_a' in report['modules']
        assert 'moxie.semi_supervised._probe_b' in report['modules']
    for name, diagnostic in [('missing_predict', "required function 'predict'"),
                             ('missing_affinity', "required function 'affinity'"),
                             ('missing_decision', 'decision_function')]:
        result = run('mojo', 'build', '-I', ROOT/'builds/deps',
                     ROOT/'tests/compile_fail'/f'{name}.mojo', '-o', ROOT/'builds/invalid', check=False)
        assert result.returncode != 0 and diagnostic in result.stderr, result.stderr
    assert (CONSUMERS[0]/'builds/deps/moxie/_dense.mojo').stat().st_ino != (CONSUMERS[1]/'builds/deps/moxie/_dense.mojo').stat().st_ino
    original = json.loads((CONSUMERS[0]/'DEPENDENCIES.lock.json').read_text())
    import copy
    for kind in ('module', 'export'):
        lock = copy.deepcopy(original)
        family = next(p for p in lock['projects'] if p['id'] == 'semi-supervised')
        other = copy.deepcopy(family)
        other['id'] = 'conflicting-owner'
        other['assets'] = {}
        if kind == 'export':
            other['modules'] = {'moxie.semi_supervised._collision': {'file': 'src/moxie/semi_supervised/_contracts.mojo'}}
            other['exports'] = {'schema': 1, 'packages': {'moxie': {}, 'moxie.semi_supervised': {'SemiSupervisedClassifier': 'moxie.semi_supervised._collision'}}}
        lock['projects'].append(other)
        c = CONSUMERS[0]
        # Paths in a lock are relative to that file; keep the bad lock at root.
        path = c/'bad-lock.json'
        before = sha(c/'builds/deps/ASSEMBLY.json')
        try:
            path.write_text(json.dumps(lock))
            result = assemble(c, path, check=False)
            assert result.returncode, f'accepted {kind}'
            expected = {'module': 'module collision', 'export': 'export collision'}[kind]
            assert expected in result.stderr, result.stderr
            assert sha(c/'builds/deps/ASSEMBLY.json') == before
        finally:
            path.unlink(missing_ok=True)
    print('PASS test: ownership, class encoding, negative capabilities, collisions')

if __name__ == '__main__':
    try:
        build()
        if sys.argv[1] == 'test':
            test()
        elif sys.argv[1] == 'start':
            print(run(ROOT/'builds/semi_supervised_example').stdout, end='')
    except subprocess.CalledProcessError as error:
        print(error.stdout, error.stderr, file=sys.stderr)
        raise
