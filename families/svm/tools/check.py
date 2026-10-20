"""Build gates for the family; live Moxie HEAD and the current shared assembler."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY=ROOT.parents[1]/'shared/assembly'
CORE = ROOT.parents[1]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command
sys.path.insert(0, str(ASSEMBLY / 'tools'))
from family_validator import assemble as compose_sources
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def env(root):
    e=os.environ.copy()
    for key,directory in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=root/directory; p.mkdir(parents=True,exist_ok=True); e[key]=str(p)
    e['PYTHONDONTWRITEBYTECODE']='1'
    e['OPENBLAS_NUM_THREADS']='1'
    e['OMP_NUM_THREADS']='1'
    return e

def run(*args,root=ROOT,check=True,capture=False):
    return subprocess.run(compiler_command(args),cwd=root,env=env(root),check=check,capture_output=capture,text=True)
def assemble(root,lock=None,check=True):
    if root == ROOT and lock is None:
        data = json.loads((CORE / 'DEPENDENCIES.json').read_text())
        for project in data['projects']:
            project['path'] = str((CORE / project['path']).resolve())
        resolved = ROOT / 'builds/resolved.json'
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(data, indent=2) + '\n')
        compose_sources(ROOT, resolved, ROOT / 'builds/deps')
        check_integration()
        return subprocess.CompletedProcess([], 0, '', '')
    lock=lock or root/'DEPENDENCIES.json'
    p=next(p for p in json.loads(lock.read_text())['projects'] if p['id']=='assembly')
    tool=(lock.parent/p['path']).resolve()
    # Every declared source is read live; only the tool location is checked.
    assert tool==ASSEMBLY.resolve(),tool
    return run(sys.executable,tool/'tools/consume_family.py','--consumer',root,'--lock',lock,'--tool',tool/'tools/assemble_family.py',root=root,check=check,capture=True)
def check_integration():
    data = json.loads((CORE / 'DEPENDENCIES.json').read_text())
    report = json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())
    exported = {}
    for project in data['projects']:
        owner = (CORE / project['path']).resolve()
        assert 'moxie-svm' not in owner.parts, owner
        assert 'retired-moxie-svm' not in owner.parts, owner
        if not project['id'].startswith('svm-'):
            continue
        assert owner.is_relative_to(ROOT), owner
        for module, spec in project['modules'].items():
            assert report['modules'][module] == [project['id']], module
            target = ROOT / 'builds/deps' / Path(*module.split('.')).with_suffix('.mojo')
            assert target.read_bytes() == (owner / spec['file']).read_bytes(), module
        for destination, source in project['assets'].items():
            assert (ROOT / 'builds/deps' / destination).read_bytes() == (owner / source).read_bytes()
        exported.update(project['exports']['packages'].get('moxie.svm', {}))
    assert exported == json.loads((ROOT / 'EXPORTS.json').read_text())['packages']['moxie.svm']
    print('PASS core SVM ownership, source bytes, licenses and public exports', flush=True)

def compile_file(source,dest,root=ROOT):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',root/'builds/deps','-I',ROOT/'examples',source,'-o',dest,root=root)
def toolchain():
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'

def native_tests():
    return sorted((ROOT/'tests').glob('test_*.mojo')) + sorted((ROOT/'algorithms/c_svm/tests').glob('test_*.mojo'))

def build():
    toolchain()
    for root in [ROOT]+CONSUMERS:
        print(assemble(root).stdout,end='')
        run('mojo','precompile','-I',root/'builds/deps',root/'builds/deps/moxie','-o',root/'builds/moxie.mojoc',root=root)
        if root!=ROOT: compile_file(root/'main.mojo',root/'builds/probe',root)
    for src in native_tests() + sorted((ROOT/'tests').glob('*_driver.mojo')):
        compile_file(src,ROOT/'builds'/src.stem)
    compile_file(ROOT/'examples/estimator.mojo',ROOT/'builds/example')
    compile_file(ROOT/'examples/plumbing.mojo',ROOT/'builds/plumbing')
def test():
    for src in native_tests(): run(ROOT/'builds'/src.stem)
    for root in CONSUMERS:
        run(root/'builds/probe',root=root)
        report=json.loads((root/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie.svm._contracts']==['svm-scaffold']
        assert report['modules']['moxie._dense']==['numerics']
        assert report['modules']['moxie._random']==['random']
        for owner, modules in {
            'svm-libsvm': ('_libsvm', '_kernels', '_cache'),
            'svm-c-svm': ('_c_svm', '_classification'),
            'svm-nu-svm': ('_nu_svm',),
            'svm-one-class': ('_one_class',),
            'svm-linear': ('_linear', '_bounds'),
            'svm-liblinear': ('_liblinear',),
        }.items():
            for module in modules:
                assert report['modules']['moxie.svm.' + module] == [owner]

        for module in ('_consumer_a','_consumer_b'): assert (root/'builds/deps/moxie/svm'/f'{module}.mojo').exists()
    for name in ('_contracts','_base','_cache','_kernels','_libsvm','_c_svm','_nu_svm','_one_class','_classification','_linear','_liblinear'):
        a=CONSUMERS[0]/f'builds/deps/moxie/svm/{name}.mojo'
        b=CONSUMERS[1]/f'builds/deps/moxie/svm/{name}.mojo'
        assert sha(a)==sha(b)
        assert a.stat().st_ino != b.stat().st_ino
    p=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail/missing_predict.mojo','-o',ROOT/'builds/invalid',check=False,capture=True)
    assert p.returncode and "required function 'predict' is not implemented" in p.stderr,p.stderr
    # Exercise our actual family manifest: missing live source, module conflict, export conflict.
    root=CONSUMERS[0]
    original=json.loads((root/'DEPENDENCIES.json').read_text())
    before=sha(root/'builds/deps/ASSEMBLY.json')
    for case in ('missing','module','export'):
        data=json.loads(json.dumps(original))
        family=next(p for p in data['projects'] if p['id']=='svm-scaffold')
        if case=='missing': next(project for project in data['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
        elif case=='module':
            duplicate=json.loads(json.dumps(family)); duplicate['id']='duplicate'; data['projects'].append(duplicate)
        else:
            p=next(p for p in data['projects'] if p['id']=='consumer-b')
            p['exports']['packages']['moxie.svm']['ConsumerA']='moxie.svm._consumer_b'
        bad=root/'bad-lock.json'
        try:
            bad.write_text(json.dumps(data))
            proc=assemble(root,bad,check=False)
            assert proc.returncode,case
            if case=='missing': assert 'missing file' in proc.stderr,proc.stderr
            assert sha(root/'builds/deps/ASSEMBLY.json')==before
        finally: bad.unlink(missing_ok=True)
    print('PASS family assembly, isolated consumers, missing-source/conflict and capability rejection')
if __name__=='__main__':
    action=sys.argv[1]
    if action in ('build-consumer','test-consumer'):
        root=Path(sys.argv[2]).resolve()
        assert root in CONSUMERS
        toolchain()
        print(assemble(root).stdout,end='')
        run('mojo','precompile','-I',root/'builds/deps',root/'builds/deps/moxie','-o',root/'builds/moxie.mojoc',root=root)
        compile_file(root/'main.mojo',root/'builds/probe',root)
        if action=='test-consumer': run(root/'builds/probe',root=root)
        sys.exit(0)
    if action not in ('build','test','test-only','parity','parity-only','start'): raise ValueError(action)
    children = ('shared/liblinear', 'algorithms/linear', 'shared/libsvm')
    if action in ('build','test','parity','start'):
        build()
        for child in children:
            run(sys.executable, ROOT / child / 'tools/check.py', 'build')
    if action in ('test','test-only'):
        test()
        for child in children:
            run(sys.executable, ROOT / child / 'tools/check.py', 'test-only')
    if action in ('parity','parity-only'):
        for child in children:
            run(sys.executable, ROOT / child / 'tools/check.py', 'parity-only')
        run(sys.executable, ROOT / 'tests/public_kernel_parity.py')
    if action == 'start':
        run(ROOT / 'builds/plumbing')
        run(ROOT / 'algorithms/linear/builds/linear')
