"""Execute pinned sklearn function bodies and compare native dense outputs."""
from pathlib import Path
import argparse, ast, importlib.util, json, subprocess, warnings, sys
import numpy as np
from scipy import special,stats
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
ORACLE=ROOT.parent/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def oracle():
    assert subprocess.check_output(['git','-C',str(ORACLE),'rev-parse','HEAD'],text=True).strip()==REV
    source=ORACLE/'sklearn/feature_selection/_univariate_selection.py'
    assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',str(source)],text=True).strip()
    def check_X_y(x,y,**kw): return np.asarray(x,dtype=kw.get('dtype')), np.asarray(y)
    class Labels:
        def __init__(self,**kw): pass
        def fit_transform(self,y):
            from scipy.sparse import csr_matrix
            y=np.asarray(y)
            if y.ndim==2 and y.shape[1]>1:return csr_matrix(y)
            y=y.ravel(); classes=np.unique(y)
            if len(classes)<=2: return csr_matrix((y[:,None]==classes[-1]).astype(int) if len(classes)==2 else np.zeros((len(y),1),int))
            return csr_matrix((y[:,None]==classes).astype(int))
    def dot(a,b):return a@b
    namespace=dict(np=np, special=special,stats=stats,warnings=warnings,
        as_float_array=lambda a,**kw:np.array(a,dtype=np.float32 if np.asarray(a).dtype==np.float32 else np.float64,copy=True),
        safe_sqr=lambda a:a*a,check_X_y=check_X_y,
        check_array=lambda x,**kw:np.asarray(x),safe_mask=lambda x,m:m,
        safe_sparse_dot=dot,row_norms=lambda x,squared=False:np.einsum('ij,ij->i',x,x) if squared else np.sqrt(np.einsum('ij,ij->i',x,x)),
        LabelBinarizer=Labels,issparse=lambda x:False,check_is_fitted=lambda x:None)
    tree=ast.parse(source.read_text()); nodes=[]
    wanted={'f_oneway','f_classif','chi2','_chisquare','r_regression','f_regression','_clean_nans'}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in wanted:
            node.decorator_list=[];nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name in {'SelectKBest','SelectPercentile','SelectFpr','SelectFdr','SelectFwe'}:
            method=next(m for m in node.body if isinstance(m,ast.FunctionDef) and m.name=='_get_support_mask')
            method.name=node.name; nodes.append(method)
    exec(compile(ast.Module(nodes,type_ignores=[]),str(source),'exec'),namespace)
    assert namespace['f_regression'].__code__.co_filename==str(source)
    return namespace

def literal(v):
    if np.isnan(v):return 'nan[DType.float64]()'
    if np.isposinf(v):return 'inf[DType.float64]()'
    if np.isneginf(v):return '-inf[DType.float64]()'
    return repr(float(v))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    assembly_root=parser.parse_args().assembly_root.resolve()
    spec=importlib.util.spec_from_file_location('check',ROOT/'tools/feature_selection_checks.py'); checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    ns=oracle(); rng=np.random.default_rng(417); lines=['import numojo as nm','from std.math import nan, inf','from moxie.feature_selection import f_classif, chi2, r_regression, f_regression','from moxie.feature_selection._univariate_probability import f_survival, chi_survival','from moxie.feature_selection._univariate_policy import UnivariatePolicy','from moxie.feature_selection._contracts import ScoreResult','from moxie.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, BuiltinScorer, SelectionParameter','def main() raises:']; expected=[]; exact=[]; transformed=[]; cases=[]
    def array(name,a,dt):
        shape=', '.join(map(str,a.shape)); lines.append(f'    var {name} = nm.zeros[DType.{dt}](nm.Shape({shape}))')
        for j,v in enumerate(a.ravel()):lines.append(f'    {name}.unsafe_set({j}, SIMD[DType.{dt}, 1]({literal(v)}))')
    def strided_array(name, a, dt, layout):
        # Spare backing elements are NaNs: only the logical view is valid input.
        if a.ndim == 1:
            shape = (a.size, 1) if layout == 2 else a.shape
            strides = (-2, 1) if layout == 2 else (-2,)
            offset = 2 * a.size
        else:
            rows, cols = a.shape
            shape = a.shape
            strides, offset = [((4 * cols, 2), 3),
                               ((-4 * cols, -2), 4 * rows * cols),
                               ((1, rows), 3), ((cols, 1), 3)][layout]
        size = 8 * a.size + 16
        lines.append(f'    var {name}_backing = nm.full[DType.{dt}](nm.Shape({size}), nan[DType.{dt}]())')
        for index in np.ndindex(shape):
            position = offset + sum(i * stride for i, stride in zip(index, strides))
            value = literal(a.reshape(shape)[index])
            lines.append(f'    {name}_backing.unsafe_set({position}, SIMD[DType.{dt}, 1]({value}))')
        dims = ', '.join(map(str, shape))
        steps = ', '.join(map(str, strides))
        lines.append(f'    var {name} = {name}_backing.view_with_layout(nm.Shape({dims}), nm.NDArrayStrides({steps}), {offset})')
    def emit(expression,values):
        idx=len(expected);lines.append(f'    print({idx}, {expression})');expected.append(float(values))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for case in range(26):
            dtype=np.float32 if case%2 else np.float64;dt=np.dtype(dtype).name
            target_dtype = (np.float64 if dtype == np.float32 else np.float32) if case >= 18 else dtype
            target_dt = np.dtype(target_dtype).name
            x=rng.uniform(.1,4,size=(12,4)).astype(dtype); y=rng.integers(0,3,size=12).astype(target_dtype)
            if case >= 18 and target_dtype == np.float64:
                # These three labels collapse to two when downcast to Float32.
                y += 2**24
            if case in (2,3):x[:,0]=0
            if case in (4,5):x[:,0]=1
            if case in (6,7):y[:]=1
            if case in (8,9):y=np.array([0,0,0,0,1,1,1,1,2,2,2,2],dtype=dtype)
            case_start=len(lines)
            if case < 10:
                array(f'x{case}',x,dt);array(f'y{case}',y,target_dt)
            else:
                layout = ((case - 10) // 2) % 4
                strided_array(f'x{case}', x, dt, layout)
                strided_array(f'y{case}', y, target_dt, layout)
            for method in ('chi2','f_classif','r_regression','f_regression'):
                for center,finite in ([(True,True)] if method in ('chi2','f_classif') else [(a,b) for a in (True,False) for b in (True,False)]):
                    kw={} if method in ('chi2','f_classif') else dict(center=center,force_finite=finite)
                    truth=ns[method](x,y,**kw);name=f'result{len(expected)}'
                    options='' if not kw else f', center={center}, force_finite={finite}'
                    lines.append(f'    var {name} = {method}(x{case}, y{case}{options})')
                    if method=='r_regression':
                        for j,v in enumerate(truth):emit(f'{name}.unsafe_get({j})',v)
                    else:
                        for field,values in zip(('scores','pvalues.value()'),truth):
                            for j,v in enumerate(values):emit(f'{name}.{field}.unsafe_get({j})',v)
                    if (case % 2 or case >= 10) and center and finite:
                        score_values = truth if method == 'r_regression' else truth[0]
                        p_values = None if method == 'r_regression' else truth[1]
                        variants = [
                            ('SelectKBest', 'k=SelectionParameter(2)', 'SelectKBest', dict(k=2)),
                            ('SelectPercentile', 'percentile=50', 'SelectPercentile', dict(percentile=50)),
                            ('GenericUnivariateSelect', 'mode="k_best", param=SelectionParameter(2)', 'SelectKBest', dict(k=2)),
                        ]
                        if p_values is not None:
                            variants += [(cls, 'alpha=0.05', cls, dict(alpha=.05)) for cls in ('SelectFpr','SelectFdr','SelectFwe')]
                        for cls, args, policy, config in variants:
                            name=f'selector{len(expected)}'
                            lines.append(f'    var {name} = {cls}(BuiltinScorer[DType.{dt}, DType.{target_dt}]("{method}"), {args})')
                            lines.append(f'    {name}.fit(x{case}, y{case}.copy())')
                            lines.append(f'    var {name}_scores: nm.NDArray[DType.float64] = {name}.scores_()')
                            for j,v in enumerate(score_values):emit(f'{name}_scores.unsafe_get({j})',v)
                            if p_values is not None:
                                lines.append(f'    var {name}_p = {name}.pvalues_()')
                                for j,v in enumerate(p_values):emit(f'{name}_p.value().unsafe_get({j})',v)
                            mask=ns[policy](SimpleNamespace(scores_=score_values,pvalues_=p_values,**config))
                            lines.append(f'    var {name}_mask = {name}.get_support()[List[Bool]].copy()')
                            for j,v in enumerate(mask):
                                exact.append(len(expected));emit(f'Int({name}_mask[{j}])',v)
                            lines.append(f'    var {name}_output: nm.NDArray[DType.{dt}] = {name}.transform(x{case})')
                            exact.append(len(expected));emit(f'{name}_output.shape[1]',int(mask.sum()))
                            for j,v in enumerate(x[:,mask].ravel()):
                                (transformed if dt == 'float32' else exact).append(len(expected));emit(f'Float64({name}_output.unsafe_get({j}))',v)
            cases.append((f'case{case}', lines[case_start:]))
            del lines[case_start:]
        for df1 in (1,2,5,30):
            for df2 in (1,2,10,1000):
                for value in (0,1e-10,.1,1,10,1e6):emit(f'f_survival({literal(value)}, {literal(df1)}, {literal(df2)})',stats.f.sf(value,df1,df2))
        for df in (1,2,5,100):
            for value in (0,1e-10,.1,1,10,100,1000):emit(f'chi_survival({literal(value)}, {literal(df)})',special.chdtrc(df,value))
        for case,scores in enumerate(([1,1,1,1],[np.nan,2,2,1],[1,2,3,4],[-np.inf,np.nan,1,np.inf])):
            a=np.asarray(scores); p=np.array([.01,.04,.03,np.nan]);array(f's{case}',a,'float64');array(f'p{case}',p,'float64')
            lines.append(f'    var sr{case} = ScoreResult[DType.float64](s{case}^, p{case}^)')
            for mode,cls,attr,params in [('k_best','SelectKBest','k',[0,1,2,4,8]),('percentile','SelectPercentile','percentile',[0,10,25,34,50,67,100]),('fpr','SelectFpr','alpha',[0,.03,.06,1]),('fwe','SelectFwe','alpha',[0,.04,.12,1]),('fdr','SelectFdr','alpha',[0,.04,.06,1])]:
                for value in params:
                    truth=ns[cls](SimpleNamespace(scores_=a,pvalues_=p,**{attr:value}));name=f'pol{len(expected)}'
                    lines.append(f'    var {name} = UnivariatePolicy[DType.float64]("{mode}", {literal(value)}).select(sr{case})')
                    for j,v in enumerate(truth):emit(f'Int({name}[{j}])',v)
    # Compile one fixture at a time: a single optimized module containing all
    # specializations previously exhausted RAM and spilled tens of GiB to swap.
    main_index = lines.index('def main() raises:')
    imports = lines[:main_index]
    cases.append(('policies', lines[main_index + 1:]))
    folder = assembly_root / 'builds/univariate-parity-source'
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, body in cases:
        source = folder / (name + '.mojo')
        source.write_text('\n'.join(imports + ['def main() raises:'] + body) + '\n')
        binary = assembly_root / 'builds' / ('univariate-parity-' + name)
        checker.run('mojo', 'build', '-O3', '-I', assembly_root/'builds/deps', source, '-o', binary)
        outputs.append(subprocess.check_output([binary], text=True))
    output = ''.join(outputs)
    actual=np.array([float(line.split()[1]) for line in output.splitlines()]); truth=np.array(expected)
    assert len(actual)==len(truth)
    np.testing.assert_array_equal(actual[exact],truth[exact])
    np.testing.assert_array_equal(actual[transformed].astype(np.float32),truth[transformed].astype(np.float32))
    try:np.testing.assert_allclose(actual,truth,rtol=5e-5,atol=2e-12,equal_nan=True)
    except AssertionError:
        bad=~np.isclose(actual,truth,rtol=5e-5,atol=2e-12,equal_nan=True)
        print('mismatches',[(int(i),actual[i],truth[i]) for i in np.where(bad)[0][:30]])
        raise
    (assembly_root/'builds/univariate-parity-results.json').write_text(json.dumps(dict(comparisons=len(truth),oracle=REV,rtol=5e-5,atol=2e-12))+'\n')
    print('PASS',len(truth),'dense numerical/policy comparisons')
if __name__=='__main__':main()
