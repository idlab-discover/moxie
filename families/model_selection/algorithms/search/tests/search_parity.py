"""End-to-end search comparison using original pinned search/validation bodies."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[4]
ORACLE = WORKSPACE / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
if '--oracle' not in sys.argv:
    python = os.environ.get('MOXIE_REFERENCE_PYTHON', str(WORKSPACE / 'moxie/.pixi/envs/default/bin/python'))
    subprocess.run([python, str(Path(__file__).resolve()), '--oracle'], check=True)
    raise SystemExit()

import numpy as np
import sklearn
import sklearn.model_selection._validation as validation
import sklearn.utils._metadata_requests as metadata
import warnings

assert sklearn.__version__ == '1.9.1'
assert subprocess.check_output(['git', '-C', str(ORACLE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
files = ['sklearn/model_selection/_search.py', 'sklearn/model_selection/_validation.py', 'sklearn/utils/_metadata_requests.py', 'sklearn/dummy.py']
assert not subprocess.check_output(['git', '-C', str(ORACLE), 'status', '--porcelain', '--', *files], text=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ORACLE / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

reference = module('sklearn.model_selection._moxie_search_reference', files[0])
dummy = module('sklearn._moxie_dummy_reference', files[3])
# Keep original CV algorithm bodies; 1.9.1 supplies validation, callbacks and tags.
ns = vars(validation).copy()
ns.update(array_device=lambda x: None, move_to=lambda x, **kwargs: np.asarray(x))
body = next(n for n in ast.parse((ORACLE / files[2]).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_manual_routing')
meta = vars(metadata).copy()
exec(compile(ast.Module(body=[body], type_ignores=[]), str(ORACLE / files[2]), 'exec'), meta)
ns['_manual_routing'] = meta['_manual_routing']
functions = [n for n in ast.parse((ORACLE / files[1]).read_text()).body if isinstance(n, ast.FunctionDef)]
for function in functions:
    function.decorator_list = []
exec(compile(ast.Module(body=functions, type_ignores=[]), str(ORACLE / files[1]), 'exec'), ns)
for name in ('_fit_and_score', '_aggregate_score_dicts', '_insert_error_scores', '_normalize_score_results', '_warn_or_raise_about_fit_failures'):
    setattr(reference, name, ns[name])

class Discrete:
    def __init__(self, values):
        self.values = values
    def rvs(self, random_state):
        return self.values[random_state.randint(len(self.values))]

def score_mse(estimator, X, y):
    return -np.mean((estimator.predict(X) - y) ** 2)

def score_mae(estimator, X, y):
    return -np.mean(abs(estimator.predict(X) - y))

def score_nan(estimator, X, y):
    return np.nan

def score_fail(estimator, X, y):
    raise ValueError('intentional scorer failure')

X = np.column_stack([np.arange(10.), 10 - np.arange(10.)])
y = np.arange(10.)
cases = []
for values in ([0., 4., 5., 9.], [4.], [9., 0., 4., 4.], [1e4, -1e4, 4.5]):
    for train in (False, True):
        cases.append(dict(values=values, train=train))
for seed in (0, 42, 113):
    for distribution in (False, True):
        cases.append(dict(values=[0., 2., 4., 5., 9.], random=True, seed=seed, distribution=distribution, n_iter=7 if distribution else 3, train=True))
cases += [dict(values=[0., 4., 5.], multi=True), dict(values=[0., 4., 5.], multi=True, refit=False), dict(values=[0., 4.], refit=False), dict(values=[0., 4., 9.], custom=True), dict(values=[0., 4.], nan=True), dict(values=[4.], failed_candidate=True), dict(values=[4.], failed_candidate=True, numeric_error=True), dict(values=[0., 4.], multi=True, scorer_failure=True)]

code = '''from search_support import Adapter, Scorer, data, grid
from search import LastCandidate
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterDomain, DiscreteDistribution, SearchRefit, SearchResults, ErrorPolicy
from numojo import Item, Shape, NDArray
from moxie.model_selection._search import _ranks
from std.memory import bitcast
comptime DummyRegressor = _DummyRegressor[]

def emit(r: SearchResults[Float64], case_id: Int) raises:
    for c in range(len(r.params)):
        for key in r.params[c].keys():
            print(case_id, "param", c, key, r.params[c][key])
        for m in range(r.scores.test_scores.shape[2]):
            print(case_id, "mean", c, m, r.mean_test_score[Item(c, m)])
            print(case_id, "std", c, m, r.std_test_score[Item(c, m)])
            print(case_id, "rank", c, m, r.rank_test_score[Item(c, m)])
            for f in range(r.scores.test_scores.shape[1]):
                print(case_id, "test", c, m, f, r.scores.test_scores[Item(c, f, m)])
                if r.scores.train_scores:
                    print(case_id, "train", c, m, f, r.scores.train_scores.value()[Item(c, f, m)])
            if r.mean_train_score:
                print(case_id, "train_mean", c, m, r.mean_train_score.value()[Item(c, m)])
                print(case_id, "train_std", c, m, r.std_train_score.value()[Item(c, m)])
        print(case_id, "time", c, r.mean_fit_time[Item(c)], r.std_fit_time[Item(c)], r.mean_score_time[Item(c)], r.std_score_time[Item(c)])

def main() raises:
    var arrays = data()
    var model = DummyRegressor()
'''
expected = {}
for i, case in enumerate(cases):
    values = case['values']
    multi = case.get('multi', False)
    scores = {'mse': score_mse, 'mae': score_fail if case.get('scorer_failure') else score_mae} if multi else score_nan if case.get('nan') else score_mse
    refit = (lambda results: len(results['params']) - 1) if case.get('custom') else case.get('refit', 'mse' if multi else True)
    settings = dict(scoring=scores, refit=refit, return_train_score=case.get('train', False), error_score=-123. if case.get('numeric_error') else np.nan)
    # The native adapter translates presence of constant into strategy='constant'.
    space = {'constant': values, 'strategy': ['constant']}
    native_grid = f'ParameterGrid[Float64](grid({values}))'
    if case.get('failed_candidate'):
        space = [{'strategy': ['invalid']}, space]
        code += f'    var bad{i} = Dict[String, List[Float64]]()\n    bad{i}["invalid"] = [1.0]\n    var spaces{i} = List[Dict[String, List[Float64]]]()\n    spaces{i}.append(bad{i}^)\n    spaces{i}.append(grid({values}))\n'
        native_grid = f'ParameterGrid[Float64](spaces{i})'
    args = []
    if multi:
        args += [f'scoring=List[Scorer]([Scorer(0), Scorer({2 if case.get("scorer_failure") else 1})])', 'metric_names=List[String](["mse", "mae"])']
        args += ['refit=SearchRefit[Float64](False)' if refit is False else 'refit=SearchRefit[Float64]("mse")']
    else:
        args += [f'scoring=List[Scorer]([Scorer({3 if case.get("nan") else 0})])']
        if refit is False:
            args += ['refit=SearchRefit[Float64](False)']
    if case.get('train'):
        args += ['return_train_score=True']
    if case.get('numeric_error'):
        args += ['error_score=ErrorPolicy(False, -123)']
    kind = 'GridSearchCV[Adapter, LastCandidate]' if case.get('custom') else 'GridSearchCV[Adapter]'
    if case.get('random'):
        kind = 'RandomizedSearchCV[Adapter]'
        args += [f'n_iter={case["n_iter"]}', f'random_state={case["seed"]}']
        if case['distribution']:
            space['constant'] = Discrete(values)
            code += f'    var axes{i} = List[ParameterDomain[DiscreteDistribution[Float64]]]()\n    axes{i}.append(ParameterDomain[DiscreteDistribution[Float64]]("constant", DiscreteDistribution[Float64]({values})))\n    var domains{i} = List[List[ParameterDomain[DiscreteDistribution[Float64]]]]()\n    domains{i}.append(axes{i}^)\n'
            native_grid = f'domains{i}'
        estimator = reference.RandomizedSearchCV(dummy.DummyRegressor(), space, n_iter=case['n_iter'], random_state=case['seed'], **settings)
    else:
        estimator = reference.GridSearchCV(dummy.DummyRegressor(), space, **settings)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        estimator.fit(X, y)
    expected[i] = estimator
    code += f'    var s{i} = {kind}(model, {native_grid}, Adapter(), '+', '.join(args)+f')\n    s{i}.fit(arrays[0], arrays[1], None, None)\n    emit(s{i}.cv_results_(), {i})\n'
    if refit or not multi:
        code += f'    print({i}, "best", s{i}.best_index_())\n'
    if refit:
        code += f'    print({i}, "prediction", s{i}.predict(arrays[0])[Item(0)])\n'
# Search-specific ranks use the original formatter, including unusual NaN/inf ties.
rank_cases = [[float('nan'), float('nan')], [float('nan'), -float('inf'), 0., float('inf')], [float('nan'), float('inf')], [float('nan'), float(2**54), float(2**54)], [4., 1., 4., 0.]]
rank_expected = {}
for k, values in enumerate(rank_cases):
    formatter = reference.GridSearchCV(dummy.DummyRegressor(), {})
    rows = [dict(test_scores=v, fit_time=0., score_time=0.) for v in values]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rank_expected[k] = formatter._format_results([{} for v in values], 1, rows)['rank_test_score']
    code += f'    var rank_input{k} = NDArray[DType.float64](Shape({len(values)}, 1))\n'
    for j, v in enumerate(values):
        value = 'bitcast[DType.float64](UInt64(0x7ff8000000000000))' if np.isnan(v) else ('-' if v < 0 else '') + 'bitcast[DType.float64](UInt64(0x7ff0000000000000))' if np.isinf(v) else repr(v)
        code += f'    rank_input{k}[Item({j}, 0)] = {value}\n'
    code += f'    var ranks{k} = _ranks(rank_input{k})\n'
    for j in range(len(values)):
        code += f'    print({k}, "edge_rank", {j}, ranks{k}[Item({j}, 0)])\n'

source = ROOT / 'builds/search_parity.mojo'
source.write_text(code)
binary = ROOT / 'builds/search_parity'
subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', str(ROOT / 'builds/deps'), '-I', str(ROOT / 'tests'), str(source), '-o', str(binary)], check=True)
counts = {}
actual_params = {}
for line in subprocess.check_output([str(binary)], text=True).splitlines():
    words = line.split()
    i, field = int(words[0]), words[1]
    counts[field] = counts.get(field, 0) + 1
    if field == 'edge_rank':
        assert int(words[3]) == rank_expected[i][int(words[2])], (i, words, rank_expected[i])
        continue
    model = expected[i]
    result = model.cv_results_
    if field == 'param':
        actual_params.setdefault(i, {}).setdefault(int(words[2]), {})[words[3]] = float(words[4])
    elif field == 'time':
        assert np.isfinite(np.asarray(words[3:], dtype=float)).all()
        assert (np.asarray(words[3:], dtype=float) >= 0).all()
    elif field == 'best':
        assert int(words[2]) == model.best_index_, (i, line, model.best_index_)
    elif field == 'prediction':
        np.testing.assert_allclose(float(words[2]), model.predict(X)[0], rtol=1e-12, atol=1e-12)
    else:
        candidate, metric = map(int, words[2:4])
        name = ['mse', 'mae'][metric] if cases[i].get('multi') else 'score'
        key = {'mean': 'mean_test_', 'std': 'std_test_', 'rank': 'rank_test_', 'train_mean': 'mean_train_', 'train_std': 'std_train_'}.get(field)
        if key is None:
            key = f'split{int(words[4])}_{field}_'
        wanted = result[key + name][candidate]
        np.testing.assert_allclose(float(words[-1]), wanted, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f'case {i} {line}')
for i, estimator in expected.items():
    params = [{k: v for k, v in p.items() if k != 'strategy'} if p['strategy'] != 'invalid' else {'invalid': 1.} for p in estimator.cv_results_['params']]
    assert [actual_params[i][j] for j in range(len(params))] == params, (i, actual_params[i], params)
classifier = reference.GridSearchCV(dummy.DummyClassifier(random_state=42), {'strategy': ['most_frequent', 'prior', 'uniform', 'stratified']}, return_train_score=True)
classifier.fit(np.zeros((15, 2)), np.array([10] * 10 + [20] * 5))
for line in subprocess.check_output([str(ROOT / 'builds/classification')], text=True).splitlines():
    words = line.split()
    if not words[0].startswith('class_'):
        continue
    field, candidate = words[0], int(words[1])
    key = f'split{words[2]}_test_score' if field == 'class_test' else 'mean_test_score' if field == 'class_mean' else 'rank_test_score'
    np.testing.assert_allclose(float(words[-1]), classifier.cv_results_[key][candidate], rtol=1e-12, atol=1e-12)
    counts[field] = counts.get(field, 0) + 1
weighted = reference.GridSearchCV(dummy.DummyRegressor(), [{}]).fit(X, y, sample_weight=np.arange(1., 11.))
for line in subprocess.check_output([str(ROOT / 'builds/search_edges')], text=True).splitlines():
    words = line.split()
    if words[0] == 'weighted_score':
        np.testing.assert_allclose(float(words[1]), weighted.best_score_, rtol=1e-12, atol=1e-12)
        counts['weighted_score'] = 1

report = dict(reference=REVISION, support_sklearn=sklearn.__version__, files={f: hashlib.sha256((ORACLE / f).read_bytes()).hexdigest() for f in files}, cases=len(cases), compared=counts, rtol=1e-12, atol=1e-12)
destination = ROOT / 'docs/private/search/parity.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS original search/CV bodies: {len(cases)} cases, {sum(counts.values())} checked records, tolerance 1e-12')
