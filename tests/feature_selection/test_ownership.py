"""Verify public exports and live source ownership in all three assemblies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = set('SelectorMixin VarianceThreshold SelectKBest SelectPercentile SelectFpr SelectFdr SelectFwe GenericUnivariateSelect SelectFromModel RFE RFECV SequentialFeatureSelector chi2 f_classif f_oneway f_regression r_regression mutual_info_classif mutual_info_regression'.split())
for consumer in [ROOT, *(ROOT / 'tests/feature_selection/consumers' / name for name in ('a', 'b'))]:
    manifest = json.loads((consumer / 'DEPENDENCIES.json').read_text())
    owner = next(project for project in manifest['projects'] if project['id'] == 'moxie')
    assert (consumer / owner['path']).resolve() == ROOT
    adapters = {'BayesianRidgeImportance', 'LinearRegressionImportance', 'LinearSVCImportance', 'LogisticRegressionImportance', 'RidgeClassifierImportance', 'SVCImportance'}
    assert PUBLIC | adapters <= owner['exports']['packages']['moxie.feature_selection'].keys()
    report = json.loads((consumer / 'builds/deps/ASSEMBLY.json').read_text())
    assert report['modules']['moxie.svm._c_svm'] == ['svm-c-svm']
    assert report['modules']['moxie.svm._libsvm'] == ['svm-libsvm']
    for project in manifest['projects']:
        assert 'moxie-feature_selection' not in project['path']
        assert 'retired-moxie-tree' not in project['path']
    for name, module in owner['modules'].items():
        if name.startswith('moxie.feature_selection.'):
            assert report['modules'][name] == ['moxie'], name
            relative = Path(*name.split('.')).with_suffix('.mojo')
            assert (consumer / 'builds/deps' / relative).read_bytes() == (ROOT / module['file']).read_bytes(), name
print('The 19 sklearn exports and estimator adapters resolve to live Moxie sources in the root and both consumers')
