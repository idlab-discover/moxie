"""Pinned reference checks for the behaviors asserted in test_api.mojo.

Column extraction is adapted to the native API's explicit string-name list;
the validation and request-selection bodies are the unedited pinned sources.
"""
import ast
import builtins
from types import SimpleNamespace
import sklearn.utils._metadata_requests as routing_support
from reference import *

validation = {'np': np, 'warnings': warnings,
              '_get_feature_names': lambda names: names}
load_functions(REFERENCE / 'sklearn/utils/validation.py',
               ['_check_feature_names'], validation)
check_names = validation['_check_feature_names']
estimator = SimpleNamespace()
names = np.array(['height', 'width'], dtype=object)
check_names(estimator, names, reset=True)
check_names(estimator, names.copy(), reset=False)
for wrong in (names[::-1], np.array(['height', 'renamed']),
              np.array(['height']), np.array(['height', 'width', 'extra'])):
    try:
        check_names(estimator, wrong, reset=False)
    except ValueError:
        pass
    else:
        raise AssertionError('reference accepted mismatched names')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    check_names(estimator, None, reset=False)
    check_names(estimator, None, reset=True)
    assert not hasattr(estimator, 'feature_names_in_')
    check_names(estimator, names, reset=False)

path = REFERENCE / 'sklearn/utils/_metadata_requests.py'
assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'status',
                                    '--porcelain', '--', str(path)], text=True).strip()
namespace = dict(vars(routing_support))
node = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == 'MethodMetadataRequest')
exec(builtins.compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
Request = namespace['MethodMetadataRequest']
metadata = {'sample_weight': 2., 'weights': 3.}
for setting, expected in [(True, {'sample_weight': 2.}), (False, {}),
                          ('weights', {'sample_weight': 3.})]:
    request = Request(owner='KernelRidge', method='fit')
    request.add_request(param='sample_weight', alias=setting)
    assert request._route_params(params=metadata, parent='Parent', caller='fit') == expected
request.add_request(param='sample_weight', alias=None)
try:
    request._route_params(params=metadata, parent='Parent', caller='fit')
except namespace['UnsetMetadataPassedError']:
    pass
else:
    raise AssertionError('reference accepted unset metadata')
print('Pinned feature-name and metadata request reference checks passed')
