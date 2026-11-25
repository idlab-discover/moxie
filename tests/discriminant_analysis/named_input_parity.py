"""Differential checks for native named tables and the sklearn DataFrame path."""

import ast
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.utils import validation

from oracle import SOURCE, _function, _module


def check_named_input(da, runner):
    path = 'sklearn/utils/validation.py'
    routing_path = 'sklearn/utils/_metadata_requests.py'
    assert not subprocess.check_output(
        ['git', '-C', str(SOURCE), 'status', '--porcelain', '--', path, routing_path], text=True
    )
    namespace = vars(validation).copy()
    _, check_names_out, check_names = _function(
        path, ['_get_feature_names', '_check_feature_names_in', '_check_feature_names'], namespace
    )
    routing = _module('sklearn.utils._da_pinned_routing', routing_path)
    output = subprocess.check_output([str(runner), '--named-parity'], text=True)
    seen = set()
    with (
        patch.object(validation, '_check_feature_names', check_names),
        patch.object(sklearn.base, '_check_feature_names_in', check_names_out),
    ):
        for line in output.splitlines():
            if '|' not in line:
                continue
            kind, dtype, names, *values = line.split('|')
            assert (kind, dtype) not in seen, (kind, dtype)
            seen.add((kind, dtype))
            assert names.startswith('[') and names.endswith(']'), names
            _compare(da, routing, kind, dtype, names[1:-1].split(', '), values)
    assert seen == {
        (kind, dtype)
        for kind in ['lda', 'qda', 'projection', 'rank_reduced_projection']
        for dtype in ['float32', 'float64']
    }, seen
    print('Named-table, labeled projection and routed score parity PASS in Float32 and Float64')


def _compare(da, routing, kind, dtype, names, values):
    columns = ['mass', 'âge']
    X = pd.DataFrame(
        [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]],
        columns=columns, dtype=dtype,
    )
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    if kind == 'rank_reduced_projection':
        X = np.array(
            [[-4, 0], [-2, 0], [-3, -1], [-3, 1],
             [-1, 0], [1, 0], [0, -1], [0, 1],
             [2, 0], [4, 0], [3, -1], [3, 1]], dtype=dtype,
        )
        y = np.repeat([0, 1, 2], 4)
    cls = da.QuadraticDiscriminantAnalysis if kind == 'qda' else da.LinearDiscriminantAnalysis
    model = cls().fit(X, y)
    # Float32 sklearn rounds intermediate values that Moxie computes in Float64.
    tol = 5e-4 if dtype == 'float32' else 1e-10
    if kind in ('projection', 'rank_reduced_projection'):
        expected = model.transform(X)
        feature_names = model.get_feature_names_out().tolist()
        assert names == feature_names[:expected.shape[1]]
        if kind == 'rank_reduced_projection':
            assert len(feature_names) == 2 and expected.shape == (12, 1)
        else:
            assert feature_names == model.get_feature_names_out(columns).tolist()
            with np.testing.assert_raises_regex(ValueError, 'input_features is not equal'):
                model.get_feature_names_out(columns[::-1])
        expected = expected[:, 0]
        actual = np.array(ast.literal_eval(values[0]))
        # A discriminant direction is identifiable up to sign.
        actual *= np.sign(np.dot(actual, expected))
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    else:
        assert names == model.feature_names_in_.tolist()
        assert ast.literal_eval(values[0]) == model.predict(X).tolist()
        expected = model.predict_proba(X)
        actual = np.array(ast.literal_eval(values[1])).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
        with np.testing.assert_raises_regex(ValueError, 'feature names should match'):
            model.predict(X[columns[::-1]])
        request = routing.MethodMetadataRequest(owner=type(model).__name__, method='score')
        request.add_request(param='sample_weight', alias='validation_weight')
        score_labels = y.copy()
        score_labels[0] = 2
        weights = np.array([4, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dtype)
        routed = request._route_params(
            {'validation_weight': weights}, parent='TestParent', caller='score'
        )
        # The weighting sum is 12. A Float32 accuracy implementation may
        # round a fractional score in single precision.
        np.testing.assert_allclose(
            float(values[2]), model.score(X, score_labels, **routed),
            atol=1e-7 if dtype == 'float32' else 1e-15, rtol=0,
        )
        model.fit(X.to_numpy(), y)
        assert not hasattr(model, 'feature_names_in_')
