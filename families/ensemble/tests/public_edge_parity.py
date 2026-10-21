"""Public forest edge cases and invalid inputs against the source-built reference.

Value lines must match exactly (same trees, same NaN routing, same sampling).
Error lines must fail on both sides with the same error class; Moxie raises
Mojo `Error`s whose text starts with the Python class name (docs/interfaces.md),
and sklearn's InvalidParameterError is a ValueError. Messages are not compared.
"""
from pathlib import Path
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import (RandomForestClassifier as RFC, RandomForestRegressor as RFR,
                              ExtraTreesClassifier as ETC, ExtraTreesRegressor as ETR)

ROOT = Path(__file__).resolve().parents[1]
# Documented differences: learned fields exist (empty) before fit, and the
# regenerated estimators_samples_ raises NotFittedError where Python's missing
# attribute raises AttributeError (NotFittedError is also an AttributeError).
DOCUMENTED = {'unfitted_importances': 'size 0', 'unfitted_samples': 'ERR NotFittedError'}


def expected():
    i = np.arange(30)
    X = np.c_[((i * 7) % 11) / 3, ((i * 5) % 13) / 4, i % 4].astype(np.float32)
    y = ((i * 7) % 3).astype(np.float64)
    out = {}

    def run(tag, f):
        try:
            out[tag] = ('OK', f())
        except NotFittedError:
            out[tag] = ('ERR', 'NotFittedError')
        except ValueError:
            out[tag] = ('ERR', 'ValueError')
    run('criterion_foo', lambda: RFC(5, criterion='foo', random_state=0).fit(X, y) and None)
    run('max_features_foo', lambda: RFC(5, max_features='foo', random_state=0).fit(X, y) and None)
    run('cw_missing', lambda: RFC(5, class_weight={0: 2., 99: 1.}, random_state=0).fit(X, y) and None)
    yn = y.copy(); yn[3] = np.nan
    run('y_nan', lambda: RFR(5, random_state=0).fit(X, yn) and None)
    Xi = X.copy(); Xi.flat[4] = np.inf
    run('x_inf', lambda: RFR(5, random_state=0).fit(Xi, y) and None)
    Q = X.copy()
    for k in range(10):
        Q[k, k % 3] = np.nan
    out['nan_predict'] = ('OK', RFR(5, random_state=0).fit(X, y).predict(Q))
    m = RFC(4, max_samples=100, random_state=0).fit(X, y)
    out['max_samples_100'] = ('OK', [len(m.estimators_samples_[0])])
    out['max_samples_100_proba'] = ('OK', m.predict_proba(X))
    out['log_loss'] = ('OK', RFC(5, criterion='log_loss', random_state=0).fit(X, y).predict_proba(X))
    out['friedman'] = ('OK', ETR(5, criterion='friedman_mse', random_state=0).fit(X, y).predict(X))
    one = RFC(3, random_state=0).fit(np.zeros((1, 3), np.float32), [4.])
    out['single'] = ('OK', [*one.predict_proba(np.zeros((1, 3), np.float32)).ravel(), one.n_classes_])
    run('zero_weights', lambda: RFR(3, random_state=0).fit(X, y, sample_weight=np.zeros(30)).predict(X))
    w = np.ones(30); w[0] = -1
    run('neg_weight', lambda: RFR(3, random_state=0).fit(X, y, sample_weight=w).predict(X))
    run('unfitted_proba', lambda: ETC(3).predict_proba(X))
    run('max_features_10', lambda: RFC(3, max_features=10, random_state=0).fit(X, y) and None)
    Xc = np.zeros((30, 3), np.float32)
    m = RFR(3, random_state=0).fit(Xc, y)
    out['constant_imp'] = ('OK', m.feature_importances_)
    out['constant_pred'] = ('OK', m.predict(Xc[:2]))
    out['oob16'] = ('OK', [RFC(6, oob_score=True, max_samples=.5, random_state=1).fit(X, y).oob_score_])
    m = RFC(4, random_state=2).fit(X, -y * 2)
    out['classes17'] = ('OK', m.classes_)
    out['pred17'] = ('OK', m.predict(X))
    return out


def main():
    assert 'builds/oracle-env' in sklearn.__file__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want = expected()
    run = subprocess.run([ROOT / 'builds/public_edge_driver'], check=True, capture_output=True, text=True)
    got = {}
    for line in run.stdout.splitlines():
        tag, _, payload = line.partition(' ')
        got[tag] = payload
    compared = 0
    for tag, payload in got.items():
        if tag in DOCUMENTED:
            assert payload.startswith(DOCUMENTED[tag]), (tag, payload)
            continue
        kind, value = want[tag]
        if kind == 'ERR':
            assert payload.startswith('ERR ' + value + ':'), (tag, payload, value)
        elif value is None:
            assert payload == 'OK', (tag, payload)
        else:
            actual = np.array(payload.split(), dtype=np.float64)
            np.testing.assert_array_equal(actual, np.asarray(value, dtype=np.float64).ravel(), err_msg=tag)
        compared += 1
    assert compared == len(want), (compared, sorted(set(want) - set(got)))
    print(f'PASS public forest edges: {compared} cases exact or same error class, {len(DOCUMENTED)} documented differences')


if __name__ == '__main__':
    main()
