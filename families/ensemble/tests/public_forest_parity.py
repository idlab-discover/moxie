"""Public forest results against the source-built scikit-learn reference."""
from pathlib import Path
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from forest_parity import make_data, bits

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert 'builds/oracle-env' in sklearn.__file__
    commands, expected = [], []
    configurations = []
    for cls in (RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor):
        classifier = cls in (RandomForestClassifier, ExtraTreesClassifier)
        configurations.extend((cls, classifier, variant) for variant in range(5))
    for index, (cls, classifier, variant) in enumerate(configurations):
        extra = cls in (ExtraTreesClassifier, ExtraTreesRegressor)
        spec = {'weights': variant in (2, 3), 'nan': variant == 4}
        if classifier:
            spec['classes'] = [2, 2] if variant == 3 else [2 if variant == 2 else 3]
        elif variant == 3:
            spec['outputs'] = 2
        X, y, w, Q = make_data(index, spec)
        criterion = ('entropy' if variant == 1 else 'gini') if classifier else ('absolute_error' if variant == 1 else 'squared_error')
        cw = ('dict' if variant == 2 else 'balanced_subsample' if variant == 3 else 'none') if classifier else 'none'
        bootstrap = not extra or variant >= 2
        oob = bootstrap
        grow = 11 if variant == 4 else 0
        params = dict(n_estimators=7, criterion=criterion, bootstrap=bootstrap, oob_score=oob, random_state=41+index)
        if classifier:
            params['class_weight'] = {-3: 2.5, 7: .5} if cw == 'dict' else None if cw == 'none' else cw
        model = cls(**params)
        outputs = 1 if y.ndim == 1 else y.shape[1]
        commands += [' '.join(map(str, [index, int(classifier), int(extra), len(X), X.shape[1], outputs, len(Q), 7, criterion, int(bootstrap), int(oob), 41+index, cw, grow])),
                     'X '+bits(X, np.float32), 'Y '+bits(y), 'Q '+bits(Q, np.float32), 'W '+('none' if w is None else bits(w))]
        expected.append(('CASE', str(index)))
        for phase in range(2 if grow else 1):
            if phase:
                model.set_params(n_estimators=grow, warm_start=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X, y, sample_weight=w)
            pred = model.predict(Q)
            expected.append(('SHAPE', f'{pred.ndim} {len(Q)} {outputs} {X.shape[1]}'))
            expected.extend([('PRED', pred), ('IMP', model.feature_importances_), ('SCORE', model.score(X, y)), ('APPLY', model.apply(Q))])
            expected.extend(('SAMPLES', a) for a in model.estimators_samples_)
            if classifier:
                with np.errstate(divide='ignore'):
                    logs = model.predict_log_proba(Q)
                probabilities = model.predict_proba(Q)
                domains = model.classes_
                if outputs == 1:
                    domains, probabilities, logs = [domains], [probabilities], [logs]
                for c, p, logp in zip(domains, probabilities, logs):
                    expected.extend([('CLASSES', c), ('PROBA', p), ('LOG', logp)])
            if oob:
                expected.append(('OOB', model.oob_score_))
                if classifier:
                    decision = model.oob_decision_function_
                    expected.extend(('OOBD', decision if outputs == 1 else decision[:, :, k]) for k in range(outputs))
                else:
                    expected.append(('OOBP', model.oob_prediction_))
    dest = ROOT/'docs/private/forest'
    dest.mkdir(parents=True, exist_ok=True)
    source = dest/'public_input.txt'
    source.write_text('\n'.join(commands)+'\n')
    output = subprocess.run([ROOT/'builds/public_forest_driver', source], check=True, capture_output=True, text=True)
    (dest/'public_native.txt').write_text(output.stdout)
    got = output.stdout.splitlines()
    assert len(got) == len(expected), (len(got), len(expected), output.stderr)
    case = None
    for line, (tag, want) in zip(got, expected):
        actual_tag, _, payload = line.partition(' ')
        assert tag == actual_tag, (tag, line)
        if tag == 'CASE': case = want
        if tag in ('CASE', 'SHAPE'):
            assert payload == want, (case, tag, payload, want)
            continue
        if tag in ('APPLY', 'SAMPLES'):
            actual = np.fromstring(payload, dtype=np.int64, sep=' ')
        elif tag in ('SCORE', 'OOB'):
            actual = np.array([float(payload)])
        else:
            actual = np.fromstring(payload, dtype=np.int64, sep=' ').view(np.float64)
        want = np.asarray(want).ravel()
        if tag in ('IMP', 'SCORE', 'OOB', 'LOG'):
            # Tree impurity ulps and libm log/score summation order differ.
            np.testing.assert_allclose(actual, want, rtol=1e-12, atol=1e-14, equal_nan=True, err_msg=f'{case} {tag}')
        else:
            np.testing.assert_array_equal(actual, want, err_msg=f'{case} {tag}')
    print(f'PASS public forests: {len(configurations)} configurations, 24 fits; labels, shapes, predictions, probabilities, leaves and bootstrap rows exact')


if __name__ == '__main__':
    main()
