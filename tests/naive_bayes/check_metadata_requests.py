"""Compare the native Naive Bayes sample-weight requests with pinned sklearn."""

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn import base, config_context, naive_bayes
from sklearn.utils import _metadata_requests as routing

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
assert subprocess.check_output(
    ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True,
).strip() == REVISION
for module, path in (
    (base, "sklearn/base.py"),
    (naive_bayes, "sklearn/naive_bayes.py"),
    (routing, "sklearn/utils/_metadata_requests.py"),
):
    pinned = subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{REVISION}:{path}"])
    assert Path(inspect.getfile(module)).read_bytes() == pinned, path

binary = Path(sys.argv[1]).resolve()
actual = {}
for line in subprocess.check_output([binary], text=True).splitlines():
    name, *values = line.split("|")
    actual[name] = [float(value) for value in values]

X = np.array([[1., 0.], [2., 0.], [0., 2.], [0., 1.]])
y = np.array([0, 0, 1, 1])
scoring_y = np.array([0, 0, 1, 0])
training = np.array([1., 2., 3., 4.])
weight = np.array([2., 4., 1., 3.])
mapping = routing.MethodMapping()
for method in ("fit", "partial_fit", "score"):
    mapping.add(caller=method, callee=method)

with config_context(enable_metadata_routing=True):
    for name in ("GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB"):
        model = getattr(naive_bayes, name)()
        initial = model.get_metadata_routing()
        for method in ("fit", "partial_fit", "score"):
            assert getattr(initial, method).requests["sample_weight"] is None

        model.set_fit_request(sample_weight="training")
        model.set_partial_fit_request(sample_weight="sample_weight")
        model.set_score_request(sample_weight=False)
        model.set_fit_request()
        cloned = base.clone(model)
        assert repr(cloned.get_metadata_routing()) == repr(model.get_metadata_routing())
        parent = routing.MetadataRouter(owner="test").add(method_mapping=mapping, child=cloned)
        fit = parent.route_params(caller="fit", params={"training": training}).child.fit
        update = parent.route_params(
            caller="partial_fit", params={"sample_weight": weight},
        ).child.partial_fit
        score = parent.route_params(caller="score", params={"sample_weight": weight}).child.score
        assert set(fit) == set(update) == {"sample_weight"}
        assert not score
        cloned.fit(X, y, **fit)
        fitted_count = cloned.class_count_[0]
        fitted_score = cloned.score(X, scoring_y, **score)
        direct_score = cloned.score(X, scoring_y, sample_weight=weight)
        cloned.partial_fit(X, y, **update)
        assert cloned.class_count_[0] == 9.0
        np.testing.assert_allclose(
            actual.pop(name),
            [fitted_count, cloned.class_count_[0], fitted_score, direct_score],
            rtol=0,
            atol=1e-15,
            err_msg=name,
        )
        print(f"PASS {name} metadata routing, clone, and scores")
assert not actual, actual.keys()
