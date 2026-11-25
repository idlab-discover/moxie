"""Check frozen fit and score sample-weight routing against pinned sklearn."""

import argparse
import subprocess
from pathlib import Path

from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.utils.metadata_routing import process_routing

from compare_reference import REVISION, ROOT, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn-root", type=Path, default=ROOT.parent / "scikit-learn")
    args = parser.parse_args()
    source = args.sklearn_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected sklearn {REVISION}, got {revision}")
    files = ("sklearn/frozen/_frozen.py", "sklearn/isotonic.py",
             "sklearn/base.py", "sklearn/utils/_metadata_requests.py",
             "sklearn/pipeline.py")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD",
                    "--", *files], check=True)
    Frozen = load("sklearn._moxie_metadata_frozen", source / files[0]).FrozenEstimator
    Isotonic = load("sklearn._moxie_metadata_isotonic", source / files[1]).IsotonicRegression

    with config_context(enable_metadata_routing=True):
        model = Isotonic()
        assert not Frozen(model).get_metadata_routing().fit.requests
        model.set_fit_request(sample_weight="train_weights")
        model.set_score_request(sample_weight="score_weights")
        frozen = Frozen(model)
        X = [0.0, 1.0, 2.0]
        changed = [0.0, 9.0, 2.0]
        weights = [1.0, 0.0, 1.0]
        try:
            frozen.fit(X, X, sample_weight=weights)
        except NotFittedError:
            pass
        else:
            raise AssertionError("Unfitted frozen estimator accepted weighted fit")
        model.fit(X, X)
        nested = Frozen(clone(frozen))
        snapshot = frozen.get_metadata_routing()
        snapshot.score.add_request(param="sample_weight", alias=False)
        pipeline = make_pipeline(nested)
        pipeline.fit(X, changed, train_weights=[1.0, 1.0, 1.0])
        assert nested.score(X, changed) < 1
        # Pipeline.score also passes its sample_weight keyword to the router.
        # Resolve the alias directly, mirroring the native typed caller.
        routed = process_routing(pipeline, "score", score_weights=weights)
        request = nested.get_metadata_routing()
        expected = {
            "fit_alias": request.fit.requests["sample_weight"],
            "score_alias": request.score.requests["sample_weight"],
            "prediction": str(nested.predict(X)[1]),
            "weighted_score": str(nested.score(X, changed, **routed.frozenestimator.score)),
        }

    output = subprocess.check_output(
        [str(ROOT / "builds/tests/frozen/test_frozen_metadata")], cwd=ROOT, text=True
    )
    observations = [line.split(maxsplit=1) for line in output.splitlines()]
    actual = dict(observations)
    assert len(actual) == len(observations)
    for key in ("prediction", "weighted_score"):
        actual[key] = str(float(actual[key]))
    assert actual == expected, (actual, expected)
    print(f"Passed {len(expected)} exact frozen routing comparisons at sklearn {revision}")


if __name__ == "__main__":
    main()
