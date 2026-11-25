"""Compare the public dense histogram classifier with the pinned sklearn build."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from hgb_splitting_parity import bits64

BUILDS = Path(__file__).resolve().parents[1] / "builds"


@dataclass
class Case:
    X: np.ndarray
    query: np.ndarray
    y: np.ndarray
    classes: int
    weights: np.ndarray | None
    class_weight: int
    mode: int
    constraints: int
    sampled: bool
    seed: int

    def explicit_validation(self):
        y_val = (np.arange(len(self.query)) % self.classes) * 11 - 7
        return y_val, np.linspace(0.5, 2, len(self.query))

    def fixture(self):
        fields = [
            *self.X.shape, len(self.query), self.classes, int(self.weights is not None),
            self.class_weight, self.mode, self.constraints, int(self.sampled), self.seed,
            *bits64(self.X.ravel()), *bits64(self.query.ravel()), *bits64(self.y),
        ]
        if self.weights is not None:
            fields.extend(bits64(self.weights))
        if self.mode == 3:
            for values in self.explicit_validation():
                fields.extend(bits64(values))
        return " ".join(map(str, fields))

    def reference(self):
        validation_size = {1: None, 4: 12}.get(self.mode, 0.1)
        validation = {}
        if self.mode == 3:
            y_val, w_val = self.explicit_validation()
            validation = dict(X_val=self.query, y_val=y_val, sample_weight_val=w_val)
        mapping = {0: 2.0, 1: 0.5} if self.classes > 1 else {0: 2.0}
        model = HistGradientBoostingClassifier(
            learning_rate=0.125, max_iter=9, max_leaf_nodes=5, max_depth=3,
            min_samples_leaf=3, l2_regularization=0.25, max_bins=16,
            random_state=self.seed, max_features=0.65 if self.sampled else 1.0,
            early_stopping="auto" if self.mode == 5 else self.mode != 0,
            validation_fraction=validation_size, n_iter_no_change=3, tol=0.02,
            monotonic_cst=[1, 0, 0, 0] if self.constraints == 1 else None,
            interaction_cst="pairwise" if self.constraints == 2 else None,
            class_weight={0: None, 1: "balanced", 2: mapping}[self.class_weight],
        ).fit(self.X, self.y, sample_weight=self.weights, **validation)
        attrs = [
            model.n_iter_, model.n_features_in_, model.n_trees_per_iteration_,
            model.do_early_stopping_, model.is_categorical_ is not None, len(model.classes_),
            len(model.train_score_), len(model.validation_score_),
        ]
        accuracy = model.score(self.X, self.y, sample_weight=self.weights)
        return attrs, [
            (False, [accuracy]), (True, model.classes_), (False, model.train_score_),
            (False, model.validation_score_),
            *((True, p) for p in model.staged_predict(self.query)),
            *((False, p) for p in model.staged_predict_proba(self.query)),
            *((False, p) for p in model.staged_decision_function(self.query)),
            (True, model.predict(self.query)), (False, model.predict_proba(self.query)),
            (False, model.decision_function(self.query)),
        ]


def cases():
    for seed in (0, 41, 421):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(183, 4))
        X[::17, 1] = np.nan
        signal = X[:, 0] + np.nan_to_num(X[:, 1]) + 0.4 * X[:, 2] + rng.normal(size=len(X))
        query = rng.normal(size=(31, 4))
        query[::4, 1] = np.nan
        weights = rng.integers(1, 8, len(X)) / 4
        weights[::29] = 0
        for classes in (2, 3, 5):
            edges = np.linspace(-0.4, 1.2, classes - 1)
            y = np.searchsorted(edges, signal) * 11 - 7
            for mode in range(5):
                for class_weight, w in ((0, None), (1, weights), (2, weights)):
                    yield Case(X, query, y, classes, w, class_weight, mode, 0, True, seed)
            constraints = 1 if classes == 2 else 2
            yield Case(X, query, y, classes, weights, 1, 0, constraints, False, seed)

    X = np.arange(25, dtype=np.float64).reshape(-1, 1)
    yield Case(X, X, np.full(len(X), -7), 1, None, 0, 0, 0, False, 7)
    X = np.zeros((10_001, 1))
    y = np.arange(len(X)) % 2 * 11 - 7
    yield Case(X, X[:5], y, 2, None, 0, 5, 0, False, 8)


def main():
    assert Path(sklearn.__file__).resolve().is_relative_to(BUILDS / "oracle-env")
    all_cases = list(cases())
    fixtures = BUILDS / "hgb_classifier_fixtures.txt"
    fixtures.write_text("".join(f"{case.fixture()}\n" for case in all_cases))
    completed = subprocess.run(
        [BUILDS / "hgb_classifier_driver", fixtures], check=True, capture_output=True, text=True,
    )
    records = iter(completed.stdout.splitlines())
    exact_count = float_count = 0
    with threadpool_limits(limits=1):
        for index, case in enumerate(all_cases):
            attrs, expected = case.reference()
            context = f"case {index}, classes={case.classes}, mode={case.mode}, seed={case.seed}"
            assert_array_equal(list(map(int, next(records).split())), attrs, err_msg=context)
            for exact, values in expected:
                values = np.asarray(values, dtype=np.float64).ravel()
                actual = np.array([next(records) for _ in values], dtype=np.int64).view(np.float64)
                if exact:
                    assert_array_equal(actual, values, err_msg=context)
                    exact_count += len(values)
                else:
                    assert_allclose(actual, values, atol=1e-12, rtol=1e-12, err_msg=context)
                    float_count += len(values)
    assert next(records, None) is None
    print(f"PASS: {len(all_cases)} classifier fits, {exact_count} exact class values, "
          f"{float_count} decisions, probabilities and scores")


if __name__ == "__main__":
    main()
