"""Compare complete public MLP incremental sequences with pinned scikit-learn."""

import argparse
from pathlib import Path
import subprocess
import warnings

import numpy as np

from mlp_parity import FAMILY, compare, indicators, load_reference


def snapshots(reference):
    for dtype in (np.float32, np.float64):
        for solver in ("sgd", "adam"):
            for explicit in (False, True):
                for kind in ("multilabel", 2, 3, "squared_error", "poisson"):
                    classifier = kind == "multilabel" or isinstance(kind, int)
                    start_with_fit = explicit if classifier else not explicit
                    rng = np.random.RandomState(42)
                    params = dict(
                        hidden_layer_sizes=(4, 3), activation="tanh", solver=solver,
                        alpha=0.03, batch_size=3, max_iter=2, random_state=rng if explicit else 42,
                        learning_rate="adaptive" if explicit else "invscaling",
                        learning_rate_init=0.03, tol=10.0, n_iter_no_change=1,
                    )
                    if classifier:
                        model = reference.MLPClassifier(**params)
                    else:
                        model = reference.MLPRegressor(loss=kind, **params)
                    for step in range(5):
                        rows = np.arange(2 if step == 2 else 6)
                        X = np.column_stack(
                            [(rows + step - 3) / 4, ((rows**2 + step) % 7 - 3) / 4],
                        ).astype(dtype)
                        weights = ((rows % 3 + 1) / 2).astype(dtype)
                        kwargs = {}
                        if kind == "multilabel":
                            y = indicators(rows, step)
                            weighted = explicit
                            if step == 0 and not explicit:
                                kwargs["classes"] = np.arange(2, -1, -1)
                        elif classifier:
                            present = kind - 1 if step == 0 and not explicit else kind
                            y = 3 * ((rows + step) % present) - 4
                            weighted = explicit
                            if step == 0 and not explicit:
                                kwargs["classes"] = 3 * np.arange(kind - 1, -1, -1) - 4
                        else:
                            diff = X[:, 0] - X[:, 1]
                            y = 2 + diff
                            if kind == "squared_error":
                                y = np.column_stack([y, 2 + diff / 2])
                            weighted = not explicit or step != 0
                        operation = model.fit if step == 0 and start_with_fit else model.partial_fit
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            operation(X, y, sample_weight=weights if weighted else None, **kwargs)
                        rng_draw = None
                        if explicit:
                            probe = np.random.RandomState()
                            probe.set_state(rng.get_state())
                            rng_draw = probe.randint(2**32, dtype=np.uint32)
                        yield model, X, y, weights, weighted, classifier, caught, rng_draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=FAMILY / "builds/mlp_incremental_dump")
    args = parser.parse_args()
    actual = iter(subprocess.check_output([args.binary], text=True).splitlines())
    count, worst = compare(actual, snapshots(load_reference()))
    print(f"PASS {count} public MLP incremental sequence snapshots against pinned sklearn")
    print("Maximum absolute differences", {str(dtype): float(error) for dtype, error in worst.items()})


if __name__ == "__main__":
    main()
