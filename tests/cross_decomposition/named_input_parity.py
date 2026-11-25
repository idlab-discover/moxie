"""Compare the named cases in serialization.mojo with the pinned estimators."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from pls.parity import load_reference, verify_reference


def main(runner):
    verify_reference()
    reference = load_reference()
    completed = subprocess.run([runner], stdout=subprocess.PIPE, text=True, check=True)
    seen = set()
    for line in completed.stdout.splitlines():
        if not line.startswith("named|"):
            continue
        _, name, names, components, x_scores, y_scores = line.split("|")
        assert name not in seen, name
        seen.add(name)
        dtype = np.float32 if name in {"PLSRegression", "PLSSVD"} else np.float64
        i = np.arange(18)
        columns = ["mass", "âge", "taux"]
        x = pd.DataFrame(
            ((i * i + 3 * i) % 19 - 9).reshape(6, 3), columns=columns, dtype=dtype
        )
        i = np.arange(6 if name == "PLSRegression" else 12)
        y = ((i * i + 2 * i) % 13 - 6).astype(dtype)
        if name != "PLSRegression":
            y = y.reshape(6, 2)
        kwargs = {"algorithm": "svd"} if name == "PLSCanonical" else {}
        model = getattr(reference, name)(n_components=2, **kwargs).fit(x, y)
        assert names.split(",") == model.feature_names_in_.tolist(), name
        assert components.split(",") == model.get_feature_names_out(columns).tolist(), name
        for values, expected in zip((x_scores, y_scores), model.transform(x, y), strict=True):
            # Float32 PLS inference normalizes in the input dtype.
            tol = 1e-7 if dtype == np.float32 else 1e-10
            np.testing.assert_allclose(
                np.fromstring(values, sep=",").reshape(expected.shape),
                expected, atol=tol, rtol=tol, err_msg=name,
            )
        try:
            model.transform(x[columns[::-1]])
        except ValueError as error:
            assert str(error).splitlines()[0] == (
                "The feature names should match those that were passed during fit."
            )
        else:
            raise AssertionError(f"{name} accepted reordered columns")
        try:
            model.get_feature_names_out(columns[::-1])
        except ValueError as error:
            assert str(error) == "input_features is not equal to feature_names_in_"
        else:
            raise AssertionError(f"{name} accepted reordered input_features")
        model.fit(x.to_numpy(), y)
        assert not hasattr(model, "feature_names_in_"), name
    assert seen == {"PLSRegression", "PLSCanonical", "CCA", "PLSSVD"}, seen
    print("PASS: named input parity for all four cross-decomposition estimators")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
