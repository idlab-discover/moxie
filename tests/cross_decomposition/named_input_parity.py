"""Compare native named inputs and table projections with the pinned estimators."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from pls.parity import REFERENCE, load_reference, verify_reference


def load_reference_with_output():
    source = "sklearn/utils/_set_output.py"
    assert not subprocess.check_output(
        ["git", "-C", REFERENCE, "status", "--porcelain", "--", source], text=True
    ).strip()
    spec = importlib.util.spec_from_file_location("pinned_set_output", REFERENCE / source)
    output = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(output)
    # The installed wheel supplies sklearn.base; use the pinned wrapping code
    # when its subclass hook decorates the pinned estimator methods.
    with patch("sklearn.utils._set_output._wrap_method_output", output._wrap_method_output):
        return load_reference()


def main(runner):
    verify_reference()
    reference = load_reference_with_output()
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
        values = ((i * i + 3 * i) % 19 - 9).reshape(6, 3)
        x = pd.DataFrame({
            columns[0]: values[:, 0],
            columns[1]: values[:, 1].astype(dtype),
            columns[2]: values[:, 2].astype(np.float64),
        })
        i = np.arange(6 if name == "PLSRegression" else 12)
        y = ((i * i + 2 * i) % 13 - 6).astype(dtype)
        if name != "PLSRegression":
            y = y.reshape(6, 2)
        kwargs = {"algorithm": "svd"} if name == "PLSCanonical" else {}
        model = getattr(reference, name)(n_components=2, **kwargs)
        model.set_output(transform="pandas")
        fit_x_scores, fit_y_scores = model.fit_transform(x, y)
        assert names.split(",") == model.feature_names_in_.tolist(), name
        assert components.split(",") == model.get_feature_names_out(columns).tolist(), name
        projected_x, projected_y = model.transform(x, y)
        assert isinstance(projected_x, pd.DataFrame), name
        assert isinstance(projected_y, np.ndarray), name
        pd.testing.assert_frame_equal(fit_x_scores, projected_x)
        np.testing.assert_array_equal(fit_y_scores, projected_y)
        assert projected_x.columns.tolist() == components.split(","), name
        assert all(projected_x.dtypes == np.float64), name
        # The Float32 native table materializes in its configured dtype. Pandas
        # promotes these mixed columns to Float64, so PLS inference may round
        # standardization differently before multiplication.
        tol = 1e-7 if dtype == np.float32 else 1e-10
        for values, expected in zip((x_scores, y_scores), (projected_x, projected_y), strict=True):
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
    print("PASS: named input and paired table output parity for all four estimators")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
