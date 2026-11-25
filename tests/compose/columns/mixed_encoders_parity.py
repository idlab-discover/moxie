"""Compare mixed OrdinalEncoder and multiclass TargetEncoder columns exactly."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/columns"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "families/preprocessing/tests/parity")]
import compose_parity_io as io
from overlay import activate

activate()
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

COMPOSE = "sklearn/compose/_column_transformer.py"
io.verify_reference([COMPOSE])
# Load this module directly so the oracle does not import unrelated compiled
# linear-model extensions through sklearn.compose.__init__.
ColumnTransformer = io.load_pinned(COMPOSE).ColumnTransformer
io.pinned_code(ColumnTransformer.fit_transform, COMPOSE)


def main():
    train = pd.DataFrame({
        "color": pd.Series(["red", "blue", "amber", "blue", "red", "amber"], dtype=object),
        "age": [10., 20., 30., 40., 50., 60.],
    })
    test = pd.DataFrame({
        "age": [15., 35., 55.],
        "color": pd.Series(["green", None, "red"], dtype=object),
    })
    y = np.tile([0., 1., 2.], 2)
    model = ColumnTransformer([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["color"]),
        ("target", TargetEncoder(smooth=0., cv=list(StratifiedKFold(2).split(train, y))), ["color"]),
        ("automatic", TargetEncoder(target_type="continuous", cv=None, shuffle=False), ["color"]),
        ("leave_one_out", TargetEncoder(cv=LeaveOneOut()), ["color"]),
        ("numeric", "passthrough", ["age"]),
    ])
    expected_fit = model.fit_transform(train, y)
    expected_transform = model.transform(test)
    output = io.run(ROOT, BUILD / "mixed_encoders", capture=True).stdout
    actual = io.parse_results(output)["mixed_encoders"]
    np.testing.assert_array_equal(actual["fit"], expected_fit)
    for label in ("transform", "after_failed_refit", "after_fit"):
        np.testing.assert_array_equal(actual[label], expected_transform)
    names = [line.split("|")[1] for line in output.splitlines() if line.startswith("name|")]
    spans = {
        name: slice(int(start), int(stop))
        for _, name, start, stop in
        (line.split("|") for line in output.splitlines() if line.startswith("span|"))
    }
    assert names == model.get_feature_names_out().tolist(), names
    assert spans == model.output_indices_, spans
    print("Mixed ordinal and target encoding parity passed: values, names and spans exact")


if __name__ == "__main__":
    main()
