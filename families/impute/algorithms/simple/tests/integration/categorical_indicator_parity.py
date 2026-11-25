"""Compare typed ColumnTable indicator results with the pinned sklearn body."""
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import load


def main(binary):
    reference = load().MissingIndicator
    records = iter(subprocess.check_output([binary], text=True).splitlines())
    datasets = [
        (["color", "color2"],
         [["red", "blue"], [np.nan, "red"], ["blue", "blue"]],
         [["green", "blue"], [np.nan, np.nan]]),
        (["count", "count2"],
         [[2**53, 2**53 + 1], [np.nan, 2**53], [2**53 + 1, 2**53 + 1]],
         [[4, 2**53 + 1], [np.nan, np.nan]]),
    ]
    for names, train, query in datasets:
        train = np.asarray(train, dtype=object)
        query = np.asarray(query, dtype=object)
        strict = reference(sparse=False).fit(train)
        try:
            strict.transform(query)
        except ValueError as error:
            assert "missing values" in str(error)
        else:
            raise AssertionError("Expected a newly missing feature to raise")
        for features in ("missing-only", "all"):
            model = reference(features=features, sparse=False, error_on_new=False).fit(train)
            count = int(next(records))
            assert [int(next(records)) for _ in range(count)] == model.features_.tolist()
            assert [next(records) for _ in range(count)] == model.get_feature_names_out(names).tolist()
            rows, columns = map(int, next(records).split())
            expected = model.transform(query)
            assert expected.dtype == np.bool_
            assert (rows, columns) == expected.shape
            assert [int(next(records)) for _ in range(rows * columns)] == expected.ravel().tolist()
    assert next(records, None) is None
    print("PASS: 4 categorical indicator comparisons and 2 newly missing feature errors")


if __name__ == "__main__":
    main(sys.argv[1])
