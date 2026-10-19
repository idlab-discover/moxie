"""Incremental curves against pinned sklearn bodies; native display moments/SVG."""
import importlib.util
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import KFold


def check_incremental_display(ns, root, oracle):
    spec = importlib.util.spec_from_file_location('sklearn._moxie_nb_reference', oracle / 'sklearn/naive_bayes.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    X = np.array([[(i * 7) % 11, i % 3 + i * .17] for i in range(12)])
    y = np.arange(12) % 2
    actual = {}
    words = {}
    for line in subprocess.check_output([str(root / 'builds/incremental_gaussian')], text=True).splitlines():
        fields = line.split()
        if fields[0] == 'gaussian_incremental':
            actual[tuple(map(int, fields[1:4]))] = list(map(float, fields[4:]))
        elif fields[0] == 'gaussian_rng':
            words[int(fields[1])] = int(fields[2])
    for mode in range(2):
        rng = np.random.RandomState(42)
        sizes, train, test = ns['learning_curve'](reference.GaussianNB(), X, y,
            cv=list(KFold(3).split(X)), train_sizes=[2, 5, 8],
            exploit_incremental_learning=True, shuffle=bool(mode), random_state=rng)
        np.testing.assert_array_equal(sizes, [2, 5, 8])
        for size, fold in np.ndindex(train.shape):
            np.testing.assert_allclose(actual[mode, size, fold], [train[size, fold], test[size, fold]], rtol=1e-13, atol=1e-13)
        assert words[mode] == int(rng.randint(0, 2**32, dtype=np.uint32))
    output = subprocess.check_output([str(root / 'builds/displays')], cwd=root, text=True)
    cells = 0
    for line in output.splitlines():
        fields = line.split()
        if fields[0] == 'display':
            artist, row = map(int, fields[1:3])
            scores = np.array([row, row + (1 if artist == 0 else 2)], dtype=float)
            # _BaseCurveDisplay._plot_curve uses population std and row means.
            np.testing.assert_allclose(list(map(float, fields[3:])), [scores.mean(), scores.std()], rtol=1e-13, atol=1e-13)
            cells += 2
    assert cells == 12
    for name in ('learning', 'validation'):
        svg = ET.parse(root / f'builds/{name}.svg').getroot()
        assert svg.tag == '{http://www.w3.org/2000/svg}svg'
        assert svg.findall('.//{http://www.w3.org/2000/svg}path')
    print('PASS incremental GaussianNB: 36 score cells and exact RNG; 12 display moments and parsed SVGs')
    return dict(incremental_score_cells=36, display_moments=cells, rtol=1e-13, atol=1e-13)
