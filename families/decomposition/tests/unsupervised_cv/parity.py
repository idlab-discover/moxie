"""Compare targetless dense CV with the pinned sklearn validation and estimators."""
import ast
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from sklearn.utils import _array_api, _metadata_requests, metadata_routing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, assert_close, compile_pinned_extension, load_pinned, run_native, verify_source


def pinned_helpers(relative, support, *names):
    path = verify_source(relative)
    functions = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = vars(support).copy()
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


# Bridge the pinned modules to the NumPy-only support wheel as in the
# validation and PCA owners' parity checks.
routing = pinned_helpers("sklearn/utils/_metadata_requests.py", _metadata_requests, "_manual_routing")
array_helpers = pinned_helpers("sklearn/utils/_array_api.py", _array_api, "_cov", "_add_to_diagonal")
with (
    patch.multiple(metadata_routing, create=True, **routing),
    patch.multiple(_array_api, create=True, array_device=lambda x: None,
                   move_to=lambda x, **kwargs: np.asarray(x), **array_helpers),
):
    validation = load_pinned("sklearn/model_selection/_validation.py")
    base_pca = load_pinned("sklearn/decomposition/_base.py")
    with patch.dict(sys.modules, {"sklearn.decomposition._base": base_pca}):
        pca = load_pinned("sklearn/decomposition/_pca.py")

split = load_pinned("sklearn/model_selection/_split.py")
# Match the pinned validator's split interpretation as well as its fold execution.
validation.check_cv = split.check_cv
fa = load_pinned("sklearn/decomposition/_factor_analysis.py")
lda_extension = compile_pinned_extension("sklearn/decomposition/_online_lda_fast.pyx")
lda = load_pinned("sklearn/decomposition/_lda.py", extensions={"_online_lda_fast": lda_extension})
kde = load_pinned("sklearn/neighbors/_kde.py")
driver = Path(os.environ.get("MOXIE_UNSUPERVISED_CV_DRIVER", ROOT / "builds/unsupervised_cv_driver"))
rng = np.random.RandomState(2718)
samples = rng.poisson(5, size=(23, 5))

for dtype in (np.float32, np.float64):
    x = samples.astype(dtype)
    got = run_native(driver, dtype, [x])
    folds = list(split.KFold(4).split(x))
    cases = [
        ("pca_default", pca.PCA(2, svd_solver="full"), None),
        ("pca_three", pca.PCA(2, svd_solver="full"), 3),
        ("pca_explicit", pca.PCA(2, svd_solver="full"), [folds[3], folds[0]]),
        ("fa", fa.FactorAnalysis(2, svd_method="lapack"), None),
        ("lda", lda.LatentDirichletAllocation(n_components=2, random_state=0, max_iter=3), 3),
        ("kde", kde.KernelDensity(bandwidth=1.3), None),
    ]
    for name, estimator, cv in cases:
        expected = validation.cross_val_score(estimator, x, cv=cv, error_score="raise")
        # Native decomposition and sklearn use different SVD/reduction kernels.
        tol = 3e-4 if dtype == np.float32 else 2e-8
        assert_close(got[name], expected, tol, f"{dtype} {name}")
print(f"PASS: 12 targetless CV cases against sklearn {REVISION}")
