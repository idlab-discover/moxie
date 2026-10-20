"""LatentDirichletAllocation versus pinned sklearn/decomposition/_lda.py.

The pinned Cython helpers (_online_lda_fast.pyx: its own digamma, Dirichlet
expectations, mean_change) are compiled from the pinned file for the oracle.
Both sides draw the same legacy-NumPy gamma variates, so all learned values are
compared directly. n_jobs=None (sequential), the only deterministic setting.
"""
import sys
import warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close, compile_pinned_extension

fast = compile_pinned_extension("sklearn/decomposition/_online_lda_fast.pyx")
module = load_pinned("sklearn/decomposition/_lda.py", extensions={"_online_lda_fast": fast})
DRIVER = ROOT / "builds/lda_driver"
count = 0


def compare(x, k=3, method="batch", max_iter=10, batch_size=128, evaluate_every=-1, seed=0, doc_prior=None,
            word_prior=None, decay=0.7, offset=10.0, doc_iter=100, mean_tol=1e-3, perp_tol=0.1,
            total_samples=1e6, mode="fit"):
    global count
    ref = module.LatentDirichletAllocation(k, learning_method=method, max_iter=max_iter, batch_size=batch_size,
        evaluate_every=evaluate_every, random_state=seed, doc_topic_prior=doc_prior, topic_word_prior=word_prior,
        learning_decay=decay, learning_offset=offset, max_doc_update_iter=doc_iter, mean_change_tol=mean_tol,
        perp_tol=perp_tol, total_samples=total_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "partial":
            ref.partial_fit(x)
            ref.partial_fit(x)
        else:
            ref.fit(x)
        expected = dict(components=ref.components_, exp_dirichlet=ref.exp_dirichlet_component_,
                        n_batch_iter=ref.n_batch_iter_, transform=ref.transform(x),
                        unnormalized=ref.transform(x, normalize=False), score=ref.score(x),
                        perplexity=ref.perplexity(x), perplexity_sub=ref.perplexity(x, sub_sampling=True))
        if mode == "fit":
            expected.update(bound=ref.bound_, n_iter=ref.n_iter_)
    got = run_native(DRIVER, x.dtype, [x], k, method, max_iter, batch_size, evaluate_every, seed,
                     -1 if doc_prior is None else doc_prior, -1 if word_prior is None else word_prior,
                     decay, offset, doc_iter, mean_tol, perp_tol, total_samples, mode)
    label = f"{x.dtype} {x.shape} {k=} {method} {max_iter=} {batch_size=} {evaluate_every=} {doc_prior=} {word_prior=} {doc_iter=} {mode}"
    # Variational updates are iterated fixed points with data-dependent
    # early exits (mean_change_tol); Float32 follows float32 NumPy/Cython.
    t = 1e-7 if x.dtype == np.float64 else 2e-3
    for key in ("n_iter", "n_batch_iter"):
        if key in expected:
            assert int(got[key][0]) == int(expected[key]), (label, key, got[key], expected[key])
    for key, value in expected.items():
        if key in ("n_iter", "n_batch_iter"): continue
        scale = max(1.0, float(np.abs(value).max()))
        assert_close(got.get(key), value, t, f"{label}: {key}", atol=t * scale)
    count += 1


rng = np.random.RandomState(31)
for dtype in (np.float64, np.float32):
    for shape in [(20, 12), (9, 30)]:
        topics = rng.dirichlet(np.ones(shape[1]) * 0.3, size=3)
        mixture = rng.dirichlet(np.ones(3) * 0.5, size=shape[0])
        x = np.vstack([rng.multinomial(40, mixture[i] @ topics) for i in range(shape[0])]).astype(dtype)
        for k in (1, 3, 5):
            compare(x, k)
        compare(x, 3, "online", batch_size=6)
        compare(x, 3, "online", batch_size=128, decay=0.5, offset=4.0)
        compare(x, 3, evaluate_every=2, max_iter=12, perp_tol=1e-3)
        compare(x, 3, evaluate_every=1, max_iter=20, perp_tol=10.0)
        compare(x, 3, doc_prior=0.3, word_prior=0.05)
        compare(x, 3, doc_iter=0)
        compare(x, 3, doc_iter=5, mean_tol=0.0)
        compare(x, 3, max_iter=0)
        compare(x, 3, seed=7)
        compare(x, 3, mode="partial", batch_size=7, total_samples=500)
    compare(np.zeros((4, 5), dtype=dtype), 2)
    compare(np.ones((1, 1), dtype=dtype), 1)
    # Documents with no words and a vocabulary term never used.
    y = x.copy()
    y[2] = 0
    y[:, 3] = 0
    compare(y, 3)
    compare(y, 3, "online", batch_size=4)
print(f"PASS: {count} dense LatentDirichletAllocation cases against local sklearn {REVISION}")
