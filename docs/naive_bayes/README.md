# Naive Bayes

Moxie owns `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, and
`CategoricalNB` under `moxie.naive_bayes`, along with their parameter records,
`NaiveBayesClassifier`, `NumericDomain`, `StringDomain`, and `IntListDomain`.
The retired sibling checkout is not a build dependency.

Implementations live in `src/moxie/_nb_*.mojo`. Shared family validation,
label adapters and prediction helpers live in `_naive_bayes_common.mojo` and
`_naive_bayes_dense.mojo`. They use Moxie's current framework, labels, numerical
infrastructure and live dependency assembler. List inputs remain supported by
the canonical estimators. The duplicate legacy estimators were removed after
moving `CategoricalNBParams` and redirecting their remaining tests and consumers.

See [the API](family.md), [coverage](api-surface.md),
[Gaussian arithmetic](gaussian/api.md), and [internal contracts](plumbing.md).
Dense boolean, integer and floating inputs, numeric/string labels, weights,
incremental fitting, probabilities, cloning and fitted copying remain supported.
Sparse input and Python-specific facilities remain deferred. This migration adds
no serialization. The API documents intentional validation and dtype differences.
Bernoulli per-feature alpha deliberately corrects the pinned reference's
broadcasting bug and uses independent scalar sklearn fits as its oracle.

From the Moxie root:

```sh
pixi run build
pixi run test
pixi run parity
```

Focused checks use `pixi run python tools/naive_bayes_checks.py test` and
`pixi run python tools/naive_bayes_checks.py parity`. Root commands run the same
checks. The source receipt verifies Moxie ownership; an isolated consumer builds
against the root manifest and exercises public imports and fluent operations.
The original missing-input and capability compilation failures, namespace
collision rejection, native numerical and lifecycle contracts, typed labels,
and child regression tests remain covered.

Reference comparisons require scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` in the local `../scikit-learn` checkout.
The family and Gaussian runners verify the installed `naive_bayes.py` bytes.
Prepare the validation environment once:

```sh
uv venv builds/naive-bayes-reference-env
uv pip install --python builds/naive-bayes-reference-env/bin/python ../scikit-learn -Ccompile-args=-j2
pixi run python tools/naive_bayes_checks.py parity
```

Alternatively set `SKLEARN_PYTHON` to that environment's Python. Python is
validation tooling only. The three retained comparisons are
`tests/naive_bayes/check_sklearn.py`, `gaussian/parity.py`, and `counts/parity.py`.
The main suite compiles batches of 32 cases and Gaussian uses batches of 16
to bound compiler memory;
`--start`, `--stop` and `--batch-size` retain reproducible case selection.
Their original tolerances remain unchanged: the main and count suites use
`1e-7 + 1e-7 * abs(expected)`, and Gaussian uses `1e-7` for Float64 and `3e-5`
for explicit Float32 arithmetic. Exact shape, label, count and lifecycle checks
supplement approximate floating comparisons.

The pre-migration family baseline failed at compilation against current Moxie
traits, so no passing native or parity baseline is claimed. The implementations
now return a mutable reference from `fit`, `partial_fit` and `set_params` and
use the current associated sample-weight type in scoring signatures. Shared
traits and numerical tolerances were not relaxed.

Moxie's compatible Mojo 1.0.0 / MAX 26.5.0 lock remains unchanged. Original
family/child locks and Git history remain in the retired checkout. Attribution
is preserved in [the sklearn notice](../../LICENSES/naive-bayes-scikit-learn.txt).

The preserved count-model benchmark uses pyperf after the focused build:
`pixi run python tools/naive_bayes_benchmark.py`. Keep results in ignored
`benchmarks/`; this integration makes no performance claim.
