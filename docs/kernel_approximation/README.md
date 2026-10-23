# Kernel approximation

Moxie owns the dense kernel feature maps in `src/moxie/kernel_approximation.mojo`
and the private `_kernel_*.mojo` modules. Import all five estimators from
`moxie.kernel_approximation`. No sibling family checkout is required.

- [RBFSampler and SkewedChi2Sampler](fourier/api.md)
- [AdditiveChi2Sampler](additive_chi2/api.md)
- [PolynomialCountSketch](polynomial/api.md)
- [Nystroem](nystroem/api.md)

Run [the example](../../examples/kernel_approximation.mojo) after `pixi run build`
with `builds/examples/kernel_approximation`.

See [internal contracts](internals.md) when extending the family.

## Composition

All five estimators implement the shared `FeatureNamesOut` and
`InputFeatureCount` traits from `moxie._feature_metadata`, so they work as
`moxie.pipeline` `Pipeline` prefixes and `FeatureUnion`/`make_union` branches,
including `get_feature_names_out()` (sklearn's `<step>__<name>` prefixes) and
`n_features_in_()` delegation. `get_n_features_in()` mirrors sklearn's
`n_features_in_` attribute: it raises an `AttributeError`-prefixed error until
`fit` (AdditiveChi2Sampler's stateless `transform` does not set it). The
estimators' `n_features_in_` fields stay public. A kernel map followed by a
classifier with a different target type (for example `RidgeClassifier` or
`LinearSVC`) is fitted with `y=None`, which sklearn's maps ignore anyway.
`PolynomialCountSketch` outputs Float64, so in a Float32 union or pipeline its
successor must accept Float64. `tests/kernel_approximation/test_composition.mojo`
covers pipelines with `RidgeClassifier`, unions, clone, nested `set_params`
and names. `moxie.model_selection` search/validation needs a typed adapter per
estimator; none is provided by this family.

## Native scope

Inputs are nonempty dense Float32 or Float64 NuMojo matrices, including valid
strided views. Outputs and fitted snapshots own their storage. Constructors,
learned attributes, clone/refit behavior and random draw order retain the family
contracts. `KernelFeatureMap` is exported from the public module and defined in
`_kernel_approximation_contracts.mojo`; validation and dimensions live in
`_kernel_features.mojo`.

Sparse containers, implicit densification, Python dictionaries, pandas output,
metadata routing, Python exception classes and device/Array API dispatch are
excluded. Randomized estimators with `random_state=None` require an explicit
mutable application RNG through `fit_with_rng`; integer seeds reset per fit.
Polynomial outputs Float64 for either input dtype and rejects negative `coef0`
at transform instead of returning NaNs. Sequential Float32 Laplacian Nystroem
requires explicit Float64 output. See each API page for numerical adaptations,
parameter updates, feature names and conditioning limits.

Nystroem reuses Moxie's integrated `_kernel_ridge` engine and its existing
mojo-linalg SVD. Polynomial uses the main `../mojo-linalg` FFT through the
`kernel-fft` manifest project. Existing shared array, RNG and numerical owners
remain unchanged. The scikit-learn notice is retained in `LICENSES/scikit-learn.txt`
and distributed by the assembly manifest.

## Validation

Root `pixi run build`, `pixi run test` and `pixi run parity` include this family.
Focused commands are `pixi run python tools/kernel_approximation_checks.py test`
and `pixi run python tools/kernel_approximation_checks.py parity`.

Tests live under `tests/kernel_approximation/`. They cover family contracts,
all five public estimators in both dtypes, cloning, ownership, strides, RNG,
invalid inputs and unsupported capabilities. `test_edges.mojo` checks the shared
numerical edges against sklearn's behavior: NaN/inf rejection in fit and transform,
one-sample fits (Nystroem clips and warns), constant data with `gamma="scale"`,
seeded determinism and refits on a different width. Isolated consumers assemble only
Moxie's current manifest and run outside the repository. Ownership checks compare
the compiled sources with Moxie's files and assert Moxie is their sole owner.
Failed missing-source, module-conflict and export-conflict assemblies must leave
the previous output intact. The old per-algorithm absence checks are replaced by
positive coexistence checks, since all five estimators now share one public module.

The four parity suites execute method bodies from the clean local scikit-learn
checkout at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. NumPy/SciPy and a support
scikit-learn wheel provide test infrastructure only. Exact RNG/index assertions,
conditioned numerical bounds and kernel approximation thresholds are retained;
each API page explains its tolerances. Reports go to ignored `docs/private/`.
These suites do not establish full scikit-learn API parity.

Side-by-side notebooks replay two gallery examples against the pinned
reference: `notebooks/kernel_approximation/plot_kernel_approximation.ipynb`
(RBFSampler and Nystroem pipelines with LinearSVC on digits) and
`plot_scalable_poly_kernels.ipynb` (PolynomialCountSketch on a synthetic
Covtype stand-in, plus every other map before LinearSVC). Rerun them with
`pixi run -e notebooks notebooks notebooks/kernel_approximation`.
