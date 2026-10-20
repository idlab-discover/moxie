# Decomposition validation

The rollup was validated on 2026-10-20 against local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Default compilation uses
Mojo 1.0.0/MAX 26.5.0 with `-O3 --fp-mode=contract=off`. The optional
Numax backend retains Mojo 1.1.0/MAX 26.6.0 and its existing Pixi lock.

## Default namespace and native contracts

From Moxie:

```sh
pixi run decomposition-test
pixi run python families/decomposition/shared/svd_engine/tools/check.py test
pixi run python families/decomposition/tools/family.py parity-only
```

Passed package precompilation, all nine estimator contract groups and examples,
SVD contracts and example, and two separately assembled public consumers.
The final native gate also ran the new generic fluent-interface contracts in
Float32/Float64, the retained layout/ownership/engine-extension contracts and
example, and the negative sparse-import check.

All 1,806 estimator differential cases passed: threshold sparse coding 144,
KernelPCA 55, PCA 349, IncrementalPCA 318, TruncatedSVD 358, FactorAnalysis 186,
FastICA 180, NMF/MiniBatchNMF 156, and LDA 60. The SVD engine passed another
917 cases, including both native factorization drivers and randomized ranges.
Per-estimator guides and the engine validation guide specify tolerances.

The migration changed public mutation methods to satisfy current Moxie fluent
traits. It also adopted current Moxie kernel/RNG contracts and the existing root
mojo-linalg owner. Numerical algorithms and unsupported modes were preserved.

## Independence

The standalone checkout was temporarily moved away from `../moxie-decomposition`.
With that path absent, the following command passed from Moxie:

```sh
pixi run python -c 'import pathlib, sys; assert not pathlib.Path("../moxie-decomposition").exists(); sys.path.insert(0,"families/decomposition/tools"); import family; family.assemble(); family.test(); print("PASS: standalone checkout absent throughout fresh assembly and native validation")'
```

This rebuilt dependency trees and compiled and executed both isolated consumers.
Ownership checks compared assembled source bytes and license assets to their
Moxie owners and rejected standalone dependency paths. Estimator and engine
contracts, including fluent chaining, passed. Numerical parity also ran with
the standalone path absent. After retirement that path remains absent.

The two feature-selection consumer manifests were redirected to this family.
Assembly passed for each, without running another family's numerical suite:

```sh
pixi run python shared/assembly/tools/assemble_family.py --consumer tests/feature_selection/consumers/a --lock tests/feature_selection/consumers/a/DEPENDENCIES.json --output tests/feature_selection/consumers/a/builds/deps
pixi run python shared/assembly/tools/assemble_family.py --consumer tests/feature_selection/consumers/b --lock tests/feature_selection/consumers/b/DEPENDENCIES.json --output tests/feature_selection/consumers/b/builds/deps
```

## Optional Numax backend

From this family directory:

```sh
pixi install --locked -e numax
pixi run -e numax numax-test
pixi run -e numax python tests/pca/parity.py --numax
pixi run -e numax python tools/check_incremental_pca.py test
MOXIE_IPCA_DRIVER="$PWD/builds/ipca_driver" pixi run -e numax python tests/incremental_pca/parity.py
```

PCA and IncrementalPCA contracts and the two IncrementalPCA consumers passed.
PCA passed 348 differential cases; IncrementalPCA passed 318. The PCA validation
driver now includes the 5-by-24 wide-matrix specialization needed by the existing
ARPACK comparisons. After adding that specialization, its exact rebuild was:

```sh
pixi run -e numax mojo build -O3 --fp-mode=contract=off -I builds/numax-deps -I ../../../numax -I backends/numax -I tests/pca -I tests/support tests/numax_pca/static_evaluate.mojo -o builds/numax_pca_static_evaluate
```

The generated-copy compatibility patch now covers the current shared RNG's
Atomic type argument as well as NuMojo. Shared source checkouts are unchanged.
The SVD parity tool creates its ignored report directory on a fresh checkout.

## Limits

These are bounded dense correctness checks, not full sklearn parity or a
performance claim. Dictionary learning, SparsePCA, regression-based sparse
coding, sparse containers and Python-only facilities remain unfinished or
explicitly unsupported as described in [scope](scope.md). Historical diagnostic
children are retained under provenance and do not define current support.
