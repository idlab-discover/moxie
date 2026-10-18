# Random projection

Moxie owns `GaussianRandomProjection`, `SparseRandomProjection`, and
`johnson_lindenstrauss_min_dim` in `src/moxie/random_projection.mojo`.
Import them, `ProjectionMatrix`, and the typed parameter records from
`moxie.random_projection`. The retired sibling is not a dependency.

Both estimators support Float32/Float64 dense and CSR inputs. Gaussian outputs
are dense; sparse projection can return CSR. The family-specific matrix,
projection and pseudoinverse helpers live under `src/moxie/_projection/`.
See [API](api.md), [numerical decisions](numerics.md), and
[license notices](THIRD_PARTY_NOTICES).

From the Moxie root:

```sh
pixi run build
pixi run test
pixi run parity
builds/examples/random_projection
```

Focused validation uses `pixi run python tools/random_projection_checks.py test`
and `parity`. Root validation includes these same checks. Tests compile against
Moxie's assembled sources and check ownership receipts. An external consumer
assembles the Moxie manifest and uses all three exports, both dtypes, dense and
CSR transformations, reconstruction, typed parameters and fluent calls.
Mixed-dtype implicit input conversion is rejected at compilation.

The original root's 365 and Gaussian child's 127 reference fixtures passed
before integration. Both suites remain, with the root suite extended by ten
mixed Gaussian/sparse advancing-stream fixtures that check the next raw RNG word.
Native contracts retain lifecycle, cache, clone, copy, invalid-input, duplicate
CSR and inverse edge cases. The child's sparse-absence check is replaced by
public sparse-consumer coverage and a mixed-dtype negative check.

The oracle remains the local scikit-learn checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The installed wheel supplies support
utilities only. Numerical tolerances and exact shape/storage comparisons remain
unchanged. See the API for those tolerances and remaining Python-specific gaps.
Moxie's compatible Mojo 1.0.0 / MAX 26.5.0 lock is unchanged. The retired checkout
retains the original locks and Git history; no copied estimator framework or
family build environment is needed by Moxie.

The preserved whole-process benchmark runs with
`pixi run python tools/random_projection_benchmark.py` after building.
Raw pyperf results and GNU time memory reports stay in ignored `benchmarks/`.
No performance claim is made.
