# Linear-model ownership

Follow ../../AGENTS.md, ../../docs/family-development.md and
../../docs/interfaces.md. Moxie owns the whole family, including the child
validation directories here and implementation modules under
../../src/moxie/linear_model. Read docs/scope.md, docs/api-surface.md and the
relevant child docs before changing an estimator or solver.

Keep a single implementation of preprocessing, prediction and coefficient
contracts in `_linear_base` and `_contracts`. `_cd_fast` owns the cyclic
coordinate engine. Smooth and SGD directories contain prerequisite probes,
not implemented engines; RidgeCV has no implementation. Keep those limits
explicit. Shared RNG is owned by ../../shared/random; `our_rand_r` now exists,
but random coordinate selection and SGD integration remain unfinished.

Use the live current Moxie contracts. Preserve the compatible toolchain locks,
reference revision, source provenance and license assets. Dependency manifests
are live source declarations, not source pins. Do not regenerate old snapshot
hashes or restore retired standalone sources.

Run the focused root linear-model build, test and parity tasks for estimator
changes. They cover both dtypes and the original source-based sklearn oracles.
The root namespace ownership check must pass before retirement is claimed.
