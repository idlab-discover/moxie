# Dense SGD engine scope

Status: blocked before engine implementation. No engine or estimator exports
are available. `EXPORTS.json` is deliberately empty.

The intended private engine supports dense Float32/Float64 datasets, explicit
sample weights, incremental coefficients/counters, stochastic loss updates,
regularizers, learning-rate schedules, averaging, seeded shuffling and early
stopping. These numerical branches are **not implemented or validated** yet.
The existing executable is a dependency probe, not a solver.

Sparse CSR/CSC/COO datasets, sparse intercept decay, sparse outputs and
coefficient sparsification are deferred by the session's dense-only decision.
Reference: sklearn `linear_model/_stochastic_gradient.py` and
`utils/_seq_dataset.pyx.tp` at 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
No sparse API is exported and sparse inputs are never implicitly densified.

The family implements centering/prediction and declares engine extension points. This engine must consume its input/weight/ownership contracts.
No factorization or sparse dependency is needed for dense SGD.

Default shuffled training is in scope, but requires the shared RNG owner's
`our_rand_r` stream, now available in shared/random. `RandomState.permutation` uses MT19937 and is not
an equivalent replacement. Setting shuffle=False would reduce required dense
coverage; it does not complete this task.

Executed checks: build and test passed for dependency probes; parity failed with
the documented RNG mismatch. These are not engine acceptance results.

Validation commands:

- `pixi run build`: compiles assembled dependencies and the native RNG probe.
- `pixi run test`: runs RNG snapshot and Float32/Float64 family ownership probes.
- `pixi run parity`: compiles the pinned local sklearn RNG primitive, records
  differences, and exits nonzero for the known missing capability. It is a
  blocker reproducer, not SGD differential validation.

There are no implemented estimator defaults or language adaptations to certify.
Engine negative-input tests, traces, incremental tests and performance baselines
remain required after implementation. No performance claim is made.
