# Dense-only scope

This owner implements the private shared `moxie._random` engine. There are no
public estimators. Supported operations are scalar integer-seeded MT19937,
Float64 uniform/normal/gamma draws, int64-bounded integers, binomial (inversion/BTPE),
categorical and multinomial draws, integer permutation, and all four sklearn
sampling-without-replacement methods. NuMojo provides dense probability and
result storage. Explicit native Philox streams delegate to the pinned stdlib.

Tests cover repeated/distinct seeds, independent and copied mutable states,
normal caching, gamma shape regimes and zero parameters, exact state advancement, 32/64-bit integer bounds, probability
validation, empty/singleton outputs, sampling dispatch boundaries, dense Dummy
single/multioutput call patterns, and integer versus mutable-state repetition.
`permutation(n)` returns an empty vector for negative integer n, as NumPy does;
negative sampling populations remain errors. Native typed signatures and the
explicit shared-context equivalent of None are documented in [random.md](random.md).

Deferred sparse branch: sklearn `utils.random._random_choice_csc`, including
CSC output, sparse target draws and Dummy's sparse prediction branch. Sources:
`sklearn/utils/random.py` and `sklearn/dummy.py` at revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b. No local sparse substitute, sparse
backend, or implicit densification is supplied. `tests/negative/sparse.mojo`
proves the sparse helper is unavailable. Inherited `_dense` only supports its
numerical owner's accepted operations; no sparse or factorization capabilities
are inferred from those dependencies.

No supported RNG default requires sparse storage. No explicit dense selection
is necessary. Dense consumers with sparse-default outputs elsewhere must still
select their documented dense mode; this engine does not override estimators.

Validation commands: `pixi run build`, `pixi run test`, `pixi run parity`.
Build compiles all owned runtime modules, tests, example and both locked consumer
probes. Test runs native contracts and the missing-sparse-capability check.
Parity checks actual output sequences and full RNG state against locked NumPy,
the pinned local sklearn Cython sampler and original dense Dummy method bodies.
These checks establish this engine's documented dense scope, not full sklearn
or every NumPy random-distribution API.
