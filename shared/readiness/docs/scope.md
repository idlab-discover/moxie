# Dense-only scope

Supported: two isolated native consumers importing shared NuMojo dense storage,
`moxie._dense` reductions/norms and `moxie._random.RandomState`. The gate tests
Float32/Float64 ownership and kernels; Float64 weighted/unweighted class counts
and probabilities; integer and Unicode string label mappings; seeded uniform
and stratified two-output Dummy-shaped workloads; empty, singleton and batched
outputs; repeated and advancing streams. There are no public estimators.
Exact tested seeds, shapes, tolerances and adaptations are in [readiness.md](readiness.md).

Deferred: `_sparse` imports, CSR/CSC/COO storage/conversions, sparse targets and
outputs, `sklearn.utils.random._random_choice_csc`, and Dummy's sparse target
fit/prediction branches. Source: `sklearn/utils/random.py`, `sklearn/dummy.py`
and sparse cases in `sklearn/tests/test_dummy.py` at reference revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. These are authorized exclusions under
`session-plan/SPARSITY-SCOPE.md`, not failed readiness requirements. The old
Dummy sparse import blocker is retained only as historical evidence. No
sparse input is implicitly densified and no sparse substitute is introduced.

Inherited restrictions remain: use only the dense numerical owner's accepted
operations; NuMojo dot, general axis reduction, general solves and seeded RNG
are not approved substitutes. Advanced factorization is not supplied by this
gate. The RNG owner provides legacy RandomState semantics, not arbitrary NumPy
Generator APIs. Dynamic Python reflection/bindings and full estimator methods
are outside this integration gate; no estimator completion is implied.

No tested default requires sparse storage, and no explicit dense setting is
needed by these private imports. Downstream estimators whose default requests
sparse output must explicitly select their supported dense option. This gate
neither changes their defaults nor certifies those estimators.

Executed and passed dense commands: `pixi run build`, `pixi run test`, `pixi run parity`.
Build compiles owned modules through the assembled package, the example and
both consumers. Test checks numerical/ownership errors, independent copies,
repeated seeds, dependency drift rejection and absence of the deferred sparse
helper. The negative sparse-helper check documents unavailability; sparse
import success is not required. Parity executes the pinned original reference
method bodies for 144 native invocations / 288 consecutive prediction batches
across the two consumers. Private reports record dependency-write audits.
