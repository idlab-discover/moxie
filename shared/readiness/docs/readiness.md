# Dense integration checks

Integration fixtures only. Real Dummy estimators now live in Moxie;
see [Dummy API](../../../docs/dummy/api.md). This gate remains useful for isolated
consumer namespaces and shared ownership checks; it does not certify estimators.

| Import | Owner |
| --- | --- |
| `numojo.NDArray`, `Shape`, `Item` | [Dense storage](../../numerics/docs/numerics.md) |
| `moxie._dense.sum_axis2d`, `row_squared_norms` | `shared/numerics` |
| `moxie._random.RandomState` | [RNG](../../random/docs/random.md) |
| `moxie._readiness.workload`, `contracts` | Local integration fixtures |

Follow [live-source policy](../../../docs/family-development.md). Current shared
assembler builds separate consumer namespaces under `builds/deps`; receipts
record consumed bytes. No revision/hash gate or historical source fallback.

Coverage: Float32/Float64 copies/views, reductions/norms; weighted class counts;
integer/Unicode fixture labels; seeded uniform/stratified two-output draws;
empty/singleton/batched outputs; repeated/advancing RNG streams. Fixtures check
all 624 MT words and position. General label handling belongs to
[labels](../../labels/docs/labels.md), not fixture mapping loops.

Parity executes original sklearn `class_distribution`, `check_random_state`,
Dummy `predict`/`predict_proba` bodies at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Reference revision/cleanliness checked.
Fitted attributes are supplied by fixtures: fitting, validation and scoring
are outside this gate. Counts, labels, draws and RNG state match exactly;
printed Float64 priors allow rtol 2e-15, atol 0 for bounded test cases.
Compile with `-O3 --fp-mode=contract=off`.

Run `pixi run build`, `pixi run test`, `pixi run parity` here. Consumers under
`tests/consumers/a` and `b` also expose these commands. Build compiles both
namespaces and `examples/seeded.mojo`. Test verifies independent outputs,
missing-source rejection, preserved valid output after failure, and unavailable
sparse helper. Cache/temp/output paths stay consumer-local; dependency-write
audits record observed changes under ignored `docs/private/readiness/`.

`tools/benchmark.py` uses pyperf for whole-process integration, including startup.
Keep raw results under `benchmarks/`; no current speed claim.
