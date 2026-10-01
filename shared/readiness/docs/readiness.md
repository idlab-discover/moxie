# Dense foundation readiness

This gate validates two isolated consumers of the frozen dense storage and RNG
owners. It implements no estimator and changes no parent API. The historical
Dummy attempt `f50e051c09af4f46a9101bef55607f76` failed to import `_random` and
`_sparse`. The RNG import now compiles and executes; sparse work is deferred.

## Imports and ownership

| Exact import | Owner and supported use |
| --- | --- |
| `from numojo import NDArray, Shape, Item` | NuMojo 0.10.0, revision `515fb2856f0ecf3d2740a34d958fe168183e1129`; accepted storage policy in `../numerics/docs/numerics.md` |
| `from moxie._dense import sum_axis2d, row_squared_norms` | `moxie/shared/numerics`; Float32/Float64 matrix reductions and norms |
| `from moxie._random import RandomState` | `moxie/shared/random`; mutable legacy MT19937, integer draws and multinomial sampling |
| `from moxie._readiness import workload, contracts` | This project; integration fixtures only, not estimator or general label-encoder APIs |
| `from moxie.consumer_a import run` / `from moxie.consumer_b import run` | Independent local consumers under `tests/consumers/a` and `b` |

`foundation.assembly` owns `../assembly/tools/consume.py` and `assemble.py`.
Locks flatten the baseline, dense owner, NuMojo, RNG owner and assembler pins,
including upstream dependency locks, export manifests and license notices.
Builds verify SHA-256 hashes and Git revisions; they never refresh pins.
Each consumer gets its own single namespace under `builds/deps`, with identical
shared implementation hashes and separate files. Only that namespace is passed
to the compiler. A partial owned `src/moxie` package cannot compose dependencies
by adding include paths; compile the assembled package instead.

## What the gate proves

Both consumers execute the same shared native implementations. Numerical arrays
use NuMojo, including the matrix of per-row weighted class contributions, its
shared axis reduction and the multinomial result. Small native numeric/string
lists hold categorical fixture metadata; they are not alternative dense buffers.
The fixed vocabularies `[-7, 2, 11]` and `["amsterdam", "paris", "東京"]` are
independently checked against the reference's sorted classes. Mapping loops
and contribution matrices are fixture code, not an exported general encoder.

Parity covers weighted/unweighted counts, zero-weight rows, numeric and Unicode
string label mappings, two-output uniform/stratified prediction draw order,
single-class distributions, empty/singleton/batched predictions, seeds 0, 42
and 2**32-1, and two consecutive predictions using one mutable state. Repeated
fresh process calls prove integer-seeded reset behavior. All 624 MT words and
the state position are compared after each prediction. Native contracts cover
Float32/Float64 copy versus shared-view mutation, empty reductions, norms,
invalid rank/axis/seed/probability/bounds and independent cached RNG snapshots.

The oracle executes original AST-selected `class_distribution`,
`check_random_state`, and `DummyClassifier.predict`/`predict_proba` bodies from
local sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. It verifies
HEAD, source cleanliness and original code filenames. NumPy/SciPy supply test
plumbing; a wheel never supplies these target implementations. Fitted attributes
are explicit fixtures and fitted-state checks are bypassed in this oracle:
this does **not** test Dummy fitting, validation, scoring or estimator parity.
Reference entry points are `sklearn/dummy.py`, `sklearn/utils/multiclass.py`,
`sklearn/utils/validation.py`, `sklearn/tests/test_dummy.py` and the
Dummy section of `doc/modules/model_evaluation.rst`. No specialized mathematical
paper is cited there for these elementary count and sampling operations.

Counts, class order, labels, draws and RNG state must match exactly. Priors
allow relative error 2e-15 (zero absolute tolerance) for printed Float64 division;
small weights are exactly representable halves. These bounded cases do not
certify arbitrary ill-conditioned sums. Compile with `-O3 --fp-mode=contract=off`,
matching the RNG owner's required arithmetic mode, Mojo 1.0.0 and MAX 26.5.0.

## Reproduce

From this project:

```sh
PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi install --locked
pixi run build
pixi run test
pixi run parity
pixi run python tools/benchmark.py
```

The independent consumer manifests also offer build/test/parity. For example,
from `tests/consumers/a`, run `PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi install
--locked` and `pixi run test`. The root gate uses its locked compiler to build
both consumers with independent import trees, cache roots and temporary storage.
The example `examples/seeded.mojo` compiles to `builds/example`.

Every root check snapshots dependency file paths, sizes and nanosecond mtimes
before/after execution, including dependency environments and generated areas.
Pinned source bytes additionally undergo SHA-256 verification. This detects
writes and newly generated caches in the observed dependency trees; it is an
integration audit, not a general process sandbox. Cache variables and TMPDIR
point inside the active consumer; Python bytecode writes are disabled. Native
outputs stay under its `builds/`. Test also rejects a corrupted dependency pin
without replacing the last valid assembled tree, and verifies shared files are
separate consumer-local copies. Isolation and oracle reports live in ignored
`docs/private/readiness/`.

The pyperf baseline measures whole-process integration, including startup,
native contracts, allocations and output formatting; compilation is excluded.
GNU time records peak RSS for 16 and 10,000 prediction rows in both consumers.
The temporary count matrix is O(6*k), output storage O(n*k), with k at most 3.
Raw results, compiler flags, hardware, versions, CPU affinity, seeds, commands
and thread settings remain in ignored `benchmarks/`. Short samples are smoke
baselines, not speed claims or a replacement for the owners' kernel benchmarks.

See [scope.md](scope.md) for explicit deferred coverage. Full family estimator
implementation, generic label encoders and unapproved numerical operations
remain their owners' work; readiness certifies only the integration above.
