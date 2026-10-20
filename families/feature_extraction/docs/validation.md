# Rollup validation

Run these commands from the Moxie root with its compatible Pixi lock.
The integration was validated on 2026-10-20 against local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, Mojo 1.0.0 and MAX 26.5.0.

| Command | Outcome |
| --- | --- |
| `pixi run feature-extraction-test` | Passed. Precompiled the assembled public namespace, built the family and both algorithms, ran native contracts and six isolated consumers, and checked missing sources, ownership/export collisions and unsupported capabilities. |
| `pixi run feature-extraction-parity` | Passed. 308 dictionary encoder cases, 43 dictionary array API cases, mixed-key and restriction probes, and 263 image differential cases. |
| `pixi run feature-extraction-start` | Passed. Vocabulary/geometry, dictionary and image examples executed. |
| `pixi run feature-extraction-independence` | Passed. All family manifest paths resolve without the standalone checkout, ownership receipts and license assets match, and copied assembly inputs compile and run dictionary, image and deferred-API consumers in a temporary directory. |
| `git diff --cached --check` | Passed. |

Dictionary assertions compare exact values after dtype conversion and compare
NaN masks separately. Image comparisons use exact shapes and a 1e-12 absolute
numeric tolerance. Native contracts additionally check large integer precision,
strides, invalid inputs, lifecycle behavior and RNG advancement.

The reference adapters execute class/function bodies from the local sklearn
source. They replace Python decorator/base/validation infrastructure only in
validation tooling. These checks do not establish Python binding parity or full
scikit-learn parity. Supported modes and remaining work are in [scope](scope.md)
and the child API documents.

The original monolithic dictionary parity compile exceeded the shared 8 GiB
compiler address-space limit. Both generated suites now compile at most 28 case
functions per batch. The generators assert that every original entry point is
present exactly once; they preserve the oracle values and assertions. All batches
must compile and execute before a suite reports success. The report directories
are created on a clean checkout. Runtime numerical implementations and compatible
Pixi locks were relocated byte-for-byte.

Generated code, receipts and detailed logs remain ignored under `builds/` and
`docs/private/`. Source history and baseline tags remain in the retired checkout;
[provenance](../PROVENANCE.json) records the source revision.
