# Moxie integration

The implementation and internal owners now live in `moxie/families/svm`.
Moxie's root `DEPENDENCIES.json` assembles their modules and public exports.
The root family build consumes that complete manifest and verifies owner IDs,
source bytes, license bytes and exports. Isolated consumer manifests exercise
missing-source errors, collisions and real estimator calls.

`PROVENANCE.json` records the original repository revision, scaffold baseline
and sklearn reference. Estimator, engine and example implementations were relocated without numerical
changes. The linear lifecycle test now checks rejection of obsolete explicit
class weights, then clears the weights before refitting new labels, matching
the pinned sklearn class-weight validation. Compatible Pixi locks were retained.
Build and oracle paths now point to their current owners.

Run `pixi run svm-build`, `pixi run svm-test`, `pixi run svm-parity` and
`pixi run svm-start` from Moxie. The `test-only` and `parity-only` runner actions
reuse binaries produced by `svm-build`; root shared checks use these actions
after building. Raw outputs stay in ignored `builds/` and `docs/private/`.

[Scope](scope.md) describes the dense contract, including sample-weight requests.
See the
[linear model file API](../algorithms/linear/docs/api.md#native-model-files) and
the [kernel model file API](../algorithms/c_svm/docs/api.md#native-model-files)
for persistence.
The retirement record in the sibling checkout identifies the integration commit.

## Acceptance evidence

The integrated build, native contracts, assembly conflict checks and examples
passed on 2026-10-20 with Mojo 1.0.0 and MAX 26.5.0. After the build and the
corrected lifecycle test's recompilation, validation ran these commands from
Moxie while the standalone checkout was temporarily unavailable:

```sh
pixi run svm-independence
pixi run python families/svm/tools/check.py test-only
pixi run python families/svm/tools/check.py parity-only
pixi run families/svm/builds/plumbing
pixi run families/svm/algorithms/linear/builds/linear
```

`svm-independence` creates a fresh assembly from Moxie's root manifest and
compiles and executes public linear and kernel lifecycle tests. Parity passed
356 linear-engine cases and 15,003 RNG words, 174 linear-wrapper cases, 32
linear edge cases, 36 bound cases, 24 kernel-arithmetic cases, 144 kernel-engine
cases and 125 public kernel-estimator cases. Existing suite tolerances and
limitations remain in the engine and estimator scope documents. This evidence
covers the tested dense contract, not complete sklearn compatibility.
