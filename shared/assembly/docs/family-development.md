# Family development assembly

Families may adopt `tools/consume_family.py` and `tools/assemble_family.py` in
place of the strict v1 launchers. The original entry points are retained for
consumers that have not adopted this policy. Pin the new launchers' hashes in the
`assembly` entry and retain `tools/family_validator.py` beside them; the new assembler
verifies this versioned validator against its embedded hash before importing it.
It retains v1 checks and accepts both existing Numax and mojo-optimize asset
namespaces. The strict v1 tools remain unchanged for unadopted consumers.

The version-1 ownership manifest remains valid. Source projects in the same Git
repository as the consumer use current working files, with actual hashes in the
generated receipt. An edit to a family engine or its documentation requires no
repinning. Adding or removing modules still requires updating the explicit
module/export manifest. Baseline-marked framework modules remain immutable.

Only consumed external modules/assets and assembly tools are build gates.
Upstream AGENTS, nested locks and documentation remain recorded provenance;
changes to those files do not invalidate a family build.

External inputs are copied with their exact locked hashes into
`FAMILY/builds/family-dependencies/`. Initial recovery uses matching working
files or matching files in Git history. Recovery never accepts different bytes.
Snapshots have their own Git objects, preserve baseline proofs and licenses,
and are shared by the family's child projects. Subsequent builds can run with
the original sibling checkout absent. Missing unrecoverable pins fail clearly;
external adoption remains a deliberate change to the source lock.

Caches are disposable. Rebuilding them requires the recorded sources/history or
an equivalent source snapshot containing the exact pins. Uncommitted upstream
bytes should be published in a durable dependency release before distributing
a family to another machine. A local verified cache alone is not a distribution.

The shared v1 validator still checks module/export collisions, explicit overrides,
source path containment, immutable baseline provenance and output placement.
Output remains `CONSUMER/builds/deps`; invalid dependencies preserve the previous
assembled namespace. Source manifests are never rewritten by a build.

Run the isolated build-tool regression tests with:

```sh
python -m unittest discover -s shared/assembly/tests -p test_family_assembly.py -v
```

The tests cover live family edits, child reuse, upstream commits, unavailable
sibling checkouts after snapshotting, missing pins, collisions and baseline
protection. Estimator contracts and parity remain each family's responsibility.
