# Shared dependency assembly, version 1

`tools/assemble.py` is the single dependency assembler for foundation projects,
family scaffolds and compact families. It is Python build tooling; all executed
native probes and inherited numerical code are pure Mojo. There is no library
runtime export from this project. The assembler uses Python's standard library
and Linux advisory file locking.

## Consumer setup

Keep owned modules in `src/moxie/`. Create `DEPENDENCIES.lock.json` with the
versioned [lock schema](../schemas/assembly-v1.schema.json), and ignore `builds/`.
The two complete examples are [consumer A](../tests/fixtures/consumer_a/DEPENDENCIES.lock.json)
and [consumer B](../tests/fixtures/consumer_b/DEPENDENCIES.lock.json). B represents
a compact family with an inherited module and overlapping baseline files.
Each has its own Pixi manifest/lock and can run `pixi run build` and `pixi run test`.

Use the exact Mojo 1.0.0 / MAX 26.5.0 / NuMojo combination from the completed
numerics owner. This project's Pixi environment reproduces that resolved lock;
no backend selection or numerical buffer implementation occurs here.

Pin the shared tool's project path, Git revision and file SHA-256 in an
`assembly` project entry, including `tools/assemble.py`, `tools/consume.py` and
both schemas. Run the shared launcher read-only, using explicit paths:

```sh
python /path/to/assembly/tools/consume.py \
  --consumer /path/to/consumer \
  --lock /path/to/consumer/DEPENDENCIES.lock.json \
  --tool /path/to/assembly/tools/assemble.py
mojo build -O3 -I /path/to/consumer/builds/deps \
  /path/to/consumer/tests/main.mojo -o /path/to/consumer/builds/test
```

The launcher checks the selected assembler's hash **before executing it**.
The assembler rechecks all pinned files and revisions, including its own tool
entry, on every invocation. Treat the launcher itself as trusted build tooling:
pin it in the consumer lock and review/verify that pin when updating the shared
tool. Do not copy an editable assembler into the consumer. A consumer may invoke
`assemble.py` directly after its own pre-execution SHA-256 check.

Compile with the single generated import root. Adding several partial `moxie`
include directories is not supported composition. The owner tool itself
bootstraps read-only from the completed numerics lock; consumers have no bootstrap
exception. `tools/pin.py` is an explicit maintainer command for this project's
fixtures, never a build step and never a general automatic update command.

## Lock and export contract

Paths in a project's `path` resolve relative to the lock file. They may refer
to sibling repositories or use absolute paths. File paths within projects and
asset destinations must be canonical relative paths: no absolute paths, `..`,
`.` segments, repeated separators or backslashes. Source symlinks may resolve
only within the specified project root. All pinned files must exist.

A lock is a flattened dependency graph: list the family, foundation baseline,
every transitive engine/algorithm actually used, the numerical source and the
tool. Include upstream dependency locks as hashed evidence. The assembler does
not discover or silently extend transitive dependencies. The compiler catches
missing imports. Authors must review the full closure before pinning.

Each project contains:

| Field | Meaning |
| --- | --- |
| `id` | Unique stable owner name |
| `path`, `revision` | Source root and Git revision (40 hexadecimal characters) |
| `files` | Relative filename to SHA-256 map; every consumed file must be pinned |
| `modules` | Full `moxie.*` module to `{ "file": "src/...mojo" }` |
| `assets` | Destination to pinned source filename, for `numojo/`, `mojo_linalg/`, `numax/` and `licenses/` |
| `exports` | Embedded version-1 manifest, or null for non-Moxie assets |

External repositories must be at the exact pinned HEAD. Projects in the same
Git repository as the consumer may pin an ancestor of HEAD: a lock cannot
contain the hash of its own commit. Every listed file still requires an exact
SHA-256 match, so later commits never authorize source drift implicitly.

The [export schema](../schemas/exports-v1.schema.json) is also usable standalone.
An embedded manifest is explicit data, for example:

```json
{"schema": 1, "packages": {
  "moxie": {"MyEstimator": "moxie.example"},
  "moxie.utils": {}
}}
```

Keys below each package are exported symbols; values name their owning modules.
The contributor must own each referenced module. Empty maps explicitly request
empty initializers. Every package ancestor needs a manifest entry. Generated
initializers contain sorted absolute imports only; they do not concatenate or
execute inherited initializer bodies. Pin original initializers as provenance
and explicitly inventory their exports. Package-level executable initialization
and symbol aliases are outside v1; put required logic in owned modules. Native
compilation verifies that exported symbols actually exist.

Duplicate JSON keys, unknown fields/schema versions, invalid identifiers,
module/package collisions and file/directory collisions are errors. The schema
specifies structural types; the tool additionally enforces ownership, provenance,
path containment and cross-reference constraints.

## Deduplication and relocation

A duplicate module is an error even if the bytes happen to match. The one
exception is an identical inherited baseline: both module entries must set
`baseline` to the same commit, and `git show BASELINE:REPO_FILE` must exactly
match each pinned source. Matching baseline exports are deduplicated too.
A baseline label alone cannot authorize deduplication.

For an authorized relocation, specify the exact module, all displaced owners,
the one replacement owner, and a nonempty reason:

```json
{"module": "moxie.dummy", "from": ["dummy-family"], "to": "consumer",
 "reason": "Relocate the inherited module to the owned implementation"}
```

Every named contributor must supply that module; unused, ambiguous or incomplete
overrides fail. The displaced source is still hash-verified. Exports referring
to its displaced module are removed, then replacement exports are applied.
Other export collisions remain errors. The original repository stays unchanged.
An override records an authorized relocation; it does not authorize a fork of
shared helpers. Consumer B replaces the real inherited dummy scaffold with a
small marker and proves the fixture-requested root estimator exports disappear.
The original dummy-family root initializer exports only base contracts; this
fixture explicitly requests two additional dummy re-exports to test their removal. The marker is
strictly a test double, not a Dummy implementation.

## Materialization and failure behavior

Output must be exactly `CONSUMER/builds/deps`, Git-ignored and free from symlink
redirection. The tool verifies and snapshots all bytes before creating output,
then writes a staging tree and replaces the generated namespace. Stale files
cannot survive a successful assembly. `ASSEMBLY.json` records the lock hash,
selected module owners and every output file hash. Invalid input leaves an
existing namespace unchanged and returns a nonzero status; consumers must stop
on failure rather than use an older tree.

Cooperating assembler writers are serialized. Assembly and compilation should
run sequentially: replacement is not a filesystem-wide atomic transaction, and
concurrent compilers or untrusted processes mutating inputs/output are outside
the contract. Process interruption during final replacement may require a rerun.
The tool is an integrity checker for trusted, reviewed locks, not a sandbox for
untrusted repositories or a cryptographic signature service.

## Validation and limits

`pixi run build` verifies dependencies and compiles all owned native modules,
the generated packages, two isolated consumers, the example and direct-import
controls. `pixi run test` runs host-side failure contracts and native tests.
The estimator-shaped probes exercise parent fitted-state checks, repeat fit,
clone independence, parameter changes, shared views versus owned copies, and
the same pinned dense kernels for Float32 and Float64.

`pixi run parity` compares both native consumers to direct imports from the
original numerics package and separately executes the original parent validation
contract. It uses exact equality on small exactly representable values. This is
assembly behavior parity, not sklearn estimator parity or another numerical
backend evaluation. The original parent and numerical projects are read-only.

Run `pixi run python tools/benchmark.py` after parity for a short pyperf baseline
of both fixture assemblies plus GNU time peak RSS. Results include process
startup, Git verification, hashing and full rematerialization; compilation is
excluded. Raw measurements and machine/tool metadata stay in ignored
`benchmarks/`. Space scales with pinned source bytes and generated copies; there
is no claim of zero-copy assembly or tuned performance.

Dense-only exclusions and executed checks are recorded in [scope.md](scope.md).

The initial short baseline on this Ryzen 9 5950X host measured approximately
108 ms (consumer A) and 111 ms (consumer B) per full assembly, with about 1.4 MB
of generated source each. These two-process, three-value samples are smoke
baselines; pyperf warned that the B sample is too small for a stable estimate.
