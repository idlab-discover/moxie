# Shared dependency assembly

`tools/assemble.py` and `tools/assemble_family.py` use the same namespace
validator. They assemble current local sources on every build, without source
revision pins, hash gates, Git recovery or historical caches. Python is build
tooling only; the assembled library remains pure Mojo.

Keep owned modules under `src/moxie/`, dependency declarations in
`DEPENDENCIES.json`, and generated output under ignored `builds/deps`.
The manifest uses [schema 1](../schemas/assembly-v1.schema.json) with projects
and explicit overrides. Each project declares `id`, `path`, `modules`, `assets`
and `exports`. Paths resolve relative to the manifest. A module maps its full
`moxie.*` name to a source `file`; assets map output paths to source paths.
All transitive runtime imports and license assets must be declared.

```sh
python /path/to/moxie/shared/assembly/tools/assemble.py \
  --consumer /path/to/consumer \
  --lock /path/to/consumer/DEPENDENCIES.json \
  --output /path/to/consumer/builds/deps
mojo build -I /path/to/consumer/builds/deps /path/to/consumer/tests/main.mojo
```

The `--lock` argument remains for CLI compatibility; it now names a source
manifest. Old version-1 manifests with `revision` and `files` remain accepted;
those fields no longer constrain consumed content. The launchers verify the
specified tool path against the assembly project, without checking its hash.
Consumers with their own old-tool bootstrap must switch to the live tool.
Missing dependency directories or source files fail explicitly; there is no
fallback to a cached revision. Compatible Pixi toolchain locks are unchanged.

## Ownership and validation

Export manifests map each public symbol to its owning module. Every package
ancestor needs an explicit manifest. Generated initializers contain sorted
imports; arbitrary package initialization is unsupported. Module, symbol, asset
and file/directory collisions fail before output is replaced. Supported external
asset namespaces include `numojo`, `mojo_linalg`, `mojo_optimize`, `numax`,
`balsa`, `mojo_snappy`, and `licenses`.

Duplicate modules require an explicit override naming all displaced owners and
one winner. Identical inherited baseline modules are the exception: both must
name the same baseline commit and match its actual Git bytes. This baseline
provenance check protects shared contract deduplication; it is not a pin on a
dependency checkout's HEAD. Published baseline tags remain unchanged.

Source paths must be canonical relative paths within their declared project;
traversal and escaping symlinks fail. Output must be exactly the consumer's
Git-ignored `builds/deps`, without symlink redirection. Input bytes are read
before output replacement, cooperating writers are serialized, and a failed
validation preserves existing output. Successful assembly removes stale files.
`ASSEMBLY.json` records module ownership and hashes of the actual output bytes
for inspection; those hashes are receipts, not future build gates.

Assembly must finish before compilation. Concurrent source mutation during a
build is unsupported. This is build tooling for trusted repositories, not a
sandbox for executing untrusted code.

## Verification

Run `pixi run build`, `pixi run test`, and `pixi run parity`. Tests exercise local
and external edits, stale legacy metadata, missing sources, namespace ownership,
baseline proofs, path containment and output preservation. Two isolated native
consumers are compiled and compared exactly against direct imports in both
Float32 and Float64. This tests assembly and shared numerical contracts, not
estimator parity. No source pins need updating to run these checks.
