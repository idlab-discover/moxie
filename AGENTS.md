# Moxie contributor rules

## Ownership and current sources

- Before family or child work, read [family development](docs/family-development.md).
  Families own their APIs, algorithms, helpers, tests and builds. Integrated
  families use this repository's Git history; global infrastructure belongs in
  Moxie's shared owners. Follow current `DEPENDENCIES.json` paths and owner
  contracts rather than retired sibling checkouts or historical request entries.
- Build against current local sources, including working edits. Preserve
  compatible toolchain locks, scikit-learn oracle revisions and baseline provenance.
  The family policy supersedes inherited containment, immutable-helper,
  separate-session, source-repinning and no-commit restrictions.
- Preserve unrelated work and coordinate overlapping edits. Record unresolved
  global needs in [common family requirements](COMMON-FAMILY-REQUIREMENTS.md)
  after checking the current owner contract. Update or remove resolved requests.

## Reference and API

Port scikit-learn to pure Mojo. Local `../scikit-learn` is correctness/API
reference; [interfaces](docs/interfaces.md) records its revision. Before changing
an estimator interface, read that contract. Before starting a family, read
[families](docs/families.md). Inspect relevant source, tests, math and papers
before implementing algorithms.

Preserve module layout under `src/moxie`, class names, constructor defaults,
parameter meanings, learned `_` attributes, shapes, dtype, class ordering,
fit/predict/transform behavior and scoring. Document language-driven differences;
cover each with contract tests. Unsupported capabilities: omit or raise clearly.
Never return plausible placeholders.

Keep constructor configuration separate from learned state. Clone configuration
into an unfitted estimator; fitted copies are separate. Ordinary fit replaces
state. Warm starts and partial fits follow each estimator's contract. Require
optional capabilities only where supported. Family contracts refine shared
contracts; algorithm details stay local. Shared contract changes belong in Moxie
with review/tests. Preserve scaffold provenance; historical tags are not build
pins.

## Native implementation and evidence

Learning/prediction: pure Mojo. Python, sklearn, NumPy, SciPy and subprocesses:
validation/tooling only, including NuMojo Python conversion/I/O helpers.

Before numerical work, read the
[shared numerical policy](shared/numerics/docs/numerics.md). Inspect libraries
actually consumed by the build. Prefer Numax CPU kernels; use Mojo stdlib,
NuMojo and MAX primitives. Use mojo-linalg when Numax cannot meet the contract.
Other numerical runtime dependencies require a user decision.

Prefer direct library operations, then bounded-memory compositions, then minimal
custom kernels in their shared owner. Use accepted NuMojo storage. Record exact
APIs considered and concrete correctness, allocation or performance reasons for
exceptions. Missing convenience functions or whitelist entries prove no blocker.
Preserve dtype, NaN, reduction-order, stride, ownership and memory contracts.
Validate new compositions at their owner; retain operation exclusions until
resolved by evidence. Enable SIMD/parallelism where contracts and measurements
support it. Use mojo-syntax skill; compile every changed Mojo module.

Before algorithm implementation, collect references/comparison sources under
local `docs/private/<algorithm>/` or ignored `references/`. Record URLs, exact
revisions, licenses and comparison commands; retain license notices.

Validate against sklearn: shapes/dtypes/order, invalid inputs, unfitted errors,
repeat fits, parameter changes, clone, determinism and numerical edges. Explain
tolerances. Performance claims require correctness and repeatable measurements:
pyperf, perf, Valgrind, Callgrind. Retain versions, flags, hardware, seeds, thread
settings, commands and raw results. Do not build substitute profiling tools.

## Persistence

Read [serialization](docs/serialization.md) before adding persistence.
Families own `ModelSchema`, state versions, validation, reconstruction and tests.
Call shared `save_model[Schema]` / `load_model[Schema]`; restore configuration and
learned state without refitting. Test fitted/unfitted round trips and invalid
state. Support is opt-in; format remains experimental.

## Completion

Keep `builds/`, `benchmarks/`, `profiling/`, `docs/private/`, `references/` ignored.
Tracked docs: current usage, extension contracts, reproducible tests/examples.
Research, downloads, raw measurements and worklogs stay untracked.

Commit completed, validated changes atomically. Stage explicit paths; inspect
staged diff. Run relevant build/contracts before committing code; parity for
changed numerical/estimator behavior. Compilation or smoke tests establish no
full sklearn parity. Scaffolds/test doubles are not implemented algorithms.
