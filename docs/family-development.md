# Family development

The family is the unit of implementation ownership. Integrated families live
in this repository, under `src/moxie` and/or `families`, and use its Git history.
Follow current dependency manifests and owner contracts for source locations;
retired sibling checkouts are not development inputs. A family may change its
API, algorithms, internal shared engines, contracts, tests, examples and build
configuration together. Internal project directories express reuse and
organization; they do not require separate authorization or sessions.
Integrate and validate changes within the family when needed. Preserve unrelated
work and coordinate overlapping edits with any session already in that family.

Use the local scikit-learn checkout as the API and correctness reference, never
as the implementation runtime. Keep the recorded reference revision, API parity,
pure-Mojo implementation, numerical evidence and existing scope requirements.
Family independence does not relax correctness or permit placeholder algorithms.

Families and algorithm child folders consume the current Moxie HEAD
for genuinely global framework contracts and shared infrastructure, even while
it is unstable. Use the live local checkout on every build, including working
source edits; no stable release, source pin, or separate adoption approval is
required. Validate consumers against those inputs and resolve compatibility
issues through their owning projects. Preserve copied baseline provenance and
common API compatibility. Family-specific helpers belong in the family, even
when a similar helper might eventually be useful elsewhere. Implement or revise them locally and validate their consumers;
a missing family helper is ordinary implementation work, not an owner handoff.
Request genuinely global additions through Moxie's COMMON-FAMILY-REQUIREMENTS.md.
Continue independent family work while an external capability is unavailable.

Each family controls its own build, test and parity commands and compatible
locked toolchain. Its working source, internal engines and local tests are live
development inputs: editing them does not require changing content pins. Keep
module/export manifests accurate when adding, removing or relocating modules;
retain namespace collision checks and explicit ownership overrides.

Local source dependencies are live development inputs. Shared assembly reads the
current files at each declared project path on every build; there are no source
revision/hash gates or cached historical fallbacks. Keep dependency paths, module
ownership, exports and license assets in `DEPENDENCIES.json`. Older v1 manifests
remain readable, but their `revision` and `files` pins are ignored. Some shared
projects retain the filename `DEPENDENCIES.lock.json` for consumer compatibility;
it names a live-source manifest, not a source lock. Missing source
checkouts or files raise an error. Generated receipts record the bytes consumed.

Retain compatible Pixi toolchain locks, correctness-reference revisions and
published baseline tags. Baseline proofs still protect deduplication of inherited
contracts. Validate affected consumers when changing shared contracts or
cross-family integration; keep genuinely global components in Moxie. Existing
consumers that bootstrap an old assembler must switch to the current shared tool
to adopt this workflow.

Use atomic commits with explicit paths for completed, validated changes within
the owning repository. Nested project folders use that repository's Git history;
creating nested Git repositories is unnecessary. Preserve published baseline
tags. Run relevant build and contract checks before committing code, and parity
checks for changed numerical/estimator behavior. Leave unrelated work untouched.

These rules supersede older child-only containment, immutable family-helper,
mandatory separate-session, local source repinning and blanket no-commit rules
in inherited AGENTS files, family guides and historical session prompts. Other
API, numerical, validation, licensing and safety requirements remain applicable.
