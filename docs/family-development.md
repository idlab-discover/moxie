# Independent family development

The family repository is the unit of development and ownership. A family may
change its root API, algorithms, internal shared engines, contracts, tests,
examples and build configuration together. Internal project directories express
reuse and organization; they do not require separate authorization or sessions.
Integrate and validate changes within the family when needed. Preserve unrelated
work and coordinate overlapping edits with any session already in that family.

Use the local scikit-learn checkout as the API and correctness reference, never
as the implementation runtime. Keep the recorded reference revision, API parity,
pure-Mojo implementation, numerical evidence and existing scope requirements.
Family independence does not relax correctness or permit placeholder algorithms.

Import Moxie for genuinely global framework contracts and shared infrastructure.
Preserve copied baseline provenance and common API compatibility. Family-specific
helpers belong in the family, even when a similar helper might eventually be
useful elsewhere. Implement or revise them locally and validate their consumers;
a missing family helper is ordinary implementation work, not an owner handoff.
Request genuinely global additions through Moxie's COMMON-FAMILY-REQUIREMENTS.md.
Continue independent family work while an external capability is unavailable.

Each family controls its own build, test and parity commands and compatible
locked toolchain. Its working source, internal engines and local tests are live
development inputs: editing them does not require changing content pins. Keep
module/export manifests accurate when adding, removing or relocating modules;
retain namespace collision checks and explicit ownership overrides.

External dependencies are deliberately adopted, reproducible inputs. Verify
consumed bytes and retain licenses and provenance; a dependency owner's newer
HEAD or unrelated edits must not invalidate an already adopted version. Cache
verified snapshots inside the family. If exact pinned bytes cannot be recovered,
report that specific dependency and deliberately adopt a validated replacement;
never silently refresh external pins during a build.

Existing cross-family reuse can consume a verified snapshot without requiring
its owner's live checkout or tests on each build. New cross-family integration
is separate work, not a prerequisite for unrelated family algorithms. Avoid
turning planned reuse into mandatory scheduling dependencies. Keep genuinely
global components in Moxie rather than creating competing framework copies.

Use atomic commits with explicit paths for completed, validated changes within
the family repository. Nested project folders use that repository's Git history;
creating nested Git repositories is unnecessary. Preserve published baseline
tags. Run relevant build and contract checks before committing code, and parity
checks for changed numerical/estimator behavior. Leave unrelated work untouched.

These rules supersede older child-only containment, immutable family-helper,
mandatory separate-session, local source repinning and blanket no-commit rules
in inherited AGENTS files, family guides and historical session prompts. Other
API, numerical, validation, licensing and safety requirements remain applicable.
