## Current family development policy

Implement each family independently in its own repository beside
`../scikit-learn`, the reference for correctness and public API surface.
Import `../moxie` for genuinely global framework components; keep
family-specific implementation and builds local.

For family work, read [docs/family-development.md](docs/family-development.md).
The whole family repository is the development boundary. That policy supersedes
older child-only containment, immutable family-helper, local source repinning
and blanket no-commit rules in inherited instructions and historical prompts.
Preserve global framework compatibility and the correctness requirements below.

# Moxie: rules for Astra and other contributors

## Mission and fixed interface

Port all of scikit-learn to pure Mojo, preserving its familiar estimator
ergonomics and public module organization. Read `docs/interfaces.md` before
changing an estimator interface and `docs/families.md` before starting a family.
The local `../scikit-learn` checkout is the reference; its revision is recorded
in `docs/interfaces.md`. Inspect the relevant source, tests, mathematical
documentation, and cited papers before implementing an algorithm.

Preserve scikit-learn names, constructor defaults, parameter meanings, learned
attributes ending in `_`, fit/predict/transform behavior, output shapes, class
ordering, and scoring semantics. Implementation strategies may differ. Every
language-driven interface difference must be explicit in the public docs and
covered by a contract test. Treat unsupported capabilities honestly: omit the
capability or raise a clear error; never return plausible placeholder results.

## Scaffold first, algorithms second

Preserve the derivation chain: Moxie mirrors scikit-learn's fundamental API;
each family copies that scaffold and refines only its family's contracts; each
algorithm starts from its family's scaffold and develops in a self-contained
project. These codebases stabilize independently. Retain explicit provenance
and compatible public interfaces so they can be integrated later; integration
is separate work. Adopt parent changes deliberately rather than coupling child
progress to the latest parent revision. Shared engines serve concrete reuse
needs; algorithm-specific implementation details belong in the algorithm.

Moxie is the shared foundation. The `scaffold-v1.1` tag identifies its frozen
starting point. Family repositories derive from that exact commit; algorithm
work derives from the appropriate family. Keep shared contracts unchanged in
family work. A required shared change belongs in moxie as a reviewed, tested,
versioned change, then is deliberately propagated to derivatives. Never move a
published baseline tag to disguise a change.

Mirror scikit-learn's package/module layout under `src/moxie`. Group related
algorithms through their real shared capabilities. Do not impose probability,
incremental fitting, transformation, or sample-weight support on estimators
that lack it. Keep constructor parameters separate from learned state. Clone
constructor configuration into an unfitted estimator; copying a fitted object
is a different operation. A normal fit replaces learned state; warm starts and
partial fits follow the individual estimator's documented rules.

## Implementation and evidence

Library code and algorithm kernels must be pure Mojo. Maximize reuse of Mojo's
standard library, NuMojo, and MAX before writing equivalent numerical code:
standard math/SIMD/algorithm primitives, NuMojo `NDArray` storage and suitable
array operations, and `max.algorithm` reductions and execution/parallelism. These
are authorized native runtime dependencies.

Before implementing a numerical operation or reporting a missing capability,
read `../session-plan/NUMERICS.md` and the shared numerical project's tested
operation policy, then inspect the pinned libraries for both complete operations
and composable primitives. Prefer a direct library call, then a bounded-memory
composition, then the smallest custom kernel needed for the remaining semantics.
Keep custom kernels on shared storage in their assigned owner. Record exact
APIs considered and concrete correctness, allocation or performance reasons for
exceptions; a missing convenience function or absent whitelist entry alone is
not evidence that library reuse is impossible. Validate new compositions in the
owning project; preserve operation-specific exclusions until evidence resolves
them. Reuse must meet the task's dtype, NaN, reduction-order, stride and memory
contracts. Enable SIMD/parallel execution where those contracts and measured
workloads support it.

Python, scikit-learn, NumPy, SciPy, and subprocesses are validation/tooling only;
they must not execute the library's learning or prediction. NuMojo's optional
Python conversion/I/O helpers are subject to the same restriction. Additional
third-party numerical runtime dependencies require a new user decision.
Use the mojo-syntax skill when writing Mojo and compile every changed module.

Before implementing an algorithm, gather its mathematical references and useful
specialized open-source implementations in the most specific local algorithm
directory, under `docs/private/<algorithm>/` or an ignored `references/` folder.
Record source URLs, exact revisions, licenses, and comparison commands there.
Retain third-party license notices. Use the local scikit-learn source first;
fetch other implementations when they provide an actual comparison target.

Validate observable behavior against scikit-learn: shapes, dtypes, ordering,
invalid inputs, unfitted errors, repeat fits, parameter changes, cloning,
determinism, and numerical edge cases. State tolerances and why they are valid.
Performance claims require correctness evidence and repeatable measurements.
Use existing tools: pyperf for benchmarks, perf for CPU profiling, Valgrind for
memory checks, and Callgrind for call costs. Keep tool versions, compiler flags,
hardware, seeds, thread settings, commands, and raw results with the local run.
Do not build replacements for these tools as part of the ML library.

## Repository hygiene and completion

`builds/`, `benchmarks/`, `profiling/`, `docs/private/`, and `references/` are
local working areas and must remain Git-ignored. Keep downloads, source mirrors,
papers, exploratory notes, raw measurements, and generated binaries there.
Tracked documentation must help users use or extend the library. Supply clear,
human-friendly examples and reproducible correctness tests. Keep development
worklogs and bulk research out of tracked docs.

Commit atomically: one completed, validated piece of work per commit, with only
its related files. Stage explicit paths and inspect the staged diff. Preserve
unrelated work. Run the project's build and contract tests before committing
code. Never present a scaffold or test double as an implemented algorithm, and
never claim full scikit-learn parity from compilation or a smoke test.
