## Current family development policy

Implement each family independently in its own repository beside
`../scikit-learn`, the reference for correctness and public API surface.
Import `../moxie` for genuinely global framework components; keep
family-specific implementation and builds local.

Read [docs/development.md](docs/development.md) before development. The whole family repository
is the ownership boundary. That policy supersedes conflicting child-only,
read-only family-helper, separate-session, local-repinning and no-commit rules
below. Preserve external dependency ownership and all correctness requirements.

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

Library code and algorithm kernels must be pure Mojo. NuMojo and MAX (including
`max.algorithm`) are authorized native runtime dependencies. Before choosing
buffers or numerical operations, read `../moxie/AGENTS.md` and
`../session-plan/NUMERICS.md`; reuse the shared numerical project's tested
storage/operation policy. Custom buffers or replacement operations need evidence.
Python, scikit-learn, NumPy, SciPy, and subprocesses remain validation/tooling
only, including NuMojo's optional Python conversion/I/O helpers. Additional
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

## Neural-network family ownership (dense session contract)

Before any child work, read the mandatory global [Moxie guidance](../moxie/AGENTS.md),
[session contract](../session-plan/SESSION.md), [dense scope](../session-plan/SPARSITY-SCOPE.md),
then [plumbing](docs/plumbing.md) and [family interfaces](docs/family.md).
The runner's containment and leave-uncommitted rules override the commit guidance above.

Reuse `moxie.neural_network._plumbing` (LayerLayout, LayerTensors,
validate_matrix, snapshot_matrix, validate_incremental) and `_contracts`
(MLPKernel, MLPOptimizer, LBFGSObjective, RBMSampling), owned by this scaffold.
Preserve `_base` and foundation `moxie.base` / `moxie.utils.validation`.
`shared/mlp_engine` owns future `_mlp_engine`, `_activation_loss`,
`_stochastic_optimizers` and `_lbfgs`; both MLP estimators consume that engine.
`algorithms/mlp` owns `_multilayer_perceptron` and uses model_selection.splitters.
`algorithms/rbm` owns `_rbm`, including persistent contrastive divergence;
it shares storage/RNG with MLP, not MLP optimizers or backpropagation.
These are reserved owners, not existing child projects or implemented engines.

Use `numojo.NDArray[DType.float32/float64]`, `Shape`, `Item` directly under the
[numerics policy](../moxie/shared/numerics/docs/numerics.md). Accepted operations:
creation/copy/view, finite elementwise multiply/column subtraction, global sum,
mean with explicit returned_dtype, finite rank-2 matrix multiplication.
Use `moxie._dense.sum_axis2d` / `row_squared_norms` for axis sums/norms;
NuMojo axis sum, dot, variance, general solve/inverse/LU and RNG are excluded.
Use `moxie._random.RandomState` from the random owner; compile with
`--fp-mode=contract=off`. New activation/loss/optimizer kernels require measured,
operation-specific evidence in their owner session, retaining these arrays.
Array ownership/layout belongs to the scaffold, not individual estimators.
NuMojo and MAX are authorized native runtime dependencies; Python is tooling only.

Children flatten and pin every consumed dependency using foundation.assembly,
explicit EXPORTS.json and the lock format in docs/plumbing.md. Run build/test
with two isolated consumers after contract changes. Child numerical sessions
also require parity. Existing baseline sources and child projects stay intact.
