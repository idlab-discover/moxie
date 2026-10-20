## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. The integrated cluster directory
is the ownership boundary within Moxie. That policy supersedes conflicting child-only,
read-only family-helper, separate-session, local-repinning and no-commit rules
below. Preserve external dependency ownership and all correctness requirements.

# Affinity propagation

Follow the workspace session contracts and cluster family plumbing before edits.
This project owns only `moxie.cluster._affinity_propagation` and its public exports
`AffinityPropagation` and `affinity_propagation`. Commit validated changes atomically.

Reuse `numojo.NDArray`, `Shape`, `Item` for Float32/Float64 data and Int64 labels;
`moxie.cluster._contracts` validation and labels; `moxie._dense.sum_axis2d` and
`row_squared_norms`; `moxie._random.RandomState` for perturbations and
`moxie.utils.validation.FittedState` for lifecycle checks. Compile with
`--fp-mode=contract=off`. The family operation exclusions still apply.

Use live DEPENDENCIES.json manifests and the current Moxie namespace builder.
No pinned launcher, source hashes or repinning step is used.
Keep references and raw evidence under ignored docs/private and benchmarks.
Completion requires build, test, parity, API docs, example and measured baseline.
