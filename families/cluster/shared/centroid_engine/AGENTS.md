## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. The integrated cluster directory
is the ownership boundary within Moxie. That policy supersedes conflicting child-only,
read-only family-helper, separate-session, local-repinning and no-commit rules
below. Preserve external dependency ownership and all correctness requirements.

# Centroid engine

Before implementation, read ../../docs/plumbing.md and docs/scope.md, then the
workspace session-plan contracts. The family repository is the development boundary.

Consume current Moxie and family sources. Reuse
moxie.cluster._contracts: Labels, AssignmentResult, CenterUpdate, CentroidResult,
CentroidAssignment, CentroidUpdate, validate_centers and validate_labels.
Use numojo.NDArray, Shape and Item with Float32/Float64 values and Int64 labels;
copy owns data and view borrows shared storage. Reuse moxie._dense.sum_axis2d
and row_squared_norms where their documented semantics apply. Follow the
numerics owner's operation exclusions; use moxie._random.RandomState for RNG.
This project owns private centroid kernels only, not estimators or generic
storage, RNG, neighbor search, linkage or family contracts.

Compile with --fp-mode=contract=off. Build reads live sources without hash gates.
Run build, test and parity before completion. Save research and raw checks in
ignored docs/private/ and benchmarks/. Record unfinished work in the runner's
exact handoff note path.
