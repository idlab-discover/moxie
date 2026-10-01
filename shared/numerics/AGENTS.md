# Shared numerics

Read `docs/numerics.md` before using numerical storage or operations. This project
owns the NuMojo evaluation and private `moxie._dense` kernels; parent source stays
unchanged. Adoption is selective: use only the documented operation subset and shared
fallbacks. New numerical capabilities need their own correctness and cost evidence.

Reuse `numojo.NDArray`, `numojo.Shape`, `numojo.Item` from the verified dependency,
and `moxie._dense.sum_axis2d` / `row_squared_norms` for the tested matrix kernels.
Use `tools/check.py` to verify the exact dependency source before compilation.
`DEPENDENCIES.lock.json` and `pixi.lock` define the single candidate toolchain.
No generic dense container or editable dependency fork belongs here.

Preserve raw research in `docs/private/numojo/`, measurements in `benchmarks/`
and `profiling/`. Run build, test and parity after native changes. Tests showing
candidate failures are evidence for excluding an operation, not full parity.
