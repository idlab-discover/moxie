# Decomposition ownership

Follow [Moxie contributor rules](../../AGENTS.md) and
[family development](../../docs/family-development.md).
Moxie owns this family's sources, internal engines, tests, examples and docs.
Use the local scikit-learn revision recorded in docs/family.md.
Preserve unsupported modes in docs/scope.md and numerical tolerances in the guides.

Run `pixi run decomposition-test` and `pixi run decomposition-parity` from Moxie.
Run the SVD engine checks through `shared/svd_engine/tools/check.py`.
The optional Numax environment retains its Mojo 1.1/MAX 26.6 lock.
Historical scaffold manifests and tools under provenance are evidence only.
