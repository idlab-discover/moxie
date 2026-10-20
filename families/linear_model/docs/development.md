# Family development

Follow [Moxie's family development policy](../../../docs/family-development.md)
and [contributor rules](../../../AGENTS.md). Moxie owns this entire family,
including implementation modules, engines, child tests, documentation and builds.
There are no child-only ownership or separate-session requirements.

Use the current local shared contracts and live dependency manifests. Preserve
compatible toolchain locks, sklearn reference revision and provenance. Run the
focused root `linear-model-build`, `linear-model-test` and `linear-model-parity`
tasks for estimator changes. See [scope](scope.md) before extending support.
