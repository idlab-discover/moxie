# Feature extraction ownership

Follow [Moxie rules](../../AGENTS.md) and [family development](../../docs/family-development.md).
Before changing behavior, read [scope](docs/scope.md) and the affected child's
`docs/api.md`. The family owns both algorithm directories and its contracts.
Global dense, random and assembly helpers retain their Moxie owners.

Validate through the `feature-extraction-*` tasks in Moxie's root Pixi manifest.
Keep unsupported defaults explicit and preserve the reference revision and
compatible toolchain locks. Do not restore standalone source dependencies.
