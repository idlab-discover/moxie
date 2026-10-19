# Model selection

Follow [Moxie contributor rules](../../AGENTS.md) and
[family development](../../docs/family-development.md). This directory and its
algorithm children belong to Moxie; the retired sibling is provenance only.

Before changing adapters or algorithms, read [plumbing](docs/plumbing.md),
[scope](docs/scope.md), and the relevant child contract. Preserve explicit
unsupported threshold/halving APIs and dense-only input rules.

Run `pixi run python tools/model_selection_checks.py all` from Moxie for
assembled contracts and local sklearn parity. Run
`pixi run python families/model_selection/tools/check.py test` for isolated
consumer ownership, negative compilation and family contracts.
